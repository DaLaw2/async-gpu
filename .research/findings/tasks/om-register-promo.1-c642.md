# om-register-promo.1: PTX Diff Analysis — Register vs Memory Placement

## Summary

Compiled 10 test kernels (owned vs borrowed values, loops, structs, arrays,
call chains) to PTX at both O1 and O3+LTO. Analyzed when LLVM promotes values
to registers vs spills to local memory (per-thread DRAM).

**Core finding**: Rust ownership semantics (move/copy vs borrow) directly
determines register placement on GPU. Borrowing forces values to local memory
because a reference requires an address; owned/moved values stay in registers.
At O3+LTO, LLVM can inline across call boundaries to recover register
placement for borrowed scalars, but NOT for non-inlined call sites or
multi-field structs.

## Findings

### 1. Owned scalar values -> pure register computation

`case1_owned_values` (let x=42; y=17; compute): LLVM constant-folded the
entire chain to a single `st.volatile.global.b32 [%rd2], 135`. Zero local
memory, zero intermediate registers. Same result at O1 and O3.

`case9_chain_owned` (value passed by-value through 3 non-inlined functions):
Constant-folded to `st.volatile.global.b32 [%rd2], 460` at both O1 and O3.
Even with `#[inline(never)]`, LLVM performed inter-procedural constant
propagation through pass-by-value parameters.

### 2. Borrowed scalars -> forced local memory spill

`case2_borrowed_values` (take &c, pass to non-inlined function): Allocates
`.local .align 4 .b8 __local_depot[4]` — 4 bytes of local memory. The value
must be materialized at an address for the reference. Even at O3+LTO, the
local memory remains because `consume_ref` uses `read_volatile` (opaque to
optimizer).

`case10_chain_ref` (&v passed through 3 non-inlined ref functions):
- O1: Allocates `.local .align 4 .b8 __local_depot[12]` (3 x 4-byte slots),
  one per intermediate value. Each intermediate result is stored to local
  memory, then a pointer is passed to the next function.
- O3+LTO: **Fully optimized away** — constant-folded to
  `st.volatile.global.b32 [%rd2], 460`. LTO inlined all three functions,
  eliminated the references, and constant-propagated.

**Key insight**: LTO can recover register placement for borrowed scalars IF it
can inline the callee. Without LTO or with truly opaque callees (FFI,
volatile, cross-crate without LTO), borrowing permanently prevents register
promotion.

### 3. Inner loops — owned vs borrowed is critical

`case3_loop_owned` (owned values in loop): Zero local memory. All loop
variables live in registers. LLVM unrolled the loop (4x at O1, 8x at O3)
using pure register arithmetic (`shl.b32`, `add.s32`). 29 registers at O1,
41 at O3 (more unrolling).

`case4_loop_borrowed` (&doubled passed to function in loop): Allocates
`.local .align 4 .b8 __local_depot[4]`. Each loop iteration:
1. Computes `doubled` in a register (`shl.b32`)
2. Spills to local memory (`st.local.b32`)
3. Passes pointer to `consume_ref`
4. Function does `ld.volatile.b32` from that address

At O3+LTO, the loop was still unrolled (4x) but **every unrolled iteration**
repeats the st.local + call sequence. The `st.local` in the hot loop is a
~100-cycle penalty per iteration (local memory = per-thread DRAM).

### 4. Structs — ABI forces local memory regardless of ownership

`case5_struct_owned` (Vec3 passed by value to `dot_owned`): Despite being
"owned", allocates `.local .align 4 .b8 __local_depot[24]` (2 x 12 bytes for
two Vec3 structs). The PTX calling convention passes structs > 1 register via
pointer — the caller materializes the struct in local memory and passes the
address.

`case6_struct_ref` (Vec3 passed by &ref to `dot_ref`): Same local memory
allocation (24 bytes). The callee body is identical — loads fields via
pointer offsets.

**Key insight**: For multi-field structs, owned vs borrowed makes NO difference
at the call site because the PTX ABI serializes both through memory. The only
way to avoid this is inlining (which both O1 and O3+LTO failed to do here due
to `#[inline(never)]`).

### 5. Arrays — owned enables full constant folding; borrowed forces memory

`case7_array_owned` ([u32; 4] never address-taken): LLVM constant-folded the
sum to `st.volatile.global.b32 [%rd2], 100`. Zero local memory.

`case8_array_borrowed` (&[u32; 4] passed to non-inlined function): Allocates
`.local .align 4 .b8 __local_depot[16]`. All 4 elements stored to local
memory via `st.local.b32`, pointer passed to callee. Even at O3+LTO, not
optimized away (callee not inlined).

## Quantitative Summary

| Case | Ownership | Local Mem (O1) | Local Mem (O3+LTO) | Registers (O1) | Registers (O3+LTO) |
|------|-----------|----------------|--------------------|-----------------|--------------------|
| 1: scalar owned | owned | 0 B | 0 B | %rd<3> | %rd<3> |
| 2: scalar borrowed | &ref | 4 B | 4 B | %r<3>, %rd<5> | %r<3>, %rd<5> |
| 3: loop owned | owned | 0 B | 0 B | %r<29>, %rd<12> | %r<41>, %rd<8> |
| 4: loop borrowed | &ref | 4 B | 4 B | %r<25>, %rd<10> | %r<39>, %rd<14> |
| 5: struct owned | owned | 24 B | 24 B | %r<2>, %rd<7> | %r<2>, %rd<7> |
| 6: struct ref | &ref | 24 B | 24 B | %r<2>, %rd<7> | %r<2>, %rd<7> |
| 7: array owned | owned | 0 B | 0 B | %rd<3> | %rd<3> |
| 8: array borrowed | &ref | 16 B | 16 B | %r<2>, %rd<5> | %r<2>, %rd<5> |
| 9: chain owned | owned | 0 B | 0 B | %rd<3> | %rd<3> |
| 10: chain ref | &ref | 12 B | 0 B | %r<4>, %rd<9> | %rd<3> |

## Actionable Rules for Register Promotion

1. **Prefer pass-by-value for scalars** in inner loops. Borrowing a u32/f32
   forces a st.local per use — ~100 cycle penalty on GPU.

2. **Structs bypass ownership optimization** because the PTX ABI serializes
   them via pointer. For hot paths, destructure into scalar fields and pass
   individually, or rely on `#[inline(always)]`.

3. **LTO can recover** borrowed-scalar register placement when callees are
   inlineable. But volatile reads, FFI, and `#[inline(never)]` block this.

4. **Arrays work like scalars when never borrowed** — LLVM can constant-fold
   or keep elements in individual registers. Taking &arr forces all elements
   to local memory.

5. **For GpuRef patterns**: the `.read(i)` method returns a Copy value (not a
   reference), which is correct — the loaded value stays in a register. No
   changes needed to GpuRef itself.

## Open Questions

1. **What about larger structs (>64 bytes)?** The PTX ABI likely always uses
   local memory for these. Destructuring may be impractical — investigate
   whether `#[inline(always)]` on small methods is sufficient.

2. **Register pressure at scale**: Our test kernels use <41 registers. Real
   kernels with many live variables may see register spilling even with owned
   values. Need to investigate the register-pressure vs occupancy tradeoff.

3. **Cross-crate boundaries**: GpuRef methods are `#[inline(always)]`, which
   should enable register promotion. But if a user writes a non-inlined
   function that takes &GpuRef, the GpuRef struct itself (ptr + len + phantoms)
   gets serialized to local memory. Worth documenting this pitfall.
