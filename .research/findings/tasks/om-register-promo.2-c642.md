# om-register-promo.2: Register Promotion Speedup in Inner-Loop Kernel

## Summary

Wrote three kernels demonstrating the register promotion spectrum for
GpuRef::read() (by-value) vs slice access vs volatile pointer access.
Compiled to PTX at O1 and O3+LTO. Confirmed that GpuRef::read() produces
optimal codegen: a single `ld.shared.u32` that keeps the value in a register
across all loop iterations. The volatile case shows 1024x more memory
traffic — one `ld.volatile.b32` per iteration.

## Findings

### 1. Three-kernel comparison

| Kernel | Load Pattern | Load Instruction | Loads in Loop | Unroll (O1) | Unroll (O3) |
|--------|-------------|------------------|---------------|-------------|-------------|
| regpromo_byval_loop | GpuRef::read() → by-value | `ld.shared.u32` | 0 | 8x | 16x |
| regpromo_slice_loop | as_generic_slice()[0] | `cvta.shared` + `ld.b32` | 0 | 8x | 16x |
| regpromo_reload_loop | read_volatile(ptr) | `ld.volatile.b32` | 1/iter | 4x | 8x |

### 2. GpuRef::read() — optimal codegen (1 instruction)

PTX before the inner loop:
```
ld.shared.u32 %r7, [%rd13];   // address-space-specific load, 1 instruction
```

The inner loop uses `%r7` as a pure register operand. No memory traffic.
LLVM unrolled 8x (O1) or 16x (O3+LTO). Each unrolled iteration is just:
```
cvt.rn.f32.u32  %r8, %r41;    // convert loop counter to f32
mul.rn.f32      %r9, %r7, %r8; // multiply with register-held value
add.rn.f32      %r10, %r40, %r9; // accumulate
```

Total for 1024 iterations: **1 memory load, 0 in loop body**.

### 3. Slice access — address conversion overhead

PTX before the inner loop (4 instructions vs 1 for byval):
```
mov.u64 %rd17, dynamic_smem;
add.u64 %rd17, %rd17, %rd18;  // shared-space base
cvta.shared.u64 %rd19, dynamic_smem; // convert to generic space
sub.s64 %rd20, %rd13, %rd17;  // compute offset
add.s64 %rd21, %rd19, %rd20;  // generic-space pointer
ld.b32  %r2, [%rd21];         // generic load (not ld.shared!)
```

The inner loop body is **identical** to byval — LLVM successfully hoisted the
load. But the load setup requires 4 extra address conversion instructions,
and the load itself is a generic `ld.b32` instead of `ld.shared.u32`.

At O1, the slice kernel uses 5 more 64-bit registers (%rd<35> vs %rd<30>)
to hold the address conversion intermediates.

### 4. Volatile reload — worst case (1024 loads)

PTX inner loop (4x unrolled at O1, 8x at O3+LTO):
```
$L__BB118_5:  // Inner Loop Header
ld.volatile.b32  %r7, [%rd4];     // RELOAD from memory
cvt.rn.f32.u32   %r8, %r28;
mul.rn.f32       %r9, %r7, %r8;
add.rn.f32       %r10, %r27, %r9;
ld.volatile.b32  %r11, [%rd4];    // RELOAD again (next unrolled iteration)
...
```

**Each unrolled iteration contains a `ld.volatile.b32`.** The loop cannot
hoist the load because volatile semantics require a fresh memory access.

Total for 1024 iterations: **1024 memory loads in the loop body**.

LLVM also unrolled less aggressively (4x at O1 vs 8x for byval/slice)
because each iteration has a memory dependency that limits ILP.

### 5. Quantitative summary

| Metric | byval (GpuRef::read) | slice (as_generic_slice) | volatile (read_volatile) |
|--------|---------------------|--------------------------|-------------------------|
| Memory loads total | 1 | 1 | 1024 |
| Loads in loop body | 0 | 0 | 1 per iteration |
| Load instruction | `ld.shared.u32` | `ld.b32` (generic) | `ld.volatile.b32` |
| Setup instructions | 1 | 6 | 6 |
| Loop unroll (O1) | 8x | 8x | 4x |
| Loop unroll (O3) | 16x | 16x | 8x |
| 32-bit registers | %r<42> | %r<42> | %r<29> |
| 64-bit registers | %rd<30> | %rd<35> | %rd<35> |
| Local memory | 0 | 0 | 0 |

### 6. Why GpuRef::read() is optimal by design

The type system naturally promotes values to registers because:

1. **Returns by value** (`fn read(&self, i: usize) -> T`): The loaded
   value is a plain `T` (Copy), not a reference. The compiler has no
   obligation to keep it in memory — it goes straight to a register.

2. **Address-space-specific load**: Because GpuRef stores a raw
   shared-space pointer (not converted via cvta.shared), the load emits
   `ld.shared.u32` — a single instruction that the GPU hardware routes
   directly to the SRAM bank, bypassing the generic address resolution.

3. **#[inline(always)]**: The read() method is inlined, so the register
   assignment happens at the caller's scope. No function call overhead,
   no ABI-forced spilling.

The slice path (`as_generic_slice()`) deliberately converts to generic
address space — that's the documented "escape hatch" behavior. It works,
but at the cost of address conversion instructions and a generic-space load.

## Open Questions

1. **Multi-value hoisting**: When multiple GpuRef::read() calls happen
   before a loop (e.g., loading 4 tile values), do they all stay in registers?
   Register pressure could force spilling if too many values are live.

2. **Non-scalar types**: GpuRef::read() returns T by value. For T=u64 or
   T=f64, this uses `ld.shared.u64` — still a single instruction. But for
   hypothetical struct types, the PTX ABI would serialize via local memory
   (same issue found in om-register-promo.1).
