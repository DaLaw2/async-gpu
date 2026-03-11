# Review rv1: atomics.3 — Performance
**Verdict**: issues_found

## Summary

The `gpu-atomics` crate implements system-scope GPU atomic primitives using inline PTX `asm!`
blocks. The core instruction choices (`st.release.sys`, `ld.acquire.sys`, `atom.cas.sys`) are
semantically correct and represent the right level of the PTX memory model for GPU-CPU
communication. However, several performance issues exist: a redundant `membar.sys` after
`st.release.sys` in the integration kernel adds unnecessary serialization cost, the host-side
polling loop omits `std::hint::spin_loop()` causing CPU efficiency loss, and the mapped-memory
allocation omits `CU_MEMHOSTALLOC_PORTABLE` which may limit performance on multi-GPU or NUMA
systems. None of these are correctness bugs, but they do impose measurable overhead that matters
for the hostcall protocol this work is building toward.

---

## Issues Found

### Issue 1: Redundant `membar.sys` after `st.release.sys` (GPU side)

**Location**: `crates/gpu-kernel/src/lib.rs` lines 154 and `crates/gpu-atomics/src/lib.rs`
`kernel_sys_store_and_signal` lines 242-246.

Both copies of the integration kernel perform:
```
st.release.sys.global.u32  [data_ptr], value   // (1)
membar.sys;                                     // (2)  ← redundant
st.release.sys.global.u32  [flag_ptr], 1       // (3)
```

`st.release.sys` (instruction 1) already carries full release semantics at system scope: all
prior stores are guaranteed to be visible to any observer in the system before store (1) itself
becomes visible. Adding `membar.sys` between (1) and (3) does not strengthen the guarantee
because (3) is itself another `st.release.sys`, which again provides a release fence before (3)
is visible. The `membar.sys` between them is therefore fully subsumed.

`membar.sys` on SM86 maps to a global memory barrier that stalls the warp until all outstanding
memory requests in the memory subsystem have drained. On Ampere this costs on the order of
hundreds of cycles per warp. For a flag-signal operation that will be invoked on every hostcall,
this gratuitous stall adds latency directly to the critical path.

**Recommendation**: Remove `membar.sys;`. The correct pattern is:
```
st.release.sys.global.u32 [data_ptr], value;
st.release.sys.global.u32 [flag_ptr], 1;
```
The release on the flag store is the architectural guarantee that the data write is ordered
before it. The `membar_sys()` helper function itself is correct and should be retained for
cases where a standalone fence is genuinely needed (e.g., before a relaxed atomic).

### Issue 2: Host polling loop lacks `std::hint::spin_loop()`

**Location**: `crates/gpu-host/src/main.rs` lines 326-334.

```rust
for i in 0..TIMEOUT_ITERS {
    flag_val = flag_atomic.load(Ordering::Acquire);
    if flag_val == 1 {
        // ...
        break;
    }
    // no spin_loop hint
}
```

On x86-64 a tight spin loop without a `PAUSE` instruction (which `std::hint::spin_loop()`
emits) wastes power, degrades SMT sibling performance, and can saturate the memory bus with
redundant load traffic. The PAUSE instruction also prevents memory ordering violations that the
CPU would otherwise need to handle on the fast path.

For mapped pinned memory, the CPU is making PCIe reads (or NVLink reads) on each iteration.
Without PAUSE, the CPU issues these as fast as the out-of-order engine allows. With PAUSE,
the hardware throttles slightly, which can actually reduce total latency to flag observation
by reducing contention on the interconnect.

**Recommendation**:
```rust
for i in 0..TIMEOUT_ITERS {
    flag_val = flag_atomic.load(Ordering::Acquire);
    if flag_val == 1 { break; }
    std::hint::spin_loop();
}
```

### Issue 3: `CU_MEMHOSTALLOC_PORTABLE` not set

**Location**: `crates/gpu-host/src/main.rs` line 377.

```rust
let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP;
```

`CU_MEMHOSTALLOC_DEVICEMAP` (0x02) makes the allocation accessible from the GPU. However, by
default the pinned mapping is only valid for the CUDA context that allocated it. On a system
with multiple GPUs or where the CUDA context migrates between devices, the device pointer
obtained via `cuMemHostGetDevicePointer_v2` may be invalid for another context.

More importantly for performance: without `CU_MEMHOSTALLOC_PORTABLE` (0x01), the allocation is
not marked as "portable" across CUDA contexts. If the hostcall protocol is later extended to
use multiple streams or contexts (which is likely for an async runtime), the absence of this
flag will require re-registration or re-mapping. Setting it now costs nothing.

**Recommended flags**:
```rust
let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
```

### Issue 4: Per-kernel PTX re-loading in smoke tests

**Location**: `crates/gpu-host/src/main.rs` lines 122, 71, 304.

`cudarc::nvrtc::Ptx::from_src(KERNEL_PTX)` and `dev.load_ptx(...)` are called repeatedly
across `run_write_thread_idx`, `run_vector_add`, `run_asm_smoke_tests`, and
`run_integration_sys_store`. `from_src` on the PTX string is cheap (it wraps a `&str`), but
`load_ptx` calls into the CUDA driver each time. The subsequent calls use `let _ = ...` to
silently swallow the "already loaded" error, which suggests this was noticed but never fixed.

For a test harness this is not critical, but for the hostcall protocol the module should be
loaded once at driver initialization time and function handles cached.

**Recommendation**: Load the PTX module once in `main` and pass the device handle (with the
module already loaded) to all sub-functions, or use a lazy-static / `OnceLock` for the module
and function handles.

### Issue 5: Three separate `mov` instructions for thread index in `gpu-atomics` kernel

**Location**: `crates/gpu-atomics/src/lib.rs` lines 222-238.

The `kernel_sys_store_and_signal` kernel reads `%tid.x`, `%ctaid.x`, and `%ntid.x` via three
separate inline `asm!` blocks, each producing one register:
```rust
core::arch::asm!("mov.u32 {tid}, %tid.x;", ...);
core::arch::asm!("mov.u32 {ctaid}, %ctaid.x;", ...);
core::arch::asm!("mov.u32 {ntid}, %ntid.x;", ...);
```

These are three virtual register allocations that will each spill to a 32-bit slot if the
compiler is already under register pressure. The PTX register file is wide (255 registers per
thread on SM86), so spilling is unlikely here. However, the intent could be expressed more
cleanly—and identically to the `gpu-kernel` crate—by using `nvptx::_thread_idx_x()` etc., or
by consolidating into a single `asm!` block that reads all three specials. This is a minor
style/maintenance concern rather than an actual performance bottleneck.

The `gpu-kernel` crate correctly uses `nvptx::_thread_idx_x()` (via `stdarch_nvptx`) while
`gpu-atomics` reimplements this with raw `asm!` because `gpu-atomics` does not import
`stdarch_nvptx`. This inconsistency may be intentional (keeping `gpu-atomics` dependency-free),
but it should be documented.

---

## Positive Observations

1. **Correct instruction selection for GPU-CPU communication**: `st.release.sys.global.u32`
   and `ld.acquire.sys.global.u32` are the correct PTX instructions for SM70+ GPU-to-CPU
   flag passing. The semantics exactly match the producer-consumer pattern needed for hostcall.

2. **`#[inline(always)]` is appropriate**: All primitives (`sys_store_release_u32`,
   `sys_load_acquire_u32`, `sys_cas_u32`, `sys_fetch_add_u32`, `membar_sys`) are leaf
   operations with a single PTX instruction body. Forcing inlining ensures zero call overhead
   and allows the compiler to fold adjacent operations without call frame setup. This is the
   correct annotation here.

3. **`options(nostack)` on all asm blocks**: The `nostack` option signals to LLVM that the
   inline asm does not touch the stack pointer, enabling better code generation in surrounding
   code and avoiding unnecessary stack frame allocation. All blocks correctly carry this option.

4. **`options(readonly)` on load operations**: `sys_load_acquire_u32` and
   `sys_load_acquire_u64` both carry `options(readonly)`, which allows LLVM to reason that the
   asm block does not write memory. This is correct and enables better instruction scheduling
   around the load.

5. **`st.release.sys` over `st.volatile + membar.sys`**: The codebase uses the modern PTX 7.x
   `st.release.sys` instruction rather than the older `st.volatile` + `membar.sys` two-step.
   On Ampere (SM86), `st.release.sys` is architecturally preferred: it is implemented as a
   single L2-coherent operation rather than two serializing operations. This is the correct
   modern approach.

6. **Host-side `Ordering::Acquire` polling**: The host uses `AtomicU32::load(Ordering::Acquire)`
   rather than `read_volatile` or a relaxed load. This is correct: it creates the acquire half
   of the GPU release / CPU acquire synchronization pair, ensuring the data write is visible
   after the flag is seen.

7. **`atom.cas.sys.global.b32` register usage**: The CAS operation passes `expected` and
   `desired` as separate `in(reg32)` constraints, which maps exactly to the PTX three-operand
   form. No register aliasing issues.

8. **NVVM intrinsic fallbacks**: Providing `nvvm_membar_sys` and `nvvm_atomic_add_sys_i32` as
   separately testable alternatives gives valuable comparison data and a fallback path if inline
   asm compilation regresses in a future toolchain version.

---

## Verdict Rationale

The fundamental instruction choices are correct and represent the minimal-overhead path for
system-scope GPU-CPU communication on SM86. No correctness bugs were found. The verdict is
`issues_found` rather than `pass` due to:

- **Issue 1** (redundant `membar.sys`): This is a performance regression on the critical path
  of every future hostcall signal. The extra barrier adds hundreds of cycles of serialization
  with zero benefit over the already-present release semantics. It should be removed before
  the hostcall protocol is built on top of this primitive.
- **Issue 2** (`spin_loop` hint): Low cost to fix, measurable benefit in a polling-heavy
  hostcall runtime.
- **Issue 3** (missing `PORTABLE` flag): Zero cost to add, prevents future correctness issues
  in multi-context scenarios.

Issues 4 and 5 are minor and do not block progress.
