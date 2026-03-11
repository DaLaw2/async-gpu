# Review rv1: atomics.3 — Correctness
**Verdict**: issues_found

## Summary

The gpu-atomics crate implements system-scope GPU atomic primitives using inline PTX assembly
and NVVM intrinsic fallbacks. The PTX instructions chosen are correct for the SM70+ target and
the release/acquire qualifiers are semantically sound. However, several correctness hazards exist
in the integration test: the host-side polling loop uses Rust `AtomicU32::load(Acquire)` on
mapped memory without any CPU-side memory fence that is guaranteed to be observed across the
PCIe/NVLink boundary; the kernel is launched with arguments that do not match the kernel
signature; the mapped memory is freed before verifying the data value in the timeout-error path;
and the CAS smoke test writes back via a plain Rust dereference rather than through the
`.global` address space, which may generate a non-global store. Multiple warp behaviour and
alignment are not validated anywhere.

---

## Issues Found

### Issue 1: Kernel argument mismatch between host and `integration_sys_store` kernel

In `gpu-host/src/main.rs` line 317 the host passes **four** arguments to `integration_sys_store`:
```rust
f.launch(cfg, (data_u64, flag_u64, EXPECTED_VALUE, 32u32))
```
But `gpu-kernel/src/lib.rs` declares `integration_sys_store` with only **three** parameters:
```rust
pub unsafe extern "ptx-kernel" fn integration_sys_store(
    data_ptr: *mut u32,
    flag_ptr: *mut u32,
    value: u32,
)
```
The extra `32u32` argument (presumably `thread_count`) is passed to a kernel that has no
corresponding parameter. At runtime this is likely ignored by the PTX ABI (extra registers
are simply not read), but it is misleading and fragile. If the PTX calling convention places
extra arguments in registers that overlap with something else, behaviour is undefined. The
`kernel_sys_store_and_signal` in `gpu-atomics/src/lib.rs` does accept four parameters and
contains the `thread_count` guard, but the host never calls that kernel; it calls the
three-parameter `integration_sys_store`.  The guard `idx == 0` inside `integration_sys_store`
still limits writes to thread 0, so the functional effect is correct for this single-block
launch, but the API contract is violated and the mismatch should be fixed.

### Issue 2: Host-side acquire load does not guarantee cross-device visibility

The CPU polls `flag_atomic.load(Ordering::Acquire)` where `flag_atomic` is an `AtomicU32`
aliased over mapped (pinned) memory. Rust's `Ordering::Acquire` compiles to an x86 plain load
(x86 TSO makes all loads acquire-loads at the hardware level), so no `MFENCE` / `SFENCE` is
emitted. For intra-CPU and intra-GPU coherence this is sufficient, but the CUDA programming
model requires that the CPU issue a **CUDA stream synchronization** (e.g., `cuStreamSynchronize`
or equivalent) or use a `CU_MEMHOSTALLOC_WRITECOMBINED`-aware fence before observing GPU writes
to mapped memory on non-coherent platforms. On architectures with an IOMMU or where the GPU's
L2 cache is not automatically flushed to host-visible DRAM without a device-side `membar.sys`,
the CPU may spin indefinitely or read a stale value even after the GPU has committed the store.

The GPU side is correct: it uses `st.release.sys` followed by `membar.sys` then another
`st.release.sys` for the flag. However, the CPU side relies entirely on hardware cache
coherence with no call to `cuStreamSynchronize` before or during the poll loop. On discrete
GPUs without hardware cache coherence (pre-Ampere or non-NVLink), this can silently fail.

A correct pattern is either:
- Use `cuStreamSynchronize` after launch (which flushes GPU caches), then read with a volatile
  or atomic load; or
- Keep the spin-poll but add `std::sync::atomic::fence(Ordering::Acquire)` *and* rely on the
  platform guarantee that `CU_MEMHOSTALLOC_DEVICEMAP` memory is cache-coherent (only guaranteed
  on NUMA/unified-memory capable systems such as Tegra or when `CU_DEVICE_ATTRIBUTE_INTEGRATED`
  is true).

As written, correctness is hardware-dependent and should be documented as a constraint, or the
approach should be made explicit with a `cuStreamSynchronize` path as a fallback.

### Issue 3: Use-after-free of mapped memory in timeout error path

In `run_integration_sys_store`, on timeout the code frees both host pointers and then returns
an error:
```rust
unsafe {
    free_mapped_mem(data_host_ptr)?;
    free_mapped_mem(flag_host_ptr)?;
}
anyhow::bail!("Integration test TIMEOUT: ...");
```
However, the GPU kernel has already been launched asynchronously. After `cuMemFreeHost` returns,
the GPU thread 0 may still be executing and writing to the now-freed (and potentially remapped)
host memory. This is a use-after-free race. The correct fix is to call `cuStreamSynchronize`
(or `dev.synchronize()` via cudarc) before freeing, ensuring the GPU kernel has fully retired.

### Issue 4: `test_asm_cas_sys` kernel writes result via plain Rust dereference

In `gpu-kernel/src/lib.rs` lines 78-79, after the inline PTX CAS the result is written back
via a plain Rust pointer dereference:
```rust
*output = result;
```
There is no `.global` address-space qualifier on this store. For nvptx64, Rust's default
dereference emits a generic-address-space store (`st.b32` without `.global`). This works if
the pointer is indeed in global memory (which it will be when passed from the host), but it is
inconsistent with the rest of the code that explicitly uses `st.global.b32` via inline PTX.
If LLVM's address-space inference fails to resolve the generic pointer to `.global`, PTX
validation can fail. This should be written as an explicit `st.global.b32` inline PTX store
(matching the pattern used in `test_asm_ld_acquire_sys`) for consistency and safety.

### Issue 5: `ld.acquire.sys` on a `CU_MEMHOSTALLOC_DEVICEMAP` buffer is valid only for `.global` address space — but `integration_sys_store` is only a producer kernel

This is a minor documentation/semantic issue: `gpu-kernel/src/lib.rs` never calls
`sys_load_acquire_u32` from the gpu-atomics crate; it re-implements the inline PTX directly.
The gpu-atomics crate functions are therefore not exercised in the integration path and are
only tested by the smoke tests. This gap means the actual crate-level API is not integration-
tested end-to-end.

### Issue 6: No alignment verification for mapped pointers

`alloc_mapped_u32` allocates exactly `size_of::<u32>()` (4 bytes) via `cuMemHostAlloc`. The
PTX ISA requires that addresses used with `st.release.sys.global.u32` and
`ld.acquire.sys.global.u32` be **naturally aligned** (4-byte aligned for u32). `cuMemHostAlloc`
is documented to return page-aligned memory, so in practice this is satisfied, but there is no
assertion or debug check confirming alignment. For the u64 variants this would require 8-byte
alignment. Adding a debug assertion (`assert_eq!(ptr as usize % align_of::<T>(), 0)`) would
make the requirement explicit and catch future bugs.

### Issue 7: `readonly` option on `ld.acquire.sys` asm block may suppress necessary reloads

`sys_load_acquire_u32` and `sys_load_acquire_u64` in `gpu-atomics/src/lib.rs` both specify
`options(nostack, readonly)`. The `readonly` option tells LLVM the asm block does not write
memory, which is correct for a load. However, combined with `#[inline(always)]`, LLVM may
hoist the load out of a polling loop if it concludes the value cannot change (since the Rust
memory model does not see the GPU writing to the same location). In the integration kernel
this does not matter (the acquire load is called only once from the GPU side), but if
`sys_load_acquire_u32` were ever used inside a GPU-side spin loop waiting for a host-written
flag, the `readonly` hint could cause LLVM to cache the result in a register and never re-read
memory. A comment documenting this hazard should be added, or `readonly` should be dropped for
the acquire variants.

---

## Positive Observations

- **PTX instruction selection is correct.** `st.release.sys.global.u32`, `ld.acquire.sys.global.u32`,
  `atom.cas.sys.global.b32`, `atom.add.sys.global.u32`, and `membar.sys` are all valid PTX 7.x
  instructions for SM70+ and their semantics match what the code claims. The `.sys` scope is the
  correct choice for GPU-CPU shared memory communication.

- **Register classes are correct.** All pointer operands use `reg64` (64-bit registers, matching
  `nvptx64` pointer size) and all u32 value operands use `reg32`. The u64 store and load
  correctly use `reg64` for the value as well. There are no obvious register-class mismatches.

- **The release/acquire semantic pairing is sound.** Writing data first with `st.release.sys`,
  then issuing `membar.sys`, then writing the flag with `st.release.sys` is a belt-and-suspenders
  approach that correctly orders data visibility before flag visibility at system scope.

- **NVVM intrinsic fallback names are plausible.** `llvm.nvvm.membar.sys` and
  `llvm.nvvm.atomic.add.gen.i.sys.i32.p0i32` are documented NVVM intrinsics and the extern "C"
  linkage convention is the correct way to call them from nvptx64 Rust.

- **Panic handler is minimal and correct** for a `no_std` GPU crate: an infinite loop is the
  only sane option since there is no stack unwinding or OS support.

- **Error propagation in the host** is thorough: every CUDA API call checks its result code
  and returns an error, and mapped memory cleanup is attempted on the error paths (though see
  Issue 3 about the ordering).

- **The smoke tests cover every primitive** individually before the integration test, making it
  easy to isolate which instruction failed during debugging.

---

## Verdict Rationale

The core PTX primitives are correctly implemented and the memory ordering theory is sound. The
issues are concentrated in the host-side integration test: a kernel argument count mismatch
(Issue 1), a potential GPU-to-CPU cache coherence gap that is hardware-dependent and
undocumented (Issue 2), a use-after-free race on timeout (Issue 3), and a non-uniform store
path in one smoke-test kernel (Issue 4). None of these require a redesign of the atomic
primitives themselves, but Issues 2 and 3 are genuine correctness bugs that can produce wrong
results or crashes on real hardware. The verdict is **issues_found** rather than needs_rework
because the crate API itself is sound; the integration test harness requires targeted fixes.
