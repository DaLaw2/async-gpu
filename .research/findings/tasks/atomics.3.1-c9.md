# atomics.3.1: Rework gpu-atomics crate — fix review issues (rv1)
**Date**: 2026-03-11
**Cycle**: 9
**Theme**: atomics
**Kind**: experiment
**Status**: done
**Spawned by**: rv1

## Summary

Addressed all critical and structural issues from the rv1 code review of atomics.3.
The gpu-atomics crate is now a proper reusable library (rlib) with no panic handler
or embedded kernels, and gpu-kernel depends on it to eliminate PTX duplication.

## Changes Made

### C1: Removed panic handler from gpu-atomics (CRITICAL)
- `gpu-atomics/src/lib.rs`: Removed `#[panic_handler]` and `use core::panic::PanicInfo`
- Also removed `#![feature(abi_ptx)]` (not needed — no kernel entry points in library)
- Panic handler remains only in `gpu-kernel/src/lib.rs` (the binary crate)

### C2: Fixed use-after-free on timeout (CRITICAL)
- `gpu-host/src/main.rs`: Added `dev.synchronize()` before `cuMemFreeHost` in timeout path
- Ensures GPU kernel has retired before freeing mapped memory

### C3: Fixed kernel argument mismatch (CRITICAL)
- `gpu-host/src/main.rs`: Changed from 4 args `(data, flag, value, 32u32)` to 3 args
  `(data, flag, value)` matching `integration_sys_store`'s 3-parameter signature

### S1: Eliminated PTX duplication (STRUCTURAL)
- `gpu-kernel/Cargo.toml`: Added `gpu-atomics = { path = "../gpu-atomics" }` dependency
- `gpu-kernel/src/lib.rs`: All test kernels now call `gpu_atomics::*` functions instead
  of duplicating inline PTX asm blocks
- `gpu-atomics/Cargo.toml`: Changed `crate-type = ["cdylib"]` to `["rlib"]`

### S2: Moved test kernel out of library (STRUCTURAL)
- Removed `kernel_sys_store_and_signal` from `gpu-atomics/src/lib.rs`
- `integration_sys_store` in `gpu-kernel` now uses `gpu_atomics::sys_store_release_u32`

### S3: Removed redundant membar.sys (PERFORMANCE)
- `integration_sys_store` no longer has `membar.sys` between two `st.release.sys` stores
- Verified in PTX output: only `st.release.sys.global.u32` instructions, no `membar.sys`
- Saves hundreds of cycles per hostcall signal on Ampere

### S4: Internalized NVVM intrinsics (STRUCTURAL)
- Changed `pub fn nvvm_membar_sys` and `pub fn nvvm_atomic_add_sys_i32` to `pub(crate)`
- Added comment explaining these are kept as fallback/comparison, not public API

### Additional fixes from review
- Added `st_global_u32()` helper to gpu-atomics for explicit `.global` address space stores
- `test_asm_cas_sys` now uses `st_global_u32()` instead of plain Rust deref (fixes Issue 4)
- `test_asm_ld_acquire_sys` now uses `st_global_u32()` for consistency
- Added `std::hint::spin_loop()` to host polling loop (Performance Issue 2)
- Added `CU_MEMHOSTALLOC_PORTABLE` flag to mapped memory allocation (Performance Issue 3)
- Added documentation note about `readonly` hazard on acquire loads (Correctness Issue 7)

## Verification

### Compilation
- `gpu-kernel` compiles successfully on nvptx64-nvidia-cuda with `-Zbuild-std=core`
- `gpu-host` compiles successfully on host target
- No duplicate panic handler errors

### PTX Output Verification
- `integration_sys_store`: 2× `st.release.sys.global.u32`, no `membar.sys` between them ✓
- `test_asm_cas_sys`: `atom.cas.sys.global.b32` + `st.global.b32` (no generic store) ✓
- `test_asm_ld_acquire_sys`: `ld.acquire.sys.global.u32` + `st.global.b32` ✓
- All gpu-atomics functions properly inlined (no call instructions) ✓
- All kernel entry points present: 10 kernels ✓

## Deferred Items (not blocking)
- `SysAtomic<T>` safe wrapper type → defer to hostcall.3 design phase
- u64 CAS/fetch_add + exchange primitive → defer to hostcall.3 design phase
- PTX module caching in host → defer to hostcall.4 experiment

## Theme Progress
The atomics theme now has a clean, reusable library crate (`gpu-atomics`) that
downstream crates (`gpu-kernel`, and future `gpu-hostcall`) can depend on without
conflict. The critical path to hostcall.3 is unblocked.
