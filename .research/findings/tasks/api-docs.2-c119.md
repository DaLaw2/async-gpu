# api-docs.2: Add cfg stubs to gpu-runtime and gpu-atomics for doc generation on x86_64
**Cycle**: 119 | **Theme**: api-docs | **Kind**: experiment | **Status**: done

## Summary

Made gpu-atomics and gpu-runtime doc-buildable on x86_64 by gating PTX inline assembly and `core::arch::nvptx` intrinsics behind `cfg(target_arch = "nvptx64")`. Non-nvptx targets get stub implementations that panic at runtime but allow `cargo doc`, clippy, and type-checking to work. All 4 public crates (gpu-protocol, gpu-atomics, gpu-runtime, warp-macro) now build docs on x86_64 with stable toolchain.

## Findings

### Q: Can cfg stubs make gpu-runtime and gpu-atomics doc-buildable on x86_64?
A: Yes. Two changes were needed:
1. **Feature gates**: `#![feature(...)]` → `#![cfg_attr(target_arch = "nvptx64", feature(...))]`
2. **Function bodies**: Each inline PTX asm block wrapped with `#[cfg(target_arch = "nvptx64")]`, with a `#[cfg(not(...))]` stub that panics. A `gpu_stub!()` macro reduces boilerplate.
**Confidence**: high

### Q: What functions need stubbing and what are the right stub signatures?
A: **gpu-atomics**: All 17 public functions (membar_sys, sys_store_release_u32/u64, sys_load_acquire_u32/u64, sys_cas_u32/u64, sys_fetch_add_u32/u64, sys_exchange_u64, sys_spin_load_acquire_u32/u64, activemask, lane_id, syncwarp, shfl_sync_idx_u32, st_global_u32). Each uses `core::arch::asm!` with PTX register classes (reg32, reg64).

**gpu-runtime**: 3 inline asm uses (trap, nanosleep) + 4 functions using `core::arch::nvptx::_block_idx_x()` / `_thread_idx_x()`. Solved with `nvptx_shim` module: real impl wraps the intrinsics, stub returns 0.
**Confidence**: high

## Changes Made
- **crates/gpu-atomics/src/lib.rs**:
  - Feature gate: `cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))`
  - Added `gpu_stub!()` macro for non-nvptx stubs
  - All 17 functions: `#[cfg(target_arch = "nvptx64")]` for asm block, `#[cfg(not(...))]` for stub
  - Added `#![allow(clippy::missing_safety_doc)]` (pre-existing issue)
- **crates/gpu-runtime/src/lib.rs**:
  - Feature gates: `cfg_attr` for both `stdarch_nvptx` and `asm_experimental_arch`
  - Added `nvptx_shim` module with `block_idx_x()` / `thread_idx_x()` wrappers
  - Replaced all `core::arch::nvptx::*` calls with `crate::nvptx_shim::*`
  - 3 `core::arch::asm!` calls gated with `cfg(target_arch = "nvptx64")`
  - Added `Default` impl for `PanicBuf` (clippy requirement)
  - Added `#![allow(clippy::missing_safety_doc)]` (pre-existing issue)

## Verification
- `cargo doc --no-deps` succeeds for all 4 public crates on x86_64
- GPU kernel still compiles to PTX (nvptx64 target)
- All GPU tests still pass on hardware
