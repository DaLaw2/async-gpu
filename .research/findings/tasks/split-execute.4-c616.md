# split-execute.4: Create gpu-kernel-io crate

**Status**: DONE
**Cycle**: 616

## Summary

Created `crates/kernel/gpu-kernel-io/` crate containing the 3 I/O kernel modules
extracted from `gpu-kernel-std`:

- `hostcall_kernels.rs` — hostcall protocol tests, benchmarks, executor/channel demos,
  file I/O, trace, session, bulk I/O, buffered print, convergence, MPSC channel
- `pipeline.rs` — WarpFuture-based file transform pipeline, branching pipeline,
  pipelined I/O+compute, parallel grep
- `hybrid.rs` — hybrid executor (WarpFuture I/O + per-thread compute blocks)

## What was done

1. Created `crates/kernel/gpu-kernel-io/` with:
   - `Cargo.toml` — cdylib crate, depends on gpu-atomics, gpu-libc, gpu-protocol,
     gpu-runtime, gpu-kernel-core
   - `.cargo/config.toml` — nvptx64 target config (identical to gpu-kernel-core)
   - `src/lib.rs` — feature flags, extern crate gpu_kernel_core, mod declarations,
     force-link stdio anchors
   - `src/hostcall_kernels.rs` — copied from gpu-kernel-std
   - `src/pipeline.rs` — copied from gpu-kernel-std
   - `src/hybrid.rs` — copied from gpu-kernel-std

2. Fixed warnings in the new crate:
   - Removed unused `use gpu_protocol::*` wildcard import from `hybrid.rs`
     (code uses fully-qualified `gpu_protocol::` paths)
   - Removed unused `PP_WAIT_COMPUTING` constant from `pipeline.rs`
   - Removed unused `asm_experimental_arch` feature flag from `lib.rs`
   - Removed unused `stdio_auto_init()` function from `lib.rs` (IO kernels
     use raw hostcall protocol, not std println!)

3. Updated `gpu-kernel-std/src/lib.rs`:
   - Removed `mod hostcall_kernels`, `mod hybrid`, `mod pipeline` declarations
   - Added comment noting these modules moved to gpu-kernel-io

4. Verified:
   - `cargo build --release` in gpu-kernel-io: PTX produced, 55 kernel entries
   - `cargo build --release --target nvptx64-nvidia-cuda` in gpu-kernel-std: still works
   - `cargo fmt -- --check` in gpu-kernel-io: clean

## Import analysis

- `pipeline.rs` imports `crate::hybrid::{hybrid_warp_print_init, hybrid_warp_wait}` —
  works correctly since both modules are in the same crate
- `hostcall_kernels.rs` imports from `gpu_kernel_core::helpers::*` — resolved via
  the gpu-kernel-core dependency
- No cross-crate import issues

## Files changed

- NEW: `crates/kernel/gpu-kernel-io/Cargo.toml`
- NEW: `crates/kernel/gpu-kernel-io/.cargo/config.toml`
- NEW: `crates/kernel/gpu-kernel-io/src/lib.rs`
- NEW: `crates/kernel/gpu-kernel-io/src/hostcall_kernels.rs`
- NEW: `crates/kernel/gpu-kernel-io/src/pipeline.rs`
- NEW: `crates/kernel/gpu-kernel-io/src/hybrid.rs`
- MODIFIED: `crates/kernel/gpu-kernel-std/src/lib.rs` (removed 3 mod declarations)
- DELETED: `crates/kernel/gpu-kernel-std/src/hostcall_kernels.rs`
- DELETED: `crates/kernel/gpu-kernel-std/src/pipeline.rs`
- DELETED: `crates/kernel/gpu-kernel-std/src/hybrid.rs`
