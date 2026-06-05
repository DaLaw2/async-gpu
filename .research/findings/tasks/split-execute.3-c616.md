# split-execute.3 — Create gpu-kernel-compute crate

**Kind**: experiment
**Status**: done

## What was done
Created `crates/kernel/gpu-kernel-compute/` with 9 compute modules moved from
gpu-kernel-std: compute_cnn, compute_demo, compute_fused, compute_gemm,
compute_mma (sm_80 feature-gated), compute_persistent, compute_physics,
compute_search, compute_transformer.

## Key decisions
- cdylib only (no rlib) — no downstream dependents
- Removed `stdio_auto_init()` and `gpu-libc` dep — none of the compute kernels
  use hostcall I/O, so these were dead code
- Kept `#[used]` stdio force-link statics (required by patched std PAL)
- Preserved `#[cfg(feature = "sm_80")]` gate on compute_mma module

## Verification
- `cargo build --release` in gpu-kernel-compute: 84 entry points in PTX
- `cargo build --release --target nvptx64-nvidia-cuda` in gpu-kernel-std: still builds
- `cargo fmt --check`: clean
- No `crate::` references in any moved file — all imports use `gpu_kernel_core::helpers::*`
