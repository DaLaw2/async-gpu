# native-api: gpu::run() one-liner launcher
**Epic**: native-rust-dx | **Status**: active | **Updated**: 2026-06-04

## Progress
- gpu::run(), gpu::run_with_output(), gpu::compute() implemented in gpu.rs
- examples/std/thread-demo created demonstrating one-liner API
- extern "gpu-kernel" ABI verified working via patched rustc + gpu_kernel_abi feature
- MIR pass auto-applies to ALL async fn on nvptx64 (no #[warp_cooperative] needed)

## Verified Conclusions
- gpu::compute(name, n, threads) wraps device init + PTX load + launch + sync
- GpuHostError::KernelNotFound requires &'static str lifetime
- AUTO_BUILD_KERNEL=0 needed when using patched-rustc PTX (build.rs overwrites otherwise)

## Rejected Approaches
- None yet

## Open Questions
- How many examples need rewriting? (native-api.2 still active)
- Should old ptx-kernel examples be kept for backward compat or fully replaced?

## Key Metrics
- gpu::compute() verified: one-liner launches kernel and returns results

## Next Steps
- native-api.2: Rewrite all examples/ to use gpu::run() native Rust style
