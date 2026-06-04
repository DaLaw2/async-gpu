# native-api: gpu::run() one-liner launcher + clean examples
**Epic**: native-rust-dx | **Status**: active | **Updated**: 2026-06-04

## Progress
- gpu::run(), gpu::run_with_output(), gpu::launch() one-liners (done)
- 3/7 hostcall examples converted to one-liner API (done)
- gpu::custom() builder API implemented: CustomLaunchBuilder -> GpuContext -> GpuResult (done)
- All 4 remaining hostcall examples rewritten to builder API (done)
- std/* examples assessed: cannot use builder API (CUDA C compile, iterative multi-launch)
- extern "gpu-kernel" ABI migration COMPLETE: 213 sites across 34 files

## Verified Conclusions
- Builder API covers all hostcall patterns: pure compute, hostcall+sideband, mapped buffers, custom PTX
- Two-phase prepare()/launch() preserves cudarc compile-time tuple type safety
- Pointer extraction pattern (u64 before launch) solves borrow-after-move cleanly
- std/* examples need different abstractions: device sharing, re-launch, NVRTC compilation

## Key Metrics
- Converted examples: 7/7 hostcall (3 one-liner + 4 builder), 1/1 std (thread-demo)
- Builder API surface: 4 types, ~250 lines, zero new dependencies
- Example line reduction: 9-15% with clearer intent
