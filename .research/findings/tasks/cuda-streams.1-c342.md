# cuda-streams.1: cudarc stream API and async launch patterns
**Cycle**: 342 | **Theme**: cuda-streams | **Kind**: investigation | **Status**: done

## Summary
cudarc 0.12.1 has full CudaStream support including `launch_on_stream()` for targeting specific streams and `dev.wait_for(&stream)` for per-stream sync. The project uses zero CUDA streams currently. The hostcall safety model depends on device-level sync, which complicates per-stream synchronization for hostcall-using kernels, but pure compute kernels can use streams immediately.

## Findings

### Q: How does cudarc expose CUDA streams?
A: cudarc 0.12.1 has a `CudaStream` type for multi-stream operations, with `CudaStream::synchronize()` for stream-specific sync and `CudaDevice::synchronize()` for device-level sync. Default is single-stream model where all operations serialize on the default stream.
**Confidence**: high

### Q: Can LaunchAsync::launch target a specific stream?
A: **YES!** cudarc 0.12.1 has `LaunchAsync::launch_on_stream(&CudaStream, config, params)` — a separate method that launches on a specific stream. The default `launch()` uses the device's default stream. Streams are created via `CudaDevice::fork_default_stream()`. Also `dev.wait_for(&stream)` for per-stream sync. Full example in `cudarc/examples/04-streams.rs`.
**Confidence**: high (verified in cudarc source)

### Q: How does stream synchronization differ from device synchronization?
A: Device sync (`dev.synchronize()`) waits for ALL pending GPU work across all streams. Stream sync (`stream.synchronize()`) waits only for work on that specific stream. async_gpu uses device sync everywhere, which is safer but prevents overlapping operations.
**Confidence**: high

### Q: What's the interaction between streams and hostcall listener?
A: The hostcall listener requires **device-level idle** before resetting packets (safety comment: "Must only be called after cuCtxSynchronize()"). Stream-specific sync alone is insufficient for hostcall packet reset — the device-idle invariant must be maintained. This is the fundamental tension: streams enable overlap, but hostcall safety needs full device quiescence between kernel launches.
**Confidence**: high

## Unexpected Discoveries
- Zero CUDA stream usage in the entire project — all "stream" references are TCP streams or tokio channels
- **CORRECTION**: cudarc DOES have `launch_on_stream()` — initial investigation missed this. Full stream API exists.
- The hostcall packet reinit safety model is the real blocker for stream adoption for hostcall-using kernels
- Pure compute kernels can use streams immediately with no architectural changes

## Open Questions
- Can we decouple hostcall packet reset from device sync? (e.g., per-stream packet pools)
- Would cudarc 0.13+ add stream-targeted launch?
- Is the complexity justified given that most GPU workloads are compute-bound, not launch-bound?

## Impact on Downstream Tasks
- CUDA stream support is harder than expected due to hostcall safety model
- Recommend: experiment task should focus on non-hostcall kernels first (pure compute overlap)
- Full hostcall+streams integration requires architectural changes (per-stream packet pools)
