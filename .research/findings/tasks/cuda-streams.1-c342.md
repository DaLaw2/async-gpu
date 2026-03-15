# cuda-streams.1: cudarc stream API and async launch patterns
**Cycle**: 342 | **Theme**: cuda-streams | **Kind**: investigation | **Status**: done

## Summary
cudarc 0.12.1 has CudaStream support but LaunchAsync does NOT accept a stream parameter — kernels always target the default stream. The project uses zero CUDA streams currently. Adding stream support requires lower-level driver API calls or cudarc upgrade. The hostcall safety model depends on device-level sync, which complicates per-stream synchronization.

## Findings

### Q: How does cudarc expose CUDA streams?
A: cudarc 0.12.1 has a `CudaStream` type for multi-stream operations, with `CudaStream::synchronize()` for stream-specific sync and `CudaDevice::synchronize()` for device-level sync. Default is single-stream model where all operations serialize on the default stream.
**Confidence**: high

### Q: Can LaunchAsync::launch target a specific stream?
A: **NO.** The current cudarc 0.12.1 `LaunchAsync::launch()` has no stream parameter — it always uses the default stream. Targeting specific streams would require using lower-level cudarc driver APIs (raw `cuLaunchKernel` with stream param) or upgrading cudarc to a newer version.
**Confidence**: high

### Q: How does stream synchronization differ from device synchronization?
A: Device sync (`dev.synchronize()`) waits for ALL pending GPU work across all streams. Stream sync (`stream.synchronize()`) waits only for work on that specific stream. async_gpu uses device sync everywhere, which is safer but prevents overlapping operations.
**Confidence**: high

### Q: What's the interaction between streams and hostcall listener?
A: The hostcall listener requires **device-level idle** before resetting packets (safety comment: "Must only be called after cuCtxSynchronize()"). Stream-specific sync alone is insufficient for hostcall packet reset — the device-idle invariant must be maintained. This is the fundamental tension: streams enable overlap, but hostcall safety needs full device quiescence between kernel launches.
**Confidence**: high

## Unexpected Discoveries
- Zero CUDA stream usage in the entire project — all "stream" references are TCP streams or tokio channels
- The hostcall packet reinit safety model is the real blocker for stream adoption, not cudarc API limitations
- Even with stream support, overlapping kernels would conflict on shared hostcall buffers unless each stream gets its own buffer

## Open Questions
- Can we decouple hostcall packet reset from device sync? (e.g., per-stream packet pools)
- Would cudarc 0.13+ add stream-targeted launch?
- Is the complexity justified given that most GPU workloads are compute-bound, not launch-bound?

## Impact on Downstream Tasks
- CUDA stream support is harder than expected due to hostcall safety model
- Recommend: experiment task should focus on non-hostcall kernels first (pure compute overlap)
- Full hostcall+streams integration requires architectural changes (per-stream packet pools)
