# cuda-streams.3: GpuStream wrapper + integration test
**Cycle**: 349 | **Theme**: cuda-streams | **Kind**: experiment | **Status**: done

## Summary

Implemented `GpuStream` wrapper in `crates/core/gpu-host/src/streams.rs` and a
passing integration test `test_cuda_stream_launch`. The wrapper provides
`launch()`, `synchronize()`, and `join_default()` methods around cudarc's
`CudaStream`. All 5 integration tests pass.

## Findings

### Q: Does cudarc's `launch_on_stream()` work with mapped memory pointers?
A: Yes, but the kernel parameter count must match exactly. `launch_on_stream()`
validates parameters more strictly than `launch()` on the default stream. Passing
1 parameter to a 2-parameter kernel works on the default stream (CUDA silently
ignores the missing param) but fails with `CUDA_ERROR_INVALID_VALUE` on a
non-default stream.
**Confidence**: high (verified empirically)

### Q: Does `CudaStream::Drop` affect subsequent CUDA operations?
A: Yes. cudarc's `CudaStream::Drop` calls `device.wait_for(self)` then
`stream::destroy()`. After the stream is destroyed, subsequent raw CUDA API calls
(like `cuMemHostAlloc` in `HostcallBuffer::new`) may fail with
`CUDA_ERROR_INVALID_CONTEXT` if the thread-local CUDA context isn't re-bound.
Fix: call `device.bind_to_thread()` before raw CUDA API operations.
**Confidence**: high

## Unexpected Discoveries

1. **Parameter strictness difference**: CUDA's `cuLaunchKernel` on a non-default
   stream is stricter about parameter validation than on the default stream. This
   is undocumented behavior.

2. **Context invalidation on stream destroy**: After `cuStreamDestroy_v2`, the
   thread-local CUDA context can become invalid for raw CUDA API calls, even
   though `CudaStream::Drop` calls `bind_to_thread()` internally. The issue is
   that the context binding doesn't persist reliably across test boundaries.

## Impact on Downstream Tasks

- **cuda-streams theme**: COMPLETED — all 3 criteria met
- **production-ready epic**: ALL 4 CRITERIA NOW MET
- `shared_device()` test helper now calls `bind_to_thread()` to prevent context
  issues between tests
