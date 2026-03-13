# host-sdk.4: Two additional examples (async-io, compute) with documentation
**Cycle**: 195 | **Theme**: host-sdk | **Kind**: experiment | **Status**: done

## Summary
Rewrote `examples/async-io/host` and `examples/vector-math/host` to use the
gpu-host SDK types (`GpuRuntime`, `HostcallBuffer`, `MappedBuffer<T>`).
Also fixed async-io kernel to use the new `Result<*mut u8, GpuError>` API,
and added all three examples to CI lint.

## Findings
### Q: What is the simplest async-io example demonstrating GPU file access?
A: The existing async-io example is ideal: write_pipeline (3 sequential files)
and transform_pipeline (read → uppercase → write). Updated host to use SDK types
and kernel to use Result-based hostcall API.
**Confidence**: high

### Q: What is the simplest compute example demonstrating GPU kernel execution?
A: The existing vector-math example covers three patterns: SAXPY (element-wise),
dot product (GPU mul + CPU sum), and softmax (multi-pass GPU-CPU cooperation).
Updated host to use `GpuRuntime` for all device operations. No hostcall needed.
**Confidence**: high

## Unexpected Discoveries
- async-io kernel was using the old tuple-based `gpu_hostcall_request` API that
  returned `(pkt, ok)`. Updated to new `Result<*mut u8, GpuError>` API.
- CI lint script only checked hello-gpu; added async-io and vector-math to both
  PTX kernel builds and host check steps.

## Open Questions
None.

## Impact on Downstream Tasks
- public-api criterion 4 ("At least 3 working examples") is now met
- host-sdk.5 can build on the pattern established across all 3 examples
