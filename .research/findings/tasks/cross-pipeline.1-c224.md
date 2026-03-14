# cross-pipeline.1: Cross-launch device buffer demo
**Cycle**: 224 | **Theme**: cross-pipeline | **Kind**: experiment | **Status**: done

## Summary
Demonstrated cross-launch data persistence: pipeline_writer_kernel writes [100,200,...,800] to mapped memory, pipeline_reader_kernel reads and multiplies by 3. Both share the same HostcallSession. Zero host-side copy — both kernels operate on the same device-visible mapped buffer.

## Findings

### Q: Can two kernels share a device buffer across launches?
A: Yes. Mapped memory (cuMemHostAlloc with DEVICEMAP) persists across kernel launches. After cuCtxSynchronize(), the written data is visible to subsequent kernels. No explicit copy needed — both kernels receive the same device pointer.
**Confidence**: high

### Q: Does Pipeline API correctly handle multi-stage launches?
A: Yes. The Pipeline::new().stage(writer).stage(reader).run() pattern correctly reinits hostcall packets between stages and shuts down the session after all stages complete. Results verified: [300,600,900,1200].
**Confidence**: high
