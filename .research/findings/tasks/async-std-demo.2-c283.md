# async-std-demo.2: End-to-end async file pipeline on GPU
**Cycle**: 283 | **Theme**: async-std-demo | **Kind**: experiment | **Status**: done

## Summary
The existing `examples/async-pipeline` runs end-to-end successfully. `#[warp_cooperative] async fn data_pipeline` reads a file, transforms data (uppercase), and writes output — all via hostcall Futures with warp-level convergence barriers. Input: "Hello from GPU async pipeline!" → Output: "HELLO FROM GPU ASYNC PIPELINE!". Verification PASSED.

## Findings

### Q: Does the async file pipeline work end-to-end?
A: **Yes.** Full pipeline execution:
1. Host creates `pipeline_input.txt` (30 bytes: "Hello from GPU async pipeline!")
2. GPU kernel `async_data_pipeline` launches with `block_on(data_pipeline(buf))`
3. `data_pipeline` (marked `#[warp_cooperative]`) executes:
   - `GpuOpenFuture::new(buf, "pipeline_input.txt", FILE_OPEN_READ).await` → fd=1
   - `GpuReadFuture::new(buf, fd, &mut buffer).await` → 30 bytes
   - Transform: uppercase each byte
   - `GpuOpenFuture::new(buf, "pipeline_output.txt", FILE_OPEN_WRITE_CREATE).await` → fd=2
   - `GpuWriteFuture::new(buf, fd, &data).await` → 30 bytes written
   - `GpuCloseFuture::new(buf, fd1).await` + `GpuCloseFuture::new(buf, fd2).await`
4. Host verifies `pipeline_output.txt` = uppercased input ✓

**Confidence**: high

### Q: PTX characteristics?
A:
- 7x `bar.warp.sync` — one at each `.await` yield point
- 1x `shfl.sync.idx.b32` — discriminant broadcast from lane 0
- PTX version 7.8, target sm_86

**Confidence**: high

### Q: Does this satisfy async-std C2 and C3?
A: **Yes.**
- **C2** (PAL async bridge / warp-cooperative I/O): The kernel uses `#[warp_cooperative] async fn` with GpuXxxFuture types. Each `.await` inserts `bar.warp.sync` for warp convergence. I/O operations yield to the scheduler between steps.
- **C3** (practical demo): Complete data pipeline: read file → compute → write file, with I/O yielding at every step. End-to-end verified with correct output.

Note: The PAL (std::fs::File) remains synchronous. The async I/O is at the kernel level using explicit Future types. This is the pragmatic design — std's sync API cannot return Futures.

**Confidence**: high

## Impact on Downstream Tasks
- **async-std-demo.3**: Can be merged with this task — PTX already verified, end-to-end already passed
- **async-std epic**: C2 and C3 criteria met. Combined with C1 (Futures exist) and C4 (codebase reorg done), all criteria satisfied.
