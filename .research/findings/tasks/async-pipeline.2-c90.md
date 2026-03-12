# async-pipeline.2: File transform demo — hand-written WarpFuture pipeline
**Cycle**: 90 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Implemented a 16-state WarpFuture (`FileTransformFuture`) that demonstrates GPU-autonomous multi-step pipeline execution. In a single kernel launch with zero CPU intervention between steps, the GPU: opens input file → reads 1024 bytes via sideband → toggles ASCII case per-thread → opens output file → writes transformed data → closes both files → prints status. All I/O is warp-cooperative, compute is per-thread divergent.

## Findings
### Q: Does the GPU-autonomous pipeline work end-to-end?
A: Yes. Hardware verified:
- 16-state machine: OPEN_IN → WAIT → BULK_READ → WAIT → COMPUTE → OPEN_OUT → WAIT → BULK_WRITE → WAIT → CLOSE_IN → WAIT → CLOSE_OUT → WAIT → PRINT → WAIT → DONE
- 8 I/O hostcall round-trips + 1 per-thread compute phase
- 1024 bytes (32 lanes × 32 bytes) case-toggled correctly
- Total time: ~4.2ms (including all hostcall round-trips)
- Host log shows sequential I/O operations without any CPU-side orchestration

**Confidence**: high

### Q: In-place sideband processing viable?
A: Yes. After BULK_READ, all lanes read/modify/write their 32-byte slices directly in the sideband buffer. BULK_WRITE then sends from the same sideband offset. No extra copy needed — cleaner than the per-thread API which copies to/from a separate buffer.

**Confidence**: high

### Q: Borrow checker compatibility with closure-based submit?
A: Requires extracting fields (fd, offset, etc.) into local `Copy` variables before passing to `warp_hostcall_submit`. The closure captures locals by value, avoiding mutable borrow conflicts with `&mut self.state` and `&mut self.pkt_idx`.

**Confidence**: high

## Unexpected Discoveries
- The demo is fully functional with hardcoded file paths (`gpu_input.txt` / `gpu_output.txt`), making it easy to run and verify from the host side.
- The `membar_sys()` requirement after per-thread compute is subtle but important: lane 0's `.sys` release only orders its own prior writes, not other lanes'.

## Open Questions
None.

## Impact on Downstream Tasks
- async-pipeline.3 (README overhaul) can now reference a working, runnable demo
