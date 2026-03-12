# async-pipeline.5: Warp-Scale Embassy (32 Threads, 1 Block)
**Cycle**: 108 | **Theme**: async-pipeline | **Kind**: investigation | **Status**: done

## Summary
Implemented and hardware-verified a warp-scale Embassy test: 32 threads in one warp (1 block × 32 threads), each running its own Embassy executor with an independent `HostcallPrintFuture`. All 32 threads completed successfully, receiving all 32 hostcall responses. This is the maximum per-thread concurrency within a single warp.

## Findings

### Q: How does per-thread CAS contention scale with 32 threads in one warp?
A: All 32 threads completed within the 500 poll-round budget. The test used 64 packets (2× thread count) to avoid pool exhaustion. CAS contention was manageable — the host listener processed all 32 messages within ~2.4ms. Key observation: per-thread Embassy futures use `activemask()` which returns a single-lane mask (each thread runs independently with warp divergence), so there is no warp-cooperative behavior — each thread operates as an independent actor competing for the global free/ready stacks.
**Confidence**: high (hardware verified — 32/32 threads completed)

### Q: What is the throughput of Embassy per-thread futures at warp scale?
A: 32 hostcall round-trips completed in ~2.4ms total (75µs per message average). This is slower than the 4-thread cross-block test due to intra-warp serialization of divergent memory operations. All 32 lanes compete for the same CAS on the free stack, ready stack, and doorbell.
**Confidence**: high (hardware verified)

### Q: Does per-block sharding sufficiently mitigate intra-warp contention?
A: Not tested in this experiment (single block). However, per-block sharding would not help here since all 32 threads are in the same block/warp and would use the same shard. The contention is inherent to intra-warp divergent memory access — this is exactly why WarpFuture (warp-cooperative) is preferred for uniform workloads.
**Confidence**: medium (inferred from architecture, not directly tested)

## Key Implementation Details
- Static arrays: `[ExecutorStorage; 32]` and `[TaskStorage<HostcallPrintFuture>; 32]` using const initialization macros
- Message format: "Warp lane XX!" with two-digit lane ID (00-31)
- Launch config: 1 block × 32 threads = exactly one warp
- Packet pool: 64 packets to avoid pool exhaustion with 32 concurrent threads
- Max poll rounds: 500 (vs 200 for 4-thread test) to account for higher contention

## Unexpected Discoveries
- All messages prefixed with `[B0.T0]` — the host listener always reports block=0, tid=0 because the packet header uses `activemask()` which returns a single-bit mask for divergent threads. The block/thread reporting in the host listener extracts from the active mask, not from the actual thread ID.

## Open Questions
- None — this completes the async-pipeline EPIC.

## Impact on Downstream Tasks
- Confirms that per-thread Embassy approach works at full warp scale (32 threads)
- Documents the performance gap: per-thread (divergent) vs warp-cooperative patterns
- Completes all 5 async-pipeline tasks — EPIC is done
