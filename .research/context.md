## Current Focus
**structured-concurrency EPIC COMPLETED** (2026-06-05). All 5 success criteria verified.
18 tasks completed across 5 themes in one session. T1 highest-priority epic done.
kernel-perf (T1, medium) still active. Next: brainstorm for T2 activation or kernel-perf wrap-up.

## Recent Decisions
- 2026-06-05: structured-concurrency epic COMPLETED — all criteria verified
- 2026-06-05: Rayon scope model with for<'scope> HRTB + PhantomData invariance
- 2026-06-05: Library-only enforcement sufficient, no MIR pass changes needed
- 2026-06-05: Cancellation chain-walk: parent_cancel_ptr + is_cancelled() walks up
- 2026-06-05: Unified ScopedOneshot/ScopedMpsc enum auto-selects CTA vs system atomics
- 2026-06-05: Fork/join warp-0-only scheduling confirmed, nested spawn not supported
- 2026-06-05: GridScope uses pre-allocated pool + BlockWorkSlot coordination for SM75

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Shuffle as channel primitive: synchronous collective, not point-to-point
- Runtime channel transport detection: shared memory can't be allocated retroactively
- Work-stealing scheduler on GPU: CAS contention + complexity not worth it
- MIR pass for scope enforcement: maintenance cost >> marginal safety gain
- Nested block_scope from worker warps: allocator not thread-safe, warp exhaustion

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32, no cp.async
- 48KB shared memory per block — BlockScope allocations limited
- Max 2 concurrent subagents (OOM risk)
- Warp 0 only for scope allocation (single-writer invariant)

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS)
- BlockScope: watermark allocator, spawn/spawn_all, join_all with STATUS_TRAPPED
- GridScope: global memory pool, completion counter, work slot dispatch
- Block channels: CTA-scope atomics ~20-50x faster than system-scope
- Unified channels: ScopedOneshot/ScopedMpsc auto-select transport
- 6 demo kernels: producer-consumer, cooperative parallel, nested scopes, combined, grid reduce, channel bench

## Next
1. Brainstorm trigger: tasks_since_brainstorm >= 10, structured-concurrency completed
2. T1 remaining: kernel-perf (medium priority) — attention, conv, e2e
3. T2 epics pending: gpu-iterator, auto-fusion, unified-runtime (depend on structured-concurrency ✓)
4. Consider tier promotion: all T1 highest satisfied → T2 activation
