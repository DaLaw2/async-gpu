## Current Focus
**T2 ACTIVATED** (2026-06-05). All T1 epics completed (5/5). Phase 2 begins.
gpu-iterator (T2, HIGH) activated — iter-design theme active, first task: iter-design.1.
auto-fusion (T2, MEDIUM) activated — fusion-analysis theme active (investigation only).
Staggered start: fusion codegen waits for iterator MIR infrastructure.

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
1. iter-design.1: Investigation — Rust Iterator/Rayon traits mapping to GPU (critical path)
2. fusion-analysis.1: Investigation — fusable MIR patterns (parallel with iter-design.1)
3. iter-design.2 + fusion-analysis.2: design tasks after investigations complete
4. Phase 3 (unified-runtime) waits for Phase 2 completion
