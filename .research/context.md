## Current Focus
**Cycle 630 — unified-runtime investigations complete** (2026-06-06). T2 unified-runtime epic first batch done: scheduler routing design + zero-copy GpuVec transfer design. Next: parallel implementation tasks (unified-scheduler.2 + unified-transfer.2).

## Recent Decisions
- 2026-06-06: unified-scheduler.1 — Scheduler is work-routing, not a magic GPU compiler. CpuScheduler/GpuScheduler/AutoScheduler with par_map/par_reduce combinators. AutoScheduler uses size-based heuristics (small → CPU, large → GPU).
- 2026-06-06: unified-transfer.1 — GpuVec<T> wraps MappedBuffer for zero-copy default path. Two-tier buffer model: MappedBuffer (zero-copy, host+device visible) vs DeviceBuffer (opt-in, for multi-read GPU-only data).
- 2026-06-06: No manual cudaMemcpy — GpuVec<T> provides From<Vec<T>> / Into<Vec<T>> transparent conversion.
- 2026-06-06: gpu-generics epic PASS — all 4 criteria met, T1 fully cleared (54 epics)

## Tried & Rejected
- Round-robin DisjointSlice partitioning: can't return contiguous &mut [T]
- Lifetime-locked get_mut (WarpIndex<'scope> only): blocked cooperative_indexed cross-scope usage

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- All T0 and T1 epics completed — T2 unified-runtime active

## Key Metrics
- unified-runtime: 3 themes, 2/8 tasks done (investigations complete)
- 791 tasks completed, 54 epics completed
- T1 complete: gpu-test, gpu-iterator, gpu-type-safety, gpu-generics

## Next
1. unified-scheduler.2: implement CpuScheduler + GpuScheduler with explicit affinity routing
2. unified-transfer.2: transparent host <-> device transfer — From<Vec<T>> / Into<Vec<T>>
3. Both can run in parallel (no dependency between them)
