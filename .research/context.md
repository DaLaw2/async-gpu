## Current Focus
**Cycle 631 — Scheduler trait + GpuVec<T> implemented** (2026-06-06). Scheduler trait with CpuScheduler/GpuScheduler + GpuVec<T> zero-copy buffer wrapping MappedBuffer. Foundation for unified-runtime laid. Next: unified-scheduler.3 (AutoScheduler) + unified-transfer.3 (par_iter integration).

## Recent Decisions
- 2026-06-06: unified-scheduler.2 — Scheduler trait with cpu()/gpu_launch(), CpuScheduler routes to tokio, GpuScheduler routes to gpu::launch. NoGpu error variant added.
- 2026-06-06: unified-transfer.2 — GpuVec<T> wraps MappedBuffer for zero-copy. from_vec, zeroed, dev_ptr, as_slice, into_vec. No cudaMemcpy in user code.
- 2026-06-06: unified-scheduler.1 — Scheduler is work-routing, not a magic GPU compiler. CpuScheduler/GpuScheduler/AutoScheduler with par_map/par_reduce combinators. AutoScheduler uses size-based heuristics (small → CPU, large → GPU).
- 2026-06-06: unified-transfer.1 — GpuVec<T> wraps MappedBuffer for zero-copy default path. Two-tier buffer model: MappedBuffer (zero-copy, host+device visible) vs DeviceBuffer (opt-in, for multi-read GPU-only data).

## Tried & Rejected
- Round-robin DisjointSlice partitioning: can't return contiguous &mut [T]
- Lifetime-locked get_mut (WarpIndex<'scope> only): blocked cooperative_indexed cross-scope usage

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- All T0 and T1 epics completed — T2 unified-runtime active

## Key Metrics
- unified-runtime: 3 themes, 4/8 tasks done (scheduler.1-2, transfer.1-2)
- 793 tasks completed, 54 epics completed
- T1 complete: gpu-test, gpu-iterator, gpu-type-safety, gpu-generics

## Next
1. unified-scheduler.3: AutoScheduler with work type heuristics (I/O → CPU, compute → GPU)
2. unified-transfer.3: lazy transfer + redundancy elimination — skip unnecessary copies
3. unified-demo.1: read → compute → write pipeline (depends on scheduler.2 + transfer.2 — now unblocked)
