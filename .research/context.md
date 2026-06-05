## Current Focus
**Cycle 632 — AutoScheduler + zero-copy GpuVec launch complete** (2026-06-06). Both scheduler and transfer foundations complete. AutoScheduler routes par_map/par_reduce by size (CPU<4096, GPU>=4096). launch_with_gpuvec() uses raw CUDA driver API for zero-copy kernel launch. Integration tests pass on real GPU (1K, 2K, 1M elements). Demo phase next: unified-demo.1 is the North Star — read → compute → write with zero explicit memory management.

## Recent Decisions
- 2026-06-06: unified-transfer.3 — launch_with_gpuvec() via raw CUDA driver API (cuLaunchKernel). GpuVec::map_gpu() for in-place transforms. Integration tests: 1K, 2K, 1M zero-copy launches pass on GPU.
- 2026-06-06: unified-scheduler.3 — AutoScheduler with par_map/par_reduce. Size threshold 4096: small → CPU (Rayon-style), large → GPU kernel. Both themes complete.
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
- unified-runtime: 3 themes (2 completed, 1 active), 6/8 tasks done (scheduler.1-3, transfer.1-3)
- 795 tasks completed, 54 epics completed
- T1 complete: gpu-test, gpu-iterator, gpu-type-safety, gpu-generics

## Next
1. unified-demo.1: read → compute → write pipeline with zero GPU concepts in user code (North Star demo)
2. unified-demo.2: performance benchmark + AutoScheduler choosing GPU for compute, CPU for I/O
