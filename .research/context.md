## Current Focus
**Cycle 633 — North Star demo complete** (2026-06-06). unified-demo.1 delivered: read → compute → write pipeline with zero GPU concepts exposed to user. Two examples (unified_pipeline, gpuvec_pipeline) + 4 integration tests, all passing on real GPU (0.28s with inline PTX JIT). AutoScheduler par_map + GpuVec zero-copy pipeline fully working. Next: unified-demo.2 (performance benchmark + AutoScheduler choosing GPU for compute, CPU for I/O), then epic verification.

## Recent Decisions
- 2026-06-06: unified-demo.1 — North Star demo complete. Two examples: unified_pipeline.rs (AutoScheduler par_map), gpuvec_pipeline.rs (GpuVec zero-copy map_gpu). 4 integration tests pass. GPU concepts hidden: kernel launch, memcpy, block/thread config, sync, PTX loading.
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
- unified-runtime: 3 themes (2 completed, 1 active), 7/8 tasks done (scheduler.1-3, transfer.1-3, demo.1)
- 796 tasks completed, 54 epics completed
- T1 complete: gpu-test, gpu-iterator, gpu-type-safety, gpu-generics

## Next
1. unified-demo.2: performance benchmark + AutoScheduler choosing GPU for compute, CPU for I/O
2. Epic verification: all 4 unified-runtime success criteria met
