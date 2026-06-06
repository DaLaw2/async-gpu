## Current Focus
**Cycle 635 — gpu-coroutines investigation complete** (2026-06-06). coro-design.1 delivered: Rust Coroutine trait and async fn share the same StateTransform MIR pass. WarpCooperativeTransform already handles generator coroutine bodies — MIR pass likely needs no changes. GpuGenerator trait should model after WarpFuture with lane-0 resume + shfl.sync broadcast. Yield values: scalar via shfl.sync, large structs via shared memory. Streaming pipeline best as direct inline warp-cooperative loop (not channels). Next: coro-design.2 (GpuGenerator trait + MIR pass extension design).

## Recent Decisions
- 2026-06-06: coro-design.1 — Coroutine and async fn use same StateTransform pass. WarpCooperativeTransform processes all coroutine bodies already. MIR pass may need no changes for generator support — yield value broadcast is a runtime concern (GpuGenerator trait).
- 2026-06-06: Brainstorm 124 — gpu-coroutines activated as next T2 epic. Stagger compile-time-cost as companion (~cycle 637). conv-perf broken dependency fixed (perf-conv.1 → perf-conv.6). gpu-hot-reload needs task redesign (NVRTC assumption wrong for Rust kernels).
- 2026-06-06: unified-runtime completed (55th epic). North Star demo: read → compute → write with zero GPU concepts. AutoScheduler + GpuVec zero-copy.

## Tried & Rejected
- Round-robin DisjointSlice partitioning: can't return contiguous &mut [T]
- Lifetime-locked get_mut (WarpIndex<'scope> only): blocked cooperative_indexed cross-scope usage
- Channels for streaming pipeline: adds buffering overhead; direct inline warp-cooperative loop is better for zero-buffering producer-consumer

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- All T0 and T1 epics completed — T2 gpu-coroutines active
- hardware-intrinsics BLOCKED on sm_75 (needs sm_90+)

## Key Metrics
- gpu-coroutines: 2 themes active, 1/4 tasks done (coro-design.1)
- 798 tasks completed, 55 epics completed
- T2 completed: unified-runtime. T2 active: gpu-coroutines.

## Next
1. coro-design.2: Design GpuGenerator trait + MIR pass extension for yield to warp context switch
2. When coro-design.2 done → coro-impl.1: implement generator compilation to PTX
3. Stagger compile-time-cost activation (~cycle 637): cost-analysis.1 investigation
