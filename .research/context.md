## Current Focus
**Cycle 637 — coro-impl.1 + cost-analysis.1 complete** (2026-06-06). Generator API implemented in gpu-runtime (generator.rs, 429 lines): WarpCoroutineState, WarpBroadcast, GpuGenerator trait, for_each_yield combinator, GeneratorTask adapter, CounterGenerator reference impl. Compiles for nvptx64. cost-analysis.1 found hybrid approach best: ptxas -v for registers/occupancy + MIR for bank conflicts. Two real kernels with 25% occupancy (255 regs, 216 regs) identified as test cases. Next: coro-impl.2 (streaming pipeline demo) + cost-analysis.2 (implement ptxas -v parsing).

## Recent Decisions
- 2026-06-06: coro-impl.1 — GpuGenerator<R=()> trait implemented with resume_warp(), WarpBroadcast for all scalar types (shfl.sync), for_each_yield zero-buffered combinator, GeneratorTask Future adapter. 429 lines in generator.rs, compiles for nvptx64.
- 2026-06-06: cost-analysis.1 — Hybrid approach: ptxas -v is the ONLY source of truth for registers (MIR locals >3x error margin, PTX .regs 2-5x overcount). Two integration points: build script ptxas -v parsing + MIR pass for bank conflict stride analysis.
- 2026-06-06: coro-design.2 — GpuGenerator API design finalized. MIR pass needs NO changes.
- 2026-06-06: Brainstorm 124 — gpu-coroutines + compile-time-cost activated as T2 epics.

## Tried & Rejected
- Round-robin DisjointSlice partitioning: can't return contiguous &mut [T]
- Channels for streaming pipeline: adds buffering overhead; direct inline loop better
- Per-lane yield values: changes SIMT model, requires MIR pass changes
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- T2 active: gpu-coroutines (coro-impl), compile-time-cost (cost-analysis)
- hardware-intrinsics BLOCKED on sm_75

## Key Metrics
- gpu-coroutines: coro-design complete, coro-impl 1/2 done, total 3/4 tasks
- compile-time-cost: cost-analysis 1/2 done, cost-warnings 0/2, total 1/4 tasks
- 801 tasks completed, 55 epics completed

## Next
1. coro-impl.2: Streaming pipeline demo (fibonacci producer + consumer on GPU)
2. cost-analysis.2: Implement ptxas -v parsing + per-kernel resource estimation
3. After both: epic verification for gpu-coroutines, continue cost-warnings for compile-time-cost
