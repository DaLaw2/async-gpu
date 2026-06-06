## Current Focus
**Cycle 638 — coro-impl.2 + cost-analysis.2 complete** (2026-06-06). GPU coroutines: FibGenerator + 3 test kernels (fibonacci streaming, square-accumulate pipeline, multi-generator with edge cases), all compile to nvptx64 PTX. Compile-time cost: kernel-resources.sh + resource_report.rs (8 unit tests) + build-kernels.sh --report integration, 173+ kernels analyzed. Both themes completed. Next: gpu-coroutines epic verification, then cost-warnings tasks.

## Recent Decisions
- 2026-06-06: coro-impl.2 — FibGenerator yields Fibonacci numbers via GpuGenerator. 3 test kernels: fibonacci streaming pipeline, counter square-accumulate, multi-generator (4 independent gens including edge cases). PTX compiles. GPU JIT blocked by 20+ min per module load (no cubin).
- 2026-06-06: cost-analysis.2 — kernel-resources.sh parses ptxas -v, calculates sm_75 occupancy with WARN/CRIT thresholds. resource_report.rs with SmConfig, KernelResources, occupancy(), parse_ptxas_output() + 8 tests. build-kernels.sh --report integration (auto in --prod). 34 kernels at 112 regs from device function inflation.
- 2026-06-06: coro-impl.1 — GpuGenerator trait, WarpBroadcast, for_each_yield, GeneratorTask, CounterGenerator (429 lines)
- 2026-06-06: cost-analysis.1 — Hybrid approach: ptxas -v for registers + MIR for bank conflicts

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation
- Channels for streaming pipeline: adds buffering, not zero-buffered
- Per-lane yield values: changes SIMT model

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min — need pre-compiled cubin for GPU runtime tests
- Device function register inflation: 34 kernels show 112 regs / 50% occ falsely

## Key Metrics
- gpu-coroutines: all 4 tasks done (coro-design.1-.2, coro-impl.1-.2), ready for epic verification
- compile-time-cost: cost-analysis complete (2/2), cost-warnings active (0/2), total 2/4 tasks
- 803 tasks completed, 55 epics completed

## Next
1. gpu-coroutines epic verification — check all 4 success criteria
2. cost-warnings.1: emit compile-time warnings for low occupancy + bank conflicts
3. cost-warnings.2: catch real perf issue via compile-time lint
