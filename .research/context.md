## Current Focus
**Cycle 638 — Hierarchy migration complete** (2026-06-06). Migrated from 3-level (Epic→Theme→Task) to 4-level (Epic→Story→Feature→Task). 3 strategic epics created: lang-complete, invisible-exec, perf-transparent. All former epics became stories. All former themes became features. cost-warnings feature active under compile-time-cost story (perf-transparent epic).

## Recent Decisions
- 2026-06-06: Hierarchy migration — 3-level → 4-level. Tier system replaced by story priority + depends_on. 3 strategic epics are active simultaneously (parallel pillars). 5 off-direction stories removed (gpu-hot-reload, hardware-intrinsics, gpu-repl, cross-vendor, actor-model). 6 new stories added (gpu-panic, gpu-dyn-dispatch, transparent-data, auto-tuning, feature-audit, whole-program-partition).
- 2026-06-06: gpu-coroutines story completed (56th). FibGenerator + 3 test kernels, all compile to PTX.
- 2026-06-06: cost-analysis feature completed. kernel-resources.sh + resource_report.rs (8 unit tests), 173+ kernels analyzed.

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation
- Channels for streaming pipeline: adds buffering, not zero-buffered
- Per-lane yield values: changes SIMT model
- Off-direction epics (hardware-intrinsics, actor-model, gpu-repl, gpu-hot-reload, cross-vendor): removed — they expose GPU concepts instead of hiding them

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min — need pre-compiled cubin for GPU runtime tests
- Device function register inflation: 34 kernels show 112 regs / 50% occ falsely

## Key Metrics
- 804 tasks completed, 56 stories completed across project history
- 3 strategic epics active: lang-complete, invisible-exec, perf-transparent
- compile-time-cost: cost-analysis complete (2/2), cost-warnings active (1/2 done)

## Next
1. cost-warnings.2: catch real perf issue in existing kernel via compile-time lint
2. After compile-time-cost story completes: brainstorm to create features for gpu-panic (highest priority pending story)
3. feature-audit: high priority — verify all 56 shipped stories still work
