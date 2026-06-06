## Current Focus
**Cycle 641 — gpu-panic 1 task left, feature-audit 2 tasks left** (2026-06-06). API encapsulation done (HostcallBuffer pub→pub(crate) + accessors). Facade re-exports added (GpuVec, schedulers). Runtime audit found 8/24 examples fail due to wrong PTX loaded. gpu_assert! deprecation designed (1 call site).

## Recent Decisions
- 2026-06-06: HostcallBuffer fields → pub(crate) + 9 accessor methods. All 8 external callers migrated.
- 2026-06-06: async-gpu facade now re-exports GpuVec, Scheduler, CpuScheduler, GpuScheduler, AutoScheduler.
- 2026-06-06: Runtime audit root cause: `ptx::KERNEL` aliases KERNEL_COMPUTE but 8 examples need KERNEL_IO or KERNEL_TEST. Fix: examples should specify correct PTX via `.ptx()`.
- 2026-06-06: gpu_assert! has only 1 live call site — safe to delete entirely (no shim needed).

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation
- Channels for streaming pipeline: adds buffering, not zero-buffered

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min
- Priority gate: gpu-panic + feature-audit (high) block compile-time-cost (medium)

## Key Metrics
- 814 tasks completed, 56 stories completed
- gpu-panic: 5/6 tasks done, 1 remaining (panic-deprecate-gpu-assert.2)
- feature-audit: 5/6 tasks done, 1 remaining (audit-runtime.2)
- Both stories very close to completion

## Next
1. panic-deprecate-gpu-assert.2: deprecate gpu_assert! and migrate the 1 caller
2. audit-runtime.2: fix 8 runtime failures (PTX kernel routing)
3. After both stories complete → Story Verification Gates → potentially close stories
4. Then: cost-warnings.2 (unblocked when high stories complete)
