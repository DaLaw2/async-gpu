## Current Focus
**Cycle 622 — kernel-split EPIC COMPLETED** (2026-06-05). 49th epic archived.
4 kernel crates (core/compute/io/test), multi-cubin loader, PTX JIT dev path.
Dev rebuild: 27.5s (was 30 min). All 5 criteria verified by Epic Verification Gate.

## Recent Decisions
- 2026-06-05: kernel-split completed — 14 tasks across 4 themes in 12 cycles (611-622)
- 2026-06-05: PTX JIT dev path (3/3 majority vote): skip ptxas for dev, cubin is CI-only
- 2026-06-05: auto-fusion GPT-2 0.5-2.7% speedup — GEMM dominates, 10% needs epilogue fusion

## Tried & Rejected
- ptxas optimization via PTX size reduction: scales with complexity not size
- Unified 11MB PTX: replaced by 4-crate split (core 1.3MB, compute 2.0MB, io 2.4MB, test 6.0MB)
- Epilogue-only fusion for 10% GPT-2: GEMM dominates ~85%

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- test-integration.2 kernels stashed in git stash@{0}

## Key Metrics
- 4 kernel crates: core (17), compute (84), io (55), test (76) entry points
- Dev rebuild: 27.5s (PTX JIT) — was 30 min (unified ptxas)
- FusionCodegen: 7 ops, 2.05x standalone, 1.61x Linear layer
- 777 tasks completed, 49 epics (kernel-split just completed)

## Next
1. Unstash test-integration.2 + verify (gpu-test epic completion)
2. T1: gpu-type-safety, gpu-generics (pending epics)
3. Brainstorm for next tier activation / new directions
