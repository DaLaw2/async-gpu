## Current Focus
**Cycle 613 SAVED** (2026-06-05). kernel-split design phase COMPLETE (3/3 tasks).
fusion-integrate theme complete (2/2). GPT-2 auto-fusion: 0.5-2.7% (below 10% target).
auto-fusion epic: all themes done but GPT-2 speedup criterion unmet — needs assessment.
Next: kernel-split execution (split-execute.1 → stdio extraction).

## Recent Decisions
- 2026-06-05: stdio infra → gpu-runtime::stdio (not gpu-kernel-core), zero circular deps
- 2026-06-05: stdio_auto_init() stays per-crate as 8-line wrapper bridging gpu-runtime + gpu-libc
- 2026-06-05: Multi-cubin: KERNEL_COMPUTE/IO/TEST constants + deprecated aliases for compat
- 2026-06-05: KernelRegistry needs zero changes (all ML_KERNELS map to compute crate)
- 2026-06-05: GPT-2 auto-fusion: GEMM dominates ~85%, epilogue fusion saves only ~0.6ms/pass

## Tried & Rejected
- bar.sync for scope join: deadlocks
- Custom GPU assert: unnecessary
- GPU par_iter single-block: 4.5% SM utilization
- GPU atomics on stack: ptxas rejects .local space atomics
- Epilogue-only fusion for 10% GPT-2 speedup: GEMM dominates, need GEMM epilogue fusion

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- test-integration.2 kernels stashed in git stash@{0}
- cubin build bottleneck: 30min unified PTX — kernel-split addresses this
- auto-fusion 10% GPT-2 target unmet — needs GEMM epilogue fusion or cross-op fusion

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS (90% cuBLAS)
- FusionCodegen: 7 ops, 2.05x standalone speedup
- nn::Linear auto-fuse: 1.61x (single layer), 0.5-2.7% (GPT-2 end-to-end)
- 766 tasks completed, 48 epics archived

## Next
1. split-execute.1: extract stdio to gpu-runtime (design complete, ready to implement)
2. split-execute.2-.5: create 4 kernel crates
3. split-loader.1-.5: multi-cubin loader + build system
4. Assess auto-fusion epic: all themes done but GPT-2 10% target unmet
