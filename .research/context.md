## Current Focus
**Cycle 612 SAVED** (2026-06-05). kernel-split design verified — 4-crate split, zero circular deps.
fusion-integrate.1 done — nn::Linear auto-fuses bias+activation, 1.61x speedup.
kernel-split execution next: stdio extraction → crate creation → loader + build system.

## Recent Decisions
- 2026-06-05: 4-crate split: core (helpers+basic), compute (ML), io (hostcall), test (demos)
- 2026-06-05: stdio infra (gpu_stdout_write, stdio_auto_init) to gpu-kernel-core as pub API
- 2026-06-05: dynamic_smem global_asm must be duplicated in each crate using shared memory
- 2026-06-05: nn::Linear::forward_auto_fused() — matmul then fused bias+activation via NVRTC
- 2026-06-05: Activation enum (Gelu, Relu, Silu, Sigmoid) in nn::layers

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Custom GPU assert: unnecessary, std panic handler already sends coordinates
- GPU par_iter single-block: 4.5% SM utilization, needs multi-block
- Float4 bias reads when col%4≠0: scalar fallback needed
- GPU atomics on stack: ptxas rejects .local space atomics

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents (OOM risk)
- test-integration.2 kernels stashed in git stash@{0}
- cubin build bottleneck: 30min for unified PTX — kernel-split addresses this

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS (90% cuBLAS)
- FusionCodegen: 7 ops, 2.05x fused speedup, float4 vectorized
- nn::Linear auto-fuse: 1.61x speedup (bias+activation fusion)
- 763 tasks completed, 48 epics archived

## Next
1. split-design.2 + .3: stdio extraction design + multi-cubin loader API design
2. split-execute.1: extract stdio to gpu-runtime (after design)
3. fusion-integrate.2: GPT-2 benchmark with auto-fusion
4. test-integration.2: unstash + verify after kernel-split
