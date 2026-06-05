## Current Focus
**Cycle 614 SAVED** (2026-06-05). split-execute.1 done — stdio extracted to gpu-runtime.
kernel-split execution in progress: stdio done, next is gpu-kernel-core crate creation.
split-design theme complete (3/3), split-execute 1/5 done.

## Recent Decisions
- 2026-06-05: stdio → gpu-runtime::stdio (5 functions + 3 atomics), stdio_auto_init stays per-crate
- 2026-06-05: #[used] force-link statics prevent LTO stripping of PAL callbacks
- 2026-06-05: gpu-runtime prelude re-exports: stdio_init, stdio_print_buffer_init, gpu_print_buffer_flush
- 2026-06-05: GPT-2 auto-fusion: 0.5-2.7% speedup (GEMM dominates), 10% needs epilogue fusion

## Tried & Rejected
- bar.sync for scope join: deadlocks
- GPU par_iter single-block: 4.5% SM utilization
- GPU atomics on stack: ptxas rejects .local space
- Epilogue-only fusion for 10% GPT-2 speedup: GEMM dominates ~85%

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- test-integration.2 kernels stashed in git stash@{0}
- dynamic_smem global_asm must be duplicated in each kernel crate using shared memory

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS (90% cuBLAS)
- FusionCodegen: 7 ops, 2.05x standalone, 1.61x Linear layer
- 767 tasks completed, 48 epics archived

## Next
1. split-execute.2: create gpu-kernel-core (helpers + basic + compute_math)
2. split-execute.3-.5: create compute, io, test crates (can parallel after .2)
3. split-loader.1-.5: multi-cubin loader + build system
