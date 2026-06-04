## Current Focus
T0 CLEARED. T1: library-api COMPLETED, cooperative-compute COMPLETED. kernel-perf active (SGEMM done, attention/conv/e2e pending).
structured-concurrency pending — needs brainstorm for themes/tasks.
tasks_since_brainstorm = 6 (lib-cleanup.5, .6, lib-toolchain.2, lib-docs.2, perf-gemm-v4.2.1, lib-cleanup.6 verify).

## Recent Decisions
- 2026-06-04: library-api epic COMPLETED — all 5 criteria verified by independent gate
- 2026-06-04: SGEMM V4.1 achieves 90% cuBLAS (2691 GFLOPS at 4096³) — was 63% due to dispatch bug
- 2026-06-04: mapped_mem + model_dir doc(hidden), ptx/cubin doc(hidden)
- 2026-06-04: Getting-started guide: 5-step SAXPY, ~17 min target, facade-only imports

## Tried & Rejected
- bar.sync removal for cross-launch fix: doesn't work (L1 cache coherence)
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Cooperative closure captures: GPU local memory per-warp isolation → ILLEGAL_ADDRESS
- gpu::run_std() loading kernel_std.ptx at runtime: JIT too slow (>10min for 6MB PTX)
- cp.async SGEMM on SM75: not available, need SM80+. Double-buffer works instead.
- Moving model/yolo/tokenizer to separate crate: 50+ call sites. pub(crate)+feature gate instead.
- Removing demo feature gate entirely: breaks test harness. Keep dual-cfg pattern.

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32, no cp.async
- kernel_std.cubin must be pre-compiled (ptxas --gpu-name sm_75)
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0
- CUDA module statics persist across launches — must load into separate modules
- cudarc types (CudaSlice<T>) not re-exported by async-gpu facade — noted API gap

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS) ← was 63% before dispatch fix
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- cooperative_map: working, kernel-side naive matmul verified
- API surface: gpu-host 6 pub modules, async-gpu clean facade

## Next
1. kernel-perf remaining: attention ≥70% cuDNN, conv2d ≥70% cuDNN, GPT-2 <25ms
2. structured-concurrency needs brainstorm to create themes/tasks
3. Many kernel-perf tasks have dependency chains with missing predecessors — need cleanup
4. Consider: which kernel-perf tasks are unblocked now that GEMM V4 is done?
