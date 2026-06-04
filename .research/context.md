## Current Focus
T0 FULLY CLEARED. T1 active: library-api (highest), kernel-perf, structured-concurrency (pending).
cooperative-compute COMPLETED (4/4 criteria pass).
library-api is the critical path — lib-cleanup + lib-docs themes active.
tasks_since_brainstorm = 0.

## Recent Decisions
- 2026-06-04: State reconciliation — archived 6 completed themes, marked native-rust-dx epic completed
- 2026-06-04: mapped_mem always pub (core CUDA API, not demo-only). Removed all #[allow(dead_code)].
- 2026-06-04: lib-cleanup.2 (model code move) superseded by lib-cleanup.5 (pub(crate) visibility approach)
- 2026-06-04: perf-gemm-v4.2 parked (SM80+ cp.async). Replaced with perf-gemm-v4.2.1 (double-buffer SM75)
- 2026-06-04: lib-docs.1 (guide design) completed — 5-step SAXPY guide targeting ~17min

## Tried & Rejected
- bar.sync removal for cross-launch fix: doesn't work (L1 cache coherence)
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Cooperative closure captures: GPU local memory per-warp isolation → ILLEGAL_ADDRESS
- gpu::run_std() loading kernel_std.ptx at runtime: JIT too slow (>10min for 6MB PTX)
- cp.async SGEMM on SM75: not available, need SM80+. Use double-buffer shared memory instead.
- Moving model/yolo/tokenizer to separate crate: 50+ call sites, too large. Use pub(crate) + feature gate.

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32, no cp.async
- kernel_std.cubin must be pre-compiled (ptxas --gpu-name sm_75)
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0
- CUDA module statics persist across launches — must load into separate modules

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- std::thread::spawn: WORKS (45 + 120 = 165, println! from spawned threads)
- kernel_std.cubin: 37MB, loads <1s
- SGEMM V4.1: 63% of cuBLAS (target 70%)
- cooperative_map: working, kernel-side naive matmul verified

## Next
1. Library-API critical path: lib-cleanup.5 → lib-cleanup.6 → lib-docs.2
2. Parallel: lib-toolchain.2 (polish setup.sh), perf-gemm-v4.2.1 (double-buffer SGEMM)
3. lib-cleanup.5 (pub(crate) visibility) and lib-docs.1 (done) are both ready — no blocked deps
4. After library-api, structured-concurrency epic needs brainstorm for themes/tasks
