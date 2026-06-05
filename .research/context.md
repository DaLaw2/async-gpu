## Current Focus
**Cycle 617 SAVED** (2026-06-05). split-execute theme COMPLETE (5/5).
4 kernel crates established: core, compute, io, test. All produce valid PTX.
split-loader phase next: multi-cubin host loader + parallel build + dev opt-level.

## Recent Decisions
- 2026-06-05: gpu-kernel-std renamed to gpu-kernel-test (full rename, all refs updated)
- 2026-06-05: compute crate removed stdio/gpu-libc deps (pure compute, no hostcall)
- 2026-06-05: PTX/cubin output filenames kept as kernel_std.* for backward compat during transition
- 2026-06-05: Each kernel crate has own stdio_auto_init + #[used] force-link + dynamic_smem

## Tried & Rejected
- bar.sync for scope join: deadlocks
- GPU par_iter single-block: 4.5% SM utilization
- GPU atomics on stack: ptxas rejects .local space
- Epilogue-only fusion for 10% GPT-2 speedup: GEMM dominates ~85%

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- test-integration.2 kernels stashed in git stash@{0} (need unstash after loader done)
- PTX still uses old kernel_std.ptx filename — split-loader.1 will add per-crate constants

## Key Metrics
- 4 kernel crates: core (17 entries), compute (84), io (55), test (~60)
- FusionCodegen: 7 ops, 2.05x standalone, 1.61x Linear layer
- 771 tasks completed, 48 epics archived

## Next
1. split-loader.1: per-crate PTX constants + backward-compat aliases
2. split-loader.2: update host loader for multi-module
3. split-loader.3: parallel build script
4. split-loader.4: dev-mode opt-level reduction
5. split-loader.5: litmus test — single-crate rebuild under 5 minutes
