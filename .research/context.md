## Current Focus
T0 epics have unmet criteria. Tier Gate blocks all T1 work until T0 clears.
- native-rust-dx: 3/4 (examples rewrite pending — native-api.2)
- std-thread-gpu: 4/5 (std::thread::spawn hang — std-thread-integration.1/.2)
tasks_since_brainstorm = 12 → brainstorm triggered. current_step = think.triage.

## Recent Decisions
- 2026-06-04: gpu::compute() renamed → gpu::launch() (not cooperative compute)
- 2026-06-04: Each gpu:: call uses fresh CUDA module (avoids stale-static hang across launches)
- 2026-06-04: Added gpu_main_poll() — bar.sync-free variant for std-compiled kernels
- 2026-06-04: Thread pool stale-state bug fixed (separate modules per kernel launch)
- 2026-06-04: north_star_demo kernel written in gpu-kernel-std (needs toolchain rebuild to test)
- 2026-06-04: MIR pass auto-applies to ALL async fn on nvptx64 (no attribute needed)

## Tried & Rejected
- bar.sync removal for cross-launch fix: doesn't work (L1 cache coherence across launches)
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Cooperative closure captures: GPU local memory per-warp isolation → ILLEGAL_ADDRESS

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- kernel_std.ptx needs patched toolchain rebuild (stage1 exists, stage2 needed)
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0
- CUDA module statics persist across launches — must load into separate modules

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- thread::spawn + reuse: verified (separate module fix)
- cooperative: 4 warps verified, stride-based data parallel verified

## Next
1. Execute think.triage → standard brainstorm (proactive trigger: tasks>=10)
2. Brainstorm must focus T0 only (Tier Gate): fix std hang, rewrite examples
3. After brainstorm: execute T0 tasks to clear both epics
4. Then: activate cooperative-compute (T1), verify north_star_demo end-to-end
