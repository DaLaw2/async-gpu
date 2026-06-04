## Current Focus
T0 epics nearly complete. std-thread-gpu: 4/5 criteria met (std::thread::spawn hangs — bar.sync + std init).
native-rust-dx: 3/4 criteria met (examples rewrite pending). T1 cooperative-compute blocked on T0 completion.
tasks_since_brainstorm = 12 → brainstorm trigger armed.

## Recent Decisions
- 2026-06-04: MIR pass auto-applies to ALL async fn on nvptx64 (no #[warp_cooperative] needed)
- 2026-06-04: extern "gpu-kernel" ABI verified working via patched rustc + gpu_kernel_abi feature
- 2026-06-04: thread::spawn maps to warp execution, gpu_main() dispatches
- 2026-06-04: gpu::compute() one-liner API wraps device init + PTX + launch + sync
- 2026-06-04: Toolchain updated to nightly-2026-06-03 (rustc 1.98.0)
- 2026-06-04: Cooperative closures must use global atomics, NOT local captures (ILLEGAL_ADDRESS)

## Tried & Rejected
- Cooperative closure captures: GPU local memory per-warp isolation → use global atomics instead
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Q in shared memory for attention: increases smem, reduces occupancy

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0 for patched-rustc PTX
- Patched std needs toolchain rebuild for full std::thread integration
- std::thread::spawn hangs at bar.sync (std init calls bar.sync before warp pool ready?)

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- thread::spawn: 2 threads + join verified, reuse verified (4 tasks on 3 warps)
- cooperative: 4 warps verified, stride-based data parallel verified

## Next
1. Route assessment: tasks_since_brainstorm >= 10 → brainstorm trigger
2. T0 remaining: std-thread-integration (hang fix), native-api.2 (examples rewrite)
3. After T0 complete → activate cooperative-compute (T1)
4. kernel-perf (T1) continues in parallel: perf-attn-v3.3, perf-fusion.2
