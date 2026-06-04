## Current Focus
T0 epics nearly complete. All 4 native-rust-dx criteria addressed. std-thread-gpu 4/5 done.
Continuing to close remaining gaps and move to T1.

## Recent Decisions
- 2026-06-04: MIR pass applies to ALL async fn on nvptx64 automatically (no #[warp_cooperative] needed)
- 2026-06-04: extern "gpu-kernel" ABI verified working via patched rustc + gpu_kernel_abi feature
- 2026-06-04: thread::spawn maps to warp execution, gpu_main() dispatches
- 2026-06-04: gpu::compute() one-liner API wraps device init + PTX + launch + sync
- 2026-06-04: Toolchain updated to nightly-2026-06-03 (rustc 1.98.0)

## Tried & Rejected
- Q in shared memory for attention: increases smem, reduces occupancy
- Out-of-place elementwise_add: SLOWER than in-place (160 vs 119 GB/s) — in-place already optimal

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0 for patched-rustc PTX
- Patched std needs toolchain rebuild for full std::thread integration

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- thread::spawn: 2 threads + join verified

## Next
1. Rebuild patched toolchain to verify std::thread::spawn + println! example
2. Rewrite examples to use extern "gpu-kernel" + remove #[warp_cooperative]
3. Advance T1 kernel-perf (attention V3.3, conv Winograd)
