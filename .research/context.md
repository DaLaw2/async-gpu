## Current Focus
T0 epics: std-thread-gpu and native-rust-dx. Thread::spawn on GPU is working —
spawns warps, joins results, reuses slots. Now implementing native Rust DX (gpu::run API).

## Recent Decisions
- 2026-06-04: thread::spawn maps to warp execution, gpu_main() dispatches — warp 0 = main, others park
- 2026-06-04: Use global memory atomics (not shared memory) for inter-warp communication
- 2026-06-04: Only lane 0 manages closure state; all 32 lanes execute in SIMT lockstep
- 2026-06-04: Fix sm_80+ PTX issue — gate MMA/bf16/tf32 kernels behind feature flag
- 2026-06-04: gpu::compute() one-liner API wraps device init + PTX + launch + sync

## Tried & Rejected
- Q in shared memory for attention: increases smem from 16KB to 25KB, reduces occupancy
- Removing p_val branch in P·V: slower because V-read savings outweigh branch cost
- bar.warp.sync PTX instruction: not valid, SIMT convergence is implicit

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- kernel.ptx must NOT contain sm_80+ instructions (MMA, bf16, tf32)
- Patched std requires toolchain rebuild (build-toolchain.sh) for full std::thread integration

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal, 600 GFLOPS bidir @ seq=512
- thread::spawn: 2 threads + join in 0.x ms (overhead dominated by nanosleep polling)
- gpu::compute(): 1-line launch confirmed working

## Next
1. Rewrite examples/ to use gpu::run() native Rust style (native-api.2)
2. Thread-api theme: verify thread::current/sleep/yield with patched std (needs toolchain rebuild)
3. Consider if extern "gpu-kernel" ABI is achievable without full compiler modification
