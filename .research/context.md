## Current Focus
T0 CLEARED. Both foundation epics completed. T1 activation next.
- native-rust-dx: COMPLETED (4/4 criteria pass — gpu-kernel ABI, MIR pass, one-liner API, examples rewritten)
- std-thread-gpu: COMPLETED (5/5 criteria pass — thread::spawn works with println!)
- Tier promotion: T1 epics now eligible (cooperative-compute, structured-concurrency, kernel-perf)
tasks_since_brainstorm = 0.

## Recent Decisions
- 2026-06-04: Migrated all kernels from extern "ptx-kernel" to extern "gpu-kernel" (abi_gpu_kernel feature)
- 2026-06-04: gpu::custom() builder API for multi-arg kernels (CustomLaunchBuilder → GpuContext → GpuResult)
- 2026-06-04: Removed #[warp_cooperative] attributes — MIR pass handles all async fn
- 2026-06-04: warp-macro crate kept — warp_*! DSL functions still need the proc macro
- 2026-06-04: SIMT lane-0 guard in gpu_main/gpu_main_poll — main_fn runs on lane 0 only
- 2026-06-04: kernel_std.cubin pre-compilation required (6MB PTX → 37MB cubin, <1s load)

## Tried & Rejected
- bar.sync removal for cross-launch fix: doesn't work (L1 cache coherence)
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Cooperative closure captures: GPU local memory per-warp isolation → ILLEGAL_ADDRESS
- gpu::run_std() loading kernel_std.ptx at runtime: JIT too slow (>10min for 6MB PTX)

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- kernel_std.cubin must be pre-compiled (ptxas --gpu-name sm_75)
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0
- CUDA module statics persist across launches — must load into separate modules

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- std::thread::spawn: WORKS (45 + 120 = 165, println! from spawned threads)
- kernel_std.cubin: 37MB, loads <1s
- Examples: 8/8 hostcall converted to gpu:: API, ~350 lines boilerplate removed

## Next
1. Activate T1 epics: cooperative-compute, structured-concurrency (kernel-perf already active)
2. cooperative-compute is the COMPUTE PILLAR — gpu::cooperative() from sequential context
3. Brainstorm needed to create themes/tasks for cooperative-compute and structured-concurrency
4. kernel-perf has existing themes but many tasks blocked on dependency chains
