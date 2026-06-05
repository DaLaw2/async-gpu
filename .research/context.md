## Current Focus
**Cycle 610 SAVED** (2026-06-05). gpu-test T0 progressing well. gpu-iterator demo complete.
test-framework theme done (2/2). test-integration.1 done (cargo test integration verified).
iter-demo theme done (2/2). GPU par_iter vs Rayon: GPU currently slower (single-block bottleneck).
fusion-codegen.2 done — NVRTC codegen engine verified.

## Recent Decisions
- 2026-06-05: #[gpu_test] proc macro: Option B, expands to #[test] + run_zero_param_with_cubin
- 2026-06-05: Standard assert!/assert_eq! work on GPU via patched std panic handler (no custom needed)
- 2026-06-05: GPU par_iter single-block is 5-1178x slower than Rayon — need multi-block + cached loads
- 2026-06-05: Shared memory bug found: par_iter kernels need shared_mem_bytes > 0 for init_shared_mem_allocator
- 2026-06-05: FusionCodegen: NVRTC template with HashMap cache, 6 elementwise ops, GPU-verified

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Re-exporting nvptx64 types from host facade: architectural mismatch
- Custom GPU assert for Phase 1: unnecessary, std panic handler already sends coordinates
- GPU par_iter single-block vs Rayon: 4.5% SM utilization, volatile loads bypass cache

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- 48KB shared memory per block
- Max 2 concurrent subagents (OOM risk)
- PTX JIT ~25min for unified 5MB PTX; cubin ~10min, only at SAVE step
- SAVE only after dispatched tasks verified-done, never mid-flight

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS)
- #[gpu_test]: 5 tests (3 GPU + 2 CPU) pass in 1.5s via cargo test
- par_iter 1M+: correctness verified, GPU 5-1178x slower than Rayon (single-block)
- FusionCodegen: 6 ops, float4 vectorized, 1e-4 to 1e-6 tolerance
- 762 tasks completed, 48 epics archived

## Next
1. test-integration.2: Write 10+ #[gpu_test] tests covering SC, channels, executor, cooperative
2. fusion-codegen.3: Register-only intermediates + float4 vectorization
3. Address par_iter performance: multi-block launch needed for competitive throughput
4. gpu-test epic nearing completion — test-integration.2 is the last task
