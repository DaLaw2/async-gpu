## Current Focus
**Cycle 608 SAVED** (2026-06-05). gpu-test (T0) activated, parallel with T1.
test-framework.2 (proc macro impl) + iter-demo.1 (1M+ par_iter) in flight.
fusion-codegen.2 done — NVRTC codegen engine with cache, GPU-verified.
iter-runtime theme completed — chained iterators, zero intermediate buffers.

## Recent Decisions
- 2026-06-05: developer-showcase epic completed and archived (47th epic)
- 2026-06-05: Epic criterion 5 rewritten: two-facade architecture (async-gpu host + gpu-runtime kernel)
- 2026-06-05: Brainstorm 117: gpu-test T0 parallel with T1, priority slot
- 2026-06-05: FusionCodegen: NVRTC template codegen with HashMap cache, float4 vectorized
- 2026-06-05: #[gpu_test] design: Option B — build.rs PTX + proc macro #[test] host runner
- 2026-06-05: Standard assert!/assert_eq! already work on GPU via patched std panic handler

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Shuffle as channel primitive: synchronous collective, not point-to-point
- Runtime channel transport detection: shared memory can't be allocated retroactively
- Work-stealing scheduler on GPU: CAS contention + complexity not worth it
- Re-exporting nvptx64 types from host facade: architectural mismatch (different targets)
- Custom GPU assert for Phase 1: unnecessary, std panic handler already sends coordinates

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- 48KB shared memory per block
- Max 2 concurrent subagents (OOM risk)
- PTX JIT ~25min for unified 5MB PTX; cubin rebuild ~10min, only at SAVE step

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS)
- Flash Attention V3: 47-60% of cuDNN FA2
- FusionCodegen: 6 elementwise ops, float4 vectorized, GPU-verified 1e-4 to 1e-6 tolerance
- par_iter chained: 3-deep map chain fully inlined to registers, 16-byte iterator regardless of depth
- 760 tasks completed, 48 epics archived

## Next
1. Wait for test-framework.2 + iter-demo.1 to complete → verify → SAVE
2. Next tasks: test-integration.1, iter-demo.2 (GPU par_iter vs CPU Rayon benchmark)
3. fusion-codegen.3 (register-only intermediates + float4 vectorization)
4. ROUTE: check brainstorm triggers after next batch
