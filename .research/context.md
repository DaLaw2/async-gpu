## Current Focus
**Cycle 623 — 3 tasks verified** (2026-06-06). safety-types.1 (investigation), test-integration.2 (16/16 GPU tests), iter-demo.3 (multi-block par_iter 2.4-3.8x faster than Rayon).

## Recent Decisions
- 2026-06-06: Multi-block par_iter dispatch: grid-stride loop + cached loads, beats Rayon ≤4M
- 2026-06-06: DisjointSlice design: WarpIndex<'scope> + DisjointSlice<'scope, T> adapted from cuda-oxide
- 2026-06-06: Test coverage: 14 GPU features with #[gpu_test], test-integration theme nearly complete
- 2026-06-05: kernel-split + auto-fusion epics completed (49th + 50th)

## Tried & Rejected
- Single-block par_iter benchmark: par_iter_map_collect requires gpu_main/hostcall setup, hangs when launched via plain cudarc
- ptxas optimization via PTX size reduction: scales with complexity not size
- Epilogue-only fusion for 10% GPT-2: GEMM dominates ~85%

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents
- PTX JIT takes ~15-20 min for 6.7MB PTX; CUDA cache eliminates repeat cost
- par_iter_map_collect (single-block) kernel requires hostcall buffer setup — cannot benchmark via plain cudarc launch

## Key Metrics
- 4 kernel crates: core (17), compute (84), io (55), test (76+14) entry points
- Dev rebuild: 27.5s (PTX JIT) — was 30 min (unified ptxas)
- Multi-block par_iter: 0.27-0.41x Rayon (≤4M), 1.43x at 16M (PCIe bottleneck)
- GPU tests: 16/16 pass (14 GPU features + 2 CPU)
- 780 tasks completed, 50 epics

## Next
1. ROUTE: check epic completion gates (gpu-test, gpu-iterator)
2. gpu-type-safety: safety-types.2 (implement DisjointSlice + WarpIndex)
3. iter-demo.4 (re-benchmark multi-block vs Rayon) — may be redundant now
4. iter-demo.5 (multi-block fold/sum via two-pass reduction)
