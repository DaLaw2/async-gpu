# showcase-examples — Theme Synthesis

## Progress
- showcase-examples.1: Investigation (done) — mapped cooperative API surface
- showcase-examples.2: Cooperative compute showcase (done) — 5 demos, all verified PASS
- showcase-examples.3: Benchmark update (done) — V4.1 section, updated summary lines

## Verified Conclusions
- Cooperative APIs (cooperative, cooperative_map, cooperative_reduce, cooperative_map_with_params) all work correctly
- `gpu::custom().cubin_file()` enables sub-second kernel loading (vs 10+ min PTX JIT)
- Benchmark auto-dispatch: matmul() routes to V4.1 (large) or cuBLAS (small M)
- SGEMM V4.1: 2691 GFLOPS at 4096^3 (90% cuBLAS), FA V3: 47-60%, Conv2D: 81-229%

## Rejected Approaches
- Keeping two-crate example structure — replaced with single-crate + async-gpu facade
- Relying on PTX JIT for showcase examples — 10+ min startup is unacceptable UX
- Redundant V2 benchmark section — matmul_v2() has same dispatch as matmul()

## Key Metrics
- 5 cooperative compute demos, all PASS
- Benchmark: 6 sections (SGEMM, V4.1, mem-ops, Conv2D, FA, GPT-2 profiling)
- Kernel load time: <1s (cubin) vs 10+ min (PTX JIT)

## Next Steps
- Remaining showcase gaps: async I/O, executor, par_iter examples
- Apply `.cubin_file()` to structured-concurrency and gpu-channels examples
