## Current Focus
**developer-showcase epic ready for Epic Verification Gate** (2026-06-05).
All 5 themes completed: showcase-api, showcase-sc, showcase-channels, showcase-examples, showcase-readme.
7 tasks completed this cycle (607): showcase-readme.1-.5, showcase-examples.2-.3.
T0 has 2 epics: developer-showcase (all criteria appear met), gpu-test (pending).
T1 active: gpu-iterator (iter-design + iter-compiler done, iter-runtime.2 + iter-demo pending).
T1 active: auto-fusion (fusion-analysis done, fusion-codegen.2 next).

## Recent Decisions
- 2026-06-05: README hero snippet: File::read → matmul → File::write (Project North Star vision)
- 2026-06-05: Feature matrix: 4 groups (Language/Runtime, I/O, Compute, ML/AI), 30 features
- 2026-06-05: Progressive snippets: hello→cooperative→SC pipeline, placed after Quick Start
- 2026-06-05: Perf table: FA V3 47-60% filled in, GTX 1660 vs A2 hardware footnotes added
- 2026-06-05: warp-cooperative: single-crate showcase with 5 demos, cubin_file() API added
- 2026-06-05: gpu_main→gpu_main_poll in thread_test.rs to prevent bar.sync deadlocks

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Shuffle as channel primitive: synchronous collective, not point-to-point
- Runtime channel transport detection: shared memory can't be allocated retroactively
- Work-stealing scheduler on GPU: CAS contention + complexity not worth it
- MIR pass for scope enforcement: maintenance cost >> marginal safety gain
- Nested block_scope from worker warps: allocator not thread-safe, warp exhaustion

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32, no cp.async
- 48KB shared memory per block — BlockScope allocations limited
- Max 2 concurrent subagents (OOM risk)
- Warp 0 only for scope allocation (single-writer invariant)

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS)
- Flash Attention V3: 47-60% of cuDNN FA2
- GPT-2 forward: 25.1ms (A2) / 39.4ms (GTX 1660)
- README: 30-feature matrix, 3 progressive snippets, North Star hero, full perf table
- ci-lint: 39/39 checks pass

## Next
1. ROUTE: Epic Verification Gate for developer-showcase
2. If PASS → cascade close developer-showcase, T0 gate check (gpu-test still pending)
3. Continue T1: iter-runtime.2 (chained fusion), iter-demo.1 (par_iter 1M+ demo)
4. Continue T1: fusion-codegen.2 (generate fused PTX from fusion graph)
