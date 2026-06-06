# at-framework — Auto-Tuning Framework Synthesis

## Tunable Parameters (by impact)
1. **Block size** — 128/256 defaults; 6 candidates [32..1024]; constrained by register count
2. **Kernel variant** — V1/V2/V3/V4.1 dispatch already exists for GEMM
3. **Shared memory** — 0 to 16640B; determines tile size for GEMM/attention
4. **Grid dims** — derived from problem size / block size / tile dims
5. **Stream** — default vs dedicated; matters only for multi-kernel pipelines

## Key Insight
`cuOccupancyMaxPotentialBlockSize` gives occupancy-optimal block size for free
(available via raw driver API). Use as baseline; empirical tuning only for hot paths.

## Existing Infrastructure
- `resource_report.rs`: occupancy calculator, register constraints, SmConfig
- `CustomLaunchBuilder`: already parameterized (.threads/.grid/.shared_mem)
- Benchmark pattern: warmup(3) + measure(10) + synchronize already in use

## Missing Pieces
- CUDA event timing (need raw cuEvent* calls for GPU-side measurement)
- Tuning cache (HashMap keyed by kernel+problem_size+device)
- Candidate generator + constraint filter
- Problem-size bucketing for GEMM/attention

## Architecture Decision
Lazy tuning at first call per (kernel, size_bucket). Cache in GpuRuntime.
Start with cuOccupancy API, add empirical tuning for GEMM/attention.
