# at-framework.2: Warmup-Based Parameter Search with Result Caching

**Status**: DONE
**Kind**: experiment

## Summary

Implemented the auto-tuning framework in `crates/core/gpu-host/src/auto_tune.rs` with
warmup-based parameter search and thread-safe result caching. Verified on real GPU
(GTX 1660, sm_75) with the `vector_add` kernel across problem sizes from 16K to 1M elements.

## Findings

### 1. Module Architecture

Three main types compose the framework:

| Type | Role |
|------|------|
| `TuningKey` | Composite cache key: (kernel_name, problem_size_bucket, device_id) |
| `TuningCache` | Thread-safe HashMap<TuningKey, TuningResult> with Mutex |
| `AutoTuner` | Orchestrator: candidate generation → warmup → measurement → selection |

Supporting types: `AutoTunerConfig` (search parameters), `TuningResult` (best config + all timings).

### 2. Algorithm

```
1. Generate candidates: [32, 64, 128, 256, 512, 1024] (configurable)
2. Filter by occupancy: reject block sizes with occupancy < min_occupancy_pct
   (uses KernelResources::occupancy from resource_report.rs)
3. For each candidate:
   a. Run warmup_iterations (default 3) — discard results
   b. Run measure_iterations (default 7) — collect wall-clock times
   c. Take the median (robust to outliers)
4. Select the candidate with lowest median time
5. Cache result keyed by (kernel_name, bucket(problem_size), device_id)
```

Problem-size bucketing uses next_power_of_two with a minimum of 32.

### 3. GPU Benchmark Results (GTX 1660, sm_75)

**vector_add kernel — memory-bound (global read + global write per element)**

| N | Best Block | Default (256) | Speedup | Notes |
|---|-----------|---------------|---------|-------|
| 16K (2^14) | 512 | 7.47µs | 1.04x | Small: scheduling overhead dominates |
| 64K (2^16) | 128 | 8.53µs | 1.01x | Medium: all configs within 15% |
| 256K (2^18) | 256 | 26.53µs | 1.00x | Default is already optimal |
| 1M (2^20) | 64/256 | ~86µs | 1.00x | Bandwidth-saturated, block size matters little |

**Key insight**: For bandwidth-bound kernels like `vector_add`, block size has modest impact
(~5% range excluding the pathological block_size=32 case). The 256 default is within 1-4% of
optimal across all sizes. Bigger speedups are expected for compute-bound kernels (GEMM, attention)
where occupancy and register pressure dominate.

### 4. Occupancy Filtering Validation

With a synthetic 255-register kernel on sm_75:
- Block sizes 512 and 1024 are correctly filtered out (occupancy < 12%)
- Smaller blocks (32-256) pass the filter
- This prevents wasting benchmark time on configurations that cannot execute efficiently

### 5. Caching Validation

- First call to `tune_or_cached` runs the full benchmark (warmup + measure for each candidate)
- Second call returns the cached result with zero benchmark invocations
- Problem-size bucketing groups nearby sizes: N=60000 and N=65536 share the same cache entry

### 6. Integration Path

The framework integrates with existing infrastructure:
- `CustomLaunchBuilder` already has `.threads()` / `.grid()` / `.shared_mem()` — auto-tuner
  produces `LaunchConfig` that can replace these
- `KernelRegistry::config_1d(n)` currently hardcodes block_size=256 — can be enhanced to
  consult TuningCache
- `resource_report.rs` occupancy calculator is used directly for candidate filtering

## Open Questions

1. **Wall-clock vs CUDA events**: Current implementation uses `Instant::now()` + `synchronize()`.
   CUDA events (`cuEventRecord` / `cuEventElapsedTime`) would be more precise for short kernels
   (sub-10µs), but require raw driver API calls not wrapped by cudarc. Wall-clock is sufficient
   for most practical tuning since we take medians over multiple runs.

2. **Compute-bound kernel validation**: The `vector_add` benchmark shows modest speedups because
   it's memory-bound. The real payoff will come from tuning GEMM/attention/LayerNorm kernels
   where occupancy directly affects performance. However, most existing kernels hardcode
   `block_x * 256 + tid` instead of using `_block_dim_x()`, which prevents block-size tuning
   without kernel modifications.

3. **Per-kernel block-dim flexibility**: To get the "2x faster" story criterion, we need kernels
   that (a) use dynamic `_block_dim_x()` and (b) have meaningful compute/occupancy sensitivity.
   The current kernel set is mostly hardcoded to 256. This is a future task.

4. **Persistence**: The current cache is in-memory only (lost on process exit). Disk persistence
   (serialize to JSON/bincode) would enable cross-session reuse. Not implemented yet.
