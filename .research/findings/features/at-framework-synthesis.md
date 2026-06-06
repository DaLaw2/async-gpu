# at-framework — Auto-Tuning Framework Synthesis

## Implemented (at-framework.2)
- `auto_tune.rs` module: `AutoTuner`, `TuningCache`, `TuningKey`, `TuningResult`
- Warmup-based parameter search: configurable candidates, warmup, measurement iterations
- Occupancy-based candidate filtering via `resource_report::KernelResources::occupancy()`
- Problem-size bucketing (next_power_of_two, min 32)
- Thread-safe cache with Mutex<HashMap>
- 12 unit tests + 4 GPU integration tests (all passing on GTX 1660)

## GPU Results (vector_add, memory-bound)
- Block size sensitivity: ~5% range (excluding pathological block_size=32)
- Default 256 is within 1-4% of optimal for bandwidth-bound kernels
- Bigger speedups expected for compute-bound kernels (GEMM, attention)

## Remaining Gaps
- Most kernels hardcode `* 256` instead of using `_block_dim_x()` — limits tuning
- CUDA event timing (more precise than wall-clock for sub-10µs kernels)
- Disk persistence for cross-session cache reuse
- Compute-bound kernel demo for the "2x faster" story criterion

## Architecture
Lazy tuning at first call per (kernel, size_bucket, device).
Wall-clock timing with median selection (robust to outliers).
Integrates with existing `resource_report.rs` for occupancy filtering.
