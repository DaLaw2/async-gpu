# at-framework — Auto-Tuning Framework Synthesis

## Implemented (at-framework.1-3)
- `auto_tune.rs`: AutoTuner, TuningCache, TuningKey, TuningResult
- Warmup-based parameter search, occupancy filtering, size bucketing
- Thread-safe cache (Mutex<HashMap>), lazy per-(kernel, bucket, device)
- 12 unit tests + 6 GPU integration tests (all passing on GTX 1660)
- **Compute-bound demo**: `iterative_math` NVRTC kernel, 1.4x best-vs-worst

## Key Results
- Memory-bound (vector_add): ~5% range — block_size barely matters
- **Compute-bound (iterative_math): 1.4x range** — auto-tuning finds 16% free speedup vs default 256
- High register pressure (16 accumulators) creates occupancy/ILP trade-off across block sizes

## Remaining Gaps
- Most kernels hardcode `* 256` — need `_block_dim_x()` for dynamic tuning
- CUDA event timing for sub-10us kernels
- Disk persistence for cross-session cache
- Story criterion asks 2x; achieved 1.4x (compute-bound) — may need reduction/stencil kernels

## Architecture
Lazy tuning at first call. Wall-clock median (robust to outliers).
Integrates with `resource_report.rs` for occupancy filtering.
