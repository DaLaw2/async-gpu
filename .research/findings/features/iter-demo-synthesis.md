# iter-demo — Iterator demos synthesis

## Status
Active. iter-demo.1 (1M+ correctness) done. iter-demo.2 (Rayon benchmark) done.

## Key results
- par_iter().map().collect_into() correct at 1M+ f32 elements
- Chained .map().map() fusion verified: zero intermediate buffers at scale
- GPU par_iter NEVER beats CPU Rayon at any size (1K-16M) with current arch
- GPU is 5-1178x slower than Rayon (worst at 1M: single-block bottleneck)
- Root cause: 1 block / 4 warps = 4.5% SM utilization + volatile loads bypass cache
- Shared memory bug found: launch_config needs shared_mem_bytes > 0

## Crossover estimate (with multi-block)
With 22-block launch + cached loads, expect GPU crossover at ~100K-1M elements.
Rayon achieves ~10 GB/s at 16M; GTX 1660 peak is 192 GB/s (19x headroom).

## What's next
- Fix shared_mem_bytes bug in existing par_iter tests
- Multi-block par_iter dispatch for higher GPU utilization
- Re-benchmark after multi-block to find real crossover point
