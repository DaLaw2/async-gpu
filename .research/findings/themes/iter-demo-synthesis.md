# iter-demo — Iterator demos synthesis

## Status
Active. iter-demo.1 (1M+ map/collect demo) done.

## Key results
- par_iter().map().collect_into() correct at 1M+ f32 elements
- Chained .map().map() fusion verified at scale: zero intermediate buffers
- Triple-map + sum (deep fusion + reduction) works at 1M elements
- f32 reduction precision: ~1e-4 relative error vs f64 reference
- Single-block architecture (4 warps) handles large N via warp stripes

## Architecture insight
Current par_iter uses 1 block / 4 warps. For large N this means only
1 of 22 SMs is utilized on GTX 1660. Multi-block dispatch would
improve throughput but requires cross-block reduction coordination.
The iterator API is correct regardless of parallelism level.

## What's next
- Rayon comparison benchmark (par_iter vs rayon::par_iter on CPU)
- Multi-block par_iter for higher GPU utilization
- Filter + collect at 1M scale (atomic compaction stress test)
