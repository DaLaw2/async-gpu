# graph-bfs.3: GPU PageRank (iterative SpMV) + convergence check

**Status**: Complete
**Date**: 2026-03-17

## Summary

Implemented GPU PageRank via iterative sparse matrix-vector multiplication (SpMV)
using CUDA NVRTC, alongside a CPU reference implementation. Both produce identical
results and converge in the same number of iterations.

## Implementation

### CPU PageRank
- Standard pull-based iterative PageRank: `PR(v) = (1-d)/N + d * sum(PR(u)/out_deg(u))`
- Damping factor d = 0.85, convergence threshold epsilon = 1e-6 (L1 norm), max 100 iterations
- Uses transposed CSR graph for in-neighbor iteration

### GPU PageRank
- One CUDA thread per vertex, each pulls from in-neighbors via transposed CSR
- Per-iteration kernel launch with host-side convergence check (download L1 delta)
- `atomicAdd` for global L1 delta accumulation across all threads
- Double-buffered PageRank vectors (swap `pr_in`/`pr_out` each iteration)

### Graph Transpose
- Added `CsrGraph::transpose()` method to reverse all edge directions
- Required because PageRank pulls from in-neighbors, but original CSR stores out-neighbors

## Results (RMAT scale=17, 131K vertices, ~1.9M edges)

| Metric           | CPU      | GPU      |
|------------------|----------|----------|
| Iterations       | 51       | 51       |
| Sum(PR)          | 0.6259   | 0.6259   |
| Time             | 167.9 ms | 193.3 ms |
| Speedup          | —        | 0.87x    |

- **Verification**: PASS — max absolute error = 0.0 (exact match within f32)
- **Sum < 1.0**: Expected — dangling nodes (out-degree=0) leak rank. Standard PageRank
  without dangling-node redistribution does not conserve probability mass.

## Observations

1. GPU is slightly slower at this graph size (131K vertices) due to kernel launch
   overhead per iteration (51 launches + synchronize). At larger scales the GPU
   SpMV will dominate.
2. The atomic-based delta accumulation works correctly and matches CPU convergence
   behavior exactly (same iteration count).
3. Both implementations produce bit-identical results (max error = 0), confirming
   the pull-based formula is deterministic when neighbor order is sorted.

## Files Modified
- `examples/std/graph-algorithms/src/main.rs` — added `transpose()`, `cpu_pagerank()`,
  `gpu_pagerank()`, PageRank CUDA kernel, and main() PageRank section
