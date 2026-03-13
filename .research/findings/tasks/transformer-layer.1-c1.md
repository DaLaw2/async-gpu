# transformer-layer.1: LayerNorm Kernel
**Cycle**: 1 | **Theme**: transformer-layer | **Kind**: experiment | **Status**: done

## Summary
Implemented LayerNorm kernel using warp-level butterfly reduction (shfl.sync.bfly). Each block (1 warp, 32 threads) processes one row of d_model=768 elements: 24 elements per thread. Two-pass reduction: first sum for mean, then sum of squared deviations for variance. Verified against CPU reference with max error 3.34e-6.

## Findings

### Q: Can warp-level reduction efficiently compute mean and variance for d_model=768?
A: Yes. 32 threads each accumulate 24 elements locally, then butterfly reduction (5 steps of shfl.sync.bfly) gives full-warp sum. Two passes: one for mean, one for variance. No shared memory needed — warp shuffle is sufficient.
**Confidence**: high

### Q: What is the numerical precision of f32 reduction for d_model=768?
A: Max error 3.34e-6 compared to CPU f32 reference. The error is at the limit of f32 precision, which means GPU and CPU f32 arithmetic are equivalent for this workload.
**Confidence**: high

### Q: How to handle the affine parameters (gamma, beta)?
A: Each thread applies gamma[idx] * normalized + beta[idx] for its assigned elements. gamma and beta are passed as separate device pointers.
**Confidence**: high

## Design Notes
- `warp_reduce_sum_f32()`: butterfly reduction using `shfl.sync.bfly.b32` with offsets 16, 8, 4, 2, 1.
- Block layout: grid_dim = (num_rows, 1, 1), block_dim = (32, 1, 1).
- No shared memory used — pure register + shuffle.

## Impact on Downstream Tasks
- **transformer-layer.6** (end-to-end): LayerNorm component ready.
