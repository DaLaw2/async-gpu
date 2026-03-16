# ag-norm-bwd.1: LayerNorm backward algorithm
**Cycle**: 424 | **Theme**: ag-norm-bwd | **Kind**: investigation | **Status**: done

## Summary
LayerNorm backward decomposes into dX, dGamma, dBeta using the Jacobian of the normalization.

## Algorithm

Given forward: `y_i = gamma_i * x_hat_i + beta_i` where `x_hat_i = (x_i - mu) / sigma`

Backward (per-row, d = dimension):
```
x_hat = (x - mean) / std
dGamma = sum_batch(dY * x_hat)           # [d]
dBeta  = sum_batch(dY)                    # [d]
dX_hat = dY * gamma                       # [batch, d]
dX = (1/std) * (dX_hat - mean(dX_hat) - x_hat * mean(dX_hat * x_hat))  # [batch, d]
```

Where `mean()` is over the d dimension (per-row mean).

## Implementation Plan
- v1: CPU-side implementation (download, compute, re-upload). Simple and correct.
- v2 (future): Fused PTX kernel for performance.
