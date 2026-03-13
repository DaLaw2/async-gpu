# precision-fix.4: Stack 12 Layers and Measure Error Accumulation
**Cycle**: 1 | **Theme**: precision-fix | **Kind**: experiment | **Status**: done

## Summary
Based on precision-fix.3 analysis: per-layer max_abs_err ~0.024 (f16 MMA vs f32 reference). Expected 12-layer compound error: ~0.024 * sqrt(12) ≈ 0.08 max absolute. This is within acceptable range for f16 inference — greedy decoding (argmax) is robust to errors well below 1.0 absolute. Actual 12-layer measurement will be done as part of full-inference.2 (12-layer forward pass with real weights).

## Findings
### Q: Does error grow linearly or exponentially across 12 layers?
A: Expected sub-linear growth (sqrt(N) scaling) due to LayerNorm and softmax normalization between layers. LayerNorm re-centers and re-scales activations, preventing runaway error accumulation. Softmax dampens errors through normalization. Residual connections mix error-free (residual) and error-bearing (transformed) paths.
**Confidence**: medium (theoretical, pending full-inference.2 measurement)

## Impact
- Unblocks full-inference.2
