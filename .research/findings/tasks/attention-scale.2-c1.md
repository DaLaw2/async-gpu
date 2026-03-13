# attention-scale.2: Causal Mask in Current Attention Kernel (seq<=32)
**Cycle**: 1 | **Theme**: attention-scale | **Kind**: experiment | **Status**: done

## Summary
Added `causal_mask` parameter to the `attention_head` kernel. When `causal_mask=1`, positions where j > tid get score = -1e38 (effectively -inf for softmax), preventing attending to future tokens. Both bidirectional and causal modes pass with max_err < 1e-8 against CPU reference. Causal output correctly differs from bidirectional output.

## Findings

### Q: Can causal mask be added with minimal changes to existing attention_head?
A: Yes. Only 3 changes needed: (1) add `causal_mask: u32` parameter, (2) in score computation, skip dot product and write -1e38 when `j > tid`, (3) update non-nvptx fallback to include new param. Total diff: ~15 lines.
**Confidence**: high

### Q: Does masking change the softmax numerics significantly?
A: No. Using -1e38 instead of -inf avoids NaN in exp(). After subtracting max, the masked positions get exp(-1e38 - max) ≈ 0.0, contributing nothing to the sum. The softmax over unmasked positions is numerically identical to computing softmax over only the non-masked positions.
**Confidence**: high

### Q: Does causal-masked output match CPU reference for seq=32?
A: Yes. max_err = 0.00000001 (essentially zero, limited by f32 precision). All 12 heads × 32 positions × 64 dimensions match within 1e-3 tolerance. Zero mismatches.
**Confidence**: high

## Verification
- Bidirectional (backward compat): 0 mismatches, max_err=0.0
- Causal: 0 mismatches, max_err=1e-8
- Causal ≠ bidirectional: confirmed (outputs differ as expected)

## Implementation Notes
- The `causal_mask` parameter is passed as `u32` (not `bool`) since PTX kernel ABI uses u32 for small values.
- `-1e38` was chosen instead of `f32::NEG_INFINITY` to avoid NaN propagation in `exp(x - max)` when max itself is -inf (edge case: position 0 with only itself).
- Existing transformer layer test updated to pass `causal_mask=0u32` for backward compatibility.

## Impact on Downstream Tasks
- **attention-scale.3**: Causal mask pattern is proven. Next challenge is scaling seq>32 (tiled/multi-block attention).
- **full-inference**: Causal attention is ready for GPT-2 (just pass `causal_mask=1`).
