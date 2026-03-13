# mixed-precision.2: BF16 mixed-precision inference validation
**Cycle**: 178 | **Theme**: mixed-precision | **Kind**: experiment | **Status**: done

## Summary
Ran 3 prompts through both BF16 MMA and f32 FMA inference pipelines. Top-1 agreement: 0/3.
Top-5 overlap: 0-2/5. BF16 is ~30% faster than f32 FMA (24ms vs 35ms). Confirms that
reduced-precision MMA (bf16 or f16) causes too much accumulated error over 12 transformer layers
for inference-quality output. This matches the f16 MMA finding from mma-fix.3.

## Findings

### Q: Does BF16 MMA inference produce same top-5 tokens as f32 FMA for 3+ prompts?
A: No. 0/3 prompts have top-1 agreement, and top-5 overlap ranges from 0 to 2 tokens.

Results per prompt:
- "The capital of France is": f32→" the", bf16→"-" (overlap 2/5)
- "In 1969, the first man to walk on the moon was": f32→" Neil", bf16→" e" (overlap 0/5)
- "The largest ocean on Earth is the": f32→" largest", bf16→"-" (overlap 0/5)

The BF16 output produces nonsensical predictions (punctuation, function words) while f32 FMA
produces contextually relevant predictions. Both have negative logits (GPT-2 small limitation).
**Confidence**: high

### Q: What is the per-token latency with BF16 MMA vs f32 FMA?
A: BF16 MMA is ~30% faster:
- BF16: 24-26ms per forward pass
- f32 FMA: 35ms per forward pass

This speedup comes from Tensor Core acceleration for GEMM operations, which dominate the
12-layer transformer pipeline.
**Confidence**: high

## Unexpected Discoveries
- BF16 advantage over F16: no host-side f16 packing needed (accepts raw f32 column-major B matrix).
  This simplifies the inference pipeline significantly.
- The logit scale differs dramatically: f32 logits are around -95 to -101, while BF16 logits are
  around -52 to -80. The accumulated GEMM precision loss cascades through residual connections.

## Open Questions
- TF32 mode (19-bit mantissa instead of 7-bit for bf16 or 10-bit for f16) might provide
  sufficient precision for inference. sm_86 supports TF32 via mma.sync with .tf32 qualifier.
- Alternatively, a hybrid approach: use MMA for the larger GEMMs (FFN up/down) and f32 FMA
  for smaller ones (QKV projection, output projection) to balance speed and precision.

## Impact on Downstream Tasks
- Mixed-precision theme success criteria NOT fully met:
  - Criterion 1 (GEMM match): MET — bf16 matches f16 within tolerance at all dims
  - Criterion 2 (inference match): NOT MET — top-5 diverges from f32 FMA
- TF32 is the logical next step if higher MMA precision is needed for inference.
- The 30% speed advantage of BF16 MMA is real but unusable without inference quality.
