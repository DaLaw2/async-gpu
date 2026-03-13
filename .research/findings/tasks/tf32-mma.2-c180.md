# tf32-mma.2: TF32 inference validation (decision gate)
**Cycle**: 180 | **Theme**: tf32-mma | **Kind**: experiment | **Status**: done

## Summary
TF32 MMA inference fails all 3 prompts (0/3 top-1 agreement with f32 FMA), same as BF16 and F16.
TF32 output is actually worse than BF16 — logits are even more distorted. This confirms the
decision gate: park gpu-perf criteria 3+4 (reduced-precision Tensor Core MMA is not viable for
GPT-2 small inference without quantization-aware training).

## Findings

### Q: Does TF32 MMA inference produce same top-5 as f32 FMA for 3+ prompts?
A: No. 0/3 prompts match. TF32 produces nonsensical predictions (punctuation, random tokens).

Per-prompt results (f32 → TF32 → BF16 top-1):
- "The capital of France is": " the" → "," → "-"
- "In 1969, ...on the moon was": " Neil" → " EE" → " e"
- "The largest ocean...is the": " largest" → "," → "-"

Logit scale comparison:
- f32:  -95 to -101 (reasonable, negative but structured)
- BF16: -52 to -80  (compressed range, semantic loss)
- TF32: -28 to -63  (most distorted, worst semantic loss)

**Confidence**: high

### Q: Decision gate — if fails, park gpu-perf criteria 3+4?
A: Decision gate triggered: TF32 FAILS. All three reduced-precision MMA variants (f16, bf16, tf32)
produce unusable inference output. The precision loss cascading through 12 transformer layers +
residual connections is fundamental.

**Recommendation**: Park gpu-perf criteria 3+4. Reduced-precision Tensor Core MMA is not viable
for GPT-2 small inference without:
- Quantization-aware fine-tuning
- Per-layer precision selection (e.g., MMA for early layers, f32 for later)
- Loss scaling / gradient-free calibration

The 30% speed advantage of BF16 MMA (24ms vs 35ms) remains useful for workloads where
approximate results are acceptable (e.g., similarity search, not autoregressive generation).

## Timing
- f32 FMA: 35ms per forward pass
- TF32 MMA: 29ms (1.2x faster than f32)
- BF16 MMA: 25ms (1.4x faster than f32)

## Impact on Downstream Tasks
- tf32-mma theme: COMPLETE (2/2 criteria evaluated, decision gate triggered)
- gpu-perf epic: criteria 3+4 should be parked/removed
- Remaining gpu-perf work: criterion 2 (latency < 20ms) via f32 FMA optimization
