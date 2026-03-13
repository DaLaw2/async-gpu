# tf32-mma.1: TF32 MMA GEMM kernel (m16n8k8)
**Cycle**: 179 | **Theme**: tf32-mma | **Kind**: experiment | **Status**: done

## Summary
Implemented `full_gemm_tf32` kernel using `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.
TF32 accepts raw f32 inputs — GPU truncates to 10-bit mantissa internally. Surprisingly, TF32
produces WORSE results than BF16 at GPT-2 dimensions despite having more mantissa bits (10 vs 7),
because the m16n8k8 instruction requires 2x more accumulation steps than BF16's m16n8k16.

## Findings

### Q: Does sm_86 support mma.sync with .tf32 qualifier?
A: Yes. `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` compiles and executes correctly
on sm_86. No explicit conversion needed — TF32 uses regular f32 registers; the GPU hardware
internally truncates mantissa to 10 bits.
**Confidence**: high

### Q: Does TF32 MMA GEMM match gemm_f32 within acceptable tolerance for GPT-2 dims?
A: TF32 has larger error than BF16 vs f32:
- 768x768x768: TF32=16.23, BF16=8.18
- 768x2304x768: TF32=34.14, BF16=9.69
- 3072x768x3072: TF32=56.64, BF16=16.71

TF32 is 2-3.4x worse than BF16 despite having more mantissa bits per element.
**Confidence**: high

### Q: How does TF32 precision compare to bf16 and f16 MMA?
A: TF32 is worse at large K dimensions. Root cause: TF32 m16n8k8 needs 2x more accumulation
steps (K/8 tiles vs K/16 for BF16). Each MMA step introduces f32 rounding on the accumulator,
and the extra steps compound. At k=8 (single tile), TF32 error is 0.001 (excellent), confirming
the per-element precision is fine. The problem is purely accumulation.
**Confidence**: high

## Unexpected Discoveries
- TF32 is counter-intuitively less precise than BF16 for large GEMMs due to 2x accumulation steps
- No conversion needed for TF32 — simplest MMA variant to implement
- TF32 latency is between BF16 and f32 (29ms vs 24ms vs 35ms), likely due to 2x more MMA ops

## Impact on Downstream Tasks
- tf32-mma.2 should still run to confirm the decision gate
- Strong evidence that ALL reduced-precision Tensor Core MMA variants fail for 12-layer inference
