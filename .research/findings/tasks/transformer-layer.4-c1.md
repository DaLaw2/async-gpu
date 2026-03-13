# transformer-layer.4: FFN Block
**Cycle**: 1 | **Theme**: transformer-layer | **Kind**: experiment | **Status**: done

## Summary
Validated full FFN pipeline: linear(768→3072) → bias → GELU → linear(3072→768) → bias. Uses 7 kernel launches: f32→f16x2 pack, GEMM, bias_add, GELU, pack, GEMM, bias_add. Verified against CPU reference: max abs error 0.0009, zero mismatches.

## Findings

### Q: Can FFN (768→3072→768) be computed with the scaled GEMM?
A: Yes. Two `full_gemm` calls handle the projections. Added helper kernels: `bias_add` (in-place bias addition), `f32_to_f16x2_pack` (f32→f16x2 conversion using PTX `cvt.rn.f16.f32`). The pipeline requires f32→f16 conversion between GEMM stages since MMA requires f16 input.
**Confidence**: high

### Q: What is the memory footprint for FFN intermediate activations?
A: For seq=32: hidden [32×3072] f32 = 384 KB, packed [32×1536] u32 = 192 KB. Total intermediate: ~576 KB. Well within GPU memory limits.
**Confidence**: high

## Design Notes
- New kernels: `bias_add` (element-parallel in-place), `f32_to_f16x2_pack` (PTX cvt instruction).
- 7 kernel launches per FFN block — could be fused but not necessary for correctness validation.
- f16 quantization introduces max abs error ~0.001 per stage, acceptable for transformer inference.

## Impact on Downstream Tasks
- **transformer-layer.6** (end-to-end): FFN component ready. All sub-components now verified.
