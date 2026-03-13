# mma-fix.3: MMA inference end-to-end validation
**Cycle**: 171 | **Theme**: mma-fix | **Kind**: experiment | **Status**: done

## Summary
MMA-based GPT-2 forward pass runs successfully (zero NaN/Inf) and is 1.8x faster
than f32 FMA (25.3ms vs 46.2ms). However, top-5 predictions diverge significantly
from the f32 baseline due to accumulated f16 precision loss over 12 transformer layers.

## Findings

### Q: Does MMA inference produce same top-5 tokens as f32 FMA for 3+ prompts?
A: **No.** For prompt "The capital of France is":
- f32 FMA top-5: [" the", " now", " a", " France", " Paris"]
- MMA top-5: ["-", " and", ".", " a", " the"]
- Only 2 of 5 tokens overlap (" a", " the"), in different positions.
- Hidden state magnitudes differ significantly (max|val|: 218.15 vs 98.97).
**Confidence**: high

### Q: What is the per-token latency with MMA vs f32 FMA?
A: MMA = 25.3ms, f32 FMA = 46.2ms → 1.83x speedup from Tensor Cores.
Note: both are single-token forward passes (seq=32, actual=5 tokens).
**Confidence**: high

## Root Cause Analysis
The `full_gemm_f32in` kernel converts A (activations) from f32 to f16 on-the-fly
before the MMA instruction, and B (weights) are pre-packed as f16x2. Although MMA
uses f32 accumulators, the f16 input precision (5-bit mantissa) loses significant
information at each GEMM. Over 12 layers × 4 GEMMs/layer = 48 f16 truncations, the
error compounds to change the output distribution.

## Possible Mitigations (future work)
1. **TF32 Tensor Cores** (sm_80+): `mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32` uses
   10-bit mantissa (vs f16's 5-bit). Ampere+ only. Would need a new kernel variant.
2. **Mixed precision**: Use MMA only for GEMM, keep residual/LN/attention in f32
   (already done), but the issue is A→f16 conversion at each layer.
3. **Stochastic rounding**: Add random rounding during f32→f16 conversion to reduce
   systematic error accumulation (complex to implement).
4. **BF16**: `mma.sync` with bf16 inputs has 8-bit exponent (same as f32) which
   prevents the magnitude drift seen here.

## Impact on Downstream Tasks
- MMA kernel is functionally correct (verified in mma-fix.1) and fast (1.8x speedup)
- Current f16 precision is insufficient for 12-layer inference quality
- KV cache optimization (kv-cache.*) should proceed with f32 FMA first — it targets
  redundant recomputation, not GEMM speed
- Future TF32/BF16 MMA kernel variant could combine speed + precision
