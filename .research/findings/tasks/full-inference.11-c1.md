# full-inference.11: Switch forward/generation from broken MMA to correct f32 GEMM
**Cycle**: 1 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Switched both `run_full_forward_test` and `run_generation_test` from the broken
`full_gemm_f32in` (MMA tensor core) kernel to the correct `gemm_f32` (pure f32 FMA)
kernel. This fixes GPT-2 inference producing completely wrong predictions.

## Findings

### Q: Does switching to gemm_f32 fix forward pass predictions?
A: YES. Forward pass now matches CPU f64 exactly:
- GPU QKV GEMM at layer 0, pos 4: `[-0.191336, -0.079322, 0.937499, 0.162505]`
- CPU f64 reference:              `[-0.191336, -0.079322, 0.937499, 0.162505]`
- Top-5 predictions: " a" (-68.42), " the" (-69.05), " not" (-69.33) — matches CPU f64
**Confidence**: high

### Q: Does generation produce coherent English text?
A: YES — partially. Generation produces 22 tokens of English text:
`" a new "un-femuls and the most likely to the most likely to be more likely."`
This is recognizable English (grammatically reasonable), not random garbage.
However, NaN appears at step 22 (all 768 values), halting generation.
**Confidence**: high

### Q: What caused the NaN at step 22?
A: The NaN is a separate issue from the GEMM bug. Hidden state norms grow with
each generation step (residual connections accumulate). At seq=22+5=27 positions,
f32 precision in attention softmax likely overflows. The CPU f64 reference shows
norms growing from 56.5 (layer 0) to 429.6 (layer 11) even for 5-token input.
At 27 tokens, the norms are larger and f32 precision becomes insufficient.
**Confidence**: medium

## Changes Made
1. `run_full_forward_test`: replaced `full_gemm_f32in` → `gemm_f32`, `pack_weight` → `to_col_major`, `CudaSlice<u32>` → `CudaSlice<f32>`, K params from `D/16` → `D`
2. `run_generation_test`: same changes as above
3. Both tests now use `gemm_shared = (32 * 16 + 16 * 16) * 4` instead of `(256 + 128) * 4`

## Root Cause Analysis
The `full_gemm_f32in` MMA kernel has an indexing/layout bug in its tensor core
instruction usage. It produces completely wrong results for the same inputs where
`gemm_f32` produces results matching CPU f64 exactly. The bug is confined to the
MMA kernel — all other components (embedding, LayerNorm, attention, GELU, bias,
split_qkv, concat_heads) are verified correct.

## Open Questions
1. What exactly is wrong with `full_gemm_f32in`? (needs separate debugging task)
2. Can the NaN at step 22 be fixed by using f32 attention accumulator or online softmax normalization?
3. Should we investigate KV cache to avoid full recompute and reduce numerical drift?

## Impact on Downstream Tasks
- GPU inference now produces correct single-step predictions
- Generation works but is limited to ~20 tokens before NaN
- The MMA kernel bug is a separate issue for the gpu-pipeline theme to fix
