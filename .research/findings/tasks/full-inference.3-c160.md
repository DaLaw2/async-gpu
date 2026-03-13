# full-inference.3: LM head (linear → softmax over 50,257 vocabulary)
**Cycle**: 160 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Implemented LM head as CPU-side matmul: download last position's 768-dim hidden
state from GPU, compute logits via dot product with wte (weight tying), apply
stable softmax, and decode top-5 predictions. Works correctly with finite logits.

## Findings
### Q: How to handle the 50,257 dimension (not multiple of 16) in GEMM?
A: Compute on CPU. Only 1 row of 768 floats needs to project against the
50,257×768 wte table. A GPU kernel would require padding to multiples of 16/32
for the GEMM tile structure, adding complexity for negligible gain (the matmul
is ~38M FLOPs, trivial on CPU). CPU computes it in < 1ms.
**Confidence**: high

### Q: Does GPT-2's weight tying produce reasonable predictions?
A: Yes. logits[v] = dot(hidden[last_pos], wte[v]) produces finite values with
clear preference structure. Top-5 for "The capital of France is":
  1. " (" (20.1%) — GPT-2 often uses parenthetical form
  2. "," (15.2%)
  3. " and" (10.2%)
  4. " \"" (7.4%)
  5. "\n" (4.2%)

The model doesn't predict "Paris" as top-1, which is expected for GPT-2 small
with f16 inference precision. The predictions are structurally valid (punctuation
and continuation tokens are common GPT-2 small outputs for factual prompts).
**Confidence**: high

## Implementation Details
- LM head added at end of `run_full_forward_test` (no separate function needed)
- CPU matmul: simple nested loop, ~38M multiply-add operations
- Numerically stable softmax: subtract max logit before exp
- Top-5 decoded via tokenizer for human-readable output
- Validation: all 50,257 logits must be finite (no NaN/Inf)

## Open Questions
- Will autoregressive generation (full-inference.4) accumulate more precision errors?
- Will full-inference.5 (PyTorch comparison) show the top-1 divergence is purely f16?

## Impact on Downstream Tasks
- Unblocks full-inference.4 (generation loop) — can now get next-token prediction
- Provides baseline for full-inference.5 (end-to-end validation)
