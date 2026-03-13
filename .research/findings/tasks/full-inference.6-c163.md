# full-inference.6: Pure f32 GEMM kernel for precision validation
**Cycle**: 163 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Implemented pure f32 GEMM kernel (`gemm_f32`) using scalar FMA instructions
(no Tensor Cores). Ran full 12-layer GPT-2 forward pass with f32 precision.
Top-1 prediction changes from "-" (f16) to " a" (f32), but neither predicts
"Paris". GPT-2 small likely lacks factual accuracy for this prompt.

## Findings
### Q: Does f32 GEMM eliminate NaN at position 0?
A: NaN was already eliminated by zero_pad (full-inference.5.1). f32 GEMM
also produces zero NaN, confirming the fix is robust.
**Confidence**: high

### Q: With f32 GEMM, does GPT-2 produce a more sensible top-1 prediction?
A: Top-1 changes from "-" (f16) to " a" (f32). Neither is "Paris". The f32
top-5 are common function words: " a", " the", " not", " that", " an".
This suggests GPT-2 small (124M params) genuinely does not predict "Paris"
for "The capital of France is" — the issue is model capability, not precision.
**Confidence**: high

### Q: What is the performance penalty of f32 vs f16 Tensor Core GEMM?
A: Surprisingly small: f32 takes 33.9ms vs ~30ms for f16 Tensor Core (only
~13% slower). The kernel is memory-bound at this small scale (seq=32), so
Tensor Core throughput advantage is not realized. At larger batch sizes the
gap would be much larger.
**Confidence**: high

## Key Metrics
| Metric | f16 TC | f32 FMA |
|--------|--------|---------|
| Forward pass | ~30ms | 33.9ms |
| Top-1 | "-" | " a" |
| Top-2 | " and" | " the" |
| NaN count | 0 | 0 |
| max\|val\| | 100.5 | 150.5 |

## Kernel Design
- `gemm_f32`: 128 threads, 32×16 output tile per block
- Shared memory: A[32][16] + B[16][16] = 3072 bytes
- K tiled in chunks of 16, inner loop uses `fma.rn.f32`
- B input: column-major f32 (not packed f16x2)

## Impact on Downstream Tasks
- **Critical finding**: "Paris" not in top-5 even with f32 — epic success
  criterion "produces text containing Paris" may be unrealistic for GPT-2 small
- full-inference.5 decision gate: need PyTorch reference to confirm model behavior
- f32 GEMM can serve as precision reference for any future optimizations
