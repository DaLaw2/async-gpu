# full-inference.5.1: Diagnostic — per-layer max-abs-value instrumentation
**Cycle**: 162 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Added per-layer diagnostic instrumentation to the 12-layer forward pass and
identified the root cause of NaN: padded positions (rows beyond actual_seq)
develop NaN through transformer layers, which then contaminates valid positions
via GEMM shared memory tiles. Fixed by adding a `zero_pad` kernel that zeros
padded rows after embedding and after each residual add.

## Findings
### Q: Which layer first produces values exceeding f16 range (65504)?
A: **None.** Max values across all layers stayed well within f16 range:
Layer 0: 19.89, Layer 2: 321.45, Layer 11: 651.43. No value exceeded 65504.
The brainstorm's hypothesis that NaN came from f16 range overflow was incorrect.
**Confidence**: high

### Q: Is the NaN from f16 range overflow or another source?
A: **GEMM shared memory tile contamination from padded positions.** The root
cause chain:
1. Padded row 8 (actual_seq=5, rows 5-31 padded) develops NaN after ~2 layers
   because padded positions carry real token-0 embeddings that transform through
   bias adds and residual connections, eventually producing unstable values
2. LayerNorm propagates NaN from padded rows to its output
3. GEMM kernel loads all 32 rows (one tile) into shared memory, including
   NaN rows. The MMA instruction produces NaN output for ALL rows in the tile
   when ANY input row contains NaN
4. This contaminates pos 0 (and all other valid positions) via shared memory

**Fix**: Added `zero_pad` kernel that zeros elements at index >= actual_seq*d_model.
Applied after embedding and after each residual add (2 per layer = 24 calls + 1).
This ensures padded rows are always zero, so LN(zeros) = beta (finite), and
GEMM never sees NaN in its shared memory tiles.

**Result**: All 12 layers produce zero NaN. Generation runs full 20 tokens.
**Confidence**: high

## Key Diagnostic Data (before fix)
```
Layer 0: max|val|=18.55
Layer 1: max|val|=234.31
  L2 hidden: row8 nan=768, NaN rows: [8]
  L2 after LN1: pos0 nan=0 (LN cleans pos0 from non-NaN input)
  L3 hidden: NaN rows: [0, 8, 16, 17, 18, 19, 20, 21]
  L3 after LN1: pos0 nan=0, NaN rows: [8]
  L3 after QKV GEMM: pos0 nan=2304  ← GEMM tile contamination!
```

## Key Diagnostic Data (after fix)
```
Layer 0-11: zero NaN in all positions
Max values: 19.89 → 651.43 (monotonically growing but within f16 range)
Generation: 20 tokens, 59ms/token, no NaN interruption
```

## Impact on Downstream Tasks
- **Critical correction**: NaN was NOT from f16 precision — it was a padding bug
- f16 GEMM precision still causes wrong top-1 prediction ("-" not "Paris")
- full-inference.6 (f32 GEMM) is now more important to validate whether
  precision alone determines correct predictions
- The zero_pad fix must be applied to any future inference code
