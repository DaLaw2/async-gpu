# transformer-layer.3: Multi-Head Attention
**Cycle**: 1 | **Theme**: transformer-layer | **Kind**: experiment | **Status**: done

## Summary
Implemented per-head scaled dot-product attention kernel. Each block (1 warp, 32 threads) processes one attention head: Q×K^T/sqrt(d_head) → softmax → ×V. Verified with 12 heads, seq=32, d_head=64 against CPU reference — max error 0.0 (perfect match).

## Findings

### Q: Can per-head attention be efficiently computed with one warp per head?
A: Yes, for small seq_len (≤32). Each thread handles one query position: computes dot products with all keys (O(seq×d_head) per thread), applies softmax over seq_len scores, then weighted sum over values. For seq=32, d_head=64: each thread does 32×64=2048 FMAs for scores + 32×64=2048 FMAs for output = 4096 FMAs. Warp-level parallelism across heads (12 blocks).
**Confidence**: high

### Q: What memory layout for Q/K/V minimizes bank conflicts?
A: Used [n_heads][seq_len][d_head] f32 layout (head-major). Each block reads only its head's data contiguously. Scores stored in shared memory [seq_len×seq_len] for the softmax normalization. For seq=32: 32×32×4=4KB shared memory per block — easily fits.
**Confidence**: high

### Q: Does the full attention mechanism match CPU reference?
A: Yes. Max error = 0.0 (perfect bit-exact match) for deterministic integer-derived test data. The all-f32 computation path (no f16 quantization in attention) preserves full precision.
**Confidence**: high

## Design Notes
- Attention is computed entirely in f32 — no f16 conversion needed since Q/K/V come from GEMM f32 output.
- For larger seq_len (>32), would need multi-warp or tiled approach.
- Causal mask not implemented (not needed for single-layer validation).

## Impact on Downstream Tasks
- **transformer-layer.6** (end-to-end): Attention component ready.
