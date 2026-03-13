# kv-cache.2: Implement KV cache for single-layer attention
**Cycle**: 175 | **Theme**: kv-cache | **Kind**: experiment | **Status**: done

## Summary
Implemented `flash_attention_kv` kernel with separate q_len and kv_len parameters.
Validated against full `flash_attention` — output is bit-identical (max error = 0.0)
for the last position across all 12 heads × 64 d_head = 768 values.

## Findings

### Q: Does cached attention output match non-cached for a single layer?
A: **Yes, exactly.** max |ref - cached| = 0.000000 across all 768 output elements.
The `flash_attention_kv(Q[1], K[32], V[32], q_offset=31)` output matches the last
row of `flash_attention(Q[32], K[32], V[32])` perfectly.
**Confidence**: high

### Q: What is the latency improvement for single-layer cached vs non-cached?
A: Not measured in isolation (both are sub-millisecond for 32-token seq). The real
benefit appears in the full generation loop where only 1 token goes through 12 layers
instead of the full sequence.
**Confidence**: medium (deferred to kv-cache.3)

## Key Design Decisions
1. Created new `flash_attention_kv` kernel instead of modifying existing `flash_attention`
   to avoid breaking 15+ existing call sites
2. Added `q_offset` parameter for causal masking: when generating token N, q_offset=N-1
   so the masking correctly prevents attending to future positions
3. K/V use `kv_len` stride for head offsets; Q uses `q_len` stride

## Impact on Downstream Tasks
- kv-cache.3 (full 12-layer cached inference) is unblocked
- Need to integrate into generation loop: per-layer KV cache buffers + append logic
