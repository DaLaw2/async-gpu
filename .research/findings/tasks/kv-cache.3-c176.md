# kv-cache.3: Full 12-layer KV-cached inference
**Cycle**: 176 | **Theme**: kv-cache | **Kind**: experiment | **Status**: done

## Summary
Implemented full 12-layer KV-cached autoregressive generation. Prefill phase processes
full prompt through all layers and populates per-layer KV caches. Decode phase processes
one token at a time using `flash_attention_kv` with cached K/V. All 50 generated tokens
match the non-cached reference exactly. 2.07x speedup achieved.

## Findings

### Q: Does 12-layer cached output match non-cached for 50+ tokens?
A: **Yes, exactly.** 50/50 tokens match between cached and non-cached generation for the
prompt "The capital of France is". Both produce identical output:
" the capital of the French Republic, and the capital of the French Republic is..."
**Confidence**: high

### Q: What is the per-token latency with KV cache?
A: **68.1ms/token (cached) vs 140.8ms/token (non-cached) = 2.07x speedup.**
The speedup comes from:
- Decode GEMM uses M=32 (padded) instead of M=128
- flash_attention_kv processes q_len=1 instead of q_len=128
- Only one token is embedded and processed per step (except prefill)

The 2.07x is lower than theoretical 4x (128/32) because:
- Prefill step still uses full M=128 (amortized over 50 steps)
- CPU round-trips for KV cache copy during prefill
- GEMM still pads to M=32 even for 1 token
- LM head is CPU-based (same cost both ways)
**Confidence**: high

## Key Design Decisions
1. **Separate kv_stride parameter**: Added `kv_stride` to `flash_attention_kv` kernel so cache
   buffers can use MAX_SEQ stride while kv_len tracks valid entries. Critical bug found and
   fixed: original kernel used `kv_len` as both stride and valid count.
2. **SEQ_STEP=32 padding**: Decode-phase GEMMs require M to be a multiple of 32 (tile size).
   Single token is padded to 32 rows with zeros.
3. **CPU embedding for decode**: Single-token embedding computed on CPU (trivial lookup+add)
   since the existing embedding kernel uses position-as-index, not explicit position parameter.
4. **kv_cache_append kernel**: GPU kernel copies row 0 from padded split_qkv output [12, 32, 64]
   into the correct position in cache [12, 128, 64].
5. **Host-side KV cache copy for prefill**: During prefill, K/V are downloaded and re-uploaded
   to cache buffers. This is a one-time cost and doesn't affect decode latency.

## Unexpected Discoveries
- The flash_attention_kv kernel's original design used kv_len for both stride and bounds,
  which works when K/V are packed but breaks with pre-allocated caches. This was a subtle bug
  that only manifested at runtime (first decode step produced garbage).

## Impact on Downstream Tasks
- kv-cache theme success criteria: 2/3 met (cached matches non-cached, per-step only processes
  new token). Per-token latency 68ms > 20ms target — needs further optimization (mixed-precision,
  M=1 GEMM kernel, or GPU LM head).
- mixed-precision.1 can now build on this: swap gemm_f32 for BF16/TF32 MMA in the cached pipeline.
