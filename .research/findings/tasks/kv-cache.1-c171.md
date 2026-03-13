# kv-cache.1: KV cache architecture and memory layout design
**Cycle**: 171 | **Theme**: kv-cache | **Kind**: design | **Status**: done

## Summary
Designed KV cache architecture for GPT-2 autoregressive generation. Current
generation loop recomputes full sequence at each step (O(N²)). KV cache reduces
this to O(N) by caching K/V projections and only processing the new token.

## Findings

### Q: What is the optimal device-side buffer layout for 12-layer KV cache?
A: Per-layer pre-allocated buffers:
```
k_cache[layer]: [n_heads=12][max_seq][d_head=64] f32
v_cache[layer]: [n_heads=12][max_seq][d_head=64] f32
filled_len[layer]: u32 (cache occupancy counter)
```
Pre-allocate at max_seq_len, track `filled_len` per layer. Layout matches existing
`flash_attention` kernel expectation (contiguous `[head][row][d_head]`).
**Confidence**: high

### Q: How much GPU memory is needed for max_seq=128 vs 1024?
A:
- max_seq=128: 12 layers × 2 × 12 heads × 128 × 64 × 4 bytes = **36 MB**
- max_seq=1024: 12 layers × 2 × 12 heads × 1024 × 64 × 4 bytes = **75 MB**
Both trivially fit in 10GB+ VRAM (< 1%).
**Confidence**: high

### Q: What kernel changes are needed for cache-append and cached attention?
A: Mostly reuse existing kernels with minor modifications:

1. **`flash_attention`**: Add a `kv_len` parameter so Q length and K/V length
   can differ. Currently assumes Q, K, V all share the same `seq_len`. For
   cached generation, Q has 1 row while K/V have `cached_len` rows. This is
   the only kernel that truly needs modification.

2. **`append_kv_cache`**: NOT a new kernel — host-side `cuMemcpyDtoD` can copy
   new K/V vectors into the cache at the correct offset. Trivial.

3. **`split_qkv`**: Already accepts `seq_len` parameter. Pass `seq=1` for
   single-token generation. No modification needed.

**Confidence**: high

## Architecture Decision

### ADR: KV Cache Design
**Context**: Generation loop is O(N²) per step. At step 50, recomputing 50×12
layers×4 GEMMs is wasteful — K/V from previous steps are identical.

**Decision**: Adopt append-only KV cache with:
- Pre-allocated device buffers at max_seq_len (128 for now)
- Prompt processing: use existing flash_attention for full sequence → populate cache
- Generation: process only new token through transformer → append K/V → cached attention

**Host-side loop**:
```
# Phase 1: Prompt processing (full sequence)
for layer in 0..12:
    run full pipeline with flash_attention(q, k, v, seq_len=prompt_len)
    append all K/V to cache[layer] (prompt_len entries)

# Phase 2: Autoregressive generation
for step in 0..max_new_tokens:
    embed new token → hidden[1][768]
    for layer in 0..12:
        ln_out = LayerNorm(hidden)         # [1][768]
        qkv = GEMM(ln_out, W_qkv)         # [1][2304]
        q, k_new, v_new = split_qkv(qkv)  # each [12][64]
        append_kv_cache(k_new, v_new, cache[layer], filled_len)
        attn_out = flash_attention_cached(q, cache[layer], filled_len+1)
        concat + proj + residual + LN + FFN + residual
    logits = LM_head(hidden)
    next_token = argmax(logits)
    filled_len += 1
```

**GEMM with single-row input**: Both gemm_f32 and full_gemm_f32in require M ≥ 32
(tile size). For single-token GEMM (M=1), either:
- Pad to M=32 (wastes 31/32 compute) — simple, works now
- Write a specialized single-row GEMM (dot product per column) — better perf, future task

**Consequences**: ~36 MB GPU memory for 128-token cache. Expected speedup: ~10-50x
for generation steps 5-50+ (avoiding full sequence recomputation). Only need 3 new
GPU kernel functions.

## Impact on Downstream Tasks
- kv-cache.2: Implement for single layer — need append_kv_cache + flash_attention_cached kernels
- kv-cache.3: Full 12-layer cached inference — straightforward extension
- Single-row GEMM optimization is a potential future theme for additional speedup
