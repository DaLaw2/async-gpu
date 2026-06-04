# perf-e2e.1 — GPT-2 End-to-End Benchmark with All Optimized Kernels

## Result

**GPT-2 forward pass (seq=128, fused=true, release): median 39.4ms, min 38.7ms**

- Previous measurement: ~221ms (baseline)
- Current (release + cublas): **38.7–39.4ms** → **5.6× speedup** over baseline
- Target was < 30ms — not yet met (39.4ms vs 30ms target)

## Benchmark Data

### Release build, cublas (fused LN+residual)

```
=== GPT-2 Forward Pass Benchmark (seq=128, fused=true) ===
  embedding          0.057 ms
  block avg          2.903 ms  (min ~2.76ms)
  ln_f               0.033 ms
  lm_head            4.926 ms
  TOTAL median      39.414 ms
  TOTAL min         38.659 ms
```

### Debug build, cublas (fused)

```
  block avg          3.147 ms
  TOTAL median      42.332 ms
  TOTAL min         42.007 ms
```

### Debug build, no cublas (non-fused, PTX flash_attn_v2)

```
  block avg (min)    ~3.09 ms   (highly variable, avg inflated to 10.1ms by JIT)
  TOTAL min         44.781 ms
  TOTAL median     118.163 ms  (JIT overhead skews median)
```

## Kernel Dispatch Analysis

For seq=128, n_embd=768, n_head=12, d_head=64, ffn_dim=3072:

### Per-block operations (with `cublas` feature)

| Operation | M×K×N | Kernel Used | Notes |
|-----------|-------|-------------|-------|
| LN1 | 128 rows × 768 | layer_norm_v3 | float4 vectorized (768 % 4 == 0) |
| QKV projection | 128×768→2304 | cuBLAS | M=128 ≤ 256 → cuBLAS path |
| split_qkv | 128×2304 → Q,K,V | split_qkv kernel | single launch |
| Attention | 12 heads, seq=128, d=64 | flash_attn_v3 (NVRTC) | cooperative 4-thread/row, 128 threads |
| concat_heads | — | concat_heads kernel | single launch |
| Out projection | 128×768→768 | cuBLAS | M ≤ 256 |
| Fused residual+LN2 | 128×768 | layer_norm_residual_dual (NVRTC) | saves 1 launch + 1 read |
| FFN up | 128×768→3072 | cuBLAS | M ≤ 256 |
| GELU | 128×3072 | gelu_forward_v2 | float4 vectorized |
| FFN down | 128×3072→768 | cuBLAS | M ≤ 256 |
| Residual add | 128×768 | elementwise_add | vectorized |

**Total kernel launches per block: ~11** (fused path saves 1 vs non-fused)
**Total for 12 blocks: ~132 launches** + embedding(1) + ln_f(1) + lm_head(1) = **~135 total**

### LM head analysis

| Operation | M×K×N | Kernel | Time |
|-----------|-------|--------|------|
| LM head | 128×768→50257 | cuBLAS | 4.9ms |

The LM head is disproportionately expensive: 128×768×50257 matmul = ~12.5% of total time.
This is expected — the output dimension (vocab size = 50257) is very large.

## Performance Breakdown

| Component | Time (ms) | % of Total |
|-----------|-----------|-----------|
| Embedding | 0.06 | 0.2% |
| 12 × TransformerBlock | 34.8 | 88.3% |
| Final LayerNorm | 0.03 | 0.1% |
| LM Head | 4.9 | 12.5% |
| **Total** | **39.4** | **100%** |

Per-block breakdown (~2.9ms each):
- cuBLAS matmuls (QKV + out + FFN up + FFN down): ~2.2ms (estimated from 4 matmul calls)
- flash_attn_v3: ~0.3ms (from individual benchmarks: 564-606 GFLOPS)
- LN + GELU + elementwise: ~0.4ms (memory-bound ops)

## Improvement from Baseline

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Forward pass (seq=128) | ~221ms | 39.4ms | 5.6× |
| Per-block average | ~18.4ms | 2.9ms | 6.3× |

## Why < 30ms Target Not Met

The 30ms target requires ~2.4ms per block. Current per-block is ~2.9ms.

**Bottlenecks**:
1. **cuBLAS matmuls dominate** (~75% of block time). For M=128 these are memory-bandwidth-bound,
   not compute-bound. cuBLAS is already near-optimal for these shapes.
2. **LM head** takes 4.9ms (128×768→50257). This is a single large matmul that benefits from
   cuBLAS but has high N dimension.
3. **Kernel launch overhead**: ~135 kernel launches at ~5-10μs each = 0.7-1.4ms overhead.

**Possible paths to < 30ms**:
- Kernel fusion (combine LN + matmul, or matmul + bias + GELU into single launch)
- CUDA graphs to eliminate launch overhead
- Use FP16/TF32 for compute
- Pre-sorted/batched cuBLAS calls with stream capture
- The `forward_fused` path exists on Linear but isn't used in TransformerBlock

## Verification: All Optimized Kernels Connected

| Kernel | Status | How Used |
|--------|--------|----------|
| SGEMM V2/V3 | ✅ Connected | `matmul()` auto-selects V2/V3 for large M |
| cuBLAS | ✅ Connected | `matmul()` dispatches to cuBLAS for M≤256 (cublas feature) |
| Flash Attention V3 | ✅ Connected | `multi_head_flash_attention()` → V3 with cublas feature |
| LayerNorm V3 | ✅ Connected | `layer_norm()` selects v3 when d_model % 4 == 0 |
| Fused LN+residual | ✅ Connected | `layer_norm_residual_dual()` in TransformerBlock with cublas |
| GELU V2 (vectorized) | ✅ Connected | `gelu()` calls `gelu_forward_v2` |
| Elementwise V2 | ✅ Connected | All elementwise ops use float4 V2 kernels |
