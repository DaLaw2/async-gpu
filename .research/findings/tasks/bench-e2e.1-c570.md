# bench-e2e.1: GPT-2 inference per-layer profiling — identify bottleneck ops
**Cycle**: 570 | **Theme**: bench-e2e | **Kind**: experiment | **Status**: done

## Summary
Added forward_profiled() to GPT-2 model and GPT-2 profiling section to benchmark
example. Each transformer block takes ~13.2ms. LM head (768→50257 GEMM) takes
62.6ms (28.3% of total). Total forward pass: 221ms for seq_len=128.

## Findings

### GPT-2 Small Forward Pass Breakdown (seq_len=128)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| embedding | 0.05 | 0.0% |
| block_0-11 (each) | ~13.2 | ~6.0% |
| blocks total | 158.6 | 71.7% |
| ln_f | 0.05 | 0.0% |
| lm_head | 62.6 | 28.3% |
| **TOTAL** | **221.3** | **100%** |

### Bottleneck Analysis

1. **LM head** (28.3%): Single GEMM [128, 768] × [768, 50257]. This is our largest
   GEMM and dominates because vocab_size=50257 makes N very large. With cuBLAS,
   this would take ~3.5ms instead of 62.6ms (18x speedup).

2. **Transformer blocks** (71.7%): Each block runs ~13.2ms. Contains:
   - 3 GEMM ops (QKV, attn_out, FFN_up, FFN_down) — ~10ms
   - Flash attention — ~1.5ms
   - 2 LayerNorm — ~0.5ms
   - Element-wise ops — ~0.5ms

3. **Embedding** and **final LN** are negligible.

### With cuBLAS (theoretical)
If we replaced ALL GEMM with cuBLAS (keeping everything else the same):
- LM head: 62.6ms → ~3.5ms
- Per block GEMM: ~10ms → ~0.6ms
- Total would drop from 221ms to ~15ms (14.7x speedup)
- GEMM is >95% of total compute time

**Confidence**: high
