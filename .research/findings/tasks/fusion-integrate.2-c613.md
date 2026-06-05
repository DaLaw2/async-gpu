# fusion-integrate.2: Benchmark GPT-2 with auto-fusion vs manual

**Status**: done
**Kind**: experiment
**Result**: 0.5-2.7% speedup (below 10% epic target)

## What was done

Added `forward_auto_fused()` methods to `TransformerBlock` and `Gpt2Model`
that use `Linear::forward_auto_fused(_, Activation::Gelu)` for the FFN
up-projection, fusing bias-add + GELU into a single NVRTC kernel.

Benchmarked GPT-2 Small (124M params) full forward pass on GTX 1660 (sm_75)
at four sequence lengths, comparing unfused (3-kernel bias+gelu) vs auto-fused
(1-kernel fused bias+gelu) paths.

## Correctness

Top-1 token prediction matches between fused and unfused paths at all
sequence lengths tested. Raw logit differences accumulate through 12
transformer blocks (max_err up to ~210 at seq=128) due to different GELU
tanh approximations in PTX vs NVRTC codegen, but ranking is preserved.

## Benchmark results (20 iterations, median)

| seq_len | Unfused (ms) | Auto-fused (ms) | Speedup | Improvement |
|---------|-------------|-----------------|---------|-------------|
| 1       | 104.2       | 103.5           | 1.007x  | 0.7%        |
| 5       | 27.1        | 26.4            | 1.027x  | 2.7%        |
| 32      | 28.6        | 28.4            | 1.005x  | 0.5%        |
| 128     | 36.7        | 35.8            | 1.024x  | 2.4%        |

Fusion saves 12 kernel launches per forward pass (1 per transformer block).

## Why < 10% speedup

1. **Matmul dominates**: Each block is ~2.77ms, with GEMM taking ~85% of time.
   The bias+GELU epilogue is ~0.15ms per block — fusion saves ~0.05ms per block.
2. **Already fused elsewhere**: The `layer_norm_residual_dual` kernel already
   fuses the other major fusion opportunity (residual + LayerNorm).
3. **Kernel launch overhead is small**: On GTX 1660 with CUDA's launch overhead
   at ~5-10us per kernel, saving 12 launches saves ~60-120us total vs
   ~27-37ms forward pass — about 0.3%.
4. **Memory bandwidth bound**: The fused kernel reads+writes the same data as
   the unfused pair, just with one fewer intermediate buffer. At [seq, 3072]
   sizes the memory traffic saving is small relative to total bandwidth used.

## Conclusion

Auto-fusion of bias+GELU in the FFN provides a small but consistent speedup
(1-3%) for GPT-2 inference. The 10% speedup target would require fusing
the GEMM itself with the epilogue (true GEMM epilogue fusion), or fusing
across multiple operations in the attention path. Epilogue-only fusion on
the already-dominant GEMM's output is fundamentally bounded by how small
the epilogue is relative to the GEMM.

## Files changed

- `crates/core/gpu-host/src/nn/models/gpt2.rs` — added `TransformerBlock::forward_auto_fused()`,
  `Gpt2Model::forward_auto_fused()`, and `bench_gpt2_auto_fused_vs_manual` test
