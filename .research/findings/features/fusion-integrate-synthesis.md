# Theme: fusion-integrate — Fusion integration with nn API

## Status
Tasks .1 and .2 done. Epilogue fusion wired in and benchmarked on GPT-2.

## What exists
- `Linear::forward_auto_fused(input, Activation)` — matmul then NVRTC-fused bias+activation
- `TransformerBlock::forward_auto_fused()` — fused FFN path for GPT-2 blocks
- `Gpt2Model::forward_auto_fused()` — full model fused forward pass
- FusionCodegen JIT-compiles BiasAdd+Activation into single kernel, caches by op chain + n_cols

## Key findings
- Single-layer speedup: 1.61x (FFN up-projection, bias+GELU fusion, [128,768]->[128,3072])
- **Full GPT-2 forward speedup: 1-3%** (0.5-2.7% across seq_len 1-128)
- Top-1 prediction agreement: 100% between fused and unfused paths
- GEMM dominates ~85% of block time; epilogue fusion saves ~0.05ms per block
- 10% target unreachable with epilogue-only fusion — needs true GEMM epilogue fusion

## Files
- `crates/core/gpu-host/src/nn/layers/linear.rs` — Activation, forward_auto_fused, with_codegen
- `crates/core/gpu-host/src/nn/models/gpt2.rs` — fused forward paths, benchmark test

## Implication for epic
Epilogue fusion alone cannot hit 10% on GPT-2. Achieving the epic target requires
either GEMM epilogue fusion (embed activation in GEMM output write-back) or fusing
across attention operations (QKV projection + split, residual + LN across blocks).
