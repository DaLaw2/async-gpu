# Theme: fusion-integrate — Fusion integration with nn API

## Status
Task fusion-integrate.1 done. FusionCodegen wired into nn::Linear.

## What exists
- `Linear::forward_auto_fused(input, Activation)` — matmul then NVRTC-fused bias+activation
- `Activation` enum: Gelu, Relu, SiLU, Sigmoid (richer than old FusedActivation)
- `Linear::with_codegen()` — shared FusionCodegen cache across layers
- FusionCodegen JIT-compiles BiasAdd+Activation into single kernel, caches by op chain + n_cols

## Key findings
- Correctness: all 4 activations match CPU ref within 2e-7 (except GELU ~6e-3 due to tanh variant)
- Performance: 1.61x speedup for GPT-2 FFN dims ([128,768]->[128,3072]+GELU)
- Epilogue fusion eliminates 1 kernel launch + 1 global memory round-trip
- Matmul dominates for large matrices; fusion benefit proportionally larger for smaller batches

## Files
- `crates/core/gpu-host/src/nn/layers/linear.rs` — Activation enum, forward_auto_fused, with_codegen
- `crates/core/gpu-host/src/nn/layers/mod.rs` — re-export Activation

## Next steps
- Wire auto-fusion into other nn layers (Conv2d epilogue, LayerNorm residual)
- Integrate with autograd tape for fused backward pass
- Benchmark with smaller batch sizes where epilogue is proportionally larger
