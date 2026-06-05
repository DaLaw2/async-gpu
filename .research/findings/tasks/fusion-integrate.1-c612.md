# fusion-integrate.1: Auto-fuse nn::Linear matmul+bias+activation

**Status**: done
**Kind**: experiment

## What was done

Wired FusionCodegen (NVRTC-based fused elementwise kernel engine) into
nn::Linear's forward pass via a new `forward_auto_fused()` method:

1. Added `Activation` enum (Gelu, Relu, SiLU, Sigmoid) to linear.rs
2. Added `FusionCodegen` field to `Linear` struct (Arc-shared, cached)
3. Added `with_codegen()` constructor for shared cache across layers
4. Implemented `forward_auto_fused(input, activation)`:
   - Step 1: Standard matmul (GEMM kernel, unchanged)
   - Step 2: FusionCodegen compiles+launches fused BiasAdd+Activation kernel
   - Single NVRTC kernel replaces 2 separate kernel launches (bias_add + activation)
   - Kernel cached after first JIT compilation

## Correctness findings

All 4 activations match CPU reference within f32 tolerance:
- GELU: max_err = 1.79e-7
- ReLU: max_err = 1.19e-7
- SiLU: max_err = 1.79e-7
- Sigmoid: max_err = 5.96e-8

Fused vs unfused (same GEMM, different epilogue implementations):
- ReLU: exact match (0.0)
- Sigmoid: 5.96e-8
- SiLU: 2.98e-8
- GELU: 5.79e-3 (different tanh approximation in NVRTC codegen vs PTX kernel)

## Benchmark

GPT-2 FFN up-projection: [128, 768] -> [128, 3072] + bias + GELU

- Unfused (3 kernels: matmul + bias_add + gelu): ~1400 us/iter
- Auto-fused (2 kernels: matmul + fused(bias+gelu)): ~870 us/iter
- Speedup: **1.61x** (eliminates 1 kernel launch + 1 global memory round-trip)

Note: matmul dominates for large matrices. The epilogue fusion benefit is
proportionally larger for smaller batch sizes or when bias+activation is
a bigger fraction of total time.

## Files changed

- `crates/core/gpu-host/src/nn/layers/linear.rs` — main implementation
- `crates/core/gpu-host/src/nn/layers/mod.rs` — re-export Activation

## Tests added (8 new, all passing)

- test_auto_fused_gelu — GELU correctness vs CPU
- test_auto_fused_relu — ReLU correctness vs CPU
- test_auto_fused_silu — SiLU correctness vs CPU
- test_auto_fused_sigmoid — Sigmoid correctness vs CPU
- test_auto_fused_matches_unfused — fused vs unfused path equivalence
- test_auto_fused_gpt2_dims — GPT-2 scale smoke test
- test_shared_codegen_cache — shared FusionCodegen across layers
- test_auto_fused_benchmark — performance measurement
