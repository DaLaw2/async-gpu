# fusion-codegen.3: Register-only intermediates + float4 vectorization

## Summary

Added ElemMul op to codegen engine, optimized BiasAdd float4 vectorized bias loads for
n_cols%4==0, verified register-only intermediates in generated kernels, and benchmarked
fused vs unfused at 2.05x speedup (1M elements, ElemMul+ElemAdd+Gelu chain on GTX 1660).

## Baseline

- 24 fusion tests passing (7 GPU, 17 unit)
- 6 supported ops: BiasAdd, ElemAdd, Gelu, Relu, Silu, Sigmoid
- BiasAdd float4 path used per-element scalar bias reads in all cases

## Findings

### Register-only intermediates (verified)
Generated CUDA C stores all intermediate values in local `float4 v` variable.
NVCC promotes these to registers. Codegen test confirms exactly 2 `output[` stores
(float4 + scalar paths) — no global memory round-trip between ops. No shared memory
or temp buffers allocated.

### Float4 vectorized BiasAdd (implemented)
When `n_cols % 4 == 0` (common: 128, 256, 512, 768, 1024...), codegen now emits
`reinterpret_cast<const float4*>(&bias[col4])` — single coalesced 128-bit load
instead of 4 scalar loads with modulo. Scalar fallback preserved for non-aligned sizes.

### ElemMul op (added)
New `OpKind::ElemMul` with full codegen support (float4 vectorized + scalar tail).
Classified as Elementwise for fusion. Backward stub added (product rule TODO).
Enables the epic's required "multiply + add + activation" fusion chain.

### Benchmark results
ElemMul + ElemAdd + Gelu chain, n=1,048,576 (1M elements):
- Fused: 101.7 us/iter (single kernel launch)
- Unfused: 208.8 us/iter (3 separate kernel launches)
- Speedup: **2.05x** (exceeds epic's >= 2x criteria)

## Test Results

35 tests passing (was 24): 14 GPU tests, 21 unit tests.
New tests:
- `test_codegen_source_elemmul` — ElemMul codegen source verification
- `test_codegen_bias_vectorized_path` — vectorized bias load for n_cols%4==0
- `test_codegen_bias_scalar_fallback` — scalar fallback for non-aligned n_cols
- `test_classify_elemmul` — ElemMul classified as Elementwise
- `test_elementwise_chain_with_elemmul` — ElemMul+ElemAdd+Gelu chain detection
- `test_gpu_fused_elemmul_gelu` — ElemMul+Gelu on GPU
- `test_gpu_fused_elemmul_elemadd_gelu` — Mul+Add+Gelu chain (epic criteria)
- `test_gpu_fused_elemmul_scalar_tail` — ElemMul scalar tail (n=17)
- `test_gpu_fused_bias_gelu_vectorized` — BiasAdd vectorized path on GPU
- `test_gpu_fused_vs_unfused_benchmark` — fused vs unfused speedup measurement
- `test_codegen_register_only_intermediates` — no global memory intermediates

## Open Questions

- BiasAdd float4 path: should non-aligned n_cols emit a runtime branch or stay scalar-only?
- ElemMul backward: product rule needs full implementation for autograd training
- Fused kernel launch overhead: benchmark with longer chains (5+ ops) to find diminishing returns

## Files Changed

- `crates/core/gpu-host/src/nn/autograd/tape.rs` — added `OpKind::ElemMul`
- `crates/core/gpu-host/src/nn/autograd/backward.rs` — added `ElemMul` backward stub
- `crates/core/gpu-host/src/nn/fusion.rs` — ElemMul codegen, vectorized BiasAdd, 11 new tests
