# fusion-codegen.2 — Experiment: Generate Fused PTX from Fusion Graph

## Status: done

## Summary

Implemented `FusionCodegen` in `crates/core/gpu-host/src/nn/fusion.rs` — a thread-safe NVRTC codegen engine that generates, compiles, and caches fused elementwise CUDA kernels from op chains. Extended `FusionOptimizer` with generic elementwise chain detection (P10 fallback). Verified correctness on GPU: fused output matches unfused for BiasAdd+Gelu, BiasAdd+ReLU, BiasAdd+SiLU, ElemAdd+Gelu, ReLU+Sigmoid chains, including scalar-tail and cache-hit paths.

## Baseline

- 13 existing fusion analysis tests: all passing
- FusionOptimizer detected P1/P3/P4 patterns only (no elementwise chain codegen)

## What was implemented

### 1. `FusedOpKind::ElementwiseChain(Vec<OpKind>)` variant

New variant stores the op chain for dynamic NVRTC codegen, distinct from fixed-pattern variants (MatmulBiasGelu, etc.).

### 2. Generic elementwise chain detection in `FusionOptimizer::try_match`

After all fixed patterns (P1/P3/P4) fail, a greedy fallback extends from the current position through consecutive elementwise ops (max 5) that satisfy data-flow and single-consumer predicates. This catches all P5-P10 patterns from the design.

### 3. `FusionCodegen` struct

- **`codegen(ops, n_cols_params, key)`**: Generates CUDA C source from op chain. Each op maps to a float4-vectorized code fragment (main path) and a scalar fragment (tail path for `n % 4 != 0`). Supported ops: BiasAdd, ElemAdd, Gelu, Relu, Silu, Sigmoid.
- **`get_or_compile(ops, n_cols_params, dev)`**: Thread-safe compile-or-cache via `Mutex<HashMap<u64, CompiledFusedKernel>>`. Cache key hashes op chain + shape-affecting params (n_cols for BiasAdd). Returns `(module_name, func_name)` for `dev.get_func()`.
- **Static function name**: All fused kernels use `"fused_kernel"` as the function name (required by cudarc's `load_ptx` which demands `&'static str` for func_names). Kernels are differentiated by module name (`fused_{hash:016x}`).

### 4. GPU correctness tests (7 new tests)

| Test | Chain | Shape | Result |
|------|-------|-------|--------|
| `test_gpu_fused_bias_gelu` | BiasAdd → Gelu | [128, 768] | max_err < 1e-4 |
| `test_gpu_fused_bias_relu` | BiasAdd → ReLU | [64, 256] | max_err < 1e-6 |
| `test_gpu_fused_bias_silu` | BiasAdd → SiLU | [32, 128] | max_err < 1e-5 |
| `test_gpu_fused_elemadd_gelu` | ElemAdd → Gelu | [2048] | max_err < 1e-4 |
| `test_gpu_fused_relu_sigmoid` | ReLU → Sigmoid | [1024] | max_err < 1e-6 |
| `test_gpu_fused_scalar_tail` | ReLU → Sigmoid | [13] (not %4) | max_err < 1e-6 |
| `test_gpu_fused_cache_hit` | ReLU → Gelu | N/A | cache hit verified |

### 5. BiasAdd float4 alignment

The design (fusion-analysis.2) proposed float4-vectorized bias reads (`reinterpret_cast<const float4*>(&bias[col])`), but this requires `col % 4 == 0` which is not guaranteed when elements are interleaved across rows. The implementation uses per-element scalar bias reads in the float4 path (`bias[idx % n_cols]` for each of x,y,z,w), which is correct for arbitrary n_cols values without alignment constraints. This trades ~5% bandwidth for correctness. A future optimization could detect `n_cols % 4 == 0` and emit the vectorized path.

## Key design decisions

1. **Static func_name constraint**: cudarc 0.12's `load_ptx` requires `&'static str` for function names. We use a single static name `"fused_kernel"` and differentiate by module name. This is a cudarc API limitation, not a fundamental issue.

2. **Greedy chain extension**: The P10 fallback extends greedily up to 5 ops. This is simpler than a separate pattern for each 2-op combination (P5-P9) and generalizes to arbitrary chains.

3. **n_cols in cache key**: For BiasAdd, the column count affects the modulo arithmetic in the kernel. Different n_cols values produce different kernels. The total element count `n` is a runtime argument (not baked into codegen), so different batch sizes reuse the same compiled kernel.

## Test results

```
running 24 tests
13 analysis tests: all ok
4 codegen unit tests: all ok  
7 GPU correctness tests: all ok

test result: ok. 24 passed; 0 failed; 0 ignored
```

## Files changed

- `crates/core/gpu-host/src/nn/fusion.rs` — added `ElementwiseChain` variant, P10 chain detection, `FusionCodegen` struct, 11 new tests
