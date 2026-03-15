# compute-extract.1: Extract compute primitives to gpu_runtime public modules
**Cycle**: 329 | **Theme**: compute-extract | **Kind**: experiment | **Status**: done

## Summary
Implemented 5 new public modules in `gpu_runtime` (lib.rs): `index`, `math`, `warp`, `block`, `nn`.
41 public functions total. All compile to valid PTX on nvptx64. Cross-crate linkage verified
via gpu-kernel build. Non-nvptx targets get stub implementations for doc builds.

## What Was Extracted

### gpu_runtime::index (12 functions, all safe)
- `thread_idx_x/y/z()`, `block_idx_x/y/z()`, `block_dim_x/y/z()`, `grid_dim_x/y/z()`
- `global_thread_idx()` — convenience: `block_idx_x * block_dim_x + thread_idx_x`
- `global_thread_count()` — `grid_dim_x * block_dim_x`
- `clock_nanos()` — `%globaltimer` PTX

### gpu_runtime::math (12 functions, all safe)
- `sqrt_f32`, `rsqrt_f32` — `sqrt.approx.f32`, `rsqrt.approx.f32`
- `exp_f32`, `log_f32` — `ex2.approx.f32` + log2(e), `lg2.approx.f32` + ln(2)
- `sin_f32`, `cos_f32` — `sin.approx.f32`, `cos.approx.f32`
- `abs_f32`, `min_f32`, `max_f32` — direct PTX
- `fma_f32` — `fma.rn.f32`
- `tanh_f32` — computed from `exp_f32`
- `sigmoid_f32` — `1/(1+exp(-x))`

### gpu_runtime::warp (10 functions, all unsafe)
- `reduce_sum_f32`, `reduce_sum_u32`, `reduce_max_f32`, `reduce_min_f32` — butterfly shuffle
- `shfl_bfly_u32`, `shfl_down_u32`, `shfl_up_u32` — raw shuffle primitives
- `ballot`, `all`, `any` — warp vote via `vote.sync.ballot.b32`

### gpu_runtime::block (3 functions, all unsafe)
- `sync()` — `bar.sync 0`
- `shared_mem_ptr()` — `cvta.shared.u64 dynamic_smem`
- `shared_mem_at::<T>(offset)` — typed shared memory access

### gpu_runtime::nn (4 functions, safe)
- `gelu_f32` — GELU activation (GPT-2/BERT style)
- `relu_f32` — max(0, x)
- `leaky_relu_f32` — with configurable alpha
- `silu_f32` — Swish: x * sigmoid(x)

## Build Verification
- `cargo +nightly-2026-03-11 build --release --target nvptx64-nvidia-cuda` — SUCCESS for both gpu-runtime and gpu-kernel
- All 41 functions compile cleanly (only pre-existing warnings in gpu-kernel)
- Cross-crate inlining works via `#[inline(always)]`

## Scope Notes
This task combined work from compute-extract.1 (warp), compute-extract.2 (block/math/index),
and compute-extract.3 (nn activation functions). The remaining extract tasks can focus on
higher-level operations (layer_norm, warp_softmax, block_reduce).

## Impact on Downstream Tasks
- **compute-extract.2**: Most content already done here. Remaining: block-level reduction.
- **compute-extract.3**: Basic NN ops done. Remaining: warp_softmax_f32, warp_layer_norm_f32.
- **demo-pipeline.1**: Can now design demos using these public utils.
