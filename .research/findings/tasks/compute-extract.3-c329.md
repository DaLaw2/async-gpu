# compute-extract.3: Warp-cooperative NN ops added to gpu_runtime::nn
**Cycle**: 329 | **Theme**: compute-extract | **Kind**: experiment | **Status**: done

## Summary
Added 2 warp-cooperative neural network operations to `gpu_runtime::nn`:
`warp_softmax_f32` and `warp_layer_norm_f32`. Both build on the warp reduction
primitives from extract.1. Build verified on nvptx64.

## Functions Added

### gpu_runtime::nn (2 new unsafe functions)
- `warp_softmax_f32(val) -> f32` — softmax across 32 warp lanes
  - max via `warp::reduce_max_f32` → exp(val - max) → sum via `warp::reduce_sum_f32` → normalize
- `warp_layer_norm_f32(val, gamma, beta) -> f32` — layer normalization across 32 lanes
  - mean via `warp::reduce_sum_f32` / 32 → variance → `rsqrt(var + eps)` → scale + shift

## Build Verification
- `cargo +nightly-2026-03-11 build --release --target nvptx64-nvidia-cuda` — SUCCESS

## Public API Total Count (all 3 extract tasks)
- `gpu_runtime::index` — 12 safe functions
- `gpu_runtime::math` — 12 safe functions
- `gpu_runtime::warp` — 10 unsafe functions
- `gpu_runtime::block` — 6 unsafe functions (3 original + 3 reductions)
- `gpu_runtime::nn` — 6 functions (4 safe activations + 2 unsafe warp-cooperative)
- **Total: 46 public compute functions**

## Impact on Downstream Tasks
- **compute-extract theme**: COMPLETE — all functions extracted
- **demo-pipeline.1**: Full API surface available for demo design
