# gc-codegen.1 — Fused Kernel Dispatch Strategy

## Question

How should the graph compiler execute fused operator nodes produced by `onnx_fusion.rs`?

- **Option A**: Template-based codegen — generate PTX at runtime from templates.
- **Option B**: Precompiled pattern library — map fused op names to existing PTX kernels.

## Recommendation: Option B (Precompiled Pattern Library)

Option B is the clear choice. The project already has the full infrastructure:

1. **Fused PTX kernels exist** — `gemm_bias_gelu` and `gemm_bias_relu` in `compute_fused.rs` with identical signatures: `(a, b, bias, d, k_dim, n_cols, status)`.

2. **Host-side launch code exists** — `matmul_fused()` in `nn/ops/gemm.rs` handles padding, B-transpose, bias padding, and kernel dispatch via `FusedActivation::{Gelu, Relu}`.

3. **Kernel registry already loads them** — `ML_KERNELS` in `registry.rs` includes both `"gemm_bias_gelu"` and `"gemm_bias_relu"`.

4. **Fusion pass produces predictable names** — `onnx_fusion.rs` emits `Fused_MatMulBias{Relu,Gelu,Sigmoid,Tanh,Silu}` for 3-node patterns and `Fused_Add{activation}` for 2-node patterns.

Template-based codegen (Option A) adds substantial complexity — a PTX template language, runtime compilation, caching — for no practical gain when the kernel set is small and well-defined.

## What Needs to Change

### 1. Add dispatch cases in `onnx_executor.rs`

The `dispatch_node` match in `onnx_executor.rs` currently falls through to `Unsupported ONNX op` for any `Fused_*` op. Two new arms are needed:

```rust
"Fused_MatMulBiasRelu" => {
    let a = get_input(&node.inputs[0], tensor_map)?;
    let b = get_input(&node.inputs[1], tensor_map)?;
    let bias = get_input(&node.inputs[2], tensor_map)?;
    let out = crate::nn::ops::matmul_fused(
        a, b, bias,
        crate::nn::ops::gemm::FusedActivation::Relu,
        registry,
    ).map_err(map_nn_err)?;
    Ok(NodeOutput::Single(out))
}
"Fused_MatMulBiasGelu" => {
    let a = get_input(&node.inputs[0], tensor_map)?;
    let b = get_input(&node.inputs[1], tensor_map)?;
    let bias = get_input(&node.inputs[2], tensor_map)?;
    let out = crate::nn::ops::matmul_fused(
        a, b, bias,
        crate::nn::ops::gemm::FusedActivation::Gelu,
        registry,
    ).map_err(map_nn_err)?;
    Ok(NodeOutput::Single(out))
}
```

### 2. Handle unsupported fused patterns gracefully

The fusion pass can produce `Fused_MatMulBiasSigmoid`, `Fused_MatMulBiasTanh`, `Fused_MatMulBiasSilu`, and `Fused_Add*` variants that have no precompiled kernel yet. Two options:

- **Reject at dispatch** (simplest): return `Unsupported ONNX op` error.
- **Fallback decomposition** (better): detect unrecognized `Fused_*` ops and decompose back into the original sequence (matmul + add + activation). This preserves correctness while the fusion pass can still identify opportunities for future kernels.

Recommendation: start with rejection; add fallback decomposition as a follow-up task.

### 3. Future expansion path

When new fused kernels are added (e.g., `gemm_bias_sigmoid`):

1. Add the kernel to `compute_fused.rs`.
2. Add a `FusedActivation::Sigmoid` variant.
3. Add a dispatch case in `matmul_fused()` and `onnx_executor.rs`.
4. Register the kernel name in `ML_KERNELS`.

This is a mechanical 4-step checklist — no architecture changes needed.

## Effort Estimate

Adding `Fused_MatMulBiasRelu` and `Fused_MatMulBiasGelu` dispatch: ~20 lines of code in `onnx_executor.rs`. No new crates, no new dependencies, no PTX changes.
