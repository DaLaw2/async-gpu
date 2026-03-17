# onnx-transformer.1 — GPT-2 Small ONNX Export & Operator Audit

## Summary

Exported GPT-2 Small (124M params) to ONNX format (`models/gpt2.onnx`, 622 MB).
The export uses a clean standalone GPT-2 implementation that loads HuggingFace
pretrained weights, since the latest `transformers` (v5.3) has SDPA attention
and DynamicCache that break both the legacy TorchScript and new dynamo-based
ONNX exporters.

- **Opset version**: 17
- **Validation**: ONNX checker passes; ORT inference matches PyTorch (max err 1.7e-4)
- **Dynamic axes**: batch and seq_len are dynamic

## All ONNX Operators (19 unique)

| # | op_type              | Category          |
|---|----------------------|-------------------|
| 1 | Add                  | Elementwise       |
| 2 | Cast                 | Type conversion   |
| 3 | Concat               | Tensor manipulation |
| 4 | Constant             | Data              |
| 5 | ConstantOfShape      | Data              |
| 6 | Gather               | Tensor manipulation |
| 7 | LayerNormalization   | Normalization     |
| 8 | MatMul               | Linear algebra    |
| 9 | Mul                  | Elementwise       |
|10 | Range                | Data generation   |
|11 | Reshape              | Tensor manipulation |
|12 | Shape                | Tensor manipulation |
|13 | Softmax              | Activation        |
|14 | Split                | Tensor manipulation |
|15 | Tanh                 | Activation        |
|16 | Transpose            | Tensor manipulation |
|17 | Trilu                | Tensor manipulation |
|18 | Unsqueeze            | Tensor manipulation |
|19 | Where                | Conditional       |

## Supported vs Missing

### Already Supported in onnx_executor.rs (8 of 19)

- Add
- Concat
- Constant
- Gather
- MatMul
- Reshape
- Shape
- Softmax
- Transpose
- Unsqueeze

(10 ops already supported)

### Missing — Must Implement (9 of 19)

| op_type            | Purpose in GPT-2                                    | Difficulty |
|--------------------|------------------------------------------------------|------------|
| **LayerNormalization** | Pre-attention and pre-MLP layer norm             | Medium     |
| **Mul**            | Scaling (1/sqrt(d_k)) in attention                   | Easy       |
| **Cast**           | dtype conversion (int64 ↔ float32 for masks)         | Easy       |
| **Tanh**           | GELU approximation (tanh-based)                      | Easy       |
| **Split**          | Split QKV projection into Q, K, V                    | Medium     |
| **Where**          | Apply causal mask (select -inf or attention score)    | Easy       |
| **Range**          | Generate position_ids sequence [0, 1, ..., T-1]      | Easy       |
| **ConstantOfShape**| Create tensors filled with a constant (e.g., -inf)   | Easy       |
| **Trilu**          | Generate upper-triangular causal mask                 | Easy       |

## Implementation Priority

**High priority** (core transformer math):
1. LayerNormalization — used 25 times (2 per layer + final)
2. Mul — used everywhere for attention scaling
3. Split — splits QKV, used 12 times
4. Tanh — GELU activation, used 12 times

**Medium priority** (masking and data flow):
5. Where — causal mask application
6. Cast — type conversions
7. Trilu — triangular mask generation

**Low priority** (can be precomputed on host):
8. Range — position ID generation
9. ConstantOfShape — constant tensor creation

## Export Script

`scripts/export_gpt2_onnx.py` — run with:
```
uv run --with torch --with transformers --with onnx --with packaging \
       --with onnxscript --with onnxruntime python scripts/export_gpt2_onnx.py
```

## Notes

- The GELU in GPT-2 uses the `tanh` approximation, which ONNX decomposes into
  Mul + Tanh + Mul + Add rather than emitting a single Gelu node. If a native
  Gelu op is preferred, the model can be re-exported with `approximate="none"`.
- The existing `Gelu` support in onnx_executor.rs is not used by this export
  because PyTorch's tanh-GELU decomposes into primitive ops at the ONNX level.
