# model-loading.1: GPT-2 Safetensors Format Analysis
**Cycle**: 1 | **Theme**: model-loading | **Kind**: investigation | **Status**: done

## Summary

Analyzed the GPT-2 small (124M params) safetensors weight format from HuggingFace.
All weights are stored as F32 (137M params total including tied lm_head). The model uses
Conv1D layers (not nn.Linear), so weight matrices are stored **transposed** compared to
standard linear layers: shape is `[in_features, out_features]`. Our kernel's `full_gemm`
already consumes weights in column-major packed f16x2 format, so loading requires:
read f32 → transpose (or reinterpret layout) → convert to f16 → pack into f16x2.

## Findings

### Q: What are the exact tensor names and shapes?

A: GPT-2 small has 148 tensors total. The naming convention in safetensors omits the
`transformer.` prefix used in PyTorch's `state_dict()` — the safetensors file uses
bare names like `h.0.ln_1.weight` (not `transformer.h.0.ln_1.weight`).

**Global tensors (4 tensors):**

| Tensor Name | Shape | Size | Notes |
|-------------|-------|------|-------|
| `wte.weight` | [50257, 768] | 38,597,376 | Token embeddings |
| `wpe.weight` | [1024, 768] | 786,432 | Positional embeddings |
| `ln_f.weight` | [768] | 768 | Final LayerNorm gamma |
| `ln_f.bias` | [768] | 768 | Final LayerNorm beta |

**Per-layer tensors (12 tensors x 12 layers = 144 tensors):**

For each layer `i` in 0..12:

| Tensor Name | Shape | Size | Notes |
|-------------|-------|------|-------|
| `h.{i}.ln_1.weight` | [768] | 768 | Pre-attention LayerNorm gamma |
| `h.{i}.ln_1.bias` | [768] | 768 | Pre-attention LayerNorm beta |
| `h.{i}.attn.c_attn.weight` | [768, 2304] | 1,769,472 | QKV projection (Conv1D!) |
| `h.{i}.attn.c_attn.bias` | [2304] | 2,304 | QKV bias |
| `h.{i}.attn.c_proj.weight` | [768, 768] | 589,824 | Attention output projection |
| `h.{i}.attn.c_proj.bias` | [768] | 768 | Output projection bias |
| `h.{i}.ln_2.weight` | [768] | 768 | Pre-FFN LayerNorm gamma |
| `h.{i}.ln_2.bias` | [768] | 768 | Pre-FFN LayerNorm beta |
| `h.{i}.mlp.c_fc.weight` | [768, 3072] | 2,359,296 | FFN up-projection (Conv1D!) |
| `h.{i}.mlp.c_fc.bias` | [3072] | 3,072 | FFN up bias |
| `h.{i}.mlp.c_proj.weight` | [3072, 768] | 2,359,296 | FFN down-projection (Conv1D!) |
| `h.{i}.mlp.c_proj.bias` | [768] | 768 | FFN down bias |

**Total parameter count**: ~124.4M unique params (lm_head shares wte.weight).
Safetensors file stores 137M params (F32) = ~548 MB.

**Confidence**: high

### Q: What dtype are weights stored in?

A: All tensors are stored as **F32** (32-bit floating point). The HuggingFace metadata
page confirms: `{ 'F32' => 137022720 }`. No F16, BF16, or quantized variants in the
default `openai-community/gpt2` safetensors file.

This means loading requires an explicit f32 → f16 conversion step before GPU upload
(our kernels use f16x2 packed format for GEMM weights).

**Confidence**: high

### Q: How to map HuggingFace names to kernel params?

A: Mapping for one transformer layer (layer `i`):

| HF Safetensors Name | Kernel Parameter | Shape (HF) | Required Transform |
|----------------------|-----------------|-------------|-------------------|
| `h.{i}.ln_1.weight` | `ln1_gamma` | [768] | None (use as f32) |
| `h.{i}.ln_1.bias` | `ln1_beta` | [768] | None (use as f32) |
| `h.{i}.attn.c_attn.weight` | `w_qkv` | [768, 2304] | See note 1 below |
| `h.{i}.attn.c_attn.bias` | `bias_qkv` | [2304] | None (use as f32) |
| `h.{i}.attn.c_proj.weight` | `w_proj` | [768, 768] | See note 1 below |
| `h.{i}.attn.c_proj.bias` | `bias_proj` | [768] | None (use as f32) |
| `h.{i}.ln_2.weight` | `ln2_gamma` | [768] | None (use as f32) |
| `h.{i}.ln_2.bias` | `ln2_beta` | [768] | None (use as f32) |
| `h.{i}.mlp.c_fc.weight` | `w_fc` | [768, 3072] | See note 1 below |
| `h.{i}.mlp.c_fc.bias` | `bias_fc` | [3072] | None (use as f32) |
| `h.{i}.mlp.c_proj.weight` | `w_fc_proj` | [3072, 768] | See note 1 below |
| `h.{i}.mlp.c_proj.bias` | `bias_fc_proj` | [768] | None (use as f32) |

**Note 1 — Weight transformation pipeline:**

Our kernel's `full_gemm` expects weights in column-major f16x2 packed format, matching
the layout produced by `make_weight_colmajor()` in tests_compute.rs. The packed buffer
stores `[N_out][N_in/2]` u32 values where each u32 packs two consecutive K-dimension
f16 values for a given output column.

GPT-2 Conv1D stores weights as `[in_features, out_features]` in row-major (C-order),
which is equivalent to `[out_features, in_features]` in column-major — this is actually
**already column-major with K=in_features as the fast dimension**. So the transformation is:

1. Read f32 tensor from safetensors (row-major [in, out])
2. Reinterpret as column-major [out, in] (no data movement needed — same memory layout)
3. Convert f32 → f16 element-wise
4. Pack consecutive f16 pairs into u32 (f16x2)

This is fortunate: Conv1D's "transposed" storage means we do NOT need an explicit
matrix transpose for GEMM weights.

**Confidence**: high

## Safetensors Format Details

### Binary Layout

```
┌─────────────────────────────────────────────────┐
│ 8 bytes: header_size (u64, little-endian)       │
├─────────────────────────────────────────────────┤
│ header_size bytes: JSON header (UTF-8)          │
├─────────────────────────────────────────────────┤
│ remainder: raw tensor data (contiguous)         │
└─────────────────────────────────────────────────┘
```

### JSON Header Schema

```json
{
  "__metadata__": { "format": "pt" },
  "tensor_name": {
    "dtype": "F32",
    "shape": [768, 2304],
    "data_offsets": [BEGIN, END]
  }
}
```

- `data_offsets`: `[BEGIN, END)` — byte offsets relative to the start of the data section
  (i.e., after the 8-byte size prefix + header)
- `dtype`: string enum — see below
- `shape`: array of dimension sizes
- `__metadata__`: optional key with string→string map

### Supported Dtype Strings

| Dtype | Bytes per element | Notes |
|-------|------------------|-------|
| `BOOL` | 1 | Boolean |
| `U8` | 1 | Unsigned 8-bit int |
| `I8` | 1 | Signed 8-bit int |
| `I16` | 2 | Signed 16-bit int |
| `U16` | 2 | Unsigned 16-bit int |
| `F16` | 2 | IEEE 754 half-precision |
| `BF16` | 2 | Brain floating point |
| `I32` | 4 | Signed 32-bit int |
| `U32` | 4 | Unsigned 32-bit int |
| `F32` | 4 | IEEE 754 single-precision |
| `F64` | 8 | IEEE 754 double-precision |
| `I64` | 8 | Signed 64-bit int |
| `U64` | 8 | Unsigned 64-bit int |
| `F8_E4M3` | 1 | 8-bit float (4-bit exp, 3-bit mantissa) |
| `F8_E5M2` | 1 | 8-bit float (5-bit exp, 2-bit mantissa) |

### Constraints

- Endianness: **little-endian** exclusively
- Memory layout: **C-order** (row-major)
- Header size limit: 100 MB (DOS prevention)
- Header must start with `{` (0x7B)
- Data offsets must not overlap
- Byte buffer must be fully indexed (no gaps)
- Empty tensors (dimension = 0) and scalars (shape = []) are allowed
- Duplicate keys are disallowed

## Rust Crate Options

### 1. `safetensors` (official — recommended)

- **Version**: 0.7.0
- **License**: Apache-2.0
- **Maintainer**: HuggingFace
- **Pros**:
  - Official implementation, well-maintained
  - Zero-copy deserialization via `SafeTensors::deserialize()`
  - Mmap-friendly: works with memory-mapped byte slices
  - No dependency on deep learning frameworks
  - 100% doc coverage
- **Cons**:
  - Returns raw `&[u8]` slices — caller must handle dtype interpretation
  - No built-in ndarray integration
- **Key API**:
  - `SafeTensors::deserialize(&[u8])` — parse from byte buffer
  - `.tensor("name")` → `TensorView` with `.data()`, `.shape()`, `.dtype()`
  - `serialize()` / `serialize_to_file()` — for writing

### 2. `ndarray-safetensors`

- **Pros**: Direct ndarray integration
- **Cons**: Extra dependency, may be overkill for our use case (we just need raw f32 slices)

### 3. Manual parsing

- **Pros**: Zero dependencies, full control
- **Cons**: Reinventing the wheel for a simple format
- **Feasibility**: Very doable — read 8 bytes, parse JSON with `serde_json`, slice into
  data buffer. But the official crate already does this cleanly.

**Recommendation**: Use the official `safetensors` crate. It's lightweight, has no heavy
dependencies, and provides zero-copy access. For our use case: mmap the file, call
`SafeTensors::deserialize()`, iterate tensors, reinterpret `&[u8]` as `&[f32]` via
`bytemuck` or pointer cast, then convert to f16x2 packed format.

## Weight Transformation Requirements

### Per-layer weight loading pipeline

```
safetensors file (F32, row-major)
    │
    ▼
mmap + deserialize → TensorView with &[u8] data
    │
    ▼
reinterpret &[u8] as &[f32]  (little-endian, safe on x86/ARM)
    │
    ├── LayerNorm gamma/beta, biases: upload as f32 directly
    │
    └── GEMM weights (c_attn, c_proj, c_fc, mlp.c_proj):
        │
        ▼
        Conv1D [in, out] row-major == [out, in] col-major
        (matches make_weight_colmajor layout — no transpose needed!)
        │
        ▼
        f32 → f16 conversion (element-wise)
        │
        ▼
        Pack consecutive f16 pairs → u32 (f16x2)
        │
        ▼
        Upload packed u32 buffer to GPU via cudarc
```

### Special considerations

1. **Conv1D transposition**: GPT-2's `Conv1D` stores `[in, out]` which is the transpose
   of `nn.Linear`'s `[out, in]`. Our GEMM kernel expects column-major `[K][N]` which
   maps to row-major `[N][K]`. Conv1D's `[in=K, out=N]` in row-major IS `[K][N]` —
   this matches directly. No explicit transpose needed.

2. **QKV split**: `c_attn.weight` [768, 2304] contains Q, K, V concatenated along the
   output dimension. The 2304 = 3 × 768 columns are ordered as [Q|K|V]. Our kernel's
   `split_qkv` handles this after the fused QKV GEMM, so we can load the full [768, 2304]
   weight as-is.

3. **Token/positional embeddings**: `wte.weight` [50257, 768] and `wpe.weight` [1024, 768]
   are used as lookup tables, not GEMM weights. They should be uploaded as f32 and
   indexed by token/position ID. For the output (lm_head), wte.weight is reused as a
   GEMM weight [768, 50257] which DOES need the f16x2 packing treatment.

4. **f16 precision**: Converting f32 weights to f16 loses precision. For GPT-2 small this
   is generally acceptable — the model was trained in f32 but inference in f16 is standard
   practice with minimal quality loss.

## GPT-2 Small Architecture Reference

| Parameter | Value |
|-----------|-------|
| Layers (n_layer) | 12 |
| Hidden size (n_embd) | 768 |
| Attention heads (n_head) | 12 |
| Head dimension | 64 (= 768/12) |
| FFN intermediate (4 × n_embd) | 3072 |
| Vocab size | 50257 |
| Max positions (n_positions) | 1024 |
| Activation function | gelu_new |
| LayerNorm epsilon | 1e-5 |
| Total unique params | ~124.4M |

## Impact on Downstream Tasks

1. **model-loading.2 (loader implementation)**: Can use the `safetensors` crate with mmap.
   Weight transformation is straightforward — the Conv1D layout aligns with our kernel's
   expected format. Main work: iterate 12 layers, extract tensors by name pattern, run
   the f32→f16x2 packing pipeline, upload to GPU.

2. **Embedding handling**: Need a new kernel or host-side lookup for token/position
   embeddings (currently not implemented in gpu-kernel). Could do GPU-side gather or
   CPU-side slice + upload.

3. **LM head (output projection)**: Reuses wte.weight transposed. Since wte is [50257, 768]
   in row-major = [768, 50257] col-major, this needs the f16x2 pack for the final GEMM.
   The large vocab dimension (50257) may need padding to align with tile sizes.

4. **Memory budget**: Full model in f16 ≈ 248 MB on GPU. Plus activations for seq_len=1024:
   ~150 MB. Well within modern GPU VRAM (even 4 GB cards).
