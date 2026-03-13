# model-loading.2: Safetensors Parser Implementation
**Cycle**: 1 | **Theme**: model-loading | **Kind**: experiment | **Status**: done

## Summary

Implemented a safetensors-based GPT-2 weight loader in `crates/gpu-host/src/model.rs`.
Used the official `safetensors` crate (v0.7.0) rather than a custom parser — it is
lightweight, well-maintained, and provides exactly what we need.

## Research Questions Answered

### Q1: Does the safetensors crate work for our needs?

**Yes.** The `safetensors` crate provides:
- `SafeTensors::deserialize(&[u8])` — parses header and provides zero-copy tensor views
- `TensorView::data()` returns `&[u8]` which we convert to `Vec<f32>` via `f32::from_le_bytes`
- `TensorView::shape()` and `TensorView::dtype()` for validation
- Error types that integrate cleanly with our custom `ModelError`

No custom parser needed. The crate adds minimal dependencies (only `serde_json` and `serde`).

**Confidence**: high

### Q2: How to handle endianness and memory alignment?

**Endianness**: Safetensors stores data in little-endian format. Our host (x86-64) is also
little-endian, so `f32::from_le_bytes()` is effectively a no-op reinterpretation. We use
the safe `from_le_bytes` approach rather than pointer casts because safetensors data may
not be 4-byte aligned within the file buffer.

**Alignment**: The raw tensor data in the safetensors buffer is NOT guaranteed to be
aligned to `f32`'s 4-byte alignment requirement. We handle this by copying via
`copy_bytes_to_f32_vec()` which reads 4 bytes at a time through `from_le_bytes`.
This is safe on all platforms and the copy cost is negligible compared to subsequent
f16 conversion and GPU upload.

**Confidence**: high

### Q3: What is the memory footprint for loading all weights?

The loader reads the entire safetensors file (~548 MB for GPT-2 small with 137M F32 params)
into a `Vec<u8>`, then copies out individual tensors as `Vec<f32>`. Peak memory during
loading:
- File buffer: ~548 MB
- Extracted f32 tensors in `Gpt2Weights`: ~497 MB (124.4M unique params × 4 bytes)
- Peak: ~1.05 GB (file buffer + extracted tensors coexist briefly)

After loading, the file buffer is dropped, leaving only ~497 MB in the `Gpt2Weights` struct.
The `Gpt2Weights::memory_bytes()` method reports the exact footprint.

For future optimization, memory-mapping (`mmap`) could eliminate the file buffer copy,
reducing peak memory to ~548 MB (shared between file and tensor views). This would require
using `SafeTensors::deserialize()` with an mmap'd slice instead of `fs::read()`.

**Confidence**: high

## Implementation Details

### Files Modified
- `crates/gpu-host/Cargo.toml` — added `safetensors = "0.7"` dependency
- `crates/gpu-host/src/lib.rs` — added `pub mod model;`
- `crates/gpu-host/src/model.rs` — new module (described below)

### Module Structure (`model.rs`)

**Constants**: `NUM_LAYERS`, `HIDDEN_DIM`, `NUM_HEADS`, `FFN_DIM`, `VOCAB_SIZE`, `MAX_SEQ_LEN`

**Error type** (`ModelError`):
- `Io` — file read errors
- `SafeTensors` — parse errors from the safetensors crate
- `MissingTensor` — expected tensor not found
- `UnexpectedDtype` — tensor is not F32
- `UnexpectedShape` — tensor dimensions don't match expected GPT-2 shapes

**Data structures**:
- `LayerNormWeights` — weight + bias (both `Vec<f32>`)
- `TransformerLayerWeights` — all 12 tensors per layer
- `Gpt2Weights` — wte, wpe, ln_f, and 12 layers

**Public functions**:
- `load_gpt2_weights(path)` — load and validate all 148 GPT-2 tensors
- `load_all_tensors(path)` — generic loader returning `HashMap<String, (Vec<f32>, Vec<usize>)>`
- `list_tensors(path)` — metadata inspection without full data loading

### Key Design Decisions

1. **Copy instead of zero-copy**: We copy tensor data into owned `Vec<f32>` rather than
   borrowing `&[f32]` from the file buffer. This avoids lifetime complexity (the
   `Gpt2Weights` struct owns all its data) and handles unaligned access safely.

2. **Strict validation**: Every tensor is validated for both dtype (must be F32) and shape
   (must match expected GPT-2 dimensions). This catches format mismatches early.

3. **No f16 conversion here**: The model module only handles loading and organizing f32
   weights. The f32 → f16 → f16x2 packing pipeline belongs in a separate conversion step
   closer to GPU upload, keeping concerns separated.

4. **No anyhow**: Uses a custom `ModelError` enum with manual `Display` and `Error` impls,
   consistent with the project's dependency policy.

## Verification

`cargo check -p gpu-host` passes with zero warnings.
