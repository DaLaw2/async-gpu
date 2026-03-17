# int4-kernel.1: INT4 quantization investigation
**Cycle**: 483 | **Theme**: int4-kernel | **Kind**: investigation | **Status**: done

## Summary
INT4 (W4A16): pack 2 values per byte (8 per u32). Per-group quantization (group_size=128)
gives much better accuracy than per-tensor. No dp4a for INT4 — dequantize to f32 on-the-fly
during GEMM via FMA.

## Findings
### Q: How does INT4 packing work?
A: Two 4-bit values per byte: low nibble = val[0], high nibble = val[1].
Pack: `byte = (q1 & 0xF) | ((q2 & 0xF) << 4)`. Unpack: `val = (byte >> shift) & 0xF`.
Signed INT4: range [-8, 7], stored as unsigned [0, 15] with zero_point=8.
**Confidence**: high

### Q: Per-group quantization?
A: Divide K dimension into groups of 128. Each group has its own (scale, zero_point).
For group g: `q = round((x - zero_point) / scale)`. Dequantize: `x ≈ q * scale + zero_point`.
Storage: weights [K/group_size, N] + scales [K/group_size/group_size, N] + zeros.
GPT-2 124M: ~62MB INT4 weights + ~4MB scales = ~66MB (vs 500MB f32).
**Confidence**: high

### Q: W4A16 GEMM algorithm?
A: For each output element C[i,j]:
1. Load packed INT4 weight bytes for row i
2. Unpack to f32: `w = (nibble - 8) * scale[group]`
3. FMA: `acc += activation[k] * w`
4. Standard f32 accumulation
This is memory-bound (INT4 loads are 8x smaller), so even with dequantize overhead,
memory bandwidth savings can give speedup on bandwidth-limited ops.
**Confidence**: high
