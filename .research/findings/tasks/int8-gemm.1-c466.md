# int8-gemm.1: dp4a instruction, quantization scheme, PTX syntax
**Cycle**: 466 | **Theme**: int8-gemm | **Kind**: investigation | **Status**: done

## Summary
dp4a computes 4×INT8 dot product + accumulate in one instruction. Available SM 61+.
PTX syntax: `dp4a.u32.u32 {out}, {a}, {b}, {c}`. Each u32 input packs 4 INT8 values.
Inline asm pattern matches existing codebase (fma_f32, atom.cas patterns).

## Findings
### Q: How does dp4a work?
A: `dp4a.u32.u32 d, a, b, c` computes:
d = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) + c
where a and b are u32 holding 4 packed bytes. Result is u32 (INT32 accumulation).
For signed: use `dp4a.s32.s32` with i8 values.
**Confidence**: high

### Q: Quantization scheme?
A: Per-channel symmetric quantization: `q = clamp(round(x / scale), -128, 127)`.
Scale = max(|x|) / 127. Dequantize: `x_approx = q * scale`.
For GEMM: C_f32 = (A_int8 × B_int8) * (scale_A * scale_B) + bias
**Confidence**: high

## Design Decision
INT8 GEMM kernel: tile A[M,K/4] and B[K/4,N] (K packed by 4), use dp4a for inner loop.
Each thread computes one output element, accumulates K/4 dp4a operations into INT32,
then dequantizes to f32 via scale multiplication.
