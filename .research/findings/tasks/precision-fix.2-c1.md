# precision-fix.2: Keep f32 Activations Between GEMM Stages
**Cycle**: 1 | **Theme**: precision-fix | **Kind**: experiment | **Status**: done

## Summary
Implemented `full_gemm_f32in` kernel that accepts f32 activation input (A matrix) and converts to f16 per-tile in shared memory, eliminating the separate `f32_to_f16x2_pack` kernel launch. Test confirms output is bit-identical to `full_gemm` for representable values. The f16→MMA precision loss is inherent to Tensor Core operations and cannot be avoided by changing where the f32→f16 conversion occurs.

## Findings

### Q: Can GEMM output f32 directly without f32→f16→f32 roundtrip between layers?
A: Yes, but the precision improvement is zero. The `full_gemm_f32in` kernel reads f32 from global memory and converts to f16 per-tile in shared memory via `cvt.rn.f16.f32` PTX instruction. Since MMA m16n8k16 requires f16 operands, the f16 conversion is unavoidable — it just happens later (per-tile instead of globally).

Test result: packed vs f32in max_abs_err = 0.0 (bit-identical output for integer-valued inputs where f32→f16 is exact). For real-valued inputs, the conversion is the same `cvt.rn.f16.f32` instruction in both paths, so results will be identical.

**Confidence**: high

### Q: What memory overhead does f32 intermediate storage add?
A: 2× memory for activation buffers (f32 uses 4 bytes vs f16 uses 2 bytes per element). For GPT-2 with seq=1024, d_model=768:
- f32 activations: 1024 × 768 × 4 = 3.0 MB per buffer
- f16 activations: 1024 × 768 × 2 = 1.5 MB per buffer
- Total overhead for all intermediate buffers: ~10-15 MB extra (negligible vs GPU memory)

The real benefit is reducing kernel launch overhead: each GEMM stage saves one `f32_to_f16x2_pack` kernel launch, eliminating 4 kernel launches per transformer layer (12 layers = 48 fewer kernel launches total).

**Confidence**: high

### Q: Does f32 storage require kernel changes or just host-side buffer management?
A: Kernel changes required. The new `full_gemm_f32in` kernel:
- Takes `*const f32` instead of `*const u32` for the A matrix
- Loads 2 f32 values per thread per tile iteration
- Converts to f16 via `cvt.rn.f16.f32` and packs into u32 in shared memory
- All other code (B loading, MMA, output) is identical to `full_gemm`
- B matrix (weights) stays as pre-packed f16x2 since weights don't change between layers

**Confidence**: high

## Key Insight: Precision Floor is from MMA, Not from Memory Format

The f32in vs CPU comparison shows max_rel_err = 0.004236, which is identical to the full_gemm vs CPU error. This confirms precision-fix.1's analysis: the dominant error source is MMA's f16×f16 multiply (mechanism 2), not the f32→f16 input conversion (mechanism 1).

To actually improve precision, the options are:
1. **TF32 MMA** (`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`): Uses 19-bit mantissa instead of f16's 10-bit. Requires sm_80+. ~4× less error per multiply.
2. **Match PyTorch's precision**: PyTorch also uses Tensor Cores with f16/bf16/tf32 for inference, so our error relative to PyTorch may be smaller than our error relative to the CPU f32 reference.
3. **Accept the error**: The 0.4% relative error per GEMM is within normal f16 inference tolerance.

## Implementation Notes
- The kernel emits 4 `cvt.rn.f16.f32` instructions per tile load (2 threads × 2 f32 values each)
- Shared memory layout is unchanged: [32][8] u32 for A, [16][8] u32 for B
- Shared memory size is unchanged: (256 + 128) × 4 = 1536 bytes
- Performance should be similar: extra cvt instructions offset by eliminated pack kernel

## Impact on Downstream Tasks
- **precision-fix.3**: The f32in kernel is ready for the transformer layer test. The key question for the decision gate is whether PyTorch (which also uses Tensor Cores) produces similar errors to ours, making the comparison fairer than CPU f32 reference.
- **full-inference**: The f32in variant simplifies the pipeline (no separate pack kernels needed between layers), reducing host-side complexity.
