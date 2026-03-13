# mma-fix.1: Diagnose MMA kernel failure at 768x768
**Cycle**: 170 | **Theme**: mma-fix | **Kind**: investigation | **Status**: done

## Summary
MMA kernel `full_gemm_f32in` passes ALL GPT-2 dimensions (768x768, 768x2304,
768x3072, 3072x768) when tested independently against `gemm_f32` reference.
Max relative error is 0.4% (expected f16 vs f32 precision gap). The kernel
itself has NO bug.

## Findings

### Q: Does full_gemm (f16x2 input) also fail at 768x768?
A: Neither `full_gemm` nor `full_gemm_f32in` fails at any tested dimension.
The 768x768 test (run_full_gemm_test) passes with 0 mismatches. The previous
failure was in the inference pipeline, not the kernel itself.
**Confidence**: high

### Q: Is the host-side grid_dim.y correctly set for large N?
A: Yes. grid_dim = (M/32, N/16, 1) is correctly set in all tests and inference.
**Confidence**: high

### Q: Does bar_sync() generate block-level or warp-level sync?
A: Block-level. Implementation is `bar.sync 0;` which synchronizes all threads
in the block.
**Confidence**: high

### Q: What do shared memory contents look like?
A: Not tested — unnecessary since all dimensions pass.
**Confidence**: n/a

## Unexpected Discoveries
The MMA kernel is correct for ALL GPT-2 dimensions. The previous inference
failure was likely due to incorrect weight packing in the inference pipeline
(the inference code was using `pack_weight()` which may have had a different
packing convention). The kernel was switched to `gemm_f32` before this was
fully diagnosed.

The diagnostic test used the same packing as the standalone GEMM test:
col-major B with f16x2 pairs packed as (lo | hi<<16) per column. If the
inference pipeline packed differently, that would explain the mismatch.

## Impact on Downstream Tasks
- mma-fix.2 (fix and validate) may be unnecessary — the kernel works
- mma-fix.3 (MMA inference) is the real task: integrate MMA into inference
  pipeline with correct weight packing
- Key risk: the `pack_weight()` function used in the old inference code may
  have had a different packing order than what the kernel expects
