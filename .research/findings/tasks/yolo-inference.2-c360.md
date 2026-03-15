# yolo-inference.2: Conv2D kernel (im2col + GEMM)
**Cycle**: 360 | **Theme**: yolo-inference | **Kind**: experiment | **Status**: done

## Summary

Implemented Conv2D as im2col → gemm_f32 pipeline. Fixed a boundary handling bug where
gemm_f32's output write has no bounds check for N < tile_width (16). Solution: pad N to
a multiple of 16 on the host side, then extract only the valid columns from the output.

## Findings

### Q: Does im2col + GEMM produce correct Conv2D results?
A: Yes, after N-padding fix. max_err = 0.000000 for Conv2D with [2, 8, 8] input, [4, 2, 3, 3]
weight, stride=1, pad=1. GEMM dimensions: M=64, K=18, N=4 (padded to N=16).
**Confidence**: high

### Q: What was the root cause of the initial 0.67 max error?
A: gemm_f32 kernel (compute_gemm.rs:1787-1790) writes 4 output elements per thread
unconditionally without checking if global_c0/global_c1 < n_cols. When N=4 < tile_width=16,
threads with col_pair >= 2 compute indices like `row * 4 + 5`, which aliases into the next
row's data, overwriting valid results with zeros.
**Confidence**: high

## Unexpected Discoveries

- The gemm_f32 kernel has no output boundary checking at all — it works for GPT-2 only because
  all dimensions there happen to be multiples of 16. This is a latent bug that would bite any
  GEMM with non-16-aligned N.
- The fix (host-side N-padding) is simple and zero-cost for the kernel.

## Impact on Downstream Tasks
- Conv2D is ready for yolo-inference.6 (backbone integration)
- All YOLO Conv2D layers have C_out >= 16 (smallest is 16), so N-padding will be a no-op
  for actual YOLO weights. The test case (C_out=4) was pathological.
- Weight layout: PyTorch [C_out, C_in*kH*kW] row-major = column-major [K, C_out] — no transpose needed
