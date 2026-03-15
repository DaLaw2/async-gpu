# nn-tensor.1-8: GpuTensor Foundation
**Cycle**: 367-376 | **Theme**: nn-tensor | **Kind**: experiment | **Status**: done

## Summary
Implemented the complete GpuTensor type with N-dimensional shape/strides (SmallVec<[usize; 4]>),
Arc<CudaDevice>, and CudaSlice<f32> ownership. All 8 tasks in the nn-tensor theme completed in a
single batch: struct definition, from_host/to_host/zeros, reshape, transpose, concat, split,
clone_tensor, NnError (manual impl Display + Error, no thiserror needed), and unit tests.

## Findings
### Q: What tensor design works for both GPT-2 (2D/3D/4D) and YOLO (4D)?
A: N-dimensional with SmallVec<[usize; 4]> for shape/strides. Avoids heap for <= 4 dims.
C-contiguous by default, transpose materializes a contiguous copy.
**Confidence**: high

### Q: How to handle non-contiguous tensors?
A: V1 always materializes contiguous copies (transpose, split all produce contiguous output).
is_contiguous() check available for future optimization.
**Confidence**: high

## Key Implementation Details
- `from_data()` for wrapping existing CudaSlice (zero-copy from existing allocations)
- `from_host()` validates numel matches shape before upload
- `reshape()` copies data (no zero-copy view yet — would need refcounted CudaSlice)
- `transpose()` uses generic N-dim index mapping on host, then uploads result
- `concat()` and `split()` use host-side data movement (GPU kernels available for optimization later)
- cudarc `dtod_copy()` requires `&mut` destination — caught by compiler

## Open Questions
- GPU-side transpose kernel could speed up transpose() (currently host roundtrip)
- concat/split also use host roundtrip — could use GPU concat_channels kernel for channel dim
