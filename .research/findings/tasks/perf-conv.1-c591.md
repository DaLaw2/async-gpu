# perf-conv.1: Winograd F(4×4, 3×3) Investigation
**Cycle**: 591 | **Theme**: perf-conv | **Kind**: investigation | **Status**: done

## Summary
Studied Winograd convolution F(4×4, 3×3) for GPU. Theoretical 2.25x FLOPs reduction,
practical 2.1-2.3x speedup. However, our Conv2D already achieves 98-228% of cuDNN via
cuBLAS GEMM fallback, so Winograd implementation is lower priority.

## Key Findings
- Transform matrices: B^T (6×6 input), G (6×3 filter), A^T (6×6→4×4 output)
- For 32×32 feature map: 64 tiles of 4×4 output, each needing 6×6 input
- Three-phase GPU kernel: input transform → batched GEMM → output transform
- F(4×4) has ~4x numerical error vs direct conv (acceptable in f32)
- Practical speedup: 2.1-2.3x on V100, limited by transform memory bandwidth

## Impact
Conv2D criterion (≥60% cuDNN) already met via cuBLAS GEMM. Winograd would
improve pure-kernel conv but is not blocking any epic criteria.

**Confidence**: high
