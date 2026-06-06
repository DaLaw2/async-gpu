# conv-wino-gemm: Feature Synthesis

**Status**: Implemented. Winograd F(2x2,3x3) restructured as 16 batched cuBLAS GEMMs.

Pipeline: input transform → `cublasGemmStridedBatched` (16 batches) → output transform.
Single cuBLAS call replaces the per-channel serial loop. Bias fused into output transform.

Results (GTX 1660): 2.4-16.1% peak (119-807 GFLOPS). Best: YOLO P3 at 807 GFLOPS.
Previous baseline: 1.3-5.7% peak. Improvement: ~1.5-3x across ResNet shapes.

Correctness: all tests pass, max_err <= 0.000002 across single, batched, and bias cases.
F(2x2) numerically perfect in FP32. F(4x4) deferred for future optimization.

Remaining gap to cuDNN target (50% peak): transform kernel overhead dominates
for small spatial sizes. Next steps: filter caching, thin-GEMM fallback for L4,
or F(4x4) for large spatial dims.
