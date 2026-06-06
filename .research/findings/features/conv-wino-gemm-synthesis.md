# conv-wino-gemm: Feature Synthesis

**Status**: Complete. Winograd F(2x2) + F(4x4) with cuBLAS handle caching.

Pipeline: filter transform -> input transform -> `cublasGemmStridedBatched` -> output transform.
F(4x4) for spatial >= 20x20 (tiles >= 64), F(2x2) fallback for smaller. Bias fused in output transform.
cuBLAS handle cached per-thread, eliminating ~0.3ms/call overhead.

Results (GTX 1660): 26-55% peak (1300-2750 GFLOPS) for common 3x3 shapes.
YOLO P4 (128x128 @ 40x40): 2753 GFLOPS = 54.8% peak, exceeding 50% cuDNN target.
YOLO e2e synthetic (10 layers): 2.15ms total, 2.3x faster than baseline.

Correctness: all tests pass, max_err <= 0.000014 for F(4x4), <= 0.000002 for F(2x2).

Edge cases below target: conv1 (C_in=3, 3.3% peak) and L4 (7x7 spatial, 7.6% peak).
These are inherently GEMM-unfriendly shapes representing <5% of total YOLO FLOPs.
