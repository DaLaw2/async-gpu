# conv-baseline: Feature Synthesis

Current conv2d achieves 1-6% of FP32 peak across all paths.
Winograd F(2x2,3x3): 65-283 GFLOPS (1.3-5.7% peak).
cuBLAS GEMM on same dimensions: 820-2232 GFLOPS (16-45% peak).
Gap: our conv is 3-23% of what cuBLAS GEMM achieves.

Root cause: per-channel serial loop in Winograd kernel.
Each thread loops over C_in channels sequentially, loading the
same input tile redundantly across 32 threads in a warp.
No shared memory, no batched GEMM — pure scalar accumulation.

Direct conv (stride=2): 73-119 GFLOPS (1.5-2.4% peak).
1x1 GEMM: 160-169 GFLOPS (3.2-3.4% peak).

Most shapes are compute-bound (AI >> 26 FLOP/B) yet achieve
<2% compute utilization. Fix is restructuring Winograd as
16 batched cuBLAS GEMMs per the brainstorm recommendation.

Target: >=50% of cuBLAS GEMM = ~1000-1100 GFLOPS for ResNet L1-L3.
Current: ~70 GFLOPS. Required improvement: ~14-15x.
