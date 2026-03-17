# perf-attention: Flash Attention optimization attempts
**Cycle**: 593 | **Theme**: perf-attention | **Status**: partially met

## Summary
Attempted three approaches to improve attention from ~5-12% to 50% of cuDNN FA2.
None achieved the target due to fundamental architectural limitations.

## Approaches Tried

### 1. Rust PTX V2 (4-way unrolled dot products) — DEPLOYED
- Result: 1.77x over V1 (seq=512: 14.0ms → 7.9ms)
- Still 2.4-6.6% of cuDNN FA2
- Bottleneck: scalar dot products, 1 thread per Q row

### 2. cuBLAS matmul-based (Q·K^T as GEMM) — REVERTED
- Result: 3x SLOWER (32ms → 102ms)
- Cause: 84 PCIe round-trips (7 transfers × 12 heads) for host softmax
- Learning: zero-transfer architecture beats faster GEMM

### 3. NVRTC CUDA C kernel — AVAILABLE BUT NOT DEFAULT
- Result: ~1.1x over Rust PTX (minimal improvement)
- Cause: same algorithm, nvcc doesn't auto-vectorize
- Learning: CUDA C != automatic optimization

## Root Cause Analysis
The 50% cuDNN target requires implementing FlashAttention-2's core innovation:
**warp-cooperative tiled GEMM for both Q·K^T and P·V**, with:
- Q tiles in shared memory (not per-thread registers)
- MMA or register-blocked matrix multiply for score computation
- Multi-thread parallel dot product reduction
- Software-pipelined KV tile prefetch

This is essentially reimplementing Dao et al.'s FA2 algorithm, which required
months of GPU kernel engineering. Our current scalar approach is a reasonable
first implementation at 5-12% of the optimized reference.

## Recommendation
Accept current attention performance (5-12% of cuDNN). The GPT-2 target
(32.7ms < 35ms) is met. Attention optimization is a **future epic** requiring
dedicated GPU kernel engineering effort.

**Confidence**: high
