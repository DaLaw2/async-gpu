# conv-direct-opt.1: Warp-level C_in Reduction + Shared Memory Tiling for Direct Conv
**Cycle**: 642 | **Feature**: conv-direct-opt | **Kind**: experiment | **Status**: done

## Summary
Implemented a multi-output-channel direct conv kernel (`direct_conv2d_warp_reduce`) that
amortizes shared memory input tile loads across CO_PER_BLOCK=4 output channels and uses
CI_CHUNK_WR=8 channel chunks. Achieved 1.04-2.55x speedup across all measured shapes.
The 3x3 stride=2 path (YOLO backbone) improved most dramatically (2.1-2.6x).

## Findings

### 1. Initial Warp-Level C_in Reduction Attempt (Failed)
First approach: distribute C_in channels across CI_WARPS=8 threads (block dim 16x8x8=1024),
each thread computing a partial sum then reducing via shared memory.

**Result**: 15-34% slower than baseline. Root causes:
- 1024 threads/block hits the SM75 limit, severely constraining registers
- Only 1 channel per thread per iteration (CI_CHUNK_WR=1) = low arithmetic intensity
- Reduction overhead (shared memory write + barrier + serial sum) not amortized

### 2. Multi-Output-Channel Approach (Successful)
Redesigned: CO_PER_BLOCK=4 output channels share one input tile.
Block: (16, 8, 4) = 512 threads. CI_CHUNK_WR=8 channels per iteration.

**Key insight**: The bottleneck was redundant global memory reads of the input tile,
not the C_in serial loop itself. By having 4 output channels reuse the same shared
memory input tile, we reduce global memory traffic by ~4x.

### 3. Performance Results (GTX 1660, FP32)

| Shape | Baseline | Optimized | Speedup |
|-------|----------|-----------|---------|
| 5x5 s1 stem Cin=3 | 296 GFLOPS | 312 GFLOPS | 1.05x |
| 5x5 s1 mid Cin=32 | 323 GFLOPS | 339 GFLOPS | 1.05x |
| 5x5 s1 deep Cin=64 | 312 GFLOPS | 395 GFLOPS | **1.26x** |
| 5x5 s2 stem Cin=3 | 220 GFLOPS | 340 GFLOPS | **1.54x** |
| 5x5 s2 mid Cin=32 | 194 GFLOPS | 325 GFLOPS | **1.67x** |
| 5x5 s2 deep Cin=64 | 216 GFLOPS | 317 GFLOPS | **1.47x** |
| 7x7 s2 ResNet Cin=3 | 270 GFLOPS | 394 GFLOPS | **1.46x** |
| 7x7 s1 stem Cin=3 | 395 GFLOPS | 410 GFLOPS | 1.04x |
| 7x7 s1 mid Cin=32 | 380 GFLOPS | 396 GFLOPS | 1.04x |
| 3x3 s2 YOLO Cin=3 | 100 GFLOPS | 214 GFLOPS | **2.14x** |
| 3x3 s2 BB down Cin=16 | 109 GFLOPS | 278 GFLOPS | **2.55x** |
| 3x3 s2 BB deep Cin=64 | 111 GFLOPS | 253 GFLOPS | **2.28x** |

### 4. Correctness Verification
All shapes verified against CPU f64 reference with relative error < 1e-5.
Tested: 5x5 (s1/s2, Cin=3/8/16/32/64), 7x7 (s1/s2, Cin=3/8/16/32), 3x3 s2 (Cin=16/32/64).

### 5. Kernel Selection Logic
- `direct_conv2d_warp_reduce`: C_out >= 4 and smem fits 48KB
- `direct_conv2d_tiled`: C_out < 4 or warp-reduce smem > 48KB, but tiled smem fits
- `direct_conv2d`: fallback, no shared memory

### 6. Bug Found and Fixed
Initial warp-reduce kernel had a shared memory layout bug: smem_filter pointer
used CI_CHUNK_WR (=1) as offset instead of CI_WARPS*CI_CHUNK_WR (=8), causing
input data and filter data to overlap. Fixed by computing offsets using
max_ci_per_iter. This is relevant for any future kernel with multi-section
shared memory layouts.

## Open Questions
1. Would larger CO_PER_BLOCK (8 or 16) give further improvement for high C_out?
   Risk: shared memory per block increases, reducing occupancy.
2. Could filter weights also be cached in shared memory for the multi-channel
   kernel? Currently each tz-thread reads its own filter weights from global memory.
3. The kernel still achieves only 5-8% of peak (vs cuDNN's ~50%). The fundamental
   structure (one thread per output element, serial C_in loop) limits performance.
   True high performance requires GEMM-based approaches (im2col+cuBLAS).

## Confidence
High — benchmarks run 20 iterations with 5 warmup on real hardware. Correctness
verified against CPU f64 reference for all kernel paths and sizes.
