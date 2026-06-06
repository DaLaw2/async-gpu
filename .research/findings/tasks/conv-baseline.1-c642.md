# conv-baseline.1: Conv2D Performance Baseline Benchmark
**Cycle**: 642 | **Feature**: conv-baseline | **Kind**: investigation | **Status**: done

## Summary
Benchmarked all conv2d routing paths (Winograd F(2x2,3x3), direct conv, 1x1 GEMM) across
ResNet-50 and YOLOv8-nano shapes on GTX 1660 (sm_75). Current performance is 0.2-5.7% of
FP32 theoretical peak (5.0 TFLOPS) for 3x3 convolutions and 3.2-3.4% for 1x1. The dominant
bottleneck is per-channel serial processing in the Winograd kernel. cuBLAS GEMM on the
equivalent im2col dimensions achieves 16-45% peak, confirming 10-30x headroom.

## Findings

### 1. Performance by Routing Path

**Winograd F(2x2,3x3) — 3x3 stride=1 shapes:**

| Shape           | Cin | Cout | HxW     | Time(ms) | GFLOPS | %Peak |
|-----------------|-----|------|---------|----------|--------|-------|
| ResNet conv1    | 3   | 64   | 224x224 | 0.613    | 282.9  | 5.7%  |
| ResNet L1       | 64  | 64   | 56x56   | 3.255    | 71.0   | 1.4%  |
| ResNet L2       | 128 | 128  | 28x28   | 3.246    | 71.2   | 1.4%  |
| ResNet L3       | 256 | 256  | 14x14   | 3.512    | 65.8   | 1.3%  |
| ResNet L4       | 512 | 512  | 7x7     | 23.755   | 9.7    | 0.2%  |
| C2f 3x3 32      | 32  | 32   | 160x160 | 5.310    | 88.9   | 1.8%  |
| C2f 3x3 64      | 64  | 64   | 80x80   | 5.286    | 89.3   | 1.8%  |
| C2f 3x3 128     | 128 | 128  | 40x40   | 5.324    | 88.6   | 1.8%  |
| C2f 3x3 256     | 256 | 256  | 20x20   | 5.437    | 86.8   | 1.7%  |

**Direct conv — 3x3 stride=2 shapes:**

| Shape              | Cin | Cout | HxW     | Time(ms) | GFLOPS | %Peak |
|---------------------|-----|------|---------|----------|--------|-------|
| Stem 3x3 s2         | 3   | 16   | 640x640 | 0.884    | 100.1  | 2.0%  |
| BB 3x3 s2 16>32     | 16  | 32   | 320x320 | 2.161    | 109.2  | 2.2%  |
| BB 3x3 s2 32>64     | 32  | 64   | 160x160 | 1.984    | 118.9  | 2.4%  |
| BB 3x3 s2 64>128    | 64  | 128  | 80x80   | 2.123    | 111.1  | 2.2%  |
| BB 3x3 s2 128>256   | 128 | 256  | 40x40   | 3.223    | 73.2   | 1.5%  |

**1x1 GEMM — pointwise convolutions:**

| Shape          | Cin | Cout | HxW   | Time(ms) | GFLOPS | %Peak |
|----------------|-----|------|-------|----------|--------|-------|
| Head 1x1 64    | 64  | 64   | 80x80 | 0.311    | 168.7  | 3.4%  |
| Head 1x1 128   | 128 | 128  | 40x40 | 0.326    | 160.8  | 3.2%  |
| Head 1x1 256   | 256 | 256  | 20x20 | 0.328    | 159.7  | 3.2%  |

### 2. cuBLAS GEMM Reference (equivalent im2col dimensions)

| Conv Shape   | M(Cout) | K(Cin*k^2) | N(HW_out) | ms    | GFLOPS | %Peak |
|--------------|---------|------------|-----------|-------|--------|-------|
| ResNet conv1 | 64      | 27         | 50176     | 0.139 | 1243.6 | 24.9% |
| ResNet L1    | 64      | 576        | 3136      | 0.114 | 2032.3 | 40.6% |
| ResNet L2    | 128     | 1152       | 784       | 0.104 | 2232.4 | 44.6% |
| ResNet L3    | 256     | 2304       | 196       | 0.125 | 1854.7 | 37.1% |
| ResNet L4    | 512     | 4608       | 49        | 0.282 | 820.1  | 16.4% |

**Our conv2d vs cuBLAS GEMM equivalent:**
- ResNet conv1: 282.9 / 1243.6 = 22.7% of cuBLAS GEMM
- ResNet L1: 71.0 / 2032.3 = 3.5%
- ResNet L2: 71.2 / 2232.4 = 3.2%
- ResNet L3: 65.8 / 1854.7 = 3.5%
- ResNet L4: 9.7 / 820.1 = 1.2%

### 3. Bottleneck Analysis

**Per-channel serial processing is the dominant bottleneck.**

Evidence from scaling test (doubling Cin at same spatial size):

| HxW   | Cin | Cout | ms     | GFLOPS | ms/channel |
|-------|-----|------|--------|--------|------------|
| 56x56 | 32  | 32   | 0.673  | 85.9   | 0.0210     |
| 56x56 | 64  | 64   | 2.614  | 88.4   | 0.0408     |
| 56x56 | 128 | 128  | 10.328 | 89.5   | 0.0807     |
| 14x14 | 128 | 128  | 0.680  | 85.0   | 0.0053     |
| 14x14 | 256 | 256  | 2.842  | 81.4   | 0.0111     |
| 14x14 | 512 | 512  | 13.054 | 70.8   | 0.0255     |

Key observations:
- Time scales with C_in * C_out (quadratic in channel count), confirming O(C_in*C_out) serial loop.
- ms/channel also doubles when C doubles — this means each per-channel iteration is proportional
  to C_out, consistent with the kernel launching `n_tiles * ceil(C_out/32)` blocks, each looping
  over C_in sequentially.
- GFLOPS stays roughly constant (~85-90) regardless of channel count for same spatial size,
  meaning the kernel is uniformly inefficient — it never reaches compute saturation.
- The Winograd kernel processes C_in channels **serially within each thread** (the `for ci in 0..C_in`
  loop on line 162 of winograd_f2x2.cu). Each thread loads a 4x4 input tile per channel, transforms it,
  and does 16 multiply-accumulate with the filter. This serial loop is the core bottleneck.

**Why the kernel is slow:**
1. **No shared memory for input tiles** — every thread in a warp loads the same 4x4 input tile
   independently from global memory. With 32 threads per block (TILE_C_OUT=32), the same 16 floats
   are loaded 32 times per warp for each spatial tile.
2. **No batched GEMM in Winograd domain** — the Winograd algorithm's key advantage is converting
   conv to 16 independent matrix multiplications (one per Winograd element). Our kernel does this
   as 16 scalar multiply-accumulate per thread, missing the opportunity for high-throughput GEMM.
3. **Low occupancy** — block size is only 32 threads (1 warp), severely limiting latency hiding.
   On sm_75, max 32 warps/SM but each block only contributes 1 warp.
4. **Register pressure** — each thread stores 16 accumulators (m[16]) + 16 input transform (u[16])
   + 16 input tile (d[4][4]) = 48 registers minimum, likely spilling to local memory.

### 4. Arithmetic Intensity Analysis

| Shape           | In(KB) | Filter(KB) | Out(KB) | AI (FLOP/B) | Bound       |
|-----------------|--------|------------|---------|-------------|-------------|
| ResNet conv1    | 588.0  | 6.8        | 12544.0 | 12.9        | Mem-bound   |
| ResNet L1       | 784.0  | 144.0      | 784.0   | 131.9       | Compute     |
| ResNet L2       | 392.0  | 576.0      | 392.0   | 166.0       | Compute     |
| ResNet L3       | 196.0  | 2304.0     | 196.0   | 83.8        | Compute     |
| ResNet L4       | 98.0   | 9216.0     | 98.0    | 24.0        | Mem-bound   |

Ridge point: 5000 GFLOPS / 192 GB/s = 26.0 FLOP/B.
Most ResNet shapes are compute-bound (AI >> 26), yet we achieve only 1-2% peak.
This confirms the bottleneck is computational inefficiency (serial processing), not memory bandwidth.

### 5. YOLOv8-nano End-to-End Estimate

No model weights available for e2e measurement. Estimated from summing individual layer times:
- 4 Winograd conv layers (C2f): ~21 ms
- 4 Direct conv layers (stride-2): ~10 ms
- 3 Head 1x1 layers: ~1 ms
- BatchNorm + SiLU + Concat + Upsample: ~5 ms (estimated)
- Detection head (DFL + NMS): ~3 ms (estimated)
- **Total estimate: ~40-50 ms per frame** (20-25 FPS)

### 6. cuDNN Availability

cuDNN is **not installed** on this system. No cuDNN reference available. Comparison uses
cuBLAS GEMM on equivalent im2col dimensions as proxy.

Published cuDNN benchmarks for GTX 1660 on ResNet-50 3x3 convolutions:
- Typically 40-60% of theoretical peak (~2000-3000 GFLOPS)
- Our 50% target = ~2500 GFLOPS (vs current ~70-90 GFLOPS = 3-4% of cuDNN)

## Open Questions
1. Would batched cuBLAS GEMMs (16 GEMMs for Winograd domain) achieve near-cuBLAS performance?
   The brainstorm recommended this approach — the data strongly supports it.
2. What is the actual YOLOv8-nano e2e time? Need model weights to measure.
3. Can the direct conv kernel (stride=2) be improved with shared memory tiling? Currently
   at 2% peak, but shared memory tiling is already implemented in `direct_conv2d_tiled`.

## Confidence
High — benchmarks run on real hardware with consistent results across 20 iterations.
