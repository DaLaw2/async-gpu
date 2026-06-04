# perf-gemm-v4.2.1: Double-buffer shared memory SGEMM (SM75-compatible)
**Cycle**: 607 | **Theme**: perf-gemm | **Kind**: experiment | **Status**: done

## Summary
Fixed and activated the V4.1 SGEMM kernel (BK=16, float4 loads, double-buffered shared memory)
that was previously defined but never dispatched. Found and fixed a critical launch configuration
bug that prevented V4.1 from running on SM75 hardware.

## Bugs Found and Fixed

### 1. V4.1 never dispatched (dead code)
The `matmul_v2` dispatch logic at line 253 called `matmul_v4` (BK=8) for large matrices.
`matmul_v4_1` (BK=16) was defined but unreachable. Fixed by changing dispatch to call
`matmul_v4_1` instead.

### 2. Incorrect shared_mem_bytes in launch config (critical bug)
Both V4 and V4.1 kernels use `__shared__` (static shared memory) in CUDA C, but the
host-side launch config also specified `shared_mem_bytes` for dynamic shared memory.
CUDA treats these as additive: total = static + dynamic.

- V4 (BK=8): static=16,640B, dynamic=16,640B → total=33,280B < 48KB → happened to work
- V4.1 (BK=16): static=33,280B, dynamic=33,280B → total=66,560B > 48KB → **CUDA_ERROR_INVALID_VALUE**

Fix: set `shared_mem_bytes: 0` for both kernels since they use static shared memory.

## Benchmark Results (SM75, GTX 1660)

| Shape | V4 (GFLOPS) | V4.1 (GFLOPS) | cuBLAS (GFLOPS) | V4 % | V4.1 % |
|-------|-------------|----------------|-----------------|------|--------|
| 512x512x512 | 1,228 | 1,584 | 1,673 | 73.4% | **94.7%** |
| 1024x1024x1024 | 1,936 | 2,079 | 2,307 | 83.9% | **90.1%** |
| 2048x2048x2048 | 2,191 | 2,412 | 2,961 | 74.0% | **81.4%** |
| 4096x4096x4096 | 2,707 | 2,691 | 2,987 | 90.6% | **90.1%** |

## V4.1 vs V4 Improvement

- 512: +29% (BK=16 better hides memory latency at small sizes)
- 1024: +7.4%
- 2048: +10.1%
- 4096: ~tied (both already compute-bound at this size)

## V4.1 Design (BK=16 double-buffered)

### Tiling
- CTA tile: 128x128, BK=16
- Thread tile: 8x8 (256 threads = 16x16 grid)
- FMA:load ratio = 8x8 * 16 / (16+16) = 1024/32 = 32:1

### Shared Memory Layout
- A stored transposed: `A_smem[BK][BM+4]` = `[16][132]` (padding avoids bank conflicts)
- B stored direct: `B_smem[BK][BN]` = `[16][128]`
- One buffer: 16*132 + 16*128 = 4160 floats = 16,640 bytes
- Double-buffered: 33,280 bytes (fits in SM75's 48KB)

### Double-Buffer Pipeline
```
Prologue: LOAD_TILE(buf=0, k=0); __syncthreads();
Main loop:
  1. LOAD_TILE(1-buf, k+1) into alternate buffer
  2. COMPUTE on current buffer (8x8 outer product x BK=16 iterations)
  3. __syncthreads()
  4. buf = 1 - buf
```

### Vectorized Loads
- A: float4 coalesced reads, transposed store to smem (2 passes, 4 elem/thread/pass)
- B: float4 coalesced reads, direct store to smem (2 passes, 4 elem/thread/pass)
- Output: float4 vectorized stores where alignment permits

## Target Achievement
- **Target**: >=70% of cuBLAS at 4096x4096
- **Achieved**: 90.1% at 4096x4096 (V4.1), exceeding the target by 20 percentage points
- All tested sizes achieve 81-95% of cuBLAS

## Known Issues
- OnceLock kernel caching is device-specific but uses global state; tests that create
  multiple CudaDevice instances will fail with KernelNotFound. Not a production issue
  (single device per process).

**Confidence**: high
