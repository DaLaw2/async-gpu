# bench-kernels.1: SGEMM benchmark harness — GFLOPS vs cuBLAS
**Cycle**: 566 | **Theme**: bench-kernels | **Kind**: experiment | **Status**: done

## Summary
Created benchmark example at `examples/std/benchmark/` measuring SGEMM GFLOPS and
memory-bound op GB/s. Our GEMM achieves ~157 GFLOPS (5.6% of cuBLAS ~2780 GFLOPS).
Memory-bound ops range from 29-153 GB/s. LayerNorm is the biggest bottleneck.

## Findings

### Q: What is our SGEMM performance vs cuBLAS?
A: Consistently ~5.6% of cuBLAS across all matrix sizes:

| Size (M=N=K) | Ours (GFLOPS) | cuBLAS (GFLOPS) | Ratio |
|--------------|---------------|-----------------|-------|
| 512          | 148.7         | 2052.6          | 7.2%  |
| 1024         | 156.7         | 2776.6          | 5.6%  |
| 2048         | 157.0         | 2784.9          | 5.6%  |
| 4096         | 159.1         | 2780.8          | 5.7%  |

GPT-2 shapes:
| Shape        | Ours (GFLOPS) | cuBLAS (GFLOPS) | Ratio |
|--------------|---------------|-----------------|-------|
| 128x768x768  | 128.0         | 1995.5          | 6.4%  |
| 128x768x3072 | 131.2         | 2608.8          | 5.0%  |
| 128x3072x768 | 136.1         | 2507.2          | 5.4%  |

**Confidence**: high

### Q: What are the memory-bound op bandwidths?
A: For 786K elements (GPT-2 1024×768):

| Operation       | GB/s  |
|-----------------|-------|
| elementwise_add | 153.0 |
| gelu_forward    | 128.0 |
| layer_norm      | 29.5  |

**Confidence**: high

## Analysis

### GEMM Gap (18x)
Our kernel does ~157 GFLOPS flat regardless of size. cuBLAS does ~2780 GFLOPS.
Key reasons for the gap:
1. Our kernel uses 128 threads, 32×16 tiles with 3072B shared memory — cuBLAS
   uses much larger tiles with multi-level register blocking
2. No double-buffering of shared memory loads
3. No vectorized loads (float4)
4. No software pipelining

### Memory-Bound Gap
- elementwise_add at 153 GB/s is decent (GPU peak ~900 GB/s for RTX 3090)
- layer_norm at 29.5 GB/s is very poor — likely due to per-row reduction overhead
  and multiple kernel launches for mean/variance computation

### Optimization Priorities
1. **LayerNorm** (29.5 GB/s → target ~500 GB/s) — biggest relative gap
2. **GEMM** (157 → target 1000+ GFLOPS) — dominates end-to-end latency
3. **GELU** (128 GB/s → target 500+ GB/s) — moderate gap

## Code Changes
- Created `examples/std/benchmark/` with SGEMM + memory-bound benchmarks
- Uses cudarc's cuBLAS bindings for comparison

## Impact on Downstream Tasks
- bench-kernels.2 (Conv2D) and .3 (Attention) can reuse this harness
- bench-e2e.1 can use GEMM numbers to predict bottlenecks
- bench-e2e.2 can synthesize improvement roadmap from these numbers
