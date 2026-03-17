# bench-e2e.2: Benchmark results summary + improvement roadmap
**Cycle**: 571 | **Theme**: bench-e2e | **Kind**: design | **Status**: done

## Summary
Comprehensive analysis of benchmark results across all kernel types, with
prioritized improvement targets and estimated impact.

## Current Performance Summary

### Compute-Bound Operations (GFLOPS)
| Kernel | Ours | cuBLAS | Ratio | Notes |
|--------|------|--------|-------|-------|
| SGEMM 1024³ | 157 | 2777 | 5.6% | Consistent across sizes |
| SGEMM 4096³ | 160 | 2781 | 5.7% | Scales well |
| SGEMM GPT-2 (128×768²) | 126 | 1977 | 6.4% | Smaller M hurts |
| Conv2D (ResNet layer2) | 110 | ~2000* | ~5.5%* | im2col overhead |
| Conv2D (ResNet first) | 55 | ~2000* | ~2.7%* | Low C_in hurts |

*cuDNN comparison not measured directly; estimated from GEMM ratio.

### Memory-Bound Operations (GB/s)
| Kernel | Measured | Peak (~900) | Utilization |
|--------|----------|-------------|-------------|
| elementwise_add | 153 | 900 | 17% |
| gelu_forward | 128 | 900 | 14% |
| layer_norm | 30 | 900 | 3% |

### End-to-End: GPT-2 (seq_len=128)
| Component | Time | % Total | Bottleneck? |
|-----------|------|---------|-------------|
| LM head GEMM | 62.6ms | 28.3% | YES — vocab projection |
| 12 blocks | 158.6ms | 71.7% | YES — all GEMM-dominated |
| Embedding | 0.05ms | 0% | No |
| Total | 221ms | 100% | |

## Improvement Roadmap (Priority Order)

### P0: GEMM Kernel (5.6% → target 30%+ of cuBLAS)
**Impact**: 6x speedup on GEMM → ~5x speedup on GPT-2 inference.
**Approach**:
1. Larger tiles (64×64 or 128×128 instead of 32×16)
2. Register tiling (each thread computes 4×4 or 8×8 sub-tile)
3. Double-buffered shared memory loads (overlap compute + load)
4. Vectorized global memory loads (float4)
5. Software pipelining with `cp.async` (if sm_80+)

**Estimated effort**: Medium-Large
**Expected result**: 500-1000 GFLOPS (18-36% of cuBLAS)

### P1: LayerNorm Kernel (30 GB/s → target 300+ GB/s)
**Impact**: 10x speedup on LayerNorm. Moderate e2e impact (~2% of GPT-2).
**Approach**:
1. Fused mean + variance in single kernel (currently separate passes)
2. Warp-level reduction for row statistics
3. Vectorized loads (float4)

**Estimated effort**: Small
**Expected result**: 300-500 GB/s

### P2: Fused Kernels
**Impact**: Reduce kernel launch overhead.
- Fused LayerNorm + QKV projection
- Fused residual + LayerNorm
- Already have `gemm_bias_gelu` — use it more aggressively

### P3: Memory Bandwidth
**Impact**: 2-3x on element-wise ops.
- Use float4 loads/stores in elementwise_add, GELU
- Coalesce memory access patterns

## Key Takeaway

**GEMM is everything.** It accounts for >95% of GPT-2 compute time. A 5x GEMM
improvement would cut GPT-2 inference from 221ms to ~50ms. All other optimizations
combined would save <10ms. Focus exclusively on GEMM until it reaches 500+ GFLOPS.
