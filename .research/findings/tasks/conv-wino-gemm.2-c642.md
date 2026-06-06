# conv-wino-gemm.2: Winograd with 16x Batched cuBLAS GEMM
**Cycle**: 642 | **Feature**: conv-wino-gemm | **Kind**: experiment | **Status**: done

## Summary
Implemented the 3-phase batched-GEMM Winograd F(2x2,3x3) pipeline, replacing the
old fused per-channel kernel. The new pipeline uses separate NVRTC kernels for
input/output transforms and a single `cublasGemmStridedBatched` call for the 16
transform-domain GEMMs. All correctness tests pass with max_err <= 0.000002.

## Findings

### 1. Implementation

Three new components:
- **`winograd_gemm_f2x2.cu`**: Two NVRTC kernels — `winograd_input_transform` (tiles →
  V[16, C_in, n_tiles]) and `winograd_output_transform` (M[16, C_out, n_tiles] → spatial
  output with fused bias).
- **`conv2d_winograd_f2x2_impl` rewrite in conv.rs**: 4-phase pipeline replacing the
  old fused kernel. Phase 1: filter transform (reused), Phase 2: input transform (new),
  Phase 3: 16x strided batched GEMM via cuBLAS, Phase 4: output transform (new).
- **`winograd_gemm_bench.rs`**: Correctness tests (single, batched, with bias) and
  performance benchmark.

### 2. cuBLAS Column-Major Mapping

The key insight for the strided batched GEMM configuration:
- We want: `M_k[C_out, n_tiles] = U_k[C_out, C_in] * V_k[C_in, n_tiles]` (row-major)
- cuBLAS sees column-major, so we compute: `C^T = V^T * U^T`
- Config: `transa=N, transb=N, m=n_tiles, n=C_out, k=C_in`
- `lda=n_tiles, ldb=C_in, ldc=n_tiles` with strides between the 16 planes
- Single cuBLAS call replaces the per-channel serial loop (previous bottleneck).

### 3. Performance Results (GTX 1660, 5027 GFLOPS peak)

| Shape | C_in | C_out | HxW | Time(ms) | GFLOPS | %Peak |
|-------------|------|-------|---------|----------|--------|-------|
| ResNet conv1| 3 | 64 | 224x224 | 1.45 | 119.3 | 2.4% |
| ResNet L1 | 64 | 64 | 56x56 | 0.51 | 453.7 | 9.0% |
| ResNet L2 | 128 | 128 | 28x28 | 0.49 | 467.8 | 9.3% |
| ResNet L3 | 256 | 256 | 14x14 | 0.50 | 463.0 | 9.2% |
| ResNet L4 | 512 | 512 | 7x7 | 0.81 | 285.6 | 5.7% |
| YOLO P3 | 64 | 64 | 80x80 | 0.58 | 807.4 | 16.1% |

Previous baseline (conv-baseline.1): 1.3-5.7% peak (65-285 GFLOPS).
New results: 2.4-16.1% peak (119-807 GFLOPS).

**Speedup over old fused kernel**: ~1.5-3x for most shapes. YOLO P3 (64x64, 80x80) shows
the best improvement at 16.1% peak (807 GFLOPS), up from the old baseline.

### 4. Correctness

All tests pass:
- `winograd_f2x2_correctness`: 6/6 shapes, max_err=0.000000
- `winograd_gemm_correctness_with_bias`: 4/4 shapes, max_err <= 0.000002
- `winograd_gemm_batched_correctness`: N=4 batch, max_err=0.000000

Bias is now fused into the output transform kernel (was previously a host-side
round-trip), eliminating one D2H+H2D copy per conv.

### 5. Architecture

The batched-GEMM approach is correct and clean but the per-element GFLOPS numbers
are moderate because:
1. **Transform overhead**: Two extra kernel launches (input + output transform) add
   latency that dominates for small spatial sizes.
2. **cuBLAS batch size 16**: For small n_tiles (L4: n_tiles=16), cuBLAS underperforms
   because each GEMM matrix is thin (512x512x16).
3. **Filter transform not cached**: Re-computed each call. Caching would save ~0.1ms.

For the highest GFLOPS shapes (YOLO P3: n_tiles=1600), cuBLAS operates efficiently
and the transform overhead is amortized.

## Open Questions
1. Filter transform caching: worth adding a cache keyed by weight pointer?
2. Should L4-style shapes (n_tiles <= 16) fall back to the fused kernel?
3. F(4x4,3x3) for large spatial dims: 36 GEMMs but 4x arithmetic advantage.

## Confidence
High — implementation complete, all tests pass, performance improvement demonstrated.
