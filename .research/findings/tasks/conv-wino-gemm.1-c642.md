# conv-wino-gemm.1: Batched-GEMM Winograd Pipeline Design
**Cycle**: 642 | **Feature**: conv-wino-gemm | **Kind**: investigation | **Status**: done

## Summary
Designed a 3-phase batched-GEMM Winograd pipeline and evaluated F(2x2,3x3) vs F(4x4,3x3).
Recommendation: start with F(2x2,3x3) using `cublasGemmStridedBatched` for 16 batched GEMMs.
F(4x4) deferred — higher arithmetic advantage but numerical instability and complexity;
F(2x2) already provides the path to 10-30x speedup over the current kernel.

## Findings

### 1. Winograd-as-GEMM Pipeline Design (F(2x2,3x3))

The pipeline has 3 phases:

**Phase 1 — Filter Transform (one-time, amortized):**
- `weight[C_out, C_in, 3, 3]` → `U[16, C_out, C_in]` via `G·g·Gᵀ`
- Already implemented in `winograd_filter_transform` kernel
- Layout: `U[k][co][ci]` with stride `C_out * C_in` between k-planes
- This is correct for strided batched GEMM: each k-plane is a contiguous matrix

**Phase 2 — Input Transform (per-invocation):**
- Tile input into overlapping 4x4 patches at stride 2
- `n_tiles = ceil(H_out/2) * ceil(W_out/2)` tiles per sample
- Transform each tile: `V[i] = Bᵀ·d·B` → 16 values per tile
- **Output layout**: `V[16, C_in, n_tiles]` — each k-plane is `[C_in, n_tiles]`
- Custom CUDA kernel: one thread per (tile, channel), writes to strided layout
- This is the NEW kernel needed — transforms + scatters into GEMM-ready layout

**Phase 3 — 16 Batched GEMMs:**
- For each k in 0..15: `M_k[C_out, n_tiles] = U_k[C_out, C_in] × V_k[C_in, n_tiles]`
- Use `cublasGemmStridedBatched` with:
  - `batch_size = 16`
  - `M = C_out, N = n_tiles, K = C_in`
  - `stride_a = C_out * C_in` (between U planes)
  - `stride_b = C_in * n_tiles` (between V planes)
  - `stride_c = C_out * n_tiles` (between M planes)
- Output: `M[16, C_out, n_tiles]`

**Phase 4 — Output Transform (per-invocation):**
- For each output tile: `Y[2x2] = Aᵀ·M·A`
- Reads from `M[16, C_out, n_tiles]`, writes to `output[C_out, H_out, W_out]`
- Custom CUDA kernel: one thread per (tile, c_out), reads 16 values, writes 4

### 2. GEMM Shapes for ResNet Benchmark Shapes

| Shape | C_out | C_in | H_out | n_tiles | GEMM (M×K×N) | FLOPs/GEMM |
|---------|-------|------|-------|---------|---------------|------------|
| conv1 | 64 | 3 | 224 | 12544 | 64×3×12544 | 4.8M |
| L1 | 64 | 64 | 56 | 784 | 64×64×784 | 6.4M |
| L2 | 128 | 128 | 28 | 196 | 128×128×196 | 6.4M |
| L3 | 256 | 256 | 14 | 49 | 256×256×49 | 6.4M |
| L4 | 512 | 512 | 7 | 16 | 512×512×16 | 8.4M |

Notes:
- n_tiles = ceil(H_out/2) * ceil(W_out/2) for square spatial dims
- Total FLOPs per conv = 16 × 2×M×K×N (16 GEMMs, each M×K×N)
- These are MUCH smaller matrices than the im2col equivalent, but there are 16 of them
- L1-L3 have well-sized GEMMs (K=64-256 is good for cuBLAS)
- L4 is pathological: N=16 means very thin matrices — cuBLAS may underperform
- conv1 is memory-bound: K=3 is extremely small (GEMV territory)

### 3. cudarc 0.12 Batched GEMM API

**Available**: `cublasGemmStridedBatched` via `cudarc::cublas::Gemm::gemm_strided_batched`

Rust API (from `cudarc::cublas::safe`):
```rust
pub struct StridedBatchedConfig<T> {
    pub gemm: GemmConfig<T>,
    pub batch_size: c_int,      // 16 for F(2x2)
    pub stride_a: c_longlong,   // C_out * C_in
    pub stride_b: c_longlong,   // C_in * n_tiles
    pub stride_c: c_longlong,   // C_out * n_tiles
}

// Usage:
unsafe {
    blas.gemm_strided_batched(cfg, &u_dev, &v_dev, &mut m_dev)?;
}
```

Key points:
- f32 version calls `cublasSgemmStridedBatched` directly — no overhead
- Strided variant is ideal: all 16 matrices are contiguous with uniform stride
- No need for `cublasGemmBatchedEx` (pointer-array variant) — strided is simpler and faster
- cuBLAS handle already used in the codebase (`matmul_cublas` in gemm.rs)

**NOT available in cudarc 0.12**: `cublasGemmBatched` (pointer-array variant).
Only the strided variant is wrapped. This is fine — our data layout is naturally strided.

### 4. F(2x2,3x3) vs F(4x4,3x3) Evaluation

| Property | F(2x2,3x3) | F(4x4,3x3) |
|------------------------|---------------------|---------------------|
| Input tile size | 4×4 = 16 | 6×6 = 36 |
| Output tile size | 2×2 = 4 | 4×4 = 16 |
| Transform count | 16 | 36 |
| Batched GEMMs | 16 | 36 |
| Arithmetic advantage | 2.25× over direct | 4.0× over direct |
| n_tiles (56×56 output) | 784 | 196 |
| n_tiles (14×14 output) | 49 | 16 |
| Numerical stability | Excellent (f32) | Problematic (f32) |
| Transform complexity | Simple (integer+0.5) | Complex (fractions) |

**F(4x4) problems:**
1. **Numerical instability**: The transform matrices involve fractions (1/6, 2/3, etc.)
   that amplify rounding errors in FP32. Published literature (Lavin & Gray 2016) notes
   measurable accuracy degradation for F(4x4) in FP32, especially deep in networks.
2. **Fewer tiles = thinner GEMMs**: For L3 (14×14), F(4x4) gives only 16 tiles (N=16
   in GEMM) vs F(2x2)'s 49. For L4 (7×7), F(4x4) gives only 4 tiles — GEMM degenerates.
3. **36 batched GEMMs** vs 16: more launch overhead, though strided batched mitigates this.
4. **Larger shared memory** for 6×6 tiles in the transform kernels.
5. **More complex implementation**: 36 transform coefficients vs 16, more room for bugs.

**F(4x4) advantages:**
1. **4× arithmetic advantage** (vs 2.25× for F(2x2)): fewer total FLOPs for same conv.
2. **Fewer tiles** = smaller N dimension = less memory for intermediate buffers.
3. **Better for large spatial dims**: at 224×224, n_tiles=3136 (F(4x4)) vs 12544 (F(2x2)).

**Recommendation: Start with F(2x2,3x3).**
- Simpler to implement correctly
- Numerically stable in FP32
- The performance bottleneck is NOT arithmetic — it's the serial per-channel loop
- Converting to batched GEMM provides 10-30× speedup regardless of F(2x2) vs F(4x4)
- F(4x4) can be added later as an optimization for large spatial dimensions

### 5. Memory Layout Design

**Allocations** (for a single conv operation):
1. `U[16, C_out, C_in]` — filter transform (amortizable across forward passes)
2. `V[16, C_in, n_tiles]` — input transform output
3. `M[16, C_out, n_tiles]` — GEMM output (Winograd domain)

**Memory overhead** (ResNet L2: C_out=128, C_in=128, n_tiles=196):
- U: 16 × 128 × 128 × 4B = 1 MB (reusable)
- V: 16 × 128 × 196 × 4B = 1.6 MB
- M: 16 × 128 × 196 × 4B = 1.6 MB
- Total transient: ~3.2 MB (modest)

**Memory layout for cuBLAS**:
- cuBLAS is column-major, but we can use transposed operations
- Store U as `[16, C_out, C_in]` row-major = `[16, C_in, C_out]` column-major
- For `C = U × V`: in cuBLAS column-major, compute `Cᵀ = Vᵀ × Uᵀ`
- Or use `CUBLAS_OP_T` flags as done in existing `matmul_cublas`
- Simplest: `transa=N, transb=N, m=n_tiles, n=C_out, k=C_in` with
  B=[C_in, n_tiles] col-major (= V[n_tiles, C_in] row-major) and
  A=[C_out, C_in] col-major (= U[C_in, C_out] row-major)

### 6. Implementation Plan

1. **Input transform kernel** (new CUDA kernel via NVRTC):
   - Grid: (n_tiles, C_in_blocks, batch_size)
   - Block: (256, 1, 1) — multiple tiles per block for occupancy
   - Each thread: load one 4x4 tile, compute B^T·d·B, scatter 16 values to V[k][ci][tile]
   - Shared memory: optional, but each tile is small (16 floats)

2. **Batched GEMM call**:
   - Create `CudaBlas` handle (reuse existing pattern from `matmul_cublas`)
   - Configure `StridedBatchedConfig` with 16 batches
   - Single cuBLAS call replaces the entire per-channel serial loop

3. **Output transform kernel** (new CUDA kernel via NVRTC):
   - Grid: (n_tiles, C_out_blocks, batch_size)
   - Block: (256, 1, 1)
   - Each thread: read 16 values from M[k][co][tile], compute A^T·M·A, write 2x2 output

4. **Integration**: replace `winograd_conv2d_f2x2` body while keeping the same signature

## Open Questions
1. Should we pre-transform filters once and cache them, or re-transform each forward pass?
   The filter transform is cheap (one kernel launch) but caching saves ~0.1ms for large C.
2. For L4 (n_tiles=16), should we fall back to a different strategy (e.g., direct conv
   or im2col GEMM) since the GEMM N-dimension is very thin?
3. Should the input/output transform kernels be fused with any other operations (e.g., bias add)?

## Confidence
High — the pipeline design is well-established in the literature (Lavin & Gray 2016), and
cudarc 0.12 provides the exact API needed (`gemm_strided_batched` for f32).
