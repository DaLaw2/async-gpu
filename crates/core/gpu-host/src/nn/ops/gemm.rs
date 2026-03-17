//! Matrix multiplication via `gemm_f32` kernel.
//!
//! The kernel expects A=[M,K] row-major, B=[K,N] column-major, output D=[M,N] row-major.
//! Padding and B-transpose are done on GPU via `matrix_pad` and `matrix_transpose` kernels.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::ops::quantize;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Matrix multiplication: C = A * B.
///
/// A: `[M, K]`, B: `[K, N]` → output: `[M, N]`.
///
/// Padding and B column-major transpose happen on GPU (no host round-trip).
pub fn matmul(a: &GpuTensor, b: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    if k != k2 {
        return Err(NnError::ShapeMismatch {
            expected: format!("a.shape[1]={k} == b.shape[0]"),
            actual: format!("b.shape[0]={k2}"),
        });
    }

    // Use V2 kernel (128×64 tile, 4×8 register blocking, row-major B) for
    // matrices large enough. V2 handles bounds checking internally — no
    // padding or transpose needed.
    if m >= 4 && n >= 4 && k >= 4 {
        let mut result = matmul_v2(a, b, registry)?;
        // Record on autograd tape
        if a.requires_grad() || b.requires_grad() {
            if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
                result.set_tensor_id(out_id);
                result.set_requires_grad(true);
                crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                    op: crate::nn::autograd::OpKind::Matmul,
                    inputs: vec![
                        a.tensor_id()
                            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                        b.tensor_id()
                            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    ],
                    output: out_id,
                    saved: vec![
                        a.tensor_id()
                            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                        b.tensor_id()
                            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    ],
                    meta: crate::nn::autograd::OpMeta::Matmul { m, k, n },
                });
            }
        }
        return Ok(result);
    }

    // Fallback: V1 kernel for tiny matrices (M<4 or N<4)
    let m_pad = m.div_ceil(32) * 32;
    let k_pad = k.div_ceil(16) * 16;
    let n_pad = n.div_ceil(16) * 16;

    let dev = registry.device();
    let status = dev.htod_sync_copy(&[0u32])?;

    // === Pad A on GPU: [m, k] → [m_pad, k_pad] ===
    let a_padded = if m == m_pad && k == k_pad {
        // No padding needed — clone the device data directly
        let mut buf = dev.alloc_zeros::<f32>(m * k)?;
        dev.dtod_copy(a.data(), &mut buf)?;
        buf
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m_pad * k_pad)?;
        let f_pad = registry.get("matrix_pad")?;
        let total = (m_pad * k_pad) as u32;
        let cfg = KernelRegistry::config_1d(total);
        unsafe {
            f_pad.launch(
                cfg,
                (
                    a.data(),
                    &mut buf,
                    m as u32,
                    k as u32,
                    m_pad as u32,
                    k_pad as u32,
                    &status,
                ),
            )?;
        }
        buf
    };

    // === Transpose B on GPU: [k, n] row-major → [n, k] row-major, then pad to [n_pad, k_pad] ===
    // The GEMM kernel expects B in column-major [K_pad, N_pad], which is the same memory
    // layout as row-major [N_pad, K_pad]. So we need: transpose B → [n, k], then pad → [n_pad, k_pad].
    let b_col_major = {
        // Step 1: Transpose B[k, n] → B_T[n, k] on GPU
        let mut b_t = dev.alloc_zeros::<f32>(n * k)?;
        let f_transpose = registry.get("matrix_transpose")?;
        let total_t = (k * n) as u32;
        let cfg_t = KernelRegistry::config_1d(total_t);
        unsafe {
            f_transpose.launch(cfg_t, (b.data(), &mut b_t, k as u32, n as u32, &status))?;
        }

        // Step 2: Pad B_T[n, k] → [n_pad, k_pad]
        if n == n_pad && k == k_pad {
            b_t
        } else {
            let mut buf = dev.alloc_zeros::<f32>(n_pad * k_pad)?;
            let f_pad = registry.get("matrix_pad")?;
            let total_p = (n_pad * k_pad) as u32;
            let cfg_p = KernelRegistry::config_1d(total_p);
            unsafe {
                f_pad.launch(
                    cfg_p,
                    (
                        &b_t,
                        &mut buf,
                        n as u32,
                        k as u32,
                        n_pad as u32,
                        k_pad as u32,
                        &status,
                    ),
                )?;
            }
            buf
        }
    };

    // === GEMM: D[m_pad, n_pad] = A_pad[m_pad, k_pad] × B_cm[k_pad, n_pad] ===
    let mut d_dev = dev.alloc_zeros::<f32>(m_pad * n_pad)?;
    let f_gemm = registry.get("gemm_f32")?;
    let gemm_cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((m_pad as u32) / 32, (n_pad as u32) / 16, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 3072,
    };
    unsafe {
        f_gemm.launch(
            gemm_cfg,
            (
                &a_padded,
                &b_col_major,
                &mut d_dev,
                k_pad as u32,
                n_pad as u32,
                &status,
            ),
        )?;
    }

    // === Extract unpadded [m, n] from [m_pad, n_pad] on GPU ===
    let output_dev = if m == m_pad && n == n_pad {
        d_dev
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m * n)?;
        let f_unpad = registry.get("matrix_unpad")?;
        let total_out = (m * n) as u32;
        let cfg_out = KernelRegistry::config_1d(total_out);
        unsafe {
            f_unpad.launch(
                cfg_out,
                (&d_dev, &mut buf, m as u32, n as u32, n_pad as u32, &status),
            )?;
        }
        buf
    };

    let mut output = GpuTensor::from_data(output_dev, &[m, n], Arc::clone(dev));

    // Record on autograd tape
    if a.requires_grad() || b.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::Matmul,
                inputs: vec![
                    a.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    b.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                ],
                output: out_id,
                saved: vec![
                    a.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    b.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                ],
                meta: crate::nn::autograd::OpMeta::Matmul { m, k, n },
            });
        }
    }

    Ok(output)
}

/// High-performance matrix multiplication using `gemm_f32_v2` kernel.
///
/// A: `[M, K]`, B: `[K, N]` → output: `[M, N]`.
///
/// Both A and B are row-major. No transpose or padding overhead.
/// Uses 128×128 tile with 8×8 register blocking, 256 threads per block.
pub fn matmul_v2(
    a: &GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    if k != k2 {
        return Err(NnError::ShapeMismatch {
            expected: format!("a.shape[1]={k} == b.shape[0]"),
            actual: format!("b.shape[0]={k2}"),
        });
    }

    let dev = registry.device();

    // Use cuBLAS for small-grid cases where our kernel underutilizes the GPU
    // Use cuBLAS for small M (inference shapes) — cuBLAS has specialized kernels
    #[cfg(feature = "cublas")]
    if m <= 256 {
        return matmul_cublas(a, b, m, k, n, dev);
    }

    // Use NVRTC V4 (float4 loads + fmaf) for large matrices
    #[cfg(feature = "cublas")]
    if m >= 512 && n >= 512 && k >= 256 {
        return matmul_v4(a, b, m, k, n, dev);
    }

    let status = dev.htod_sync_copy(&[0u32])?;

    let mut d_dev = dev.alloc_zeros::<f32>(m * n)?;

    // V3 (128×128, 8×8) for large grids, V2 (128×64, 4×8) for smaller
    let v3_blocks = m.div_ceil(128) * n.div_ceil(128);
    let use_v3 = m >= 128 && n >= 128 && v3_blocks >= 16;
    let f_gemm = if use_v3 {
        registry.get("gemm_f32_v3")?
    } else {
        registry.get("gemm_f32_v2")?
    };

    let gemm_cfg = if use_v3 {
        cudarc::driver::LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 16640,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, n.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 12544,
        }
    };
    unsafe {
        f_gemm.launch(
            gemm_cfg,
            (
                a.data(),
                b.data(),
                &mut d_dev,
                m as u32,
                n as u32,
                k as u32,
                &status,
            ),
        )?;
    }

    Ok(GpuTensor::from_data(d_dev, &[m, n], Arc::clone(dev)))
}

/// cuBLAS-based matmul for small-grid cases where custom kernels underutilize GPU.
///
/// A: [M, K] row-major, B: [K, N] row-major → output: [M, N] row-major.
/// cuBLAS expects column-major, so we compute C^T = B^T × A^T to get row-major result.
#[cfg(feature = "cublas")]
fn matmul_cublas(
    a: &GpuTensor,
    b: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};

    let blas = CudaBlas::new(Arc::clone(dev)).map_err(|e| NnError::ShapeMismatch {
        expected: "cuBLAS init".to_string(),
        actual: format!("{e:?}"),
    })?;
    let mut c_dev = dev.alloc_zeros::<f32>(m * n).map_err(NnError::Cuda)?;

    // Row-major C = A × B is equivalent to column-major C^T = B^T × A^T
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };

    unsafe {
        blas.gemm(cfg, b.data(), a.data(), &mut c_dev)
            .map_err(|e| NnError::ShapeMismatch {
                expected: "cuBLAS gemm".to_string(),
                actual: format!("{e:?}"),
            })?;
    }

    Ok(GpuTensor::from_data(c_dev, &[m, n], Arc::clone(dev)))
}

/// NVRTC-compiled GEMM V4: 128×128, 8×8 register blocking, float4 loads.
///
/// Uses CUDA C compiled at runtime for proper float4 vectorization.
/// This avoids Rust→PTX compiler limitations with vectorized loads.
#[cfg(feature = "cublas")]
pub fn matmul_v4(
    a: &GpuTensor,
    b: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx;

    static GEMM_V4_SRC: &str = r#"
// GEMM V4: 128×128 tile, BK=8, 8×8 register blocking, float4 loads
// A: [M, K] row-major, B: [K, N] row-major, D: [M, N] row-major
// 256 threads = 16×16 thread grid, each thread computes 8×8 outputs

#define BM 128
#define BN 128
#define BK 8
#define A_STRIDE (BM + 4)  // 132 — padding

extern "C" __global__ void gemm_f32_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ D,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int tid = threadIdx.x;
    unsigned int bm = blockIdx.x;
    unsigned int bn = blockIdx.y;

    // Shared memory: double-buffered A[BK][A_STRIDE] + B[BK][BN]
    __shared__ float smem[2 * (BK * A_STRIDE + BK * BN)];
    const unsigned int STAGE = BK * A_STRIDE + BK * BN;

    // Thread mapping: 16×16 grid, each 8×8 output
    unsigned int tr = tid / 16;  // 0..15
    unsigned int tc = tid % 16;  // 0..15

    // 64 accumulators
    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            c[i][j] = 0.0f;

    unsigned int a_base = bm * BM;
    unsigned int b_base = bn * BN;
    unsigned int k_tiles = (K + BK - 1) / BK;

    // === Tile loading helper (inlined) ===
    // Load A: 128×8 = 1024 elems, 256 threads → 4 each
    // Load B: 8×128 = 1024 elems, 256 threads → 4 each
    #define LOAD_TILE(buf, k_start) do { \
        float* a_smem = smem + (buf) * STAGE; \
        float* b_smem = smem + (buf) * STAGE + BK * A_STRIDE; \
        /* Load A with float4 where possible (4 consecutive K elements per row) */ \
        { \
            unsigned int flat = tid * 4; \
            unsigned int ar = flat / BK; \
            unsigned int ak = flat % BK; \
            unsigned int gr = a_base + ar; \
            unsigned int gk = (k_start) + ak; \
            /* Load 4 scalar elements (scattered to different k-rows in smem) */ \
            for (int ii = 0; ii < 4; ii++) { \
                unsigned int cur_flat = tid * 4 + ii; \
                unsigned int cur_ar = cur_flat / BK; \
                unsigned int cur_ak = cur_flat % BK; \
                unsigned int cur_gr = a_base + cur_ar; \
                unsigned int cur_gk = (k_start) + cur_ak; \
                float val = (cur_gr < M && cur_gk < K) ? A[cur_gr * K + cur_gk] : 0.0f; \
                a_smem[cur_ak * A_STRIDE + cur_ar] = val; \
            } \
        } \
        /* Load B with float4 (4 consecutive N elements) */ \
        { \
            unsigned int flat = tid * 4; \
            unsigned int bk = flat / BN; \
            unsigned int bc = flat % BN; \
            unsigned int gk = (k_start) + bk; \
            unsigned int gc = b_base + bc; \
            if (gk < K && gc + 3 < N) { \
                /* float4 load — 128-bit coalesced */ \
                float4 bv = *reinterpret_cast<const float4*>(&B[gk * N + gc]); \
                b_smem[bk * BN + bc] = bv.x; \
                b_smem[bk * BN + bc + 1] = bv.y; \
                b_smem[bk * BN + bc + 2] = bv.z; \
                b_smem[bk * BN + bc + 3] = bv.w; \
            } else { \
                for (int jj = 0; jj < 4; jj++) { \
                    unsigned int cur_flat = tid * 4 + jj; \
                    unsigned int cur_bk = cur_flat / BN; \
                    unsigned int cur_bc = cur_flat % BN; \
                    unsigned int cur_gk = (k_start) + cur_bk; \
                    unsigned int cur_gc = b_base + cur_bc; \
                    float val = (cur_gk < K && cur_gc < N) ? B[cur_gk * N + cur_gc] : 0.0f; \
                    b_smem[cur_bk * BN + cur_bc] = val; \
                } \
            } \
        } \
    } while(0)

    // Load first tile
    unsigned int buf = 0;
    LOAD_TILE(0, 0);
    __syncthreads();

    for (unsigned int t = 0; t < k_tiles; t++) {
        // Prefetch next tile
        if (t + 1 < k_tiles) {
            LOAD_TILE(1 - buf, (t + 1) * BK);
        }

        float* a_s = smem + buf * STAGE;
        float* b_s = smem + buf * STAGE + BK * A_STRIDE;

        unsigned int arb = tr * 8;
        unsigned int bcb = tc * 8;

        // Inner K loop with register blocking
        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
            if (t * BK + kk >= K) break;

            // Load A fragment: 8 values
            float a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                a[i] = a_s[kk * A_STRIDE + arb + i];

            // Load B fragment: 8 values
            float b[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                b[j] = b_s[kk * BN + bcb + j];

            // 8×8 outer product
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    c[i][j] = fmaf(a[i], b[j], c[i][j]);
        }

        __syncthreads();
        buf = 1 - buf;
    }

    // Write output
    unsigned int or_ = a_base + tr * 8;
    unsigned int oc_ = b_base + tc * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (or_ + i < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (oc_ + j < N)
                    D[(or_ + i) * N + oc_ + j] = c[i][j];
            }
        }
    }
}
"#;

    // Cache compiled kernel
    use std::sync::OnceLock;
    static COMPILED: OnceLock<bool> = OnceLock::new();
    COMPILED.get_or_init(|| {
        let ptx = compile_ptx(GEMM_V4_SRC).expect("NVRTC GEMM V4 compile failed");
        dev.load_ptx(ptx, "gemm_v4", &["gemm_f32_v4"])
            .expect("GEMM V4 PTX load failed");
        true
    });

    let func = dev
        .get_func("gemm_v4", "gemm_f32_v4")
        .ok_or(NnError::KernelNotFound {
            name: "gemm_f32_v4",
        })?;

    let mut d_dev = dev.alloc_zeros::<f32>(m * n).map_err(NnError::Cuda)?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 2 * (8 * 132 + 8 * 128) * 4,
    };

    unsafe {
        func.launch(
            config,
            (a.data(), b.data(), &mut d_dev, m as u32, n as u32, k as u32),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(GpuTensor::from_data(
        d_dev,
        &[m, n],
        std::sync::Arc::clone(dev),
    ))
}

/// Activation type for fused GEMM.
#[derive(Copy, Clone, Debug)]
pub enum FusedActivation {
    /// GELU activation (approximation: x * sigmoid(1.702 * x)).
    Gelu,
    /// ReLU activation (max(0, x)).
    Relu,
}

/// Fused matrix multiplication + bias + activation: `activation(A × B + bias)`.
///
/// Eliminates 2 kernel launches vs separate matmul + bias_add + activation.
/// A: `[M, K]`, B: `[K, N]`, bias: `[N]` → output: `[M, N]`.
pub fn matmul_fused(
    a: &GpuTensor,
    b: &GpuTensor,
    bias: &GpuTensor,
    activation: FusedActivation,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    if a.shape()[1] != b.shape()[0] {
        return Err(NnError::ShapeMismatch {
            expected: format!("a.shape[1]={} == b.shape[0]", a.shape()[1]),
            actual: format!("b.shape[0]={}", b.shape()[0]),
        });
    }
    if bias.ndim() != 1 || bias.shape()[0] != n {
        return Err(NnError::ShapeMismatch {
            expected: format!("bias.shape=[{n}]"),
            actual: format!("bias.shape={:?}", bias.shape()),
        });
    }

    let m_pad = m.div_ceil(32) * 32;
    let k_pad = k.div_ceil(16) * 16;
    let n_pad = n.div_ceil(16) * 16;

    let dev = registry.device();
    let status = dev.htod_sync_copy(&[0u32])?;

    // Pad A
    let a_padded = if m == m_pad && k == k_pad {
        let mut buf = dev.alloc_zeros::<f32>(m * k)?;
        dev.dtod_copy(a.data(), &mut buf)?;
        buf
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m_pad * k_pad)?;
        let f_pad = registry.get("matrix_pad")?;
        unsafe {
            f_pad.launch(
                KernelRegistry::config_1d((m_pad * k_pad) as u32),
                (
                    a.data(),
                    &mut buf,
                    m as u32,
                    k as u32,
                    m_pad as u32,
                    k_pad as u32,
                    &status,
                ),
            )?;
        }
        buf
    };

    // Transpose + pad B
    let b_col_major = {
        let mut b_t = dev.alloc_zeros::<f32>(n * k)?;
        let f_transpose = registry.get("matrix_transpose")?;
        unsafe {
            f_transpose.launch(
                KernelRegistry::config_1d((k * n) as u32),
                (b.data(), &mut b_t, k as u32, n as u32, &status),
            )?;
        }
        if n == n_pad && k == k_pad {
            b_t
        } else {
            let mut buf = dev.alloc_zeros::<f32>(n_pad * k_pad)?;
            let f_pad = registry.get("matrix_pad")?;
            unsafe {
                f_pad.launch(
                    KernelRegistry::config_1d((n_pad * k_pad) as u32),
                    (
                        &b_t,
                        &mut buf,
                        n as u32,
                        k as u32,
                        n_pad as u32,
                        k_pad as u32,
                        &status,
                    ),
                )?;
            }
            buf
        }
    };

    // Pad bias: [n] → [n_pad]
    let bias_padded = if n == n_pad {
        let mut buf = dev.alloc_zeros::<f32>(n)?;
        dev.dtod_copy(bias.data(), &mut buf)?;
        buf
    } else {
        let mut buf = dev.alloc_zeros::<f32>(n_pad)?;
        // Copy first n elements (rest are zero-padded)
        let f_pad = registry.get("matrix_pad")?;
        unsafe {
            f_pad.launch(
                KernelRegistry::config_1d(n_pad as u32),
                (
                    bias.data(),
                    &mut buf,
                    1u32,
                    n as u32,
                    1u32,
                    n_pad as u32,
                    &status,
                ),
            )?;
        }
        buf
    };

    // Fused GEMM + bias + activation
    let mut d_dev = dev.alloc_zeros::<f32>(m_pad * n_pad)?;
    let kernel_name = match activation {
        FusedActivation::Gelu => "gemm_bias_gelu",
        FusedActivation::Relu => "gemm_bias_relu",
    };
    let f_gemm = registry.get(kernel_name)?;
    let gemm_cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((m_pad as u32) / 32, (n_pad as u32) / 16, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 3072,
    };
    unsafe {
        f_gemm.launch(
            gemm_cfg,
            (
                &a_padded,
                &b_col_major,
                &bias_padded,
                &mut d_dev,
                k_pad as u32,
                n_pad as u32,
                &status,
            ),
        )?;
    }

    // Unpad
    let output_dev = if m == m_pad && n == n_pad {
        d_dev
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m * n)?;
        let f_unpad = registry.get("matrix_unpad")?;
        unsafe {
            f_unpad.launch(
                KernelRegistry::config_1d((m * n) as u32),
                (&d_dev, &mut buf, m as u32, n as u32, n_pad as u32, &status),
            )?;
        }
        buf
    };

    Ok(GpuTensor::from_data(output_dev, &[m, n], Arc::clone(dev)))
}

/// Matrix multiplication with pre-computed column-major padded B.
///
/// A: `[M, K]`, b_prepadded: pre-transposed+padded `[N_pad, K_pad]` col-major.
/// Skips B transpose and B pad for ~2 fewer kernel launches per call.
#[allow(clippy::too_many_arguments)]
pub fn matmul_prepadded_b(
    a: &GpuTensor,
    b_prepadded: &cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    k_pad: usize,
    n_pad: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let m_pad = m.div_ceil(32) * 32;
    let dev = registry.device();
    let status = dev.htod_sync_copy(&[0u32])?;

    // Pad A on GPU
    let a_padded = if m == m_pad && k == k_pad {
        let mut buf = dev.alloc_zeros::<f32>(m * k)?;
        dev.dtod_copy(a.data(), &mut buf)?;
        buf
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m_pad * k_pad)?;
        let f_pad = registry.get("matrix_pad")?;
        let cfg = KernelRegistry::config_1d((m_pad * k_pad) as u32);
        unsafe {
            f_pad.launch(
                cfg,
                (
                    a.data(),
                    &mut buf,
                    m as u32,
                    k as u32,
                    m_pad as u32,
                    k_pad as u32,
                    &status,
                ),
            )?;
        }
        buf
    };

    // GEMM
    let mut d_dev = dev.alloc_zeros::<f32>(m_pad * n_pad)?;
    let f_gemm = registry.get("gemm_f32")?;
    let gemm_cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((m_pad as u32) / 32, (n_pad as u32) / 16, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 3072,
    };
    unsafe {
        f_gemm.launch(
            gemm_cfg,
            (
                &a_padded,
                b_prepadded,
                &mut d_dev,
                k_pad as u32,
                n_pad as u32,
                &status,
            ),
        )?;
    }

    // Extract unpadded
    let output_dev = if m == m_pad && n == n_pad {
        d_dev
    } else {
        let mut buf = dev.alloc_zeros::<f32>(m * n)?;
        let f_unpad = registry.get("matrix_unpad")?;
        let cfg = KernelRegistry::config_1d((m * n) as u32);
        unsafe {
            f_unpad.launch(
                cfg,
                (&d_dev, &mut buf, m as u32, n as u32, n_pad as u32, &status),
            )?;
        }
        buf
    };

    Ok(GpuTensor::from_data(output_dev, &[m, n], Arc::clone(dev)))
}

/// INT8 matrix multiplication via dp4a: C = A_i8 × B_i8, dequantized to f32.
///
/// Quantizes A and B from f32 to INT8 (per-tensor for A, per-column for B),
/// packs into u32 (4 INT8 per u32), runs dp4a GEMM, then dequantizes.
///
/// A: `[M, K]`, B: `[K, N]` → output: `[M, N]` f32.
/// K must be divisible by 4.
pub fn int8_matmul(
    a: &GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    if !k.is_multiple_of(4) {
        return Err(NnError::ShapeMismatch {
            expected: "K divisible by 4 for INT8 packing".to_string(),
            actual: format!("K={k}"),
        });
    }
    let k_div4 = k / 4;

    // Download to host for quantization
    let a_host = a.to_host()?;
    let b_host = b.to_host()?;

    // Quantize A: per-tensor symmetric, then pack rows into u32 [M, K/4]
    let (a_q, a_scale) = quantize::quantize_int8_per_tensor(&a_host);
    let mut a_packed = vec![0u32; m * k_div4];
    for row in 0..m {
        let row_slice = &a_q[row * k..(row + 1) * k];
        let row_packed = quantize::pack_int8_to_u32(row_slice);
        a_packed[row * k_div4..(row + 1) * k_div4].copy_from_slice(&row_packed);
    }

    // Quantize B: per-column symmetric, stored column-major [N, K/4]
    let mut b_scales = vec![0.0f32; n];
    let mut b_packed = vec![0u32; n * k_div4];
    for col in 0..n {
        // Extract column
        let col_data: Vec<f32> = (0..k).map(|row| b_host[row * n + col]).collect();
        let (col_q, col_scale) = quantize::quantize_int8_per_column(&col_data);
        b_scales[col] = col_scale;

        // Pack column into u32
        let col_packed = quantize::pack_int8_to_u32(&col_q);
        for j in 0..k_div4 {
            b_packed[col * k_div4 + j] = col_packed[j];
        }
    }

    let dev = registry.device();

    // Upload packed data
    let a_dev = dev.htod_copy(a_packed)?;
    let b_dev = dev.htod_copy(b_packed)?;
    let c_dev = dev.alloc_zeros::<i32>(m * n)?;
    let status = dev.htod_sync_copy(&[0u32])?;

    // Launch INT8 GEMM
    let func = registry.get("int8_gemm_dp4a")?;
    let total = (m * n) as u32;
    let config = KernelRegistry::config_1d(total);
    unsafe {
        func.launch(
            config,
            (
                &a_dev,
                &b_dev,
                &c_dev,
                m as u32,
                n as u32,
                k_div4 as u32,
                &status,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    // Dequantize: f32_out = int32_out * a_scale * b_scale[col]
    let b_scales_dev = dev.htod_copy(b_scales)?;
    let out_dev = dev.alloc_zeros::<f32>(m * n)?;
    let status2 = dev.htod_sync_copy(&[0u32])?;

    let func_deq = registry.get("int8_dequantize")?;
    let config_deq = KernelRegistry::config_1d(total);
    unsafe {
        func_deq
            .launch(
                config_deq,
                (
                    &c_dev,
                    &out_dev,
                    a_scale,
                    &b_scales_dev,
                    total,
                    n as u32,
                    &status2,
                ),
            )
            .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(GpuTensor::from_data(out_dev, &[m, n], Arc::clone(dev)))
}

/// INT4 (W4A16) matrix multiplication: C = A_f32 × dequant(W_int4).
///
/// Quantizes B from f32 to INT4 per-group, packs into u32 (8 values per u32),
/// runs dequantize-on-the-fly GEMM kernel.
///
/// A: `[M, K]`, B: `[K, N]` → output: `[M, N]` f32.
/// K must be divisible by 8. group_size default: 128.
pub fn int4_matmul(
    a: &GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    if !k.is_multiple_of(8) {
        return Err(NnError::ShapeMismatch {
            expected: "K divisible by 8 for INT4 packing".to_string(),
            actual: format!("K={k}"),
        });
    }

    /// Default group size for INT4 per-group quantization.
    const INT4_GROUP_SIZE: usize = 128;
    let group_size = INT4_GROUP_SIZE;
    let n_groups = k.div_ceil(group_size);
    let k_packed = k / 8;

    // Download B for quantization
    let b_host = b.to_host()?;

    // Quantize B to INT4 per-group: packed [K/8, N] + scales [n_groups, N]
    // We quantize each column independently, then interleave into row-major packed layout.
    let mut packed = vec![0u32; k_packed * n];
    let mut scales = vec![0.0f32; n_groups * n];

    for col in 0..n {
        // Extract column
        let col_data: Vec<f32> = (0..k).map(|row| b_host[row * n + col]).collect();
        let (col_packed, col_scales) = quantize::quantize_int4_per_group(&col_data, group_size);

        // Scatter column's packed u32 values into row-major layout [K/8, N]
        for j in 0..k_packed {
            packed[j * n + col] = col_packed[j];
        }
        // Scatter column's scales into [n_groups, N]
        for g in 0..n_groups {
            scales[g * n + col] = col_scales[g];
        }
    }

    let dev = registry.device();
    let a_dev = a.data();
    let w_dev = dev.htod_copy(packed)?;
    let scales_dev = dev.htod_copy(scales)?;
    let c_dev = dev.alloc_zeros::<f32>(m * n)?;
    let status = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("int4_gemm_w4a16")?;
    let total = (m * n) as u32;
    let config = KernelRegistry::config_1d(total);
    unsafe {
        func.launch(
            config,
            (
                a_dev,
                &w_dev,
                &scales_dev,
                &c_dev,
                m as u32,
                n as u32,
                k as u32,
                group_size as u32,
                &status,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(GpuTensor::from_data(c_dev, &[m, n], Arc::clone(dev)))
}
