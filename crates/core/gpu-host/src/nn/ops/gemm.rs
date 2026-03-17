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
    let status = dev.htod_sync_copy(&[0u32])?;

    let mut d_dev = dev.alloc_zeros::<f32>(m * n)?;

    // Use V3 (128×128, 8×8) for large matrices, V2 (128×64, 4×8) for smaller
    let use_v3 = m >= 128 && n >= 128;
    let f_gemm = if use_v3 {
        registry.get("gemm_f32_v3")?
    } else {
        registry.get("gemm_f32_v2")?
    };

    let gemm_cfg = if use_v3 {
        cudarc::driver::LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 16640, // 2 * (8*132 + 8*128) * 4
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, n.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 12544, // 2 * (8*132 + 8*64) * 4
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
