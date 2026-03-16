//! Matrix multiplication via `gemm_f32` kernel.
//!
//! The kernel expects A=[M,K] row-major, B=[K,N] column-major, output D=[M,N] row-major.
//! Padding and B-transpose are done on GPU via `matrix_pad` and `matrix_transpose` kernels.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
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
    if k % 4 != 0 {
        return Err(NnError::ShapeMismatch {
            expected: "K divisible by 4 for INT8 packing".to_string(),
            actual: format!("K={k}"),
        });
    }
    let k_div4 = k / 4;

    // Download to host for quantization
    let a_host = a.to_host()?;
    let b_host = b.to_host()?;

    // Quantize A: per-tensor symmetric
    let a_max = a_host.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
    let a_scale = if a_max < 1e-12 { 1.0 } else { a_max / 127.0 };

    // Pack A into u32: [M, K/4]
    let mut a_packed = vec![0u32; m * k_div4];
    for row in 0..m {
        for j in 0..k_div4 {
            let mut packed = 0u32;
            for b_idx in 0..4 {
                let val = a_host[row * k + j * 4 + b_idx];
                let q = (val / a_scale).round().clamp(-128.0, 127.0) as i8;
                packed |= (q as u8 as u32) << (b_idx * 8);
            }
            a_packed[row * k_div4 + j] = packed;
        }
    }

    // Quantize B: per-column symmetric, stored column-major [N, K/4]
    let mut b_scales = vec![0.0f32; n];
    for col in 0..n {
        let mut col_max = 0.0f32;
        for row in 0..k {
            col_max = col_max.max(b_host[row * n + col].abs());
        }
        b_scales[col] = if col_max < 1e-12 {
            1.0
        } else {
            col_max / 127.0
        };
    }

    // Pack B column-major: [N, K/4]
    let mut b_packed = vec![0u32; n * k_div4];
    for col in 0..n {
        let inv_scale = 1.0 / b_scales[col];
        for j in 0..k_div4 {
            let mut packed = 0u32;
            for b_idx in 0..4 {
                let val = b_host[(j * 4 + b_idx) * n + col];
                let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                packed |= (q as u8 as u32) << (b_idx * 8);
            }
            b_packed[col * k_div4 + j] = packed;
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
