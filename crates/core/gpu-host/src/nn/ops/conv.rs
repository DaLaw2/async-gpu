//! Convolution via im2col + GEMM pipeline.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// 2D convolution via im2col → GEMM pipeline.
///
/// Input: `[C_in, H, W]`, weight: `[C_out, C_in, kH, kW]`, optional bias: `[C_out]`.
/// Output: `[C_out, H_out, W_out]`.
///
/// Uses the `im2col` kernel to unfold input patches, then `gemm_f32` for the
/// matrix multiplication.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if input.ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "3D input [C_in, H, W]".to_string(),
            actual: format!("ndim={}", input.ndim()),
        });
    }
    if weight.ndim() != 4 {
        return Err(NnError::ShapeMismatch {
            expected: "4D weight [C_out, C_in, kH, kW]".to_string(),
            actual: format!("ndim={}", weight.ndim()),
        });
    }

    let c_in = input.shape()[0];
    let h = input.shape()[1];
    let w = input.shape()[2];
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let col_h = c_in * kh * kw; // K dimension
    let col_w = h_out * w_out; // spatial output positions (M dimension for GEMM)

    let dev = registry.device();

    // 1. im2col: input [C_in, H, W] → col [col_h, col_w]
    let mut col_dev = dev.alloc_zeros::<f32>(col_h * col_w)?;
    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let f_im2col = registry.get("im2col")?;
    let im2col_total = (col_h * col_w) as u32;
    let config_im2col = KernelRegistry::config_1d(im2col_total);
    unsafe {
        f_im2col
            .launch(
                config_im2col,
                (
                    input.data(),
                    &mut col_dev,
                    c_in as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    stride as u32,
                    padding as u32,
                    h_out as u32,
                    w_out as u32,
                    &status_dev,
                ),
            )
            .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    // 2. GEMM: weight_cm [col_h, C_out] col-major × col [col_h, col_w] → result [col_w, C_out]
    // Actually: gemm_f32 computes D = A * B where A=[M,K] row-major, B=[K,N] col-major
    // We want: output = weight_reshaped * col_matrix
    //   weight is [C_out, col_h] row-major → use as A with M=C_out, K=col_h
    //   col is [col_h, col_w] row-major → need col-major B with K=col_h, N=col_w
    let m = c_out;
    let k = col_h;
    let n = col_w;

    // Pad to tile boundaries
    let m_pad = m.div_ceil(32) * 32;
    let k_pad = k.div_ceil(16) * 16;
    let n_pad = n.div_ceil(16) * 16;

    // Prepare A (weight reshaped to [C_out, col_h] = [C_out, C_in*kH*kW]) padded
    let w_host = weight.to_host()?;
    let mut a_padded = vec![0.0f32; m_pad * k_pad];
    for r in 0..m {
        for c in 0..k {
            a_padded[r * k_pad + c] = w_host[r * k + c];
        }
    }
    let a_dev = dev.htod_sync_copy(&a_padded)?;

    // Prepare B (col matrix) in column-major, padded
    let col_host = dev.dtoh_sync_copy(&col_dev)?;
    let mut b_cm = vec![0.0f32; k_pad * n_pad];
    for r in 0..k {
        for c in 0..n {
            b_cm[c * k_pad + r] = col_host[r * n + c];
        }
    }
    let b_dev = dev.htod_sync_copy(&b_cm)?;

    let mut d_dev = dev.alloc_zeros::<f32>(m_pad * n_pad)?;

    let f_gemm = registry.get("gemm_f32")?;
    let gemm_config = cudarc::driver::LaunchConfig {
        grid_dim: (m_pad as u32 / 32, n_pad as u32 / 16, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 3072,
    };
    let status_dev2 = dev.htod_sync_copy(&[0u32])?;
    unsafe {
        f_gemm
            .launch(
                gemm_config,
                (
                    &a_dev,
                    &b_dev,
                    &mut d_dev,
                    k_pad as u32,
                    n_pad as u32,
                    &status_dev2,
                ),
            )
            .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    // 3. Extract [C_out, col_w] from padded output and reshape to [C_out, H_out, W_out]
    let d_host = dev.dtoh_sync_copy(&d_dev)?;
    let mut result = vec![0.0f32; c_out * h_out * w_out];
    for r in 0..c_out {
        for c in 0..col_w {
            result[r * col_w + c] = d_host[r * n_pad + c];
        }
    }

    // 4. Add bias if present
    if let Some(bias_tensor) = bias {
        let bias_host = bias_tensor.to_host()?;
        for ch in 0..c_out {
            let b = bias_host[ch];
            for i in 0..(h_out * w_out) {
                result[ch * h_out * w_out + i] += b;
            }
        }
    }

    GpuTensor::from_host(&result, &[c_out, h_out, w_out], dev)
}
