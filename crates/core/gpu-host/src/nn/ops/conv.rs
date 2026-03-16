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
    // Support both [C, H, W] and [N, C, H, W]
    if input.ndim() == 4 {
        return conv2d_batched(input, weight, bias, stride, padding, registry);
    }
    if input.ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "3D [C_in, H, W] or 4D [N, C_in, H, W]".to_string(),
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

    // 2. GEMM via nn::ops::matmul (handles all padding correctly)
    // im2col outputs [spatial, K] row-major (spatial = h_out*w_out, K = c_in*kh*kw)
    // We need: Output[C_out, spatial] = Weight[C_out, K] × Col[K, spatial]
    // So transpose the im2col output from [spatial, K] to [K, spatial]
    let w_host = weight.to_host()?;
    let col_raw = dev.dtoh_sync_copy(&col_dev)?;

    // Transpose col from [spatial, K] to [K, spatial]
    let mut col_t = vec![0.0f32; col_h * col_w];
    for s in 0..col_w {
        for k in 0..col_h {
            col_t[k * col_w + s] = col_raw[s * col_h + k];
        }
    }

    let w_tensor = GpuTensor::from_host(&w_host, &[c_out, col_h], dev)?;
    let col_tensor = GpuTensor::from_host(&col_t, &[col_h, col_w], dev)?;
    let gemm_out = super::matmul(&w_tensor, &col_tensor, registry)?;

    // 3. Result is [C_out, col_w] = [C_out, h_out * w_out] — already in CHW layout
    let mut result = gemm_out.to_host()?;

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

    let mut output = GpuTensor::from_host(&result, &[c_out, h_out, w_out], dev)?;

    // Record on autograd tape
    if input.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            let in_id = input
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            let w_id = weight
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::Conv2d,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id, w_id],
                meta: crate::nn::autograd::OpMeta::Conv2d {
                    c_in,
                    c_out,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                },
            });
        }
    }

    Ok(output)
}

/// Batched conv2d: input [N, C_in, H, W] → output [N, C_out, H_out, W_out].
///
/// Runs im2col per sample, concatenates columns, then ONE matmul.
#[allow(clippy::too_many_arguments)]
fn conv2d_batched(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let col_h = c_in * kh * kw;
    let col_w = h_out * w_out;

    let dev = registry.device();
    let input_host = input.to_host()?;

    // im2col per sample, collect columns
    let mut all_cols_t = vec![0.0f32; batch * col_h * col_w];
    let status_dev = dev.htod_sync_copy(&[0u32])?;

    for b in 0..batch {
        let sample = &input_host[b * c_in * h * w..(b + 1) * c_in * h * w];
        let sample_gpu = GpuTensor::from_host(sample, &[c_in, h, w], dev)?;

        let mut col_dev = dev.alloc_zeros::<f32>(col_h * col_w)?;
        let f_im2col = registry.get("im2col")?;
        let total = (col_h * col_w) as u32;
        let config = KernelRegistry::config_1d(total);
        unsafe {
            cudarc::driver::LaunchAsync::launch(
                f_im2col,
                config,
                (
                    sample_gpu.data(),
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

        // Transpose from [spatial, K] to [K, spatial]
        let col_raw = dev.dtoh_sync_copy(&col_dev)?;
        let offset = b * col_h * col_w;
        for s in 0..col_w {
            for k in 0..col_h {
                all_cols_t[offset + k * col_w + s] = col_raw[s * col_h + k];
            }
        }
    }

    // ONE big matmul: W[c_out, K] × BigCol[K, batch*spatial]
    let big_col_w = batch * col_w;
    let w_host = weight.to_host()?;
    let w_tensor = GpuTensor::from_host(&w_host, &[c_out, col_h], dev)?;
    let big_col = GpuTensor::from_host(&all_cols_t, &[col_h, big_col_w], dev)?;
    let gemm_out = super::matmul(&w_tensor, &big_col, registry)?;

    // Result [c_out, batch*spatial] → [batch, c_out, h_out, w_out]
    let mut result = gemm_out.to_host()?;

    // Add bias
    if let Some(bias_tensor) = bias {
        let bias_host = bias_tensor.to_host()?;
        for b in 0..batch {
            for ch in 0..c_out {
                let b_val = bias_host[ch];
                for i in 0..col_w {
                    result[b * c_out * col_w + ch * col_w + i] += b_val;
                }
            }
        }
    }

    // Reshape from [c_out, batch*spatial] to [batch, c_out, h_out, w_out]
    // The GEMM output is [c_out, batch*col_w]. We need to rearrange.
    let mut output_data = vec![0.0f32; batch * c_out * h_out * w_out];
    for b in 0..batch {
        for ch in 0..c_out {
            for s in 0..col_w {
                // GEMM result: result[ch * big_col_w + b * col_w + s]
                output_data[b * c_out * col_w + ch * col_w + s] =
                    result[ch * big_col_w + b * col_w + s];
            }
        }
    }

    GpuTensor::from_host(&output_data, &[batch, c_out, h_out, w_out], dev)
}
