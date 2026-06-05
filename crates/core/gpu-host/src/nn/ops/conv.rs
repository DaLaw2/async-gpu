//! Convolution via im2col + GEMM pipeline, with Winograd F(2×2, 3×3) fast path
//! and direct convolution kernels for 1×1, 5×5, 7×7 (and other non-3×3 sizes).

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

    // Route 1×1 convolutions to GEMM (no im2col overhead).
    if kh == 1 && kw == 1 {
        let mut output = conv2d_1x1(input, weight, bias, stride, padding, registry)?;

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
        return Ok(output);
    }

    // Route 3×3 stride=1 convolutions to Winograd F(2×2,3×3) when available.
    #[cfg(feature = "cublas")]
    if kh == 3 && kw == 3 && stride == 1 {
        let dev = registry.device();
        let mut output = conv2d_winograd_f2x2(input, weight, bias, padding, dev)?;

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
                // h_out and w_out used implicitly by autograd backward
                // h_out = h + 2*padding - 2, w_out = w + 2*padding - 2
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
        return Ok(output);
    }

    // Route 5×5, 7×7 (and other non-3×3) convolutions to direct conv kernel
    // when cublas feature is available (NVRTC compilation).
    #[cfg(feature = "cublas")]
    if kh * kw > 1 {
        let dev = registry.device();
        let mut output = conv2d_direct(input, weight, bias, stride, padding, dev)?;

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
        return Ok(output);
    }

    // Fallback: im2col + GEMM path (used when cublas feature is disabled
    // and kernel is not 1×1). This path has higher memory usage due to
    // im2col expansion but works for all kernel sizes and strides.
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

    // 2. Transpose im2col output on GPU: [spatial, K] → [K, spatial]
    let mut col_transposed = dev.alloc_zeros::<f32>(col_h * col_w)?;
    let f_transpose = registry.get("matrix_transpose")?;
    let transpose_total = (col_w * col_h) as u32;
    let transpose_cfg = KernelRegistry::config_1d(transpose_total);
    unsafe {
        f_transpose
            .launch(
                transpose_cfg,
                (
                    &col_dev,
                    &mut col_transposed,
                    col_w as u32, // rows of im2col output (spatial)
                    col_h as u32, // cols of im2col output (K)
                    &status_dev,
                ),
            )
            .map_err(NnError::Cuda)?;
    }

    // 3. GEMM: W[c_out, K] × Col_T[K, spatial] → [c_out, spatial]
    // Weight [c_out, c_in, kh, kw] has same flat data as [c_out, K] — just reshape on GPU
    let w_reshaped = weight.reshape(&[c_out, col_h])?;
    let col_tensor = GpuTensor::from_data(col_transposed, &[col_h, col_w], Arc::clone(dev));
    let gemm_out = super::matmul(&w_reshaped, &col_tensor, registry)?;

    // 3. Result is [C_out, col_w] = [C_out, h_out * w_out] — reshape to [C_out, H_out, W_out]
    let mut output = gemm_out.reshape(&[c_out, h_out, w_out])?;

    // 4. Add bias if present (GPU kernel)
    if let Some(bias_tensor) = bias {
        output = super::bias_add_chw(&output, bias_tensor, registry)?;
    }

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
/// Routes to optimized single-launch kernels for each conv variant.
/// Fallback im2col path runs im2col + transpose per sample on GPU,
/// then ONE big matmul for all samples combined.
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

    // Route 1×1 batched convolutions through single-launch GEMM.
    if kh == 1 && kw == 1 {
        return conv2d_batched_direct(input, weight, bias, stride, padding, registry, true);
    }

    // Route 3×3 stride=1 batched convolutions through single-launch Winograd.
    #[cfg(feature = "cublas")]
    if kh == 3 && kw == 3 && stride == 1 {
        return conv2d_batched_winograd(input, weight, bias, padding, registry);
    }

    // Route 5×5, 7×7 (and other non-3×3) batched convolutions through
    // single-launch direct conv.
    #[cfg(feature = "cublas")]
    if kh * kw > 1 {
        return conv2d_batched_direct(input, weight, bias, stride, padding, registry, false);
    }

    // Fallback: im2col path (no cublas feature, non-1×1 kernel)
    // Run im2col + GPU transpose per sample, assemble on host,
    // then ONE big matmul for all samples combined.
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let col_h = c_in * kh * kw;
    let col_w = h_out * w_out;

    let dev = registry.device();
    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let big_col_w = batch * col_w;
    let input_host = input.to_host()?;
    let mut all_cols_t = vec![0.0f32; col_h * big_col_w];

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

        // Transpose on GPU, then download
        let mut col_transposed = dev.alloc_zeros::<f32>(col_h * col_w)?;
        let f_transpose = registry.get("matrix_transpose")?;
        let transpose_total = (col_w * col_h) as u32;
        let transpose_cfg = KernelRegistry::config_1d(transpose_total);
        unsafe {
            f_transpose
                .launch(
                    transpose_cfg,
                    (
                        &col_dev,
                        &mut col_transposed,
                        col_w as u32,
                        col_h as u32,
                        &status_dev,
                    ),
                )
                .map_err(NnError::Cuda)?;
        }

        // Download transposed column and place into big matrix
        let col_t_host = dev.dtoh_sync_copy(&col_transposed)?;
        for k in 0..col_h {
            let dst_start = k * big_col_w + b * col_w;
            let src_start = k * col_w;
            all_cols_t[dst_start..dst_start + col_w]
                .copy_from_slice(&col_t_host[src_start..src_start + col_w]);
        }
    }

    // ONE big matmul: W[c_out, K] × BigCol[K, batch*spatial]
    let w_reshaped = weight.reshape(&[c_out, col_h])?;
    let big_col = GpuTensor::from_host(&all_cols_t, &[col_h, big_col_w], dev)?;
    let gemm_out = super::matmul(&w_reshaped, &big_col, registry)?;

    // Result [c_out, batch*spatial] → [batch, c_out, h_out, w_out]
    let mut result = gemm_out.to_host()?;

    // Add bias — result is [c_out, big_col_w] row-major
    if let Some(bias_tensor) = bias {
        let bias_host = bias_tensor.to_host()?;
        for ch in 0..c_out {
            let b_val = bias_host[ch];
            for j in 0..big_col_w {
                result[ch * big_col_w + j] += b_val;
            }
        }
    }

    // Reshape from [c_out, batch*spatial] to [batch, c_out, h_out, w_out]
    let mut output_data = vec![0.0f32; batch * c_out * h_out * w_out];
    for b in 0..batch {
        for ch in 0..c_out {
            for s in 0..col_w {
                output_data[b * c_out * col_w + ch * col_w + s] =
                    result[ch * big_col_w + b * col_w + s];
            }
        }
    }

    let mut output = GpuTensor::from_host(&output_data, &[batch, c_out, h_out, w_out], dev)?;

    // Record on autograd tape (same as single-sample conv2d)
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

/// Batched Winograd F(2×2, 3×3): single kernel launch for all N samples.
///
/// Input `[N, C_in, H, W]` → output `[N, C_out, H_out, W_out]`.
/// The Winograd kernel uses `grid.z = N` to process all batch samples
/// in one launch, eliminating per-sample launch overhead.
#[cfg(feature = "cublas")]
fn conv2d_batched_winograd(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let c_out = weight.shape()[0];

    let dev = registry.device();

    // Single-launch batched Winograd: input/output treated as contiguous
    // [N, C_in, H, W] / [N, C_out, H_out, W_out] with batch offset in kernel.
    let mut output = conv2d_winograd_f2x2_impl(input, weight, bias, padding, dev, batch)?;

    // Record on autograd tape
    if input.requires_grad() {
        let kh = 3;
        let kw = 3;
        let stride = 1;
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

/// 1×1 convolution as matrix multiplication (no im2col needed).
///
/// A 1×1 conv is equivalent to reshaping input as `[C_in, H*W]` and computing
/// `weight[C_out, C_in] × input_reshaped[C_in, H*W]`.
///
/// With stride > 1, we subsample the spatial output after the matmul.
/// With padding > 0, we pad the spatial dimensions (though padding > 0 is
/// unusual for 1×1 convolutions).
fn conv2d_1x1(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let c_in = input.shape()[0];
    let h = input.shape()[1];
    let w = input.shape()[2];
    let c_out = weight.shape()[0];

    let h_out = (h + 2 * padding - 1) / stride + 1;
    let w_out = (w + 2 * padding - 1) / stride + 1;

    // Reshape weight from [C_out, C_in, 1, 1] to [C_out, C_in]
    let w_2d = weight.reshape(&[c_out, c_in])?;

    if stride == 1 && padding == 0 {
        // Fast path: plain matmul. Reshape input to [C_in, H*W].
        let input_2d = input.reshape(&[c_in, h * w])?;
        let gemm_out = super::matmul(&w_2d, &input_2d, registry)?;
        let mut output = gemm_out.reshape(&[c_out, h_out, w_out])?;

        if let Some(bias_tensor) = bias {
            output = super::bias_add_chw(&output, bias_tensor, registry)?;
        }
        Ok(output)
    } else {
        // Stride > 1 or padding > 0: subsample after matmul.
        // First do the full matmul, then subsample.
        let input_2d = input.reshape(&[c_in, h * w])?;
        let gemm_out = super::matmul(&w_2d, &input_2d, registry)?;
        // gemm_out is [c_out, h*w], reshaped to [c_out, h, w]
        let full_out = gemm_out.reshape(&[c_out, h, w])?;

        // Subsample on host (padding + stride for 1x1 is rare, correctness > speed)
        let full_host = full_out.to_host()?;
        let mut output_data = vec![0.0f32; c_out * h_out * w_out];
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let ih = oh * stride;
                    let iw = ow * stride;
                    let ih = ih as isize - padding as isize;
                    let iw = iw as isize - padding as isize;
                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                        output_data[co * h_out * w_out + oh * w_out + ow] =
                            full_host[co * h * w + ih as usize * w + iw as usize];
                    }
                }
            }
        }
        let dev = registry.device();
        let mut output = GpuTensor::from_host(&output_data, &[c_out, h_out, w_out], dev)?;

        if let Some(bias_tensor) = bias {
            output = super::bias_add_chw(&output, bias_tensor, registry)?;
        }
        Ok(output)
    }
}

/// Batched 1×1 convolution as a single matrix multiplication.
///
/// Input `[N, C_in, H, W]` is reshaped to `[C_in, N*H*W]`, then a single
/// matmul `W[C_out, C_in] × input[C_in, N*H*W]` produces `[C_out, N*H*W]`.
/// The result is rearranged to `[N, C_out, H_out, W_out]`.
///
/// For stride=1, padding=0 (the common case), no rearrangement is needed
/// beyond reshaping.
fn conv2d_1x1_batched(
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

    let h_out = (h + 2 * padding - 1) / stride + 1;
    let w_out = (w + 2 * padding - 1) / stride + 1;

    // Reshape weight from [C_out, C_in, 1, 1] to [C_out, C_in]
    let w_2d = weight.reshape(&[c_out, c_in])?;

    if stride == 1 && padding == 0 {
        // Fast path: input [N, C_in, H, W] is contiguous as [C_in, N*H*W]
        // when we treat batch and spatial dims together.
        // However, the memory layout is [N][C_in][H][W], so to get [C_in][N*H*W]
        // we need to transpose batch and channel dims.
        //
        // Actually, input layout: for sample b, channel c, row r, col k:
        //   input[b * C_in*H*W + c * H*W + r * W + k]
        //
        // We want matrix [C_in, N*H*W] where column (b*H*W + r*W + k) has all C_in values.
        // This means: result[c][b*H*W + r*W + k] = input[b][c][r][k]
        //
        // The input is stored as [b][c][h][w], so for a fixed (b, h, w), the C_in values
        // are NOT contiguous — they're strided by H*W.
        //
        // Approach: transpose input from [N, C_in, H, W] to [C_in, N, H, W] = [C_in, N*H*W]
        // using the existing transpose op.
        let input_transposed = input.transpose(0, 1)?;
        // input_transposed is [C_in, N, H, W], contiguous
        let input_2d = input_transposed.reshape(&[c_in, batch * h * w])?;
        let gemm_out = super::matmul(&w_2d, &input_2d, registry)?;
        // gemm_out is [C_out, N*H*W] = [C_out, N, H, W] in memory
        // Reshape to [C_out, N, H_out, W_out], then transpose to [N, C_out, H_out, W_out]
        let gemm_reshaped = gemm_out.reshape(&[c_out, batch, h_out, w_out])?;
        let mut output = gemm_reshaped.transpose(0, 1)?;

        if let Some(bias_tensor) = bias {
            // Bias add: broadcast bias[C_out] across batch and spatial dims
            let bias_host = bias_tensor.to_host()?;
            let mut out_host = output.to_host()?;
            for b in 0..batch {
                let sample_offset = b * c_out * h_out * w_out;
                for co in 0..c_out {
                    let bv = bias_host[co];
                    let base = sample_offset + co * h_out * w_out;
                    for i in 0..h_out * w_out {
                        out_host[base + i] += bv;
                    }
                }
            }
            let dev = registry.device();
            output = GpuTensor::from_host(&out_host, &[batch, c_out, h_out, w_out], dev)?;
        }
        Ok(output)
    } else {
        // Stride > 1 or padding > 0: transpose + full matmul + subsample
        let input_transposed = input.transpose(0, 1)?;
        let input_2d = input_transposed.reshape(&[c_in, batch * h * w])?;
        let gemm_out = super::matmul(&w_2d, &input_2d, registry)?;
        // gemm_out is [C_out, N*H*W], reshape to [C_out, N, H, W]
        let gemm_reshaped = gemm_out.reshape(&[c_out, batch, h, w])?;
        // Transpose to [N, C_out, H, W]
        let full_out = gemm_reshaped.transpose(0, 1)?;

        // Subsample on host (padding + stride for 1x1 is rare, correctness > speed)
        let full_host = full_out.to_host()?;
        let mut output_data = vec![0.0f32; batch * c_out * h_out * w_out];
        for b in 0..batch {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let ih = oh * stride;
                        let iw = ow * stride;
                        let ih = ih as isize - padding as isize;
                        let iw = iw as isize - padding as isize;
                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            output_data[b * c_out * h_out * w_out
                                + co * h_out * w_out
                                + oh * w_out
                                + ow] = full_host
                                [b * c_out * h * w + co * h * w + ih as usize * w + iw as usize];
                        }
                    }
                }
            }
        }
        let dev = registry.device();
        let mut output = GpuTensor::from_host(&output_data, &[batch, c_out, h_out, w_out], dev)?;

        if let Some(bias_tensor) = bias {
            let bias_host = bias_tensor.to_host()?;
            let mut out_host = output.to_host()?;
            for b in 0..batch {
                let sample_offset = b * c_out * h_out * w_out;
                for co in 0..c_out {
                    let bv = bias_host[co];
                    let base = sample_offset + co * h_out * w_out;
                    for i in 0..h_out * w_out {
                        out_host[base + i] += bv;
                    }
                }
            }
            output = GpuTensor::from_host(&out_host, &[batch, c_out, h_out, w_out], dev)?;
        }
        Ok(output)
    }
}

/// Direct convolution CUDA kernel source (NVRTC compiled).
///
/// Handles arbitrary kernel sizes (5×5, 7×7, etc.) with register tiling.
/// Each thread computes one output element. Input tiles are loaded to shared
/// memory, filter weights are loaded per-thread into registers.
///
/// Supports both single-sample and batched modes via `batch_size` parameter.
/// In batched mode, `blockIdx.z` encodes `batch_idx * C_out + c_out`.
///
/// This avoids im2col's memory expansion (25× for 5×5, 49× for 7×7).
#[cfg(feature = "cublas")]
static DIRECT_CONV_SRC: &str = r#"
// Direct convolution kernel with batching support.
//
// Grid:  (ceil(W_out / TW), ceil(H_out / TH), batch_size * C_out)
// Block: (TW, TH, 1)  where TW=16, TH=16
//
// blockIdx.z = batch_idx * C_out + c_out
//
// Each thread computes ONE output element.
// Filter weights in registers.
//
// Parameters:
//   input:   [N, C_in, H, W] (row-major, or [C_in, H, W] when N=1)
//   weight:  [C_out, C_in, KH, KW] (row-major)
//   bias:    [C_out] or nullptr
//   output:  [N, C_out, H_out, W_out] (row-major, or [C_out, H_out, W_out] when N=1)

#define TILE_W 16
#define TILE_H 16

extern "C" __global__ void direct_conv2d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int C_in,
    unsigned int H, unsigned int W,
    unsigned int C_out,
    unsigned int KH, unsigned int KW,
    unsigned int stride, unsigned int padding,
    unsigned int H_out, unsigned int W_out,
    unsigned int has_bias
) {
    // Output element this thread computes
    unsigned int ow = blockIdx.x * TILE_W + threadIdx.x;
    unsigned int oh = blockIdx.y * TILE_H + threadIdx.y;

    // Decode batch index and output channel from blockIdx.z
    unsigned int batch_idx = blockIdx.z / C_out;
    unsigned int co = blockIdx.z % C_out;

    if (ow >= W_out || oh >= H_out) return;

    // Per-sample offsets
    unsigned int in_sample_offset = batch_idx * C_in * H * W;
    unsigned int out_sample_offset = batch_idx * C_out * H_out * W_out;

    float acc = 0.0f;

    // Iterate over input channels
    for (unsigned int ci = 0; ci < C_in; ci++) {
        // Weight base for this (co, ci) pair
        const float* w_ptr = weight + ((co * C_in + ci) * KH) * KW;

        // Iterate over filter window
        for (unsigned int fh = 0; fh < KH; fh++) {
            for (unsigned int fw = 0; fw < KW; fw++) {
                int ih = (int)(oh * stride + fh) - (int)padding;
                int iw = (int)(ow * stride + fw) - (int)padding;

                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                    float in_val = input[in_sample_offset + ci * H * W + ih * W + iw];
                    float w_val = w_ptr[fh * KW + fw];
                    acc = fmaf(in_val, w_val, acc);
                }
            }
        }
    }

    // Add bias
    if (has_bias) {
        acc += bias[co];
    }

    output[out_sample_offset + co * H_out * W_out + oh * W_out + ow] = acc;
}

// Tiled direct convolution with shared memory for input data.
//
// Each thread block computes a TILE_H x TILE_W output tile for one output
// channel and one batch sample. blockIdx.z = batch_idx * C_out + c_out.
// Input data for the required receptive field is loaded into shared memory.
// Filter weights are loaded into registers per-thread.
//
// Shared memory layout: [C_in_chunk][smem_h][smem_w]
// where smem_h = TILE_H * stride + KH - stride
//       smem_w = TILE_W * stride + KW - stride
//
// C_in is processed in chunks to limit shared memory usage.

#define C_IN_CHUNK 4

extern "C" __global__ void direct_conv2d_tiled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int C_in,
    unsigned int H, unsigned int W,
    unsigned int C_out,
    unsigned int KH, unsigned int KW,
    unsigned int stride, unsigned int padding,
    unsigned int H_out, unsigned int W_out,
    unsigned int has_bias
) {
    // Shared memory: sized dynamically to hold the input tile
    extern __shared__ float smem[];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int ow = blockIdx.x * TILE_W + tx;
    unsigned int oh = blockIdx.y * TILE_H + ty;

    // Decode batch index and output channel from blockIdx.z
    unsigned int batch_idx = blockIdx.z / C_out;
    unsigned int co = blockIdx.z % C_out;

    // Per-sample offsets
    unsigned int in_sample_offset = batch_idx * C_in * H * W;
    unsigned int out_sample_offset = batch_idx * C_out * H_out * W_out;

    // Shared memory tile dimensions
    unsigned int smem_h = TILE_H * stride + KH - stride;
    unsigned int smem_w = TILE_W * stride + KW - stride;
    unsigned int smem_plane = smem_h * smem_w;

    // Base input position for the top-left corner of this block's receptive field
    int in_h_base = (int)(blockIdx.y * TILE_H * stride) - (int)padding;
    int in_w_base = (int)(blockIdx.x * TILE_W * stride) - (int)padding;

    float acc = 0.0f;

    // Process input channels in chunks to limit shared memory
    unsigned int ci_chunks = (C_in + C_IN_CHUNK - 1) / C_IN_CHUNK;
    unsigned int tid = ty * TILE_W + tx;
    unsigned int block_size = TILE_W * TILE_H;

    for (unsigned int chunk = 0; chunk < ci_chunks; chunk++) {
        unsigned int ci_start = chunk * C_IN_CHUNK;
        unsigned int ci_end = ci_start + C_IN_CHUNK;
        if (ci_end > C_in) ci_end = C_in;
        unsigned int ci_count = ci_end - ci_start;

        // Cooperatively load input tile into shared memory
        unsigned int total_smem = ci_count * smem_plane;
        for (unsigned int idx = tid; idx < total_smem; idx += block_size) {
            unsigned int ci_local = idx / smem_plane;
            unsigned int spatial = idx % smem_plane;
            unsigned int sh = spatial / smem_w;
            unsigned int sw = spatial % smem_w;

            int ih = in_h_base + (int)sh;
            int iw = in_w_base + (int)sw;
            unsigned int ci_global = ci_start + ci_local;

            float val = 0.0f;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W && ci_global < C_in) {
                val = input[in_sample_offset + ci_global * H * W + ih * W + iw];
            }
            smem[ci_local * smem_plane + sh * smem_w + sw] = val;
        }
        __syncthreads();

        // Compute convolution for this chunk
        if (ow < W_out && oh < H_out) {
            for (unsigned int ci_local = 0; ci_local < ci_count; ci_local++) {
                unsigned int ci_global = ci_start + ci_local;
                const float* w_ptr = weight + ((co * C_in + ci_global) * KH) * KW;
                const float* s_ptr = smem + ci_local * smem_plane;

                // Position in shared memory for this thread's output element
                unsigned int sh_base = ty * stride;
                unsigned int sw_base = tx * stride;

                for (unsigned int fh = 0; fh < KH; fh++) {
                    for (unsigned int fw = 0; fw < KW; fw++) {
                        float in_val = s_ptr[(sh_base + fh) * smem_w + sw_base + fw];
                        float w_val = w_ptr[fh * KW + fw];
                        acc = fmaf(in_val, w_val, acc);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (ow < W_out && oh < H_out) {
        if (has_bias) {
            acc += bias[co];
        }
        output[out_sample_offset + co * H_out * W_out + oh * W_out + ow] = acc;
    }
}
"#;

/// Direct convolution via NVRTC-compiled kernel.
///
/// Handles arbitrary kernel sizes with shared memory tiling for input data
/// and register blocking for filter weights. Avoids im2col memory expansion.
///
/// Supports both single-sample and batched modes:
/// - Single: `input [C_in, H, W]` → `output [C_out, H_out, W_out]`
/// - Batched: `input [N, C_in, H, W]` → `output [N, C_out, H_out, W_out]`
///
/// The `batch_size` parameter controls the grid z-dimension:
/// `blockIdx.z = batch_idx * C_out + c_out`.
#[cfg(feature = "cublas")]
fn conv2d_direct(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    conv2d_direct_impl(input, weight, bias, stride, padding, dev, 1)
}

/// Inner implementation of direct convolution supporting batching.
///
/// When `batch_size > 1`, the input must be `[N, C_in, H, W]` and the output
/// is `[N, C_out, H_out, W_out]`. The kernel uses `blockIdx.z` to encode
/// `batch_idx * C_out + c_out`.
#[cfg(feature = "cublas")]
fn conv2d_direct_impl(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
    batch_size: usize,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx_with_opts;

    let (c_in, h, w) = if batch_size > 1 {
        (input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    };
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    // Compile direct conv kernels via NVRTC (cached per device).
    // Check if already loaded by probing for the function; compile + load if missing.
    if dev.get_func("direct_conv", "direct_conv2d").is_none() {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_75"),
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(DIRECT_CONV_SRC, opts)
            .expect("NVRTC direct_conv2d compile failed");
        dev.load_ptx(
            ptx,
            "direct_conv",
            &["direct_conv2d", "direct_conv2d_tiled"],
        )
        .expect("direct_conv PTX load failed");
    }

    let output_shape = if batch_size > 1 {
        vec![batch_size, c_out, h_out, w_out]
    } else {
        vec![c_out, h_out, w_out]
    };
    let mut output = GpuTensor::zeros(&output_shape, dev)?;

    // Decide whether to use the tiled (shared memory) or simple kernel.
    // Tiled kernel needs shared memory: C_IN_CHUNK * smem_h * smem_w * 4 bytes
    // smem_h = TILE_H * stride + KH - stride = 16 * stride + KH - stride
    // smem_w = TILE_W * stride + KW - stride = 16 * stride + KW - stride
    let tile_w = 16u32;
    let tile_h = 16u32;
    let c_in_chunk: u32 = 4;
    let smem_h = tile_h * stride as u32 + kh as u32 - stride as u32;
    let smem_w = tile_w * stride as u32 + kw as u32 - stride as u32;
    let smem_bytes = c_in_chunk * smem_h * smem_w * 4; // bytes

    // Use tiled kernel if shared memory fits (48KB limit on SM75)
    let use_tiled = smem_bytes <= 48 * 1024;

    // Create a dummy bias buffer if needed (the kernel needs a valid pointer)
    let bias_dev;
    let bias_ptr = if let Some(b) = bias {
        b.data()
    } else {
        bias_dev = dev.alloc_zeros::<f32>(1)?;
        &bias_dev
    };

    let has_bias: u32 = if bias.is_some() { 1 } else { 0 };

    // Grid z-dim: batch_size * C_out (kernel decodes batch_idx and c_out)
    let grid_z = (batch_size * c_out) as u32;

    // Build raw kernel parameter array (15 params exceeds cudarc's 12-tuple limit,
    // so we use the raw void-pointer launch interface).
    let mut c_in_v = c_in as u32;
    let mut h_v = h as u32;
    let mut w_v = w as u32;
    let mut c_out_v = c_out as u32;
    let mut kh_v = kh as u32;
    let mut kw_v = kw as u32;
    let mut stride_v = stride as u32;
    let mut padding_v = padding as u32;
    let mut h_out_v = h_out as u32;
    let mut w_out_v = w_out as u32;
    let mut has_bias_v = has_bias;

    // Device pointers — we need raw CUdeviceptr values
    use cudarc::driver::DevicePtr;
    let mut in_ptr = *input.data().device_ptr();
    let mut w_ptr = *weight.data().device_ptr();
    let mut bias_ptr_raw = *bias_ptr.device_ptr();
    let mut out_ptr = *output.data_mut().device_ptr();

    let mut params: Vec<*mut std::ffi::c_void> = vec![
        &mut in_ptr as *mut _ as *mut std::ffi::c_void,
        &mut w_ptr as *mut _ as *mut std::ffi::c_void,
        &mut bias_ptr_raw as *mut _ as *mut std::ffi::c_void,
        &mut out_ptr as *mut _ as *mut std::ffi::c_void,
        &mut c_in_v as *mut _ as *mut std::ffi::c_void,
        &mut h_v as *mut _ as *mut std::ffi::c_void,
        &mut w_v as *mut _ as *mut std::ffi::c_void,
        &mut c_out_v as *mut _ as *mut std::ffi::c_void,
        &mut kh_v as *mut _ as *mut std::ffi::c_void,
        &mut kw_v as *mut _ as *mut std::ffi::c_void,
        &mut stride_v as *mut _ as *mut std::ffi::c_void,
        &mut padding_v as *mut _ as *mut std::ffi::c_void,
        &mut h_out_v as *mut _ as *mut std::ffi::c_void,
        &mut w_out_v as *mut _ as *mut std::ffi::c_void,
        &mut has_bias_v as *mut _ as *mut std::ffi::c_void,
    ];

    if use_tiled {
        let func =
            dev.get_func("direct_conv", "direct_conv2d_tiled")
                .ok_or(NnError::KernelNotFound {
                    name: "direct_conv2d_tiled",
                })?;

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (
                (w_out as u32).div_ceil(tile_w),
                (h_out as u32).div_ceil(tile_h),
                grid_z,
            ),
            block_dim: (tile_w, tile_h, 1),
            shared_mem_bytes: smem_bytes,
        };

        unsafe {
            func.launch(config, &mut params).map_err(NnError::Cuda)?;
        }
    } else {
        // Fallback: simple direct conv (no shared memory, each thread reads global)
        let func = dev
            .get_func("direct_conv", "direct_conv2d")
            .ok_or(NnError::KernelNotFound {
                name: "direct_conv2d",
            })?;

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (
                (w_out as u32).div_ceil(tile_w),
                (h_out as u32).div_ceil(tile_h),
                grid_z,
            ),
            block_dim: (tile_w, tile_h, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, &mut params).map_err(NnError::Cuda)?;
        }
    }

    Ok(output)
}

/// Batched direct/1×1 convolution via single kernel launch.
///
/// When `is_1x1` is true, routes through batched GEMM (`conv2d_1x1_batched`).
/// Otherwise routes through `conv2d_direct_impl` with `batch_size=N` for a
/// single kernel launch covering all samples.
fn conv2d_batched_direct(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
    is_1x1: bool,
) -> Result<GpuTensor> {
    let _batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    let _dev = registry.device();

    let mut output = if is_1x1 {
        conv2d_1x1_batched(input, weight, bias, stride, padding, registry)?
    } else {
        #[cfg(feature = "cublas")]
        {
            conv2d_direct_impl(input, weight, bias, stride, padding, _dev, _batch)?
        }
        #[cfg(not(feature = "cublas"))]
        {
            // Fallback: should not reach here since routing only calls this
            // with is_1x1=false when cublas feature is enabled.
            return Err(NnError::KernelNotFound {
                name: "direct_conv2d (cublas feature required)",
            });
        }
    };

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

/// Winograd F(2×2, 3×3) convolution — single sample wrapper.
///
/// Delegates to `conv2d_winograd_f2x2_impl` with `batch_size=1`.
#[cfg(feature = "cublas")]
fn conv2d_winograd_f2x2(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    padding: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    conv2d_winograd_f2x2_impl(input, weight, bias, padding, dev, 1)
}

/// Winograd F(2×2, 3×3) convolution — NVRTC compiled.
///
/// Reduces FLOPs for 3×3 stride=1 convolutions by ~2.25× compared to direct.
///
/// Supports both single-sample and batched modes:
/// - `batch_size=1`: input `[C_in, H, W]` → output `[C_out, H_out, W_out]`
/// - `batch_size=N`: input `[N, C_in, H, W]` → output `[N, C_out, H_out, W_out]`
///
/// The kernel uses `grid.z = batch_size` to process all samples in one launch.
///
/// The algorithm:
/// 1. Filter transform: G·g·Gᵀ (done once, 3×3 → 4×4 Winograd domain)
/// 2. Input transform: Bᵀ·d·B (per 4×4 input tile)
/// 3. Element-wise multiply in Winograd domain (16 muls per tile)
/// 4. Output transform: Aᵀ·m·A (4×4 → 2×2 output tile)
#[cfg(feature = "cublas")]
fn conv2d_winograd_f2x2_impl(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    padding: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
    batch_size: usize,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx_with_opts;

    let (c_in, h, w) = if batch_size > 1 {
        (input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    };
    let c_out = weight.shape()[0];

    let h_out = h + 2 * padding - 2; // stride=1, kh=3: (h + 2p - 3)/1 + 1 = h + 2p - 2
    let w_out = w + 2 * padding - 2;

    // Number of 2×2 output tiles covering the output
    let n_tile_y = h_out.div_ceil(2);
    let n_tile_x = w_out.div_ceil(2);
    let total_tiles = n_tile_x * n_tile_y;

    // Compile Winograd CUDA kernels via NVRTC (cached per device).
    // Check if already loaded by probing for the function; compile + load if missing.
    static WINOGRAD_SRC: &str = include_str!("winograd_f2x2.cu");

    if dev
        .get_func("winograd_f2x2", "winograd_filter_transform")
        .is_none()
    {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_75"),
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx =
            compile_ptx_with_opts(WINOGRAD_SRC, opts).expect("NVRTC winograd_f2x2 compile failed");
        dev.load_ptx(
            ptx,
            "winograd_f2x2",
            &["winograd_filter_transform", "winograd_conv2d_f2x2"],
        )
        .expect("winograd_f2x2 PTX load failed");
    }

    // 1. Filter transform: weight[C_out, C_in, 3, 3] → filter_wino[16, C_out, C_in]
    let filter_plane = c_out * c_in;
    let mut filter_wino = dev.alloc_zeros::<f32>(16 * filter_plane)?;

    let ft_func = dev
        .get_func("winograd_f2x2", "winograd_filter_transform")
        .ok_or(NnError::KernelNotFound {
            name: "winograd_filter_transform",
        })?;
    let ft_config = cudarc::driver::LaunchConfig {
        grid_dim: ((filter_plane as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        ft_func
            .launch(
                ft_config,
                (weight.data(), &mut filter_wino, c_out as u32, c_in as u32),
            )
            .map_err(NnError::Cuda)?;
    }

    // 2. Winograd convolution: input tiles × transformed filters → output tiles
    // grid.z = batch_size to process all samples in one launch
    let output_shape = if batch_size > 1 {
        vec![batch_size, c_out, h_out, w_out]
    } else {
        vec![c_out, h_out, w_out]
    };
    let mut output = GpuTensor::zeros(&output_shape, dev)?;

    let tile_c_out: u32 = 32;
    let conv_func = dev
        .get_func("winograd_f2x2", "winograd_conv2d_f2x2")
        .ok_or(NnError::KernelNotFound {
            name: "winograd_conv2d_f2x2",
        })?;
    let conv_config = cudarc::driver::LaunchConfig {
        grid_dim: (
            total_tiles as u32,
            (c_out as u32).div_ceil(tile_c_out),
            batch_size as u32,
        ),
        block_dim: (tile_c_out, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        conv_func
            .launch(
                conv_config,
                (
                    input.data(),
                    &filter_wino,
                    output.data_mut(),
                    c_in as u32,
                    c_out as u32,
                    h as u32,
                    w as u32,
                    h_out as u32,
                    w_out as u32,
                    n_tile_x as u32,
                    n_tile_y as u32,
                    padding as u32,
                ),
            )
            .map_err(NnError::Cuda)?;
    }

    // 3. Add bias if present
    if let Some(bias_tensor) = bias {
        // Bias add for all samples: bias[C_out] broadcast across batch and spatial dims
        let bias_host = bias_tensor.to_host()?;
        let mut out_host = output.to_host()?;
        for b in 0..batch_size {
            let sample_offset = b * c_out * h_out * w_out;
            for co in 0..c_out {
                let bv = bias_host[co];
                let base = sample_offset + co * h_out * w_out;
                for i in 0..h_out * w_out {
                    out_host[base + i] += bv;
                }
            }
        }
        output = GpuTensor::from_host(&out_host, &output_shape, dev)?;
    }

    Ok(output)
}

/// Conv2d backward pass on GPU: computes dInput and dWeight.
///
/// Uses im2col + matmul for dWeight, matmul + col2im for dInput.
/// Supports both single-sample [C,H,W] and batched [N,C,H,W] inputs.
///
/// Returns `(d_input, d_weight)` with same shapes as `input` and `weight`.
pub fn conv2d_backward(
    d_output: &GpuTensor,
    input: &GpuTensor,
    weight: &GpuTensor,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor)> {
    let (c_in, h, w) = if input.ndim() == 4 {
        (input.shape()[1], input.shape()[2], input.shape()[3])
    } else {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    };
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    crate::nn::autograd::backward::conv2d_backward_dispatch(
        d_output, input, weight, c_in, c_out, h, w, kh, kw, stride, padding, registry,
    )
}
