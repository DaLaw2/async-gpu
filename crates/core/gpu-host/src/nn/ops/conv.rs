//! Convolution via im2col + GEMM pipeline, with Winograd F(2×2, 3×3) fast path.

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

    // Route 3×3 stride=1 batched convolutions through Winograd per-sample.
    #[cfg(feature = "cublas")]
    if kh == 3 && kw == 3 && stride == 1 {
        return conv2d_batched_winograd(input, weight, bias, padding, registry);
    }

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let col_h = c_in * kh * kw;
    let col_w = h_out * w_out;

    let dev = registry.device();
    let input_host = input.to_host()?;

    // im2col per sample, collect columns into [K, batch*col_w] layout.
    // Target layout: all_cols_t[k * big_col_w + b * col_w + s] for batch b,
    // K-row k, spatial position s.
    let big_col_w = batch * col_w;
    let mut all_cols_t = vec![0.0f32; col_h * big_col_w];
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

        // Transpose from [spatial, K] to [K, spatial] and place into
        // the correct batch-column slice of the big matrix.
        let col_raw = dev.dtoh_sync_copy(&col_dev)?;
        for s in 0..col_w {
            for k in 0..col_h {
                all_cols_t[k * big_col_w + b * col_w + s] = col_raw[s * col_h + k];
            }
        }
    }

    // ONE big matmul: W[c_out, K] × BigCol[K, batch*spatial]
    let w_host = weight.to_host()?;
    let w_tensor = GpuTensor::from_host(&w_host, &[c_out, col_h], dev)?;
    let big_col = GpuTensor::from_host(&all_cols_t, &[col_h, big_col_w], dev)?;
    let gemm_out = super::matmul(&w_tensor, &big_col, registry)?;

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

/// Batched Winograd F(2×2, 3×3): processes each sample through the Winograd path,
/// then assembles into `[N, C_out, H_out, W_out]`.
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
    let h_out = h + 2 * padding - 2; // stride=1, kh=3
    let w_out = w + 2 * padding - 2;

    let dev = registry.device();
    let sample_out_size = c_out * h_out * w_out;
    let mut output_host = vec![0.0f32; batch * sample_out_size];
    let input_host = input.to_host()?;

    for b in 0..batch {
        let sample_start = b * c_in * h * w;
        let sample_end = sample_start + c_in * h * w;
        let sample =
            GpuTensor::from_host(&input_host[sample_start..sample_end], &[c_in, h, w], dev)?;
        let result = conv2d_winograd_f2x2(&sample, weight, bias, padding, dev)?;
        let result_host = result.to_host()?;
        output_host[b * sample_out_size..(b + 1) * sample_out_size].copy_from_slice(&result_host);
    }

    let mut output = GpuTensor::from_host(&output_host, &[batch, c_out, h_out, w_out], dev)?;

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

/// Winograd F(2×2, 3×3) convolution — NVRTC compiled.
///
/// Reduces FLOPs for 3×3 stride=1 convolutions by ~2.25× compared to direct.
/// Input: `[C_in, H, W]`, weight: `[C_out, C_in, 3, 3]`, optional bias: `[C_out]`.
/// Output: `[C_out, H_out, W_out]`.
///
/// The algorithm:
/// 1. Filter transform: G·g·Gᵀ (done once, 3×3 → 4×4 Winograd domain)
/// 2. Input transform: Bᵀ·d·B (per 4×4 input tile)
/// 3. Element-wise multiply in Winograd domain (16 muls per tile)
/// 4. Output transform: Aᵀ·m·A (4×4 → 2×2 output tile)
#[cfg(feature = "cublas")]
fn conv2d_winograd_f2x2(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&GpuTensor>,
    padding: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx_with_opts;

    let c_in = input.shape()[0];
    let h = input.shape()[1];
    let w = input.shape()[2];
    let c_out = weight.shape()[0];

    let h_out = h + 2 * padding - 2; // stride=1, kh=3: (h + 2p - 3)/1 + 1 = h + 2p - 2
    let w_out = w + 2 * padding - 2;

    // Number of 2×2 output tiles covering the output
    let n_tile_y = h_out.div_ceil(2);
    let n_tile_x = w_out.div_ceil(2);
    let total_tiles = n_tile_x * n_tile_y;

    // Compile Winograd CUDA kernels via NVRTC (cached)
    static WINOGRAD_SRC: &str = include_str!("winograd_f2x2.cu");

    use std::sync::OnceLock;
    static COMPILED: OnceLock<bool> = OnceLock::new();
    COMPILED.get_or_init(|| {
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
        true
    });

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
    let mut output = GpuTensor::zeros(&[c_out, h_out, w_out], dev)?;

    let tile_c_out: u32 = 32;
    let conv_func = dev
        .get_func("winograd_f2x2", "winograd_conv2d_f2x2")
        .ok_or(NnError::KernelNotFound {
            name: "winograd_conv2d_f2x2",
        })?;
    let conv_config = cudarc::driver::LaunchConfig {
        grid_dim: (total_tiles as u32, (c_out as u32).div_ceil(tile_c_out), 1),
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
        // Simple host-side bias add for correctness; uses existing GPU bias_add_chw
        // when called from full conv2d path (which wraps this).
        let bias_host = bias_tensor.to_host()?;
        let mut out_host = output.to_host()?;
        for co in 0..c_out {
            let b = bias_host[co];
            let base = co * h_out * w_out;
            for i in 0..h_out * w_out {
                out_host[base + i] += b;
            }
        }
        output = GpuTensor::from_host(&out_host, &[c_out, h_out, w_out], dev)?;
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
