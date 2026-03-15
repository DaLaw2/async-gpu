// CNN building blocks: BatchNorm+SiLU fused, im2col, MaxPool2D, Upsample, Concat.
// Used by YOLOv8-nano inference pipeline.

use crate::helpers::gpu_exp_f32;
use core::arch::nvptx;

// ============================================================
// Fused BatchNorm + SiLU kernel (yolo-inference.3)
// ============================================================

/// Fused BatchNorm + SiLU: out[i] = SiLU(gamma * (x - running_mean) / sqrt(running_var + eps) + beta)
///
/// Operates on a CHW tensor: `n = C * H * W` total elements.
/// `gamma`, `beta`, `running_mean`, `running_var` are per-channel (length C).
/// `hw` = H * W (spatial size per channel).
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Launch: grid_dim = (n.div_ceil(256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batchnorm_silu(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    running_mean: *const f32,
    running_var: *const f32,
    n: u32,
    hw: u32,
    eps: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let c = global_id / hw; // channel index
            let x = *input.add(global_id as usize);
            let g = *gamma.add(c as usize);
            let b = *beta.add(c as usize);
            let mean = *running_mean.add(c as usize);
            let var = *running_var.add(c as usize);

            // BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
            let inv_std = 1.0 / crate::helpers::gpu_sqrtf(var + eps);
            let bn_out = g * (x - mean) * inv_std + b;

            // SiLU: silu(x) = x / (1 + exp(-x))
            let sigmoid = 1.0 / (1.0 + gpu_exp_f32(-bn_out));
            let result = bn_out * sigmoid;

            *output.add(global_id as usize) = result;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, gamma, beta, running_mean, running_var, n, hw, eps);
    }

    if tid == 0 {
        *status = 0;
    }
}

/// Standalone SiLU activation: out[i] = x[i] / (1 + exp(-x[i]))
///
/// Launch: grid_dim = (n.div_ceil(256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn silu_forward(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            let sigmoid = 1.0 / (1.0 + gpu_exp_f32(-x));
            *output.add(global_id as usize) = x * sigmoid;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// im2col kernel (yolo-inference.2)
// ============================================================

/// im2col: rearrange input [C, H, W] patches into columns [H_out*W_out, C*kH*kW].
///
/// For each output position (oh, ow) and each filter tap (c, kh, kw),
/// copies the corresponding input value to the column matrix.
///
/// Parameters:
/// - `input`: [C_in, H, W] tensor
/// - `output`: [H_out * W_out, C_in * kH * kW] matrix
/// - `c_in, h, w`: input dimensions
/// - `kh, kw`: filter size (typically 3x3 or 1x1)
/// - `stride, pad`: convolution stride and padding
/// - `h_out, w_out`: output spatial dimensions
///
/// Launch: grid_dim = (total_output_elements.div_ceil(256), 1, 1), block_dim = (256, 1, 1)
/// where total_output_elements = H_out * W_out * C_in * kH * kW
#[no_mangle]
pub unsafe extern "ptx-kernel" fn im2col(
    input: *const f32,
    output: *mut f32,
    c_in: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad: u32,
    h_out: u32,
    w_out: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        let total_cols = h_out * w_out;
        let col_width = c_in * kh * kw;
        let total = total_cols * col_width;

        if global_id < total {
            // Decompose global_id into (row, col) in the output matrix
            let out_row = global_id / col_width; // output spatial position
            let out_col = global_id % col_width; // position within the filter patch

            let oh = out_row / w_out;
            let ow = out_row % w_out;

            let c = out_col / (kh * kw);
            let kk = out_col % (kh * kw);
            let fh = kk / kw;
            let fw = kk % kw;

            let ih = oh * stride + fh;
            let iw = ow * stride + fw;

            let val = if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                let real_h = ih - pad;
                let real_w = iw - pad;
                *input.add((c * h * w + real_h * w + real_w) as usize)
            } else {
                0.0f32 // zero-padding
            };

            *output.add((out_row * col_width + out_col) as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, c_in, h, w, kh, kw, stride, pad, h_out, w_out);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// MaxPool2D kernel (yolo-inference.4)
// ============================================================

/// MaxPool2D: for each output position, compute max over a kxk window.
///
/// Input: [C, H, W], Output: [C, H_out, W_out]
/// where H_out = (H + 2*pad - k) / stride + 1
///
/// Launch: grid_dim = (C * H_out * W_out).div_ceil(256), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn maxpool2d(
    input: *const f32,
    output: *mut f32,
    c: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    pad: u32,
    h_out: u32,
    w_out: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = c * h_out * w_out;

        if global_id < total {
            let ch = global_id / (h_out * w_out);
            let rem = global_id % (h_out * w_out);
            let oh = rem / w_out;
            let ow = rem % w_out;

            let mut max_val = f32::NEG_INFINITY;

            for fh in 0..k {
                for fw in 0..k {
                    let ih = oh * stride + fh;
                    let iw = ow * stride + fw;
                    if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                        let real_h = ih - pad;
                        let real_w = iw - pad;
                        let val = *input.add((ch * h * w + real_h * w + real_w) as usize);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
            }

            *output.add(global_id as usize) = max_val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, c, h, w, k, stride, pad, h_out, w_out);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// Upsample 2x nearest-neighbor kernel (yolo-inference.4)
// ============================================================

/// Upsample 2x nearest neighbor: output[c, y, x] = input[c, y/2, x/2]
///
/// Input: [C, H, W], Output: [C, 2*H, 2*W]
///
/// Launch: grid_dim = (C * 2*H * 2*W).div_ceil(256), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn upsample_nearest_2x(
    input: *const f32,
    output: *mut f32,
    c: u32,
    h_in: u32,
    w_in: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        let h_out = h_in * 2;
        let w_out = w_in * 2;
        let total = c * h_out * w_out;

        if global_id < total {
            let ch = global_id / (h_out * w_out);
            let rem = global_id % (h_out * w_out);
            let oy = rem / w_out;
            let ox = rem % w_out;

            let iy = oy / 2;
            let ix = ox / 2;

            let val = *input.add((ch * h_in * w_in + iy * w_in + ix) as usize);
            *output.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, c, h_in, w_in);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// Concat along channel dimension kernel (yolo-inference.4)
// ============================================================

/// Concat two tensors along channel dimension: output = [a; b] in C dimension.
///
/// Input a: [C_a, H, W], Input b: [C_b, H, W]
/// Output: [C_a + C_b, H, W]
///
/// Launch: grid_dim = ((C_a + C_b) * H * W).div_ceil(256), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn concat_channels(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    c_a: u32,
    c_b: u32,
    hw: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        let c_total = c_a + c_b;
        let total = c_total * hw;

        if global_id < total {
            let ch = global_id / hw;
            let spatial = global_id % hw;

            let val = if ch < c_a {
                *a.add((ch * hw + spatial) as usize)
            } else {
                *b.add(((ch - c_a) * hw + spatial) as usize)
            };

            *output.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, output, c_a, c_b, hw);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// Sigmoid kernel (yolo-inference.7 — detect head classification)
// ============================================================

/// Element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
///
/// Launch: grid_dim = (n.div_ceil(256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sigmoid_forward(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            *output.add(global_id as usize) = 1.0 / (1.0 + gpu_exp_f32(-x));
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// Bias add kernel (yolo-inference.7 — detect head bare Conv2d)
// ============================================================

/// Add per-channel bias to CHW tensor: output[c, h, w] = input[c, h, w] + bias[c]
///
/// Total elements: n = C * H * W. hw = H * W.
/// Launch: grid_dim = (n.div_ceil(256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bias_add_chw(
    input: *const f32,
    output: *mut f32,
    bias: *const f32,
    n: u32,
    hw: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let ch = global_id / hw;
            let val = *input.add(global_id as usize) + *bias.add(ch as usize);
            *output.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, bias, n, hw);
    }

    if tid == 0 {
        *status = 0;
    }
}
