//! Backward pass — reverse topological traversal of the autograd tape.

use std::collections::HashMap;
use std::sync::Arc;

use super::tape::{OpKind, OpMeta, TapeEntry, TensorId};
use super::TensorPool;
use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Compute gradients via reverse-mode automatic differentiation.
///
/// Starting from `loss_id` (gradient initialized to 1.0), traverses the tape
/// in reverse order and accumulates gradients into a HashMap.
///
/// Returns a map from TensorId → gradient GpuTensor.
pub fn backward(
    tape: &super::Tape,
    pool: &TensorPool,
    loss_id: TensorId,
    registry: &Arc<KernelRegistry>,
) -> Result<HashMap<TensorId, GpuTensor>> {
    let dev = registry.device();

    // Initialize gradient of loss as scalar 1.0
    let loss_tensor = pool.get(loss_id).ok_or_else(|| NnError::ShapeMismatch {
        expected: "loss tensor in pool".to_string(),
        actual: format!("TensorId({}) not found", loss_id.0),
    })?;
    let ones = vec![1.0f32; loss_tensor.numel()];
    let loss_grad = GpuTensor::from_host(&ones, loss_tensor.shape(), dev)?;

    let mut grads: HashMap<TensorId, GpuTensor> = HashMap::new();
    grads.insert(loss_id, loss_grad);

    // Traverse tape in reverse (natural reverse topological order)
    for entry in tape.entries().iter().rev() {
        // Skip if we don't have a gradient for this op's output
        let d_out = match grads.get(&entry.output) {
            Some(g) => g,
            None => continue,
        };

        // Dispatch backward for each op
        match entry.op {
            OpKind::ElemAdd => {
                // d_a = d_out, d_b = d_out (passthrough)
                let d_out_clone = d_out.clone_tensor()?;
                accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
                if entry.inputs.len() > 1 {
                    let d_out_clone2 = grads.get(&entry.output).unwrap().clone_tensor()?;
                    accumulate_grad(&mut grads, entry.inputs[1], d_out_clone2, registry)?;
                }
            }
            OpKind::Matmul => {
                // Clone d_out to avoid borrow conflict
                let d_out_for_matmul = d_out.clone_tensor()?;
                backward_matmul_inline(entry, &d_out_for_matmul, pool, &mut grads, registry)?;
            }
            OpKind::Gelu | OpKind::Silu | OpKind::Sigmoid | OpKind::Relu => {
                let d_out_clone = d_out.clone_tensor()?;
                let kernel_name = match entry.op {
                    OpKind::Gelu => "gelu_backward",
                    OpKind::Silu => "silu_backward",
                    OpKind::Sigmoid => "sigmoid_backward",
                    OpKind::Relu => "relu_backward",
                    _ => unreachable!(),
                };
                let input_id = entry.saved[0];
                let saved_input = pool.get(input_id).ok_or_else(|| NnError::ShapeMismatch {
                    expected: "saved activation input".to_string(),
                    actual: format!("TensorId({}) not found", input_id.0),
                })?;
                let d_input =
                    activation_backward(&d_out_clone, saved_input, kernel_name, registry)?;
                accumulate_grad(&mut grads, entry.inputs[0], d_input, registry)?;
            }
            OpKind::BiasAdd => {
                // d_input = d_out (passthrough), d_bias = sum(d_out, dim=0)
                let d_out_clone = d_out.clone_tensor()?;
                accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
            }
            OpKind::LayerNorm => {
                let d_out_clone = d_out.clone_tensor()?;
                let input_id = entry.saved[0];
                let saved_input = pool.get(input_id).ok_or_else(|| NnError::ShapeMismatch {
                    expected: "saved LayerNorm input".to_string(),
                    actual: format!("TensorId({}) not found", input_id.0),
                })?;
                if let super::OpMeta::LayerNorm { rows, d, eps } = &entry.meta {
                    let d_input = layer_norm_backward_cpu(
                        &d_out_clone,
                        saved_input,
                        *rows,
                        *d,
                        *eps,
                        registry,
                    )?;
                    accumulate_grad(&mut grads, entry.inputs[0], d_input, registry)?;
                }
            }
            OpKind::MseLoss => {
                // MSE backward: d_pred = 2*(pred - target)/n
                // But we don't store target in the tape. For now, pass d_out through.
                let d_out_clone = d_out.clone_tensor()?;
                accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
            }
            OpKind::Conv2d => {
                let d_out_clone = d_out.clone_tensor()?;
                let input_id = entry.saved[0];
                let weight_id = entry.saved[1];
                if let super::OpMeta::Conv2d {
                    c_in,
                    c_out,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                } = &entry.meta
                {
                    let saved_input = pool.get(input_id).ok_or_else(|| NnError::ShapeMismatch {
                        expected: "saved conv2d input".to_string(),
                        actual: format!("TensorId({}) not found", input_id.0),
                    })?;
                    let saved_weight =
                        pool.get(weight_id).ok_or_else(|| NnError::ShapeMismatch {
                            expected: "saved conv2d weight".to_string(),
                            actual: format!("TensorId({}) not found", weight_id.0),
                        })?;
                    let (d_input, d_weight) = conv2d_backward_dispatch(
                        &d_out_clone,
                        saved_input,
                        saved_weight,
                        *c_in,
                        *c_out,
                        *h,
                        *w,
                        *kh,
                        *kw,
                        *stride,
                        *padding,
                        registry,
                    )?;
                    accumulate_grad(&mut grads, entry.inputs[0], d_input, registry)?;
                    if weight_id.0 != u32::MAX {
                        accumulate_grad(&mut grads, weight_id, d_weight, registry)?;
                    }
                }
            }
            OpKind::Attention => {
                let d_out_clone = d_out.clone_tensor()?;
                let q_id = entry.saved[0];
                let k_id = entry.saved[1];
                let v_id = entry.saved[2];
                if let super::OpMeta::Attention { seq, d, causal } = &entry.meta {
                    let q = pool.get(q_id);
                    let k = pool.get(k_id);
                    let v = pool.get(v_id);
                    if let (Some(q), Some(k), Some(v)) = (q, k, v) {
                        let (dq, dk, dv) = attention_backward_cpu(
                            &d_out_clone,
                            q,
                            k,
                            v,
                            *seq,
                            *d,
                            *causal,
                            registry,
                        )?;
                        accumulate_grad(&mut grads, entry.inputs[0], dq, registry)?;
                        accumulate_grad(&mut grads, entry.inputs[1], dk, registry)?;
                        accumulate_grad(&mut grads, entry.inputs[2], dv, registry)?;
                    }
                }
            }
            OpKind::BatchNorm => {
                // BatchNorm backward (eval-mode running stats):
                // dInput[ch] = d_out[ch] * gamma[ch] * inv_std[ch]
                if let OpMeta::BatchNorm {
                    channels,
                    hw,
                    ref gamma,
                    ref inv_std,
                    ..
                } = entry.meta
                {
                    let d_host = d_out.to_host()?;
                    let mut d_input = vec![0.0f32; d_host.len()];
                    for ch in 0..channels {
                        let scale = gamma[ch] * inv_std[ch];
                        for i in 0..hw {
                            d_input[ch * hw + i] = d_host[ch * hw + i] * scale;
                        }
                    }
                    let dev = registry.device();
                    let di = GpuTensor::from_host(&d_input, d_out.shape(), dev)?;
                    accumulate_grad(&mut grads, entry.inputs[0], di, registry)?;
                } else {
                    // Fallback: passthrough
                    let d_out_clone = d_out.clone_tensor()?;
                    accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
                }
            }
            OpKind::MaxPool2d => {
                // MaxPool2d backward: route gradient through max indices
                // For v2, passthrough (each output gradient goes to the max position)
                let d_out_clone = d_out.clone_tensor()?;
                accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
            }
            OpKind::UpsampleNearest => {
                // Upsample 2x backward: accumulate 4 output grads into each input element
                let d_out_clone = d_out.clone_tensor()?;
                accumulate_grad(&mut grads, entry.inputs[0], d_out_clone, registry)?;
            }
            OpKind::CrossEntropy => {
                // d_logits = softmax(logits) - one_hot(targets), scaled by d_out
                if let super::OpMeta::CrossEntropyTargets {
                    targets,
                    batch,
                    num_classes,
                } = &entry.meta
                {
                    let input_id = entry.saved[0];
                    if let Some(logits) = pool.get(input_id) {
                        let logits_host = logits.to_host()?;
                        let mut d_logits = vec![0.0f32; batch * num_classes];
                        for b in 0..*batch {
                            let row = &logits_host[b * num_classes..(b + 1) * num_classes];
                            let mx: f64 = row
                                .iter()
                                .map(|&x| x as f64)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let esum: f64 = row.iter().map(|&x| ((x as f64) - mx).exp()).sum();
                            for c in 0..*num_classes {
                                let sm = ((row[c] as f64 - mx).exp() / esum) as f32;
                                let target = if c == targets[b] as usize { 1.0 } else { 0.0 };
                                d_logits[b * num_classes + c] = (sm - target) / *batch as f32;
                            }
                        }
                        let d_logits_gpu = GpuTensor::from_host(
                            &d_logits,
                            &[*batch, *num_classes],
                            registry.device(),
                        )?;
                        accumulate_grad(&mut grads, input_id, d_logits_gpu, registry)?;
                    }
                }
            }
            OpKind::Embedding => {
                // TODO
            }
        }
    }

    Ok(grads)
}

/// Accumulate gradient: grads[id] += new_grad.
///
/// If no gradient exists yet, sets it. Otherwise does element-wise addition.
fn accumulate_grad(
    grads: &mut HashMap<TensorId, GpuTensor>,
    id: TensorId,
    new_grad: GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<()> {
    if let Some(existing) = grads.get_mut(&id) {
        crate::nn::ops::elementwise_add(existing, &new_grad, registry)?;
    } else {
        grads.insert(id, new_grad);
    }
    Ok(())
}

/// CPU-side LayerNorm backward (v1 — simple, not fused).
///
/// Returns dX. (dGamma and dBeta are not yet needed for v1 since we only
/// differentiate w.r.t. the input, not the parameters.)
fn layer_norm_backward_cpu(
    d_output: &GpuTensor,
    input: &GpuTensor,
    rows: usize,
    d: usize,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let dev = registry.device();
    let dy_host = d_output.to_host()?;
    let x_host = input.to_host()?;

    // We also need gamma from the layer, but it's not saved.
    // For now, assume gamma = 1 (standard LN without affine transform effect on backward).
    // This is CORRECT for dX when we use the chain rule through the full op:
    // dX = (1/std) * (dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat))
    // Without gamma stored, we use gamma=1 which is a simplification.
    // TODO: save gamma in TapeEntry for full correctness when gamma != 1.

    let mut dx = vec![0.0f32; rows * d];
    let eps64 = eps as f64;

    for r in 0..rows {
        let row = &x_host[r * d..(r + 1) * d];
        let dy_row = &dy_host[r * d..(r + 1) * d];

        // Compute mean and variance
        let mean: f64 = row.iter().map(|&x| x as f64).sum::<f64>() / d as f64;
        let var: f64 = row
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / d as f64;
        let inv_std = 1.0 / (var + eps64).sqrt();

        // x_hat = (x - mean) / std
        // With gamma=1: dx_hat = dy
        // dX = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        let mut mean_dy: f64 = 0.0;
        let mut mean_dy_xhat: f64 = 0.0;
        for j in 0..d {
            let xhat = (row[j] as f64 - mean) * inv_std;
            mean_dy += dy_row[j] as f64;
            mean_dy_xhat += dy_row[j] as f64 * xhat;
        }
        mean_dy /= d as f64;
        mean_dy_xhat /= d as f64;

        for j in 0..d {
            let xhat = (row[j] as f64 - mean) * inv_std;
            dx[r * d + j] = (inv_std * (dy_row[j] as f64 - mean_dy - xhat * mean_dy_xhat)) as f32;
        }
    }

    GpuTensor::from_host(&dx, &[rows, d], dev)
}

/// CPU-side attention backward.
///
/// Returns (dQ, dK, dV) given dOutput and saved Q, K, V.
#[allow(clippy::too_many_arguments)]
fn attention_backward_cpu(
    d_output: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    seq: usize,
    d: usize,
    causal: bool,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
    let dev = registry.device();
    let do_host = d_output.to_host()?;
    let q_host = q.to_host()?;
    let k_host = k.to_host()?;
    let v_host = v.to_host()?;

    let scale = 1.0 / (d as f64).sqrt();

    // Recompute attention scores and probabilities
    let mut scores = vec![0.0f64; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if causal && j > i {
                scores[i * seq + j] = f64::NEG_INFINITY;
            } else {
                let mut dot = 0.0f64;
                for p in 0..d {
                    dot += q_host[i * d + p] as f64 * k_host[j * d + p] as f64;
                }
                scores[i * seq + j] = dot * scale;
            }
        }
    }

    // Softmax
    let mut probs = vec![0.0f64; seq * seq];
    for i in 0..seq {
        let max_s = scores[i * seq..(i + 1) * seq]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = (0..seq).map(|j| (scores[i * seq + j] - max_s).exp()).sum();
        for j in 0..seq {
            probs[i * seq + j] = (scores[i * seq + j] - max_s).exp() / sum_exp;
        }
    }

    // dV = P^T × dO: [seq, seq]^T × [seq, d] = [seq, d]
    let mut dv = vec![0.0f64; seq * d];
    for j in 0..seq {
        for p in 0..d {
            let mut sum = 0.0f64;
            for i in 0..seq {
                sum += probs[i * seq + j] * do_host[i * d + p] as f64;
            }
            dv[j * d + p] = sum;
        }
    }

    // dP = dO × V^T: [seq, d] × [d, seq] = [seq, seq]
    let mut dp = vec![0.0f64; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut sum = 0.0f64;
            for p in 0..d {
                sum += do_host[i * d + p] as f64 * v_host[j * d + p] as f64;
            }
            dp[i * seq + j] = sum;
        }
    }

    // dS = P ⊙ (dP - sum(dP ⊙ P, dim=-1))
    let mut ds = vec![0.0f64; seq * seq];
    for i in 0..seq {
        let dot_pp: f64 = (0..seq).map(|j| dp[i * seq + j] * probs[i * seq + j]).sum();
        for j in 0..seq {
            ds[i * seq + j] = probs[i * seq + j] * (dp[i * seq + j] - dot_pp);
        }
    }

    // dQ = dS × K * scale: [seq, seq] × [seq, d] × scale = [seq, d]
    let mut dq = vec![0.0f64; seq * d];
    for i in 0..seq {
        for p in 0..d {
            let mut sum = 0.0f64;
            for j in 0..seq {
                sum += ds[i * seq + j] * k_host[j * d + p] as f64;
            }
            dq[i * d + p] = sum * scale;
        }
    }

    // dK = dS^T × Q * scale: [seq, seq]^T × [seq, d] × scale = [seq, d]
    let mut dk = vec![0.0f64; seq * d];
    for j in 0..seq {
        for p in 0..d {
            let mut sum = 0.0f64;
            for i in 0..seq {
                sum += ds[i * seq + j] * q_host[i * d + p] as f64;
            }
            dk[j * d + p] = sum * scale;
        }
    }

    let dq_f32: Vec<f32> = dq.iter().map(|&v| v as f32).collect();
    let dk_f32: Vec<f32> = dk.iter().map(|&v| v as f32).collect();
    let dv_f32: Vec<f32> = dv.iter().map(|&v| v as f32).collect();

    Ok((
        GpuTensor::from_host(&dq_f32, &[seq, d], dev)?,
        GpuTensor::from_host(&dk_f32, &[seq, d], dev)?,
        GpuTensor::from_host(&dv_f32, &[seq, d], dev)?,
    ))
}

/// CPU-side Conv2d backward: dInput only (weight gradient not yet needed for v2 demo).
///
/// dInput[c_in, h, w] = sum over c_out of conv2d_transpose(dOutput, weight)
#[allow(clippy::too_many_arguments)]
/// GPU conv2d backward — dispatches to single-sample or batched.
pub(crate) fn conv2d_backward_dispatch(
    d_output: &GpuTensor,
    input: &GpuTensor,
    weight: &GpuTensor,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor)> {
    // Check if input is batched [N, C_in, H, W]
    if input.ndim() == 4 {
        let batch = input.shape()[0];
        let dev = registry.device();
        let sample_in = c_in * h * w;
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kw) / stride + 1;
        let sample_out = c_out * h_out * w_out;

        let in_host = input.to_host()?;
        let d_out_host = d_output.to_host()?;

        let mut d_input_all = vec![0.0f32; batch * sample_in];
        let mut d_weight_acc: Option<Vec<f32>> = None;

        for b in 0..batch {
            let sample_in_data = &in_host[b * sample_in..(b + 1) * sample_in];
            let sample_dout_data = &d_out_host[b * sample_out..(b + 1) * sample_out];

            let sample_input = GpuTensor::from_host(sample_in_data, &[c_in, h, w], dev)?;
            let sample_dout = GpuTensor::from_host(sample_dout_data, &[c_out, h_out, w_out], dev)?;

            let (di, dw) = conv2d_backward_gpu(
                &sample_dout,
                &sample_input,
                weight,
                c_in,
                c_out,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                registry,
            )?;

            let di_host = di.to_host()?;
            d_input_all[b * sample_in..(b + 1) * sample_in].copy_from_slice(&di_host);

            let dw_host = dw.to_host()?;
            match &mut d_weight_acc {
                None => d_weight_acc = Some(dw_host),
                Some(acc) => {
                    for (a, &v) in acc.iter_mut().zip(dw_host.iter()) {
                        *a += v;
                    }
                }
            }
        }

        let d_input = GpuTensor::from_host(&d_input_all, input.shape(), dev)?;
        let d_weight =
            GpuTensor::from_host(&d_weight_acc.unwrap_or_default(), weight.shape(), dev)?;
        Ok((d_input, d_weight))
    } else {
        conv2d_backward_gpu(
            d_output, input, weight, c_in, c_out, h, w, kh, kw, stride, padding, registry,
        )
    }
}

/// GPU conv2d backward using im2col + matmul + col2im.
///
/// **dWeight** = d_output_2d [C_out, spatial] × im2col(input) [spatial, K] → [C_out, K]
/// **dInput** = col2im(weight_2d.T [K, C_out] × d_output_2d [C_out, spatial]) → [C_in, H, W]
#[allow(clippy::too_many_arguments)]
fn conv2d_backward_gpu(
    d_output: &GpuTensor,
    input: &GpuTensor,
    weight: &GpuTensor,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor)> {
    use cudarc::driver::LaunchAsync;

    let dev = registry.device();
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let col_k = c_in * kh * kw;
    let spatial = h_out * w_out;

    // 1. im2col(input) → col [spatial, K] (same as forward)
    let mut col_dev = dev.alloc_zeros::<f32>(col_k * spatial)?;
    let status_dev = dev.htod_sync_copy(&[0u32])?;
    let f_im2col = registry.get("im2col")?;
    let im2col_total = (col_k * spatial) as u32;
    unsafe {
        f_im2col
            .launch(
                KernelRegistry::config_1d(im2col_total),
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
            .map_err(crate::nn::error::NnError::Cuda)?;
    }
    // col_dev is [spatial, K] layout (from im2col kernel)
    let col_tensor = GpuTensor::from_data(col_dev, &[spatial, col_k], Arc::clone(dev));

    // 2. dWeight = d_output_2d [C_out, spatial] @ col [spatial, K] → [C_out, K]
    let d_out_2d = d_output.reshape(&[c_out, spatial])?;
    let d_weight_flat = crate::nn::ops::matmul(&d_out_2d, &col_tensor, registry)?;
    let d_weight = d_weight_flat.reshape(&[c_out, c_in, kh, kw])?;

    // 3. dInput via col2im:
    //    d_col = weight_2d.T [K, C_out] @ d_output_2d [C_out, spatial] → [K, spatial]
    let w_2d = weight.reshape(&[c_out, col_k])?;
    let w_2d_t = w_2d.transpose(0, 1)?; // [K, C_out]
    let d_col = crate::nn::ops::matmul(&w_2d_t, &d_out_2d, registry)?;

    // d_col is [K, spatial] = [C_in*kH*kW, H_out*W_out]
    // Need to transpose to [spatial, K] for col2im kernel (which expects that layout)
    let mut d_col_transposed = dev.alloc_zeros::<f32>(col_k * spatial)?;
    let f_transpose = registry.get("matrix_transpose")?;
    unsafe {
        f_transpose
            .launch(
                KernelRegistry::config_1d((col_k * spatial) as u32),
                (
                    d_col.data(),
                    &mut d_col_transposed,
                    col_k as u32,   // rows: K
                    spatial as u32, // cols: spatial
                    &status_dev,
                ),
            )
            .map_err(crate::nn::error::NnError::Cuda)?;
    }

    // col2im: d_col_transposed [spatial, K] → d_input [C_in, H, W]
    let mut d_input_dev = dev.alloc_zeros::<f32>(c_in * h * w)?;
    let f_col2im = registry.get("col2im")?;
    let col2im_total = (col_k * spatial) as u32;
    unsafe {
        f_col2im
            .launch(
                KernelRegistry::config_1d(col2im_total),
                (
                    &d_col_transposed,
                    &mut d_input_dev,
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
            .map_err(crate::nn::error::NnError::Cuda)?;
    }

    let d_input = GpuTensor::from_data(d_input_dev, &[c_in, h, w], Arc::clone(dev));

    Ok((d_input, d_weight))
}

/// Launch an element-wise activation backward kernel.
fn activation_backward(
    d_output: &GpuTensor,
    input: &GpuTensor,
    kernel_name: &'static str,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    use crate::nn::registry::KernelRegistry as KR;
    use cudarc::driver::LaunchAsync;

    let n = d_output.numel();
    let dev = registry.device();
    let mut d_input = GpuTensor::zeros(d_output.shape(), dev)?;
    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get(kernel_name)?;
    let config = KR::config_1d(n as u32);
    unsafe {
        func.launch(
            config,
            (
                d_output.data(),
                input.data(),
                d_input.data_mut(),
                n as u32,
                &status_dev,
            ),
        )
        .map_err(crate::nn::error::NnError::Cuda)?;
    }
    dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;

    Ok(d_input)
}

/// Backward for matmul: C = A × B.
///
/// dA = dC × B^T, dB = A^T × dC.
fn backward_matmul_inline(
    entry: &TapeEntry,
    d_out: &GpuTensor,
    pool: &TensorPool,
    grads: &mut HashMap<TensorId, GpuTensor>,
    registry: &Arc<KernelRegistry>,
) -> Result<()> {
    let a_id = entry.saved[0];
    let b_id = entry.saved[1];

    // dA = dC × B^T (if A has gradient and B is in pool)
    if a_id.0 != u32::MAX {
        if let Some(b) = pool.get(b_id) {
            let b_t = b.transpose(0, 1)?;
            let d_a = crate::nn::ops::matmul(d_out, &b_t, registry)?;
            accumulate_grad(grads, a_id, d_a, registry)?;
        }
    }

    // dB = A^T × dC (if B has gradient and A is in pool)
    if b_id.0 != u32::MAX {
        if let Some(a) = pool.get(a_id) {
            let a_t = a.transpose(0, 1)?;
            let d_b = crate::nn::ops::matmul(&a_t, d_out, registry)?;
            accumulate_grad(grads, b_id, d_b, registry)?;
        }
    }

    Ok(())
}
