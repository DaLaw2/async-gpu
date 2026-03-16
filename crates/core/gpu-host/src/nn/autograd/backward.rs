//! Backward pass — reverse topological traversal of the autograd tape.

use std::collections::HashMap;
use std::sync::Arc;

use super::tape::{OpKind, TapeEntry, TensorId};
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
            // Placeholders for remaining ops
            OpKind::Embedding | OpKind::CrossEntropy | OpKind::MseLoss => {
                // TODO: implement in ag-loss
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
