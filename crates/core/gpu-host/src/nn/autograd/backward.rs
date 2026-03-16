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
            // Placeholder for ops that need backward kernels (implemented in later tasks)
            OpKind::Gelu
            | OpKind::Silu
            | OpKind::Sigmoid
            | OpKind::Relu
            | OpKind::LayerNorm
            | OpKind::BiasAdd
            | OpKind::Embedding
            | OpKind::CrossEntropy
            | OpKind::MseLoss => {
                // TODO: dispatch to backward kernels in ag-elemwise-bwd, ag-norm-bwd, etc.
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

    let a = pool.get(a_id).ok_or_else(|| NnError::ShapeMismatch {
        expected: "saved tensor A in pool".to_string(),
        actual: format!("TensorId({}) not found", a_id.0),
    })?;
    let b = pool.get(b_id).ok_or_else(|| NnError::ShapeMismatch {
        expected: "saved tensor B in pool".to_string(),
        actual: format!("TensorId({}) not found", b_id.0),
    })?;

    // dA = dC × B^T: [m,n] × [n,k] → [m,k]
    let b_t = b.transpose(0, 1)?;
    let d_a = crate::nn::ops::matmul(d_out, &b_t, registry)?;
    accumulate_grad(grads, a_id, d_a, registry)?;

    // dB = A^T × dC: [k,m] × [m,n] → [k,n]
    let a_t = a.transpose(0, 1)?;
    let d_b = crate::nn::ops::matmul(&a_t, d_out, registry)?;
    accumulate_grad(grads, b_id, d_b, registry)?;

    Ok(())
}
