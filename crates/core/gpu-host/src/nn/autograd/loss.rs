//! Loss functions — cross-entropy and MSE with autograd integration.

use std::sync::Arc;

use super::{TapeEntry, TensorId};
use crate::nn::error::Result;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Cross-entropy loss: -sum(target * log_softmax(logits)) / batch_size.
///
/// `logits`: `[batch, num_classes]`, `targets`: `[batch]` (class indices as f32).
/// Returns a scalar tensor `[1]`.
pub fn cross_entropy_loss(
    logits: &GpuTensor,
    targets: &[u32],
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let dev = registry.device();
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let logits_host = logits.to_host()?;

    // Compute log_softmax + NLL on CPU
    let mut loss = 0.0f64;
    for b in 0..batch {
        let row = &logits_host[b * num_classes..(b + 1) * num_classes];
        let max_val: f64 = row
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp: f64 = row
            .iter()
            .map(|&x| ((x as f64) - max_val).exp())
            .sum::<f64>();
        let log_sum_exp = max_val + log_sum_exp.ln();
        let target_class = targets[b] as usize;
        loss -= logits_host[b * num_classes + target_class] as f64 - log_sum_exp;
    }
    loss /= batch as f64;

    let mut result = GpuTensor::from_host(&[loss as f32], &[1], dev)?;

    // Record on tape
    if logits.requires_grad() {
        if let Some(out_id) = super::alloc_tensor_id() {
            result.set_tensor_id(out_id);
            result.set_requires_grad(true);
            let in_id = logits.tensor_id().unwrap_or(TensorId(u32::MAX));
            super::record_op(TapeEntry {
                op: super::OpKind::CrossEntropy,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id],
                meta: super::OpMeta::CrossEntropyTargets {
                    targets: targets.to_vec(),
                    batch,
                    num_classes,
                },
            });
        }
    }

    Ok(result)
}

/// Mean squared error loss: mean((predictions - targets)^2).
///
/// Both `predictions` and `targets` have the same shape.
/// Returns a scalar tensor `[1]`.
pub fn mse_loss(
    predictions: &GpuTensor,
    targets: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let dev = registry.device();
    let pred_host = predictions.to_host()?;
    let target_host = targets.to_host()?;
    let n = pred_host.len();

    let mse: f64 = pred_host
        .iter()
        .zip(target_host.iter())
        .map(|(&p, &t)| {
            let d = p as f64 - t as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;

    let mut result = GpuTensor::from_host(&[mse as f32], &[1], dev)?;

    // Record on tape
    if predictions.requires_grad() {
        if let Some(out_id) = super::alloc_tensor_id() {
            result.set_tensor_id(out_id);
            result.set_requires_grad(true);
            let in_id = predictions.tensor_id().unwrap_or(TensorId(u32::MAX));
            super::record_op(TapeEntry {
                op: super::OpKind::MseLoss,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id],
                meta: super::OpMeta::Loss {
                    reduction: super::Reduction::Mean,
                },
            });
        }
    }

    Ok(result)
}

/// Compute cross-entropy backward: d_logits = softmax(logits) - one_hot(targets).
///
/// This is the gradient of mean cross-entropy w.r.t. logits.
pub fn cross_entropy_backward(logits: &GpuTensor, targets: &[u32]) -> Result<Vec<f32>> {
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let logits_host = logits.to_host()?;

    let mut d_logits = vec![0.0f32; batch * num_classes];
    for b in 0..batch {
        let row = &logits_host[b * num_classes..(b + 1) * num_classes];
        let max_val: f64 = row
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&x| ((x as f64) - max_val).exp()).sum();

        for c in 0..num_classes {
            let softmax_c = ((row[c] as f64 - max_val).exp() / exp_sum) as f32;
            let target_indicator = if targets[b] as usize == c { 1.0 } else { 0.0 };
            d_logits[b * num_classes + c] = (softmax_c - target_indicator) / batch as f32;
        }
    }

    Ok(d_logits)
}

/// Compute MSE backward: d_predictions = 2 * (predictions - targets) / n.
pub fn mse_backward(predictions: &GpuTensor, targets: &GpuTensor) -> Result<Vec<f32>> {
    let pred_host = predictions.to_host()?;
    let target_host = targets.to_host()?;
    let n = pred_host.len() as f32;

    let d_pred: Vec<f32> = pred_host
        .iter()
        .zip(target_host.iter())
        .map(|(&p, &t)| 2.0 * (p - t) / n)
        .collect();

    Ok(d_pred)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dev_registry() -> (
        std::sync::Arc<cudarc::driver::CudaDevice>,
        Arc<KernelRegistry>,
    ) {
        let dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let reg = Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).unwrap());
        (dev, reg)
    }

    #[test]
    fn test_mse_loss_zero() {
        let (dev, reg) = test_dev_registry();
        let pred = GpuTensor::from_host(&[1.0, 2.0, 3.0], &[3], &dev).unwrap();
        let target = GpuTensor::from_host(&[1.0, 2.0, 3.0], &[3], &dev).unwrap();
        let loss = mse_loss(&pred, &target, &reg).unwrap();
        let val = loss.to_host().unwrap();
        assert!(
            val[0].abs() < 1e-6,
            "MSE of identical tensors should be 0, got {}",
            val[0]
        );
    }

    #[test]
    fn test_cross_entropy_correct_class() {
        let (dev, reg) = test_dev_registry();
        // logits strongly favor class 1
        let logits = GpuTensor::from_host(&[-10.0, 10.0, -10.0], &[1, 3], &dev).unwrap();
        let targets = vec![1u32];
        let loss = cross_entropy_loss(&logits, &targets, &reg).unwrap();
        let val = loss.to_host().unwrap();
        // Loss should be very small when the correct class has high logit
        assert!(
            val[0] < 0.01,
            "CE loss for correct prediction should be ~0, got {}",
            val[0]
        );
    }
}
