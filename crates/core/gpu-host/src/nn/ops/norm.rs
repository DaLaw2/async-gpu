//! Normalization operations: layer_norm, batch_norm, batch_norm_silu.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Layer normalization over the last dimension.
///
/// Input: `[*, d_model]`, gamma/beta: `[d_model]` → output: same shape as input.
pub fn layer_norm(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let ndim = input.ndim();
    let d_model = input.shape()[ndim - 1];
    let num_rows = input.numel() / d_model;

    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("layer_norm")?;
    let config = KernelRegistry::config_layernorm(num_rows as u32);
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                gamma.data(),
                beta.data(),
                d_model as u32,
                eps,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    // Record on autograd tape
    if input.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            let in_id = input
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::LayerNorm,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id], // save input for backward
                meta: crate::nn::autograd::OpMeta::LayerNorm {
                    rows: num_rows,
                    d: d_model,
                    eps,
                },
            });
        }
    }

    Ok(output)
}

/// Batch normalization for CHW tensors.
///
/// Input: `[C, H, W]`, gamma/beta/mean/var: `[C]` → output: same shape.
pub fn batch_norm(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    running_mean: &GpuTensor,
    running_var: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    // Use batchnorm_silu kernel but we need a separate batchnorm-only kernel.
    // For now, fall back to host-side implementation.
    // TODO: add a dedicated batchnorm (no SiLU) kernel, or use batchnorm_silu
    // and apply inverse SiLU. For v1, compute on host.
    let inp = input.to_host()?;
    let g = gamma.to_host()?;
    let b = beta.to_host()?;
    let mean = running_mean.to_host()?;
    let var = running_var.to_host()?;

    let c = input.shape()[0];
    let hw: usize = input.shape()[1..].iter().product();
    let mut out = vec![0.0f32; inp.len()];

    for ch in 0..c {
        let inv_std = 1.0 / (var[ch] + eps).sqrt();
        for i in 0..hw {
            let idx = ch * hw + i;
            out[idx] = g[ch] * (inp[idx] - mean[ch]) * inv_std + b[ch];
        }
    }

    let dev = registry.device();
    GpuTensor::from_host(&out, input.shape(), dev)
}

/// Fused BatchNorm + SiLU for CHW tensors.
///
/// Input: `[C, H, W]`, gamma/beta/mean/var: `[C]` → output: same shape.
/// Uses the fused `batchnorm_silu` GPU kernel.
pub fn batch_norm_silu(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    running_mean: &GpuTensor,
    running_var: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let c = input.shape()[0];
    let hw: usize = input.shape()[1..].iter().product();
    let n = input.numel();

    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("batchnorm_silu")?;
    let config = KernelRegistry::config_1d(n as u32);
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                gamma.data(),
                beta.data(),
                running_mean.data(),
                running_var.data(),
                n as u32,
                hw as u32,
                eps,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    let _ = c; // used indirectly via hw calculation
    Ok(output)
}
