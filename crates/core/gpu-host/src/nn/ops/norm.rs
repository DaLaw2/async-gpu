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

    // Use V2 LayerNorm (256 threads, single-pass Welford, coalesced access)
    let func = registry.get("layer_norm_v2")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 2048, // 256 * 2 * 4 for partial sums
    };
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
    let mut inv_stds = vec![0.0f32; c];
    let mut x_norm = vec![0.0f32; inp.len()];

    for ch in 0..c {
        let is = 1.0 / (var[ch] + eps).sqrt();
        inv_stds[ch] = is;
        for i in 0..hw {
            let idx = ch * hw + i;
            let xn = (inp[idx] - mean[ch]) * is;
            x_norm[idx] = xn;
            out[idx] = g[ch] * xn + b[ch];
        }
    }

    let dev = registry.device();
    let mut output = GpuTensor::from_host(&out, input.shape(), dev)?;

    // Record on autograd tape
    if input.requires_grad() {
        let in_id = input
            .tensor_id()
            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::BatchNorm,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![],
                meta: crate::nn::autograd::OpMeta::BatchNorm {
                    channels: c,
                    hw,
                    gamma: g.clone(),
                    inv_std: inv_stds,
                    x_norm,
                },
            });
        }
    }

    Ok(output)
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
    let _c = input.shape()[0];
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

    Ok(output)
}
