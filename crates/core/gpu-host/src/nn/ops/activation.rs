//! Activation functions: GELU, SiLU, sigmoid, ReLU.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
///
/// Uses vectorized V2 kernel (4 elements per thread) for higher throughput.
pub fn gelu(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "gelu_forward_v2", registry)
}

/// SiLU (Swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x)).
///
/// Uses vectorized V2 kernel (4 elements per thread) for higher throughput.
pub fn silu(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "silu_forward_v2", registry)
}

/// Sigmoid activation: y = 1 / (1 + exp(-x)).
///
/// Uses vectorized V2 kernel (4 elements per thread) for higher throughput.
pub fn sigmoid(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "sigmoid_forward_v2", registry)
}

/// ReLU activation: y = max(0, x).
///
/// Uses vectorized float4 GPU kernel (4 elements per thread).
pub fn relu(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    let n = input.numel();
    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("relu_forward")?;
    // V2-style: 4 elements per thread, 256 threads per block
    let config = cudarc::driver::LaunchConfig {
        grid_dim: ((n as u32 + 1023) / 1024, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            config,
            (input.data(), output.data_mut(), n as u32, &status_dev),
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
                op: crate::nn::autograd::OpKind::Relu,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id],
                meta: crate::nn::autograd::OpMeta::None,
            });
        }
    }

    Ok(output)
}

/// Generic element-wise activation launcher.
fn elementwise_activation(
    input: &GpuTensor,
    kernel_name: &'static str,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let n = input.numel();
    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get(kernel_name)?;
    // V2 kernels process 4 elements per thread → grid = ceil(n/1024)
    let is_v2 = kernel_name.ends_with("_v2");
    let config = if is_v2 {
        cudarc::driver::LaunchConfig {
            grid_dim: ((n as u32 + 1023) / 1024, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        KernelRegistry::config_1d(n as u32)
    };
    unsafe {
        func.launch(
            config,
            (input.data(), output.data_mut(), n as u32, &status_dev),
        )
        .map_err(NnError::Cuda)?;
    }

    // Record on autograd tape
    if input.requires_grad() {
        let op = match kernel_name {
            "gelu_forward" | "gelu_forward_v2" => crate::nn::autograd::OpKind::Gelu,
            "silu_forward" | "silu_forward_v2" => crate::nn::autograd::OpKind::Silu,
            "sigmoid_forward" | "sigmoid_forward_v2" => crate::nn::autograd::OpKind::Sigmoid,
            _ => crate::nn::autograd::OpKind::Relu,
        };
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            let in_id = input
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id], // save input for backward (activation derivative)
                meta: crate::nn::autograd::OpMeta::None,
            });
        }
    }

    Ok(output)
}
