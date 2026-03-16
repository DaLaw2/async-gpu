//! Activation functions: GELU, SiLU, sigmoid, ReLU.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
pub fn gelu(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "gelu_forward", registry)
}

/// SiLU (Swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x)).
pub fn silu(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "silu_forward", registry)
}

/// Sigmoid activation: y = 1 / (1 + exp(-x)).
pub fn sigmoid(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    elementwise_activation(input, "sigmoid_forward", registry)
}

/// ReLU activation: y = max(0, x).
///
/// No dedicated kernel — computed on host for now.
pub fn relu(input: &GpuTensor, _registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    let host = input.to_host()?;
    let out: Vec<f32> = host.iter().map(|&x| x.max(0.0)).collect();
    GpuTensor::from_host(&out, input.shape(), input.device())
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
    let config = KernelRegistry::config_1d(n as u32);
    unsafe {
        func.launch(
            config,
            (input.data(), output.data_mut(), n as u32, &status_dev),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    // Record on autograd tape
    if input.requires_grad() {
        let op = match kernel_name {
            "gelu_forward" => crate::nn::autograd::OpKind::Gelu,
            "silu_forward" => crate::nn::autograd::OpKind::Silu,
            "sigmoid_forward" => crate::nn::autograd::OpKind::Sigmoid,
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
