//! Normalization layers: LayerNorm, BatchNorm2d.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// Layer normalization.
///
/// Normalizes over the last dimension. gamma/beta: `[d_model]`.
pub struct LayerNorm {
    gamma: GpuTensor,
    beta: GpuTensor,
    eps: f32,
    registry: Arc<KernelRegistry>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    pub fn new(
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        let d = gamma.len();
        Ok(Self {
            gamma: GpuTensor::from_host(gamma, &[d], dev)?,
            beta: GpuTensor::from_host(beta, &[d], dev)?,
            eps,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::layer_norm(input, &self.gamma, &self.beta, self.eps, &self.registry)
    }
}

/// Batch normalization for CHW tensors with optional fused SiLU.
///
/// gamma/beta/running_mean/running_var: `[C]`.
pub struct BatchNorm2d {
    gamma: GpuTensor,
    beta: GpuTensor,
    running_mean: GpuTensor,
    running_var: GpuTensor,
    eps: f32,
    fuse_silu: bool,
    registry: Arc<KernelRegistry>,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer.
    ///
    /// If `fuse_silu` is true, uses the fused `batchnorm_silu` kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gamma: &[f32],
        beta: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f32,
        fuse_silu: bool,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        let c = gamma.len();
        Ok(Self {
            gamma: GpuTensor::from_host(gamma, &[c], dev)?,
            beta: GpuTensor::from_host(beta, &[c], dev)?,
            running_mean: GpuTensor::from_host(running_mean, &[c], dev)?,
            running_var: GpuTensor::from_host(running_var, &[c], dev)?,
            eps,
            fuse_silu,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        if self.fuse_silu {
            ops::batch_norm_silu(
                input,
                &self.gamma,
                &self.beta,
                &self.running_mean,
                &self.running_var,
                self.eps,
                &self.registry,
            )
        } else {
            ops::batch_norm(
                input,
                &self.gamma,
                &self.beta,
                &self.running_mean,
                &self.running_var,
                self.eps,
                &self.registry,
            )
        }
    }
}
