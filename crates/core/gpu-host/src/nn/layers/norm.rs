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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_registry() -> Arc<KernelRegistry> {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX load"))
    }

    /// CPU f64 reference for LayerNorm.
    fn cpu_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], d: usize, eps: f32) -> Vec<f32> {
        let n = input.len() / d;
        let mut out = vec![0.0f32; input.len()];
        for i in 0..n {
            let row = &input[i * d..(i + 1) * d];
            let mean: f64 = row.iter().map(|&x| x as f64).sum::<f64>() / d as f64;
            let var: f64 = row
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / d as f64;
            let std = (var + eps as f64).sqrt();
            for j in 0..d {
                let norm = (row[j] as f64 - mean) / std;
                out[i * d + j] = (norm * gamma[j] as f64 + beta[j] as f64) as f32;
            }
        }
        out
    }

    #[test]
    fn test_layer_norm_matches_cpu() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 4;
        let d = 64;
        let eps = 1e-5;

        let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let beta: Vec<f32> = (0..d).map(|i| (i as f32) * 0.001).collect();
        let input: Vec<f32> = (0..batch * d)
            .map(|i| ((i as f32) - (batch * d / 2) as f32) * 0.01)
            .collect();

        let expected = cpu_layer_norm(&input, &gamma, &beta, d, eps);

        let layer = LayerNorm::new(&gamma, &beta, eps, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[batch, d], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        assert_eq!(output_tensor.shape(), &[batch, d]);

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max absolute error {max_err} exceeds 1e-3");
    }
}
