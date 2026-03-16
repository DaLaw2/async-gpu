//! Linear (fully connected) layer: y = xW^T + b.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// Linear layer: y = xW^T + b.
///
/// Weight: `[out_features, in_features]`, bias: `[out_features]` (optional).
pub struct Linear {
    weight_t: GpuTensor, // [in_features, out_features] — pre-transposed for matmul
    bias: Option<GpuTensor>,
    registry: Arc<KernelRegistry>,
}

impl Linear {
    /// Create a new Linear layer.
    ///
    /// `weight` is `[out_features, in_features]` (PyTorch convention).
    /// Internally transposes to `[in_features, out_features]` for efficient matmul.
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();

        // Transpose weight from [out, in] to [in, out]
        let mut wt = vec![0.0f32; in_features * out_features];
        for r in 0..out_features {
            for c in 0..in_features {
                wt[c * out_features + r] = weight[r * in_features + c];
            }
        }
        let weight_t = GpuTensor::from_host(&wt, &[in_features, out_features], dev)?;

        let bias = if let Some(b) = bias {
            Some(GpuTensor::from_host(b, &[out_features], dev)?)
        } else {
            None
        };

        Ok(Self {
            weight_t,
            bias,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for Linear {
    /// Forward pass: input `[*, in_features]` → output `[*, out_features]`.
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let ndim = input.ndim();
        let in_features = input.shape()[ndim - 1];
        let batch: usize = input.shape()[..ndim - 1].iter().product();

        // Reshape to [batch, in_features] for matmul
        let input_2d = if ndim == 2 {
            input.clone_tensor()?
        } else {
            input.reshape(&[batch, in_features])?
        };

        // matmul: [batch, in_features] x [in_features, out_features] = [batch, out_features]
        let mut output = ops::matmul(&input_2d, &self.weight_t, &self.registry)?;

        // Add bias
        if let Some(ref bias) = self.bias {
            ops::bias_add(&mut output, bias, &self.registry)?;
        }

        // Reshape back to [..., out_features]
        if ndim > 2 {
            let out_features = self.weight_t.shape()[1];
            let mut out_shape: Vec<usize> = input.shape()[..ndim - 1].to_vec();
            out_shape.push(out_features);
            output.reshape(&out_shape)
        } else {
            Ok(output)
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

    /// CPU reference: y = x * W^T + b
    fn cpu_linear(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        in_f: usize,
        out_f: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_f];
        for b in 0..batch {
            for o in 0..out_f {
                let mut sum = 0.0f64;
                for i in 0..in_f {
                    // weight is [out_f, in_f] (row-major), so W[o][i] = weight[o * in_f + i]
                    sum += input[b * in_f + i] as f64 * weight[o * in_f + i] as f64;
                }
                if let Some(bias) = bias {
                    sum += bias[o] as f64;
                }
                out[b * out_f + o] = sum as f32;
            }
        }
        out
    }

    #[test]
    fn test_linear_forward_matches_cpu() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 4;
        let in_f = 8;
        let out_f = 6;

        // Deterministic weights: small values for numerical stability
        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f32) - 24.0) * 0.01)
            .collect();
        let bias: Vec<f32> = (0..out_f).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i as f32) - 16.0) * 0.1)
            .collect();

        // CPU reference
        let expected = cpu_linear(&input, &weight, Some(&bias), batch, in_f, out_f);

        // GPU
        let layer = Linear::new(&weight, Some(&bias), in_f, out_f, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[batch, in_f], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        assert_eq!(output_tensor.shape(), &[batch, out_f]);

        // Compare with tolerance (f32 GEMM)
        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max absolute error {max_err} exceeds 1e-3");
    }

    #[test]
    fn test_linear_no_bias() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 2;
        let in_f = 4;
        let out_f = 3;

        let weight: Vec<f32> = (0..out_f * in_f).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.5).collect();

        let expected = cpu_linear(&input, &weight, None, batch, in_f, out_f);

        let layer = Linear::new(&weight, None, in_f, out_f, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[batch, in_f], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max absolute error {max_err} exceeds 1e-3");
    }
}
