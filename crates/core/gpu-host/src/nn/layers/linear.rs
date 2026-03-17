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
/// Stores pre-transposed+padded weight for fast matmul (skip per-forward transpose).
pub struct Linear {
    weight_t: GpuTensor, // [in_features, out_features] — pre-transposed for matmul
    /// Pre-computed column-major padded weight for direct GEMM launch.
    /// Layout: [N_pad, K_pad] row-major = [K_pad, N_pad] col-major.
    weight_prepadded: Option<cudarc::driver::CudaSlice<f32>>,
    k_pad: usize,
    n_pad: usize,
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

        // Pre-compute column-major padded weight for fast GEMM
        let k = in_features;
        let n = out_features;
        let k_pad = k.div_ceil(16) * 16;
        let n_pad = n.div_ceil(16) * 16;

        // Transpose: weight_t [K, N] row-major → [N, K] row-major = col-major [K, N]
        // Then pad to [N_pad, K_pad]
        let weight_prepadded = {
            let status = dev.htod_sync_copy(&[0u32])?;
            let mut b_t = dev.alloc_zeros::<f32>(n * k)?;
            let f_transpose = registry.get("matrix_transpose")?;
            let cfg = crate::nn::registry::KernelRegistry::config_1d((k * n) as u32);
            unsafe {
                cudarc::driver::LaunchAsync::launch(
                    f_transpose,
                    cfg,
                    (weight_t.data(), &mut b_t, k as u32, n as u32, &status),
                )?;
            }
            if n == n_pad && k == k_pad {
                Some(b_t)
            } else {
                let mut buf = dev.alloc_zeros::<f32>(n_pad * k_pad)?;
                let f_pad = registry.get("matrix_pad")?;
                let cfg_p = crate::nn::registry::KernelRegistry::config_1d((n_pad * k_pad) as u32);
                unsafe {
                    cudarc::driver::LaunchAsync::launch(
                        f_pad,
                        cfg_p,
                        (
                            &b_t,
                            &mut buf,
                            n as u32,
                            k as u32,
                            n_pad as u32,
                            k_pad as u32,
                            &status,
                        ),
                    )?;
                }
                Some(buf)
            }
        };

        let bias = if let Some(b) = bias {
            Some(GpuTensor::from_host(b, &[out_features], dev)?)
        } else {
            None
        };

        Ok(Self {
            weight_t,
            weight_prepadded,
            k_pad,
            n_pad,
            bias,
            registry: Arc::clone(registry),
        })
    }
}

impl Linear {
    /// Fused forward: matmul + bias + activation in a single kernel launch.
    ///
    /// Saves 2 kernel launches vs `forward()` + `activation()`.
    pub fn forward_fused(
        &self,
        input: &GpuTensor,
        activation: ops::FusedActivation,
    ) -> Result<GpuTensor> {
        let ndim = input.ndim();
        let in_features = input.shape()[ndim - 1];
        let batch: usize = input.shape()[..ndim - 1].iter().product();

        let input_2d = if ndim == 2 {
            input.clone_tensor()?
        } else {
            input.reshape(&[batch, in_features])?
        };

        // Use fused matmul (always non-prepadded path — fused kernel handles padding)
        let output = if let Some(ref bias) = self.bias {
            ops::matmul_fused(&input_2d, &self.weight_t, bias, activation, &self.registry)?
        } else {
            // No bias → fall back to unfused
            let out = ops::matmul(&input_2d, &self.weight_t, &self.registry)?;
            match activation {
                ops::FusedActivation::Gelu => return ops::gelu(&out, &self.registry),
                ops::FusedActivation::Relu => return ops::relu(&out, &self.registry),
            }
        };

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
        // V2 kernel: both A and B row-major, handles bounds internally (no pad/transpose)
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
    fn test_linear_gpt2_dims() {
        // Test with GPT-2-like dimensions: batch=5, in=768, out=768
        let registry = test_registry();
        let dev = registry.device();

        let batch = 5;
        let in_f = 768;
        let out_f = 768;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.0001)
            .collect();
        let bias: Vec<f32> = (0..out_f)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
            .collect();
        let input: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.01)
            .collect();

        let expected = cpu_linear(&input, &weight, Some(&bias), batch, in_f, out_f);

        let layer = Linear::new(&weight, Some(&bias), in_f, out_f, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[batch, in_f], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        assert_eq!(output_tensor.shape(), &[batch, out_f]);

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("GPT-2 dims Linear max error: {max_err}");
        assert!(
            max_err < 0.1,
            "max absolute error {max_err} exceeds 0.1 for GPT-2 dims"
        );
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
