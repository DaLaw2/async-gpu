//! Linear (fully connected) layer: y = xW^T + b.
//!
//! Supports automatic fusion of the bias+activation epilogue via NVRTC codegen.
//! When [`Activation`] is specified, [`Linear::forward_auto_fused`] compiles and
//! caches a single fused kernel for `bias_add + activation`, eliminating extra
//! kernel launches compared to the unfused path.

use std::sync::Arc;

use crate::nn::autograd::OpKind;
use crate::nn::error::Result;
use crate::nn::fusion::FusionCodegen;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// Activation function to fuse with Linear's bias-add epilogue.
///
/// Richer than [`ops::FusedActivation`] — supports all elementwise activations
/// the fusion codegen engine can handle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    /// GELU (Gaussian Error Linear Unit).
    Gelu,
    /// ReLU (Rectified Linear Unit).
    Relu,
    /// SiLU (Sigmoid Linear Unit / Swish).
    Silu,
    /// Sigmoid.
    Sigmoid,
}

/// Linear layer: y = xW^T + b.
///
/// Weight: `[out_features, in_features]`, bias: `[out_features]` (optional).
/// Stores pre-transposed+padded weight for fast matmul (skip per-forward transpose).
///
/// # Auto-fusion
///
/// When calling [`forward_auto_fused`](Self::forward_auto_fused), the layer
/// uses [`FusionCodegen`] to JIT-compile a single fused kernel for the
/// `bias_add + activation` epilogue. The compiled kernel is cached, so only
/// the first invocation pays the NVRTC compilation cost.
pub struct Linear {
    weight_t: GpuTensor, // [in_features, out_features] — pre-transposed for matmul
    bias: Option<GpuTensor>,
    registry: Arc<KernelRegistry>,
    codegen: Arc<FusionCodegen>,
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
            codegen: Arc::new(FusionCodegen::new()),
        })
    }

    /// Create a new Linear layer with a shared [`FusionCodegen`] cache.
    ///
    /// Use this when multiple layers should share the same kernel cache
    /// (e.g., all layers in a transformer block).
    pub fn with_codegen(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        registry: &Arc<KernelRegistry>,
        codegen: &Arc<FusionCodegen>,
    ) -> Result<Self> {
        let dev = registry.device();

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
            codegen: Arc::clone(codegen),
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

    /// Auto-fused forward: matmul then NVRTC-fused bias+activation epilogue.
    ///
    /// After the matmul kernel produces the raw output, the bias addition and
    /// activation are compiled into a **single** fused NVRTC kernel and launched
    /// together. This eliminates the extra kernel launches that the unfused path
    /// (`forward()` + separate activation) would require.
    ///
    /// The fused kernel is JIT-compiled on first use and cached for subsequent
    /// calls with the same op chain and column count.
    ///
    /// # Arguments
    /// * `input` — `[*, in_features]` tensor.
    /// * `activation` — activation to apply after bias-add.
    ///
    /// # Returns
    /// Output tensor `[*, out_features]` = `activation(matmul(input, W^T) + bias)`.
    pub fn forward_auto_fused(
        &self,
        input: &GpuTensor,
        activation: Activation,
    ) -> Result<GpuTensor> {
        use cudarc::driver::LaunchAsync;

        let ndim = input.ndim();
        let in_features = input.shape()[ndim - 1];
        let batch: usize = input.shape()[..ndim - 1].iter().product();

        let input_2d = if ndim == 2 {
            input.clone_tensor()?
        } else {
            input.reshape(&[batch, in_features])?
        };

        // Step 1: matmul (unchanged — this is the GEMM kernel)
        let matmul_out = ops::matmul(&input_2d, &self.weight_t, &self.registry)?;

        // Step 2: fused bias+activation epilogue via FusionCodegen
        let out_features = self.weight_t.shape()[1];
        let n = matmul_out.numel();
        let dev = self.registry.device();

        let act_op = match activation {
            Activation::Gelu => OpKind::Gelu,
            Activation::Relu => OpKind::Relu,
            Activation::Silu => OpKind::Silu,
            Activation::Sigmoid => OpKind::Sigmoid,
        };

        let output = if let Some(ref bias) = self.bias {
            // Fuse: BiasAdd + Activation
            let ops_chain = [OpKind::BiasAdd, act_op];
            let n_cols_params = [out_features];

            let (module_name, func_name) =
                self.codegen
                    .get_or_compile(&ops_chain, &n_cols_params, dev)?;

            let cuda_func = dev.get_func(&module_name, &func_name).ok_or(
                crate::nn::NnError::KernelNotFound {
                    name: "fused_kernel",
                },
            )?;

            let mut output = GpuTensor::zeros(matmul_out.shape(), dev)?;

            let threads = 256u32;
            let total_threads = (n as u32).div_ceil(4);
            let grid = (total_threads.div_ceil(threads), 1, 1);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: grid,
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                cuda_func
                    .launch(
                        config,
                        (
                            matmul_out.data(),
                            output.data_mut(),
                            bias.data(),
                            out_features as u32,
                            n as u32,
                        ),
                    )
                    .map_err(crate::nn::NnError::Cuda)?;
            }

            output
        } else {
            // No bias — fuse just the activation (still saves nothing vs
            // a standalone activation kernel, but keeps the API uniform).
            // Fall back to the standard activation ops.
            match activation {
                Activation::Gelu => ops::gelu(&matmul_out, &self.registry)?,
                Activation::Relu => ops::relu(&matmul_out, &self.registry)?,
                Activation::Silu => ops::silu(&matmul_out, &self.registry)?,
                Activation::Sigmoid => ops::sigmoid(&matmul_out, &self.registry)?,
            }
        };

        // Reshape back to [..., out_features] if needed
        if ndim > 2 {
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

    /// CPU GELU approximation (tanh-based).
    fn cpu_gelu(x: f32) -> f32 {
        let s = 0.7978845608_f32;
        let c = 0.044715_f32;
        let inner = s * (x + c * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// CPU ReLU.
    fn cpu_relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// CPU Sigmoid.
    fn cpu_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU SiLU.
    fn cpu_silu(x: f32) -> f32 {
        x * cpu_sigmoid(x)
    }

    /// Apply a CPU activation function to a slice.
    fn apply_cpu_activation(data: &[f32], activation: Activation) -> Vec<f32> {
        data.iter()
            .map(|&x| match activation {
                Activation::Gelu => cpu_gelu(x),
                Activation::Relu => cpu_relu(x),
                Activation::Silu => cpu_silu(x),
                Activation::Sigmoid => cpu_sigmoid(x),
            })
            .collect()
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

    // -----------------------------------------------------------------------
    // Auto-fusion correctness tests
    // -----------------------------------------------------------------------

    /// Test auto-fused forward with a specific activation against CPU reference.
    #[cfg(feature = "cublas")]
    fn check_auto_fused_correctness(activation: Activation, tol: f32) {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 16;
        let in_f = 64;
        let out_f = 128;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let bias: Vec<f32> = (0..out_f)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
            .collect();
        let input: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.01)
            .collect();

        // CPU reference: linear + activation
        let linear_out = cpu_linear(&input, &weight, Some(&bias), batch, in_f, out_f);
        let expected = apply_cpu_activation(&linear_out, activation);

        // GPU: auto-fused
        let layer = Linear::new(&weight, Some(&bias), in_f, out_f, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[batch, in_f], dev).unwrap();
        let output_tensor = layer.forward_auto_fused(&input_tensor, activation).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        assert_eq!(output_tensor.shape(), &[batch, out_f]);

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "auto_fused {:?} max_err={max_err:.6e} (tol={tol:.0e})",
            activation
        );
        assert!(
            max_err < tol,
            "auto_fused {:?} max error {max_err} exceeds tolerance {tol}",
            activation
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_gelu() {
        check_auto_fused_correctness(Activation::Gelu, 5e-3);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_relu() {
        check_auto_fused_correctness(Activation::Relu, 5e-3);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_silu() {
        check_auto_fused_correctness(Activation::Silu, 5e-3);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_sigmoid() {
        check_auto_fused_correctness(Activation::Sigmoid, 5e-3);
    }

    /// Verify fused and unfused paths produce bitwise-close results.
    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_matches_unfused() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 8;
        let in_f = 128;
        let out_f = 256;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.001)
            .collect();
        let bias: Vec<f32> = (0..out_f)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
            .collect();
        let input_data: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.01)
            .collect();

        let layer = Linear::new(&weight, Some(&bias), in_f, out_f, &registry).unwrap();

        for activation in [
            Activation::Gelu,
            Activation::Relu,
            Activation::Silu,
            Activation::Sigmoid,
        ] {
            let input_tensor = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();

            // Unfused path: forward() + separate activation
            let unfused_linear = layer.forward(&input_tensor).unwrap();
            let unfused_out = match activation {
                Activation::Gelu => ops::gelu(&unfused_linear, &registry).unwrap(),
                Activation::Relu => ops::relu(&unfused_linear, &registry).unwrap(),
                Activation::Silu => ops::silu(&unfused_linear, &registry).unwrap(),
                Activation::Sigmoid => ops::sigmoid(&unfused_linear, &registry).unwrap(),
            };
            let unfused = unfused_out.to_host().unwrap();

            // Auto-fused path
            let input_tensor = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
            let fused_out = layer.forward_auto_fused(&input_tensor, activation).unwrap();
            let fused = fused_out.to_host().unwrap();

            let max_err: f32 = unfused
                .iter()
                .zip(fused.iter())
                .map(|(u, f)| (u - f).abs())
                .fold(0.0f32, f32::max);
            eprintln!("fused vs unfused {:?}: max_err={max_err:.6e}", activation);
            // Both paths use the same matmul but different bias+activation
            // implementations (PTX kernel vs NVRTC codegen), so small numerical
            // differences are expected — especially for GELU (tanh approximation).
            assert!(
                max_err < 1e-2,
                "fused vs unfused {:?} mismatch: max_err={max_err}",
                activation
            );
        }
    }

    /// Test auto-fused forward with GPT-2-like dimensions (batch=128, 768→3072→768).
    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_gpt2_dims() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 128;
        let in_f = 768;
        let out_f = 3072;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.00005)
            .collect();
        let bias: Vec<f32> = (0..out_f)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
            .collect();
        let input_data: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.01)
            .collect();

        let layer = Linear::new(&weight, Some(&bias), in_f, out_f, &registry).unwrap();
        let input_tensor = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
        let output = layer
            .forward_auto_fused(&input_tensor, Activation::Gelu)
            .unwrap();

        assert_eq!(output.shape(), &[batch, out_f]);

        // Spot-check: values should be reasonable (GELU output in [-0.17, +inf))
        let host = output.to_host().unwrap();
        let has_nonzero = host.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "output is all zeros — fusion likely broken");
        let all_finite = host.iter().all(|x| x.is_finite());
        assert!(all_finite, "output contains NaN or Inf");
    }

    /// Test shared FusionCodegen via `with_codegen`.
    #[test]
    #[cfg(feature = "cublas")]
    fn test_shared_codegen_cache() {
        let registry = test_registry();
        let dev = registry.device();
        let codegen = Arc::new(FusionCodegen::new());

        let batch = 4;
        let in_f = 32;
        let out_f = 64;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 41) as f32 - 20.0) * 0.01)
            .collect();
        let bias: Vec<f32> = (0..out_f).map(|i| i as f32 * 0.01).collect();
        let input_data: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
            .collect();

        // Two layers sharing the same codegen cache
        let layer1 =
            Linear::with_codegen(&weight, Some(&bias), in_f, out_f, &registry, &codegen).unwrap();
        let layer2 =
            Linear::with_codegen(&weight, Some(&bias), in_f, out_f, &registry, &codegen).unwrap();

        let input_tensor = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
        let out1 = layer1
            .forward_auto_fused(&input_tensor, Activation::Gelu)
            .unwrap();

        let input_tensor = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
        let out2 = layer2
            .forward_auto_fused(&input_tensor, Activation::Gelu)
            .unwrap();

        let h1 = out1.to_host().unwrap();
        let h2 = out2.to_host().unwrap();
        assert_eq!(h1.len(), h2.len());
        let max_diff: f32 = h1
            .iter()
            .zip(h2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "shared codegen cache mismatch: {max_diff}");
    }

    // -----------------------------------------------------------------------
    // Benchmark: auto-fused vs unfused forward
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "cublas")]
    fn test_auto_fused_benchmark() {
        let registry = test_registry();
        let dev = registry.device();

        // GPT-2 FFN up-projection: [128, 768] → [128, 3072] + bias + GELU
        let batch = 128;
        let in_f = 768;
        let out_f = 3072;

        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.00005)
            .collect();
        let bias_data: Vec<f32> = (0..out_f)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
            .collect();
        let input_data: Vec<f32> = (0..batch * in_f)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.01)
            .collect();

        let layer = Linear::new(&weight, Some(&bias_data), in_f, out_f, &registry).unwrap();

        // Warm up both paths
        for _ in 0..3 {
            let t = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
            let _ = layer.forward_auto_fused(&t, Activation::Gelu).unwrap();
            dev.synchronize().unwrap();
        }
        for _ in 0..3 {
            let t = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
            let out = layer.forward(&t).unwrap();
            let _ = ops::gelu(&out, &registry).unwrap();
            dev.synchronize().unwrap();
        }

        // Benchmark: unfused (forward + gelu = matmul + bias_add + gelu = 3 kernels)
        let iters = 50;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let t = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
            let out = layer.forward(&t).unwrap();
            let _ = ops::gelu(&out, &registry).unwrap();
        }
        dev.synchronize().unwrap();
        let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

        // Benchmark: auto-fused (matmul + fused(bias+gelu) = 2 kernels)
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let t = GpuTensor::from_host(&input_data, &[batch, in_f], dev).unwrap();
            let _ = layer.forward_auto_fused(&t, Activation::Gelu).unwrap();
        }
        dev.synchronize().unwrap();
        let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

        let speedup = unfused_us / fused_us;
        eprintln!(
            "\n=== Linear Auto-Fused Benchmark ([{batch},{in_f}]→[{batch},{out_f}] + GELU) ===\n\
             Unfused (3 kernels): {unfused_us:.1} us/iter\n\
             Fused   (2 kernels): {fused_us:.1} us/iter\n\
             Speedup: {speedup:.2}x\n"
        );

        // We save 1 kernel launch (bias+gelu fused into 1 instead of 2).
        // The matmul dominates for large matrices, so expect modest speedup.
        // But the epilogue fusion still matters, especially for smaller batch sizes.
        // We don't assert a hard speedup threshold here because the matmul
        // dominates — the real win shows in the epilogue-only benchmark above.
    }
}
