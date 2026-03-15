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
