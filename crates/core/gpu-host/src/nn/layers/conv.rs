//! Conv2d layer: 2D convolution with weight ownership.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// 2D convolution layer.
///
/// Weight: `[C_out, C_in, kH, kW]`, bias: `[C_out]` (optional).
pub struct Conv2d {
    weight: GpuTensor,
    bias: Option<GpuTensor>,
    stride: usize,
    padding: usize,
    registry: Arc<KernelRegistry>,
}

impl Conv2d {
    /// Create a new Conv2d layer.
    ///
    /// `weight`: `[c_out, c_in, kh, kw]`, `bias`: `[c_out]` (optional).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        let weight_tensor = GpuTensor::from_host(weight, &[c_out, c_in, kh, kw], dev)?;
        let bias_tensor = if let Some(b) = bias {
            Some(GpuTensor::from_host(b, &[c_out], dev)?)
        } else {
            None
        };

        Ok(Self {
            weight: weight_tensor,
            bias: bias_tensor,
            stride,
            padding,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for Conv2d {
    /// Forward pass: input `[C_in, H, W]` → output `[C_out, H_out, W_out]`.
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            &self.registry,
        )
    }
}
