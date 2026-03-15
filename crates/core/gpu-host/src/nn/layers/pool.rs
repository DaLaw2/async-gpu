//! Pooling layers: MaxPool2d.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// 2D max pooling layer.
pub struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    registry: Arc<KernelRegistry>,
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer.
    pub fn new(
        kernel_size: usize,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            registry: Arc::clone(registry),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            &self.registry,
        )
    }
}
