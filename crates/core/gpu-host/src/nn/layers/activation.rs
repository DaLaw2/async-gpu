//! Activation layers: GELU, SiLU, Sigmoid, ReLU.
//!
//! These are zero-parameter Module wrappers around the activation ops.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// GELU activation layer.
pub struct GELU {
    registry: Arc<KernelRegistry>,
}

impl GELU {
    /// Create a new GELU activation layer.
    pub fn new(registry: &Arc<KernelRegistry>) -> Self {
        Self {
            registry: Arc::clone(registry),
        }
    }
}

impl Module for GELU {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::gelu(input, &self.registry)
    }
}

/// SiLU (Swish) activation layer.
pub struct SiLU {
    registry: Arc<KernelRegistry>,
}

impl SiLU {
    /// Create a new SiLU activation layer.
    pub fn new(registry: &Arc<KernelRegistry>) -> Self {
        Self {
            registry: Arc::clone(registry),
        }
    }
}

impl Module for SiLU {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::silu(input, &self.registry)
    }
}

/// Sigmoid activation layer.
pub struct Sigmoid {
    registry: Arc<KernelRegistry>,
}

impl Sigmoid {
    /// Create a new Sigmoid activation layer.
    pub fn new(registry: &Arc<KernelRegistry>) -> Self {
        Self {
            registry: Arc::clone(registry),
        }
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::sigmoid(input, &self.registry)
    }
}

/// ReLU activation layer.
pub struct ReLU {
    registry: Arc<KernelRegistry>,
}

impl ReLU {
    /// Create a new ReLU activation layer.
    pub fn new(registry: &Arc<KernelRegistry>) -> Self {
        Self {
            registry: Arc::clone(registry),
        }
    }
}

impl Module for ReLU {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::relu(input, &self.registry)
    }
}
