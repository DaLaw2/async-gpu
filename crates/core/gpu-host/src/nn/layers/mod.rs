//! Composable neural network layers with the [`Module`] trait.
//!
//! Each layer owns its weights (uploaded once at construction) and implements
//! `forward()` for inference. Layers can be composed into models by calling
//! them in sequence.
//!
//! # Example
//!
//! ```no_run
//! # use gpu_host::nn::layers::Module;
//! # use gpu_host::nn::GpuTensor;
//! // Any type implementing Module can be used:
//! // let output = layer.forward(&input)?;
//! ```

pub mod sequential;

use super::error::Result;
use super::tensor::GpuTensor;

pub use sequential::Sequential;

/// Trait for neural network layers — PyTorch-like `forward()`.
///
/// All layers implement this trait so they can be composed uniformly.
/// Each `forward()` call takes an input tensor and returns an output tensor.
pub trait Module {
    /// Run the forward pass on the input tensor.
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor>;
}
