//! Composable neural network layers with the [`Module`] trait.
//!
//! Each layer owns its weights (uploaded once at construction) and implements
//! `forward()` for inference. Layers can be composed into models by calling
//! them in sequence.

pub mod activation;
pub mod attention;
pub mod conv;
pub mod embedding;
pub mod int4_linear;
pub mod linear;
pub mod lora;
pub mod norm;
pub mod pool;
pub mod sequential;

use super::error::Result;
use super::tensor::GpuTensor;

pub use activation::{ReLU, SiLU, Sigmoid, GELU};
pub use attention::MultiHeadAttention;
pub use conv::Conv2d;
pub use embedding::Embedding;
pub use int4_linear::Int4Linear;
pub use linear::{Activation, Linear};
pub use lora::LoraLinear;
pub use norm::{BatchNorm2d, LayerNorm};
pub use pool::MaxPool2d;
pub use sequential::Sequential;

/// Trait for neural network layers — PyTorch-like `forward()`.
///
/// All layers implement this trait so they can be composed uniformly.
/// Each `forward()` call takes an input tensor and returns an output tensor.
pub trait Module {
    /// Run the forward pass on the input tensor.
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor>;
}
