//! Sequential container — chains multiple modules in order.

use super::Module;
use crate::nn::error::Result;
use crate::nn::tensor::GpuTensor;

/// Sequential container that runs modules in order.
///
/// Each module's output becomes the next module's input.
///
/// # Example
///
/// ```no_run
/// # use gpu_host::nn::layers::{Module, Sequential};
/// # use gpu_host::nn::GpuTensor;
/// let seq = Sequential::new(vec![
///     // Box::new(linear),
///     // Box::new(gelu),
/// ]);
/// // let output = seq.forward(&input)?;
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create a new sequential container from a list of boxed modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let mut x = input.clone_tensor()?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}
