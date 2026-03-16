//! Pre-built model architectures using the nn API.
//!
//! Each model provides a config struct, a model struct with `from_weights()`,
//! and a `forward()` method for inference.

#[cfg(feature = "gpt2")]
pub mod gpt2;
pub mod resnet;
#[cfg(feature = "gpt2")]
pub mod yolov8;
