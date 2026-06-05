//! Neural network module — reusable compute API for GPU inference.
//!
//! Provides [`GpuTensor`] for N-dimensional device tensors, a [`Module`] trait
//! for composable layers, and stateless operation functions wrapping GPU kernels.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌───────────────────┐
//! │  GpuTensor   │────▶│  KernelRegistry   │
//! │  (N-dim f32) │     │  (auto launch)    │
//! └──────┬──────┘     └────────┬──────────┘
//!        │                     │
//!        ▼                     ▼
//! ┌─────────────┐     ┌───────────────────┐
//! │  ops::*      │────▶│  layers::*        │
//! │  (stateless) │     │  (Module trait)   │
//! └─────────────┘     └───────────────────┘
//! ```
//!
//! # Feature gate
//!
//! This module is gated behind the `nn` feature flag:
//!
//! ```toml
//! [dependencies]
//! gpu-host = { path = "...", features = ["nn"] }
//! ```

pub mod autograd;
/// CPU f64 reference implementations for numerical verification.
///
/// Internal testing utility — compiled only for tests or with the `demo` feature.
#[cfg(any(test, feature = "demo"))]
pub mod cpu_ref;
pub mod error;
pub mod fusion;
pub mod layers;
/// Pre-built model architectures (GPT-2, ResNet, YOLOv8).
///
/// These are demo/showcase models, not part of the stable public API.
/// Compiled only for tests or with the `demo` feature.
#[cfg(any(test, feature = "demo"))]
pub mod models;
pub mod ops;
pub mod registry;
pub mod tensor;
/// Numerical comparison utilities for GPU vs CPU testing.
///
/// Internal testing utility — compiled only for tests or with the `demo` feature.
#[cfg(any(test, feature = "demo"))]
pub mod test_utils;

pub use error::{NnError, Result};
pub use layers::Module;
pub use registry::KernelRegistry;
pub use tensor::GpuTensor;
