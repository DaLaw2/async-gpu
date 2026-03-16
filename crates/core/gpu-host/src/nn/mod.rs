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
pub mod cpu_ref;
pub mod error;
pub mod layers;
pub mod models;
pub mod ops;
pub mod registry;
pub mod tensor;
pub mod test_utils;

pub use error::{NnError, Result};
pub use layers::Module;
pub use registry::KernelRegistry;
pub use tensor::GpuTensor;
