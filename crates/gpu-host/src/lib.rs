//! gpu-host — host-side GPU runtime library.
//!
//! Provides a high-level SDK for GPU kernel management, hostcall communication,
//! and pinned memory allocation.
//!
//! # Core modules
//! - [`runtime`] — [`GpuRuntime`] for device init, PTX loading, kernel launch
//! - [`memory`] — [`MappedBuffer`] for RAII pinned device-mapped memory
//! - [`hostcall`] — [`HostcallBuffer`] for GPU-host RPC communication
//! - [`error`] — Error types ([`GpuHostError`])
//!
//! # Optional modules (feature-gated)
//! - `model` — GPT-2 weight loading from safetensors (feature = `gpt2`)
//! - `tokenizer` — GPT-2 BPE tokenizer (feature = `gpt2`)

#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod hostcall;
pub mod memory;
pub mod runtime;

#[cfg(feature = "gpt2")]
pub mod model;
#[cfg(feature = "gpt2")]
pub mod tokenizer;

// Convenience re-exports for common types.
pub use error::{GpuHostError, Result};
pub use hostcall::HostcallBuffer;
pub use memory::MappedBuffer;
pub use runtime::GpuRuntime;
