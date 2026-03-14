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

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod hostcall;
/// Low-level CUDA mapped memory allocation helpers.
pub mod mapped_mem;
pub mod memory;
pub mod runtime;

#[cfg(feature = "async")]
pub mod async_rt;

#[cfg(feature = "gpt2")]
pub mod model;
#[cfg(feature = "gpt2")]
pub mod tokenizer;

/// Embedded PTX sources for GPU kernels.
///
/// These are compiled from the various kernel crates and embedded at build time.
pub mod ptx {
    /// Main GPU kernel PTX (from crates/gpu-kernel).
    pub const KERNEL: &str = include_str!("../kernel.ptx");
    /// Embassy async/await test PTX (from crates/embassy-test).
    pub const EMBASSY_TEST: &str = include_str!("../embassy_test.ptx");
    /// Async hostcall test PTX (from crates/async-hostcall-test).
    pub const ASYNC_HOSTCALL_TEST: &str = include_str!("../async_hostcall_test.ptx");
    /// Std-build-test PTX (from crates/std-build-test, -Zbuild-std=std).
    pub const STD_BUILD_TEST: &str = include_str!("../std_build_test.ptx");
    /// Async pipeline test PTX (from crates/async-pipeline-test).
    pub const ASYNC_PIPELINE_TEST: &str = include_str!("../async_pipeline_test.ptx");
    /// Multi-warp scaling test PTX (from crates/multi-warp-test).
    pub const MULTI_WARP_TEST: &str = include_str!("../multi_warp_test.ptx");
    /// Kernel-std PTX (from crates/gpu-kernel-std, -Zbuild-std=std).
    pub const KERNEL_STD: &str = include_str!("../kernel_std.ptx");
}

// Convenience re-exports for common types.
pub use error::{GpuHostError, Result};
pub use hostcall::HostcallBuffer;
pub use memory::MappedBuffer;
pub use runtime::GpuRuntime;
