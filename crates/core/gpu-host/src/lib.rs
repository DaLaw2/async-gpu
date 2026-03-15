//! gpu-host — host-side GPU runtime library.
//!
//! This crate provides the host-side SDK for launching GPU kernels and managing
//! GPU-host communication via the hostcall protocol. It handles CUDA device
//! initialization, PTX module loading, pinned memory allocation, and the
//! hostcall listener that services GPU requests (print, file I/O, networking).
//!
//! # Typical usage pattern
//!
//! ```text
//! 1. Create a GpuRuntime           → initialize CUDA device
//! 2. Load PTX module               → register kernel functions
//! 3. Allocate HostcallBuffer       → shared memory for GPU-host RPC
//! 4. Start HostcallSession         → spawn listener thread
//! 5. Launch kernel                 → pass hostcall dev_ptr as kernel arg
//! 6. Synchronize + collect results → read back from mapped memory
//! 7. Shutdown session              → stop listener, close file handles
//! ```
//!
//! # Example
//!
//! ```no_run
//! use gpu_host::{GpuRuntime, ptx};
//! use gpu_host::hostcall::HostcallSession;
//!
//! // 1. Init GPU
//! let rt = GpuRuntime::new(0).expect("CUDA init");
//!
//! // 2. Load PTX
//! rt.load_ptx(ptx::KERNEL, "kernel", &["my_kernel"]).expect("PTX load");
//!
//! // 3-4. Start hostcall session (allocates buffer + spawns listener)
//! let session = HostcallSession::start(64).expect("hostcall init");
//!
//! // 5. Launch kernel with hostcall pointer
//! let dev = rt.device();
//! let f = dev.get_func("kernel", "my_kernel").unwrap();
//! let cfg = cudarc::driver::LaunchConfig::for_num_elems(32);
//! unsafe { cudarc::driver::LaunchAsync::launch(f, cfg, (session.dev_ptr(),)) }
//!     .expect("launch");
//!
//! // 6. Wait for completion
//! dev.synchronize().expect("sync");
//!
//! // 7. Shutdown
//! session.shutdown();
//! ```
//!
//! # Key types
//!
//! - [`GpuRuntime`] — CUDA device wrapper (init, PTX loading, kernel launch, memory ops)
//! - [`HostcallBuffer`] — Shared pinned memory packet pool for GPU-host RPC
//! - [`hostcall::HostcallSession`] — Persistent hostcall listener across kernel launches
//! - [`MappedBuffer`] — RAII wrapper for CUDA pinned device-mapped memory
//! - [`hostcall::Pipeline`] — Multi-stage kernel pipeline with automatic packet reinit
//! - [`hostcall::FlightRecorder`] — Mapped-memory ring buffer for post-mortem tracing
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
pub mod streams;

#[cfg(feature = "async")]
pub mod async_rt;

#[cfg(feature = "nn")]
pub mod nn;

#[cfg(feature = "gpt2")]
pub mod model;
#[cfg(feature = "gpt2")]
pub mod model_yolo;
#[cfg(feature = "gpt2")]
pub mod tokenizer;
#[cfg(feature = "gpt2")]
pub mod yolo_backbone;

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
pub use hostcall::{HostcallBuffer, HostcallSession, Pipeline};
pub use memory::MappedBuffer;
pub use runtime::GpuRuntime;
