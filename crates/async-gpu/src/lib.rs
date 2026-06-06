//! async-gpu — Async/Await on GPU.
//!
//! Write GPU programs using familiar Rust patterns: `gpu::run()` for one-liner
//! kernel launches, `gpu::custom()` for builder-style configuration, and
//! `GpuRuntime` for full control.
//!
//! # Quick Start
//!
//! ```no_run
//! use async_gpu::gpu;
//!
//! fn main() -> async_gpu::Result<()> {
//!     // Run a hostcall-enabled kernel (supports println!, file I/O)
//!     gpu::run("my_kernel")?;
//!
//!     // Pure compute with output buffer
//!     let result: Vec<u32> = gpu::launch("compute_kernel", 1024, 256)?;
//!
//!     // Builder API for custom signatures
//!     let ctx = gpu::custom("my_kernel")
//!         .threads(256)
//!         .elements(1024)
//!         .hostcall()
//!         .prepare()?;
//!     let input = ctx.upload(&[1.0f32; 1024])?;
//!     let mut output = ctx.alloc_zeros::<f32>(1024)?;
//!     let result = unsafe { ctx.launch((ctx.hostcall_ptr(), &input, &mut output)) }?;
//!     let data = result.download(&output)?;
//!     Ok(())
//! }
//! ```
//!
//! # Advanced Usage
//!
//! For full control over CUDA device management, PTX loading, and kernel
//! launch configuration:
//!
//! ```no_run
//! use async_gpu::{GpuRuntime, HostcallSession, MappedBuffer};
//! ```
//!
//! # Feature Flags
//!
//! - `nn` — Neural network module: tensors, layers, ops, autograd, pre-built models
//! - `async` — Tokio integration: `AsyncGpuRuntime`, `GpuTask`, async kernel launch

#![warn(missing_docs)]

// ============================================================
// Core re-exports — always available
// ============================================================

/// One-liner GPU launch API.
///
/// - `gpu::run("kernel")` — hostcall-enabled kernel (println!, file I/O)
/// - `gpu::run_with_output("kernel", n)` — hostcall + output buffer
/// - `gpu::launch("kernel", n, threads)` — pure compute, output only
/// - `gpu::custom("kernel")` — builder API for custom signatures
pub use gpu_host::gpu;

// Error types
pub use gpu_host::error::error_category_name;
pub use gpu_host::error::GpuHostError;
pub use gpu_host::error::GpuKernelErrorInfo;

pub use gpu_host::Result;

// Runtime & session types — for users who need full control
pub use gpu_host::GpuRuntime;
pub use gpu_host::GpuVec;
pub use gpu_host::HostcallBuffer;
pub use gpu_host::HostcallSession;
pub use gpu_host::MappedBuffer;
pub use gpu_host::Pipeline;

// Scheduler types — unified CPU/GPU work routing
pub use gpu_host::scheduler::AutoScheduler;
pub use gpu_host::scheduler::CpuScheduler;
pub use gpu_host::scheduler::GpuScheduler;
pub use gpu_host::scheduler::Scheduler;

// Hostcall error type — returned by HostcallBuffer/HostcallSession/Pipeline constructors
pub use gpu_host::hostcall::HostcallError;

// Debugging tools
pub use gpu_host::hostcall::FlightRecorder;

pub use gpu_host::streams::GpuStream;

/// Embedded PTX sources for GPU kernels.
#[doc(hidden)]
pub use gpu_host::ptx;

/// Returns the path to the workspace `models/` directory.
#[doc(hidden)]
pub use gpu_host::model_dir;

// ============================================================
// Feature-gated re-exports
// ============================================================

/// Neural network module — tensors, layers, ops, autograd, pre-built models.
///
/// Requires the `nn` feature flag:
/// ```toml
/// [dependencies]
/// async-gpu = { path = "...", features = ["nn"] }
/// ```
#[cfg(feature = "nn")]
pub use gpu_host::nn;

/// Async/await integration for GPU runtime (requires tokio).
///
/// Requires the `async` feature flag:
/// ```toml
/// [dependencies]
/// async-gpu = { path = "...", features = ["async"] }
/// ```
#[cfg(feature = "async")]
pub use gpu_host::async_rt;

// Convenience re-exports of the most-used async types at root level,
// so users can write `use async_gpu::AsyncGpuRuntime` directly.
#[cfg(feature = "async")]
pub use gpu_host::async_rt::AsyncGpuRuntime;
#[cfg(feature = "async")]
pub use gpu_host::async_rt::AsyncHostcallSession;
#[cfg(feature = "async")]
pub use gpu_host::async_rt::GpuTask;
#[cfg(feature = "async")]
pub use gpu_host::async_rt::HostcallEvent;
