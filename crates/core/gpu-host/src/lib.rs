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
//! - [`GpuVec`] — High-level GPU buffer with zero-copy semantics (wraps MappedBuffer)
//! - [`MappedBuffer`] — Low-level RAII wrapper for CUDA pinned device-mapped memory
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
//! - `nn` — Neural network module: tensors, layers, ops, autograd, pre-built models (feature = `nn`)
//! - `async_rt` — Tokio async/await integration (feature = `async`)
//! - [`streams`] — CUDA stream overlap support

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod hostcall;
/// Low-level CUDA mapped memory allocation helpers.
///
/// Provides pinned device-mapped host memory allocation used by hostcall buffers,
/// command buffers, and other CUDA memory primitives.
///
/// Most users should use [`MappedBuffer`] instead of these raw allocation functions.
#[doc(hidden)]
pub mod mapped_mem;
pub mod memory;
pub mod resource_report;
pub mod runtime;
pub mod scheduler;
pub mod streams;

#[cfg(feature = "async")]
pub mod async_rt;

#[cfg(feature = "nn")]
pub mod nn;

#[cfg(all(feature = "gpt2", feature = "demo"))]
pub mod model;
#[cfg(all(feature = "gpt2", not(feature = "demo")))]
pub(crate) mod model;
#[cfg(all(feature = "gpt2", feature = "demo"))]
pub mod model_generic;
#[cfg(all(feature = "gpt2", not(feature = "demo")))]
pub(crate) mod model_generic;
#[cfg(all(feature = "gpt2", feature = "demo"))]
pub mod model_yolo;
#[cfg(all(feature = "gpt2", not(feature = "demo")))]
pub(crate) mod model_yolo;
#[cfg(all(feature = "gpt2", feature = "demo"))]
pub mod tokenizer;
#[cfg(all(feature = "gpt2", not(feature = "demo")))]
pub(crate) mod tokenizer;
#[cfg(all(feature = "gpt2", feature = "demo"))]
pub mod yolo_backbone;
#[cfg(all(feature = "gpt2", not(feature = "demo")))]
pub(crate) mod yolo_backbone;

#[cfg(feature = "onnx")]
#[allow(missing_docs)]
pub mod onnx_rt;

/// Backwards-compat re-export: `gpu_host::onnx` → `gpu_host::onnx_rt::proto`
#[cfg(feature = "onnx")]
pub use onnx_rt as onnx;

/// Backwards-compat re-export
#[cfg(all(feature = "onnx", feature = "nn"))]
pub use onnx_rt::executor as onnx_executor;

/// Backwards-compat re-export
#[cfg(feature = "onnx")]
pub use onnx_rt::fusion as onnx_fusion;

/// Embedded PTX sources for GPU kernels.
///
/// These are compiled from the various kernel crates and embedded at build time.
///
/// Per-crate PTX constants:
/// - `KERNEL_CORE`    — core kernels (basic ops, math helpers, infrastructure)
/// - `KERNEL_COMPUTE` — ML/compute kernels (GEMM, transformer, CNN, physics)
/// - `KERNEL_IO`      — I/O kernels (hostcall, pipeline, hybrid warp print)
/// - `KERNEL_TEST`    — test/demo kernels (std demos, warp tests, par_iter)
///
/// Backward-compatible aliases:
/// - `KERNEL`     → `KERNEL_COMPUTE` (the most-used module)
/// - `KERNEL_STD` → `KERNEL_TEST`    (test/demo kernels, formerly gpu-kernel-std)
#[doc(hidden)]
pub mod ptx {
    // ── Per-crate PTX (canonical) ──────────────────────────────
    /// Core kernels: basic ops, math helpers, infrastructure.
    pub const KERNEL_CORE: &str = include_str!("../kernel_core.ptx");
    /// ML/compute kernels: GEMM, transformer, CNN, physics, search, fused ops.
    pub const KERNEL_COMPUTE: &str = include_str!("../kernel_compute.ptx");
    /// I/O kernels: hostcall, pipeline, hybrid warp print.
    pub const KERNEL_IO: &str = include_str!("../kernel_io.ptx");
    /// Test/demo kernels: std demos, warp tests, thread tests, par_iter, SC demos.
    pub const KERNEL_TEST: &str = include_str!("../kernel_test.ptx");

    // ── Backward-compatible aliases ────────────────────────────
    /// Alias: `KERNEL` → `KERNEL_COMPUTE` (the largest, most-used module).
    ///
    /// All existing call sites that use `ptx::KERNEL` (KernelRegistry, gpu.rs,
    /// integration tests) reference ML/compute kernels, so this alias preserves
    /// behavior with zero code changes.
    pub const KERNEL: &str = KERNEL_COMPUTE;
    /// Alias: `KERNEL_STD` → `KERNEL_TEST` (test/demo kernels).
    ///
    /// The `#[gpu_test]` macro and gpu-test-harness use `KERNEL_STD` to launch
    /// test kernels. This alias keeps them working unchanged.
    pub const KERNEL_STD: &str = KERNEL_TEST;

    // ── Auto-discovery catalog ─────────────────────────────────
    /// A PTX module with a human-readable name, for auto-discovery APIs.
    pub struct PtxModule {
        /// Human-readable module name (e.g., "core", "compute").
        pub name: &'static str,
        /// PTX source string for this module.
        pub ptx: &'static str,
    }

    /// All PTX modules, for APIs that search across modules.
    ///
    /// Iterate over this to try loading a kernel from every available module.
    pub const ALL: &[PtxModule] = &[
        PtxModule {
            name: "core",
            ptx: KERNEL_CORE,
        },
        PtxModule {
            name: "compute",
            ptx: KERNEL_COMPUTE,
        },
        PtxModule {
            name: "io",
            ptx: KERNEL_IO,
        },
        PtxModule {
            name: "test",
            ptx: KERNEL_TEST,
        },
    ];

    // ── Legacy test PTX constants (unchanged) ──────────────────
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
}

// Convenience re-exports for common types.

/// One-liner GPU launch API — `gpu::run()`, `gpu::launch()`, `gpu::custom()`.
pub mod gpu;
pub use error::{GpuHostError, Result};
pub use hostcall::{HostcallBuffer, HostcallSession, Pipeline};
pub use memory::{GpuVec, MappedBuffer};
pub use runtime::GpuRuntime;

/// Returns the path to the repository-root `models/` directory.
///
/// Resolution order:
/// 1. `ASYNC_GPU_MODELS` environment variable (if set)
/// 2. Walk up from `start` (or current dir) until a `Cargo.toml` containing
///    `[workspace]` is found, then append `models/`.
///
/// # Example
///
/// ```no_run
/// // From any crate in the workspace:
/// let dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR")));
/// let gpt2 = dir.join("model.safetensors");
/// ```
#[doc(hidden)]
pub fn model_dir(start: Option<&str>) -> std::path::PathBuf {
    // 1. Env var override
    if let Ok(dir) = std::env::var("ASYNC_GPU_MODELS") {
        return std::path::PathBuf::from(dir);
    }

    // 2. Walk up to workspace root
    let start_path = match start {
        Some(s) => std::path::PathBuf::from(s),
        None => std::env::current_dir().unwrap_or_default(),
    };

    let mut dir = start_path.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
                // Look for the root workspace (has `members`), not standalone workspace stubs
                if contents.contains("[workspace]") && contents.contains("members") {
                    return dir.join("models");
                }
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }

    // Fallback: assume CWD is workspace root
    std::path::PathBuf::from("models")
}
