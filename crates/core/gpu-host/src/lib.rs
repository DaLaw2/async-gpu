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
///
/// Provides pinned device-mapped host memory allocation used by hostcall buffers,
/// command buffers, and other CUDA memory primitives.
pub mod mapped_mem;
pub mod memory;
pub mod runtime;
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
/// `KERNEL` and `KERNEL_STD` now point to the same unified PTX — the former
/// gpu-kernel and gpu-kernel-std crates have been merged into a single
/// gpu-kernel-std crate.
#[doc(hidden)]
pub mod ptx {
    /// Unified GPU kernel PTX (from crates/gpu-kernel-std).
    ///
    /// Contains all kernel entry points: compute, hostcall, warp, thread,
    /// and std-based kernels (println!, Vec, File I/O, HashMap, thread::spawn).
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
    /// Alias for `KERNEL` — kept for backward compatibility.
    ///
    /// After the kernel crate merge, this is the same PTX as `KERNEL`.
    pub const KERNEL_STD: &str = include_str!("../kernel_std.ptx");
}

/// Pre-compiled cubin binaries for fast module loading.
///
/// Loading cubin skips the CUDA JIT compilation step, reducing kernel load
/// time from 10+ minutes (for the unified 9MB PTX) to under 1 second.
///
/// Cubin is generated by `scripts/build-kernel-std.sh` via `ptxas`.
/// If the cubin file does not exist, `KERNEL_CUBIN` will be an empty slice.
#[doc(hidden)]
pub mod cubin {
    /// Pre-compiled cubin for the unified kernel (sm_75).
    ///
    /// Use with `gpu::load_module_cubin()` for instant module loading.
    /// Falls back to PTX JIT if empty.
    pub const KERNEL_CUBIN: &[u8] = include_bytes!("../kernel_std.cubin");
}

// Convenience re-exports for common types.
pub mod gpu;
pub use error::{GpuHostError, Result};
pub use hostcall::{HostcallBuffer, HostcallSession, Pipeline};
pub use memory::MappedBuffer;
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
