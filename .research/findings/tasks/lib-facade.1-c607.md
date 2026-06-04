# lib-facade.1 — Design: async-gpu facade crate structure and re-export surface

**Task**: Investigation — design `crates/async-gpu/` facade crate  
**Status**: done  
**Cycle**: 607  

---

## 1. Current State: gpu-host's Public API Surface

### Core types (always available)
| Module | Type | Role |
|--------|------|------|
| `error` | `GpuHostError`, `GpuKernelErrorInfo`, `Result<T>` | Top-level error type |
| `error` | `check_kernel_result()`, `error_category_name()` | Error helpers |
| `gpu` | `run()`, `run_with_output()`, `launch()` | One-liner launch API |
| `gpu` | `custom()` → `CustomLaunchBuilder` → `GpuContext` → `GpuResult` | Builder launch API |
| `runtime` | `GpuRuntime` | Low-level CUDA device wrapper |
| `memory` | `MappedBuffer<T>` | RAII pinned device-mapped memory |
| `hostcall` | `HostcallSession`, `HostcallBuffer`, `Pipeline` | GPU-host RPC management |
| `hostcall` | `CommandBuffer`, `Command`, `FlightRecorder` | Advanced hostcall features |
| `hostcall` | `HostcallError` | Hostcall-specific errors |
| `hostcall` | `StdinSource`, `RealStdin`, `CannedStdin` | Stdin abstraction |
| `streams` | `GpuStream` | CUDA stream overlap |
| `mapped_mem` | `alloc_mapped_u32()`, `free_mapped_mem()`, `alloc_mapped_result_array()` | Raw CUDA allocation helpers |
| `ptx` | `KERNEL`, `EMBASSY_TEST`, etc. (7 consts) | Embedded PTX binaries |
| root | `model_dir()` | Model file path resolver |

### Feature-gated: `nn`
| Module | Type | Role |
|--------|------|------|
| `nn` | `GpuTensor`, `KernelRegistry`, `Module` trait | Core nn types |
| `nn::error` | `NnError`, `Result<T>` | nn error types |
| `nn::ops` | stateless operation functions | GPU kernel wrappers |
| `nn::layers` | `Linear`, `Conv2D`, `LayerNorm`, `Attention`, etc. | Composable layers |
| `nn::autograd` | `Tape`, `Variable`, optimizers, loss fns | Autograd engine |
| `nn::models` | `Gpt2`, `ResNet`, `YoloV8` | Pre-built models |

### Feature-gated: `async` (tokio integration)
| Module | Type | Role |
|--------|------|------|
| `async_rt` | `AsyncGpuRuntime` | Tokio-compatible GPU runtime |
| `async_rt` | `AsyncHostcallSession` | Async hostcall listener |
| `async_rt` | `GpuTask` | Async kernel launch orchestrator |
| `async_rt` | `HostcallEvent` | Typed GPU events |

### Feature-gated: `gpt2` (model loading)
| Module | Type | Role |
|--------|------|------|
| `model` | GPT-2 weight loading | Safetensors weight loader |
| `model_generic` | Generic model loader | Model loading abstraction |
| `model_yolo` | YOLO model loader | YOLO-specific loader |
| `tokenizer` | BPE tokenizer | GPT-2 tokenizer |
| `yolo_backbone` | YOLO backbone | YOLO feature extraction |

### Feature-gated: `onnx` (ONNX runtime)
| Module | Type | Role |
|--------|------|------|
| `onnx_rt` | `proto`, `executor`, `fusion` | ONNX model execution |

---

## 2. Design Decision: What Goes In vs. What Stays Out

### Principle: The facade exposes the USER-FACING API only

Users of async-gpu fall into two tiers:
1. **Normal users**: `use async_gpu::gpu;` then `gpu::run()`, `gpu::launch()`, `gpu::custom()`
2. **Advanced users**: Need `GpuRuntime`, `HostcallSession`, `MappedBuffer` for custom setups

### IN (re-exported by async-gpu)

```
async_gpu::gpu              — The gpu module (run, launch, custom, etc.)
async_gpu::GpuHostError     — Primary error type
async_gpu::GpuKernelErrorInfo — Kernel error details
async_gpu::Result           — Result<T, GpuHostError>
async_gpu::GpuRuntime       — Low-level device wrapper (advanced)
async_gpu::HostcallSession  — Persistent hostcall listener (advanced)
async_gpu::MappedBuffer     — RAII pinned memory (advanced)
async_gpu::Pipeline         — Multi-stage kernel pipeline (advanced)
async_gpu::GpuStream        — CUDA stream overlap (advanced)
async_gpu::model_dir        — Model file path helper

// Feature-gated:
async_gpu::nn               — Full nn module (feature = "nn")
async_gpu::async_rt         — Tokio integration (feature = "async")
```

### OUT (stay in gpu-host only, not re-exported)

| What | Why |
|------|-----|
| `mapped_mem` module | Raw CUDA allocation helpers; `MappedBuffer` covers the safe API |
| `ptx` module | Embedded PTX binaries are implementation detail; users with custom kernels use `gpu::custom().ptx()` |
| `model`, `model_generic`, `model_yolo` | Model-specific loading internals; users access via `nn::models` |
| `tokenizer`, `yolo_backbone` | Model pipeline internals |
| `onnx_rt` | Low-level ONNX proto/executor; users access via `nn` ops |
| `HostcallBuffer` | Internal buffer management; `HostcallSession` is the public API |
| `CommandBuffer`, `Command` | Host→GPU command protocol internals |
| `FlightRecorder` | Debug tracing internals |
| `StdinSource`, `RealStdin`, `CannedStdin` | Stdin abstraction for test harness |
| `HostcallError` | Internal error; wrapped by `GpuHostError::Hostcall` |
| `check_kernel_result`, `error_category_name` | Internal error utilities |

### Rationale

- `ptx` is excluded because the facade user never loads PTX manually — `gpu::run()` and `gpu::custom()` handle it. Users with their own PTX use `gpu::custom().ptx(include_str!("my.ptx"))`.
- `HostcallBuffer` is excluded because `HostcallSession::start(n)` is the only public API users need. The buffer is an implementation detail.
- `model*`, `tokenizer`, `yolo_backbone` are excluded because they are model-pipeline internals. The `nn::models` module provides the user-facing API.
- `onnx_rt` is excluded because it's a low-level proto/executor layer. Future work should route ONNX through the nn module.
- `mapped_mem` is excluded because `MappedBuffer<T>` (from `memory`) provides the safe typed API. The raw `alloc_mapped_u32` / `free_mapped_mem` are for internal use.

---

## 3. Cargo.toml

```toml
[package]
name = "async-gpu"
version = "0.1.0"
edition = "2021"
description = "Async/Await on GPU — write Rust GPU programs with familiar APIs"

[features]
default = []
nn = ["gpu-host/nn"]
async = ["gpu-host/async"]

[dependencies]
gpu-host = { path = "../core/gpu-host" }
```

Note:
- `gpt2`, `onnx`, `cublas` features are NOT forwarded — they are internal to gpu-host.
- The `nn` feature gates the entire nn module re-export.
- The `async` feature gates the tokio integration re-export.
- No new dependencies introduced; the facade is pure re-exports.

---

## 4. lib.rs

```rust
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
//!     let result: Vec<f32> = gpu::launch("compute_kernel", 1024, 256)?;
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
pub use gpu_host::error::GpuHostError;
pub use gpu_host::error::GpuKernelErrorInfo;

/// Convenience type alias: `Result<T, GpuHostError>`.
pub use gpu_host::Result;

// Advanced types — for users who need full control
pub use gpu_host::GpuRuntime;
pub use gpu_host::HostcallSession;
pub use gpu_host::MappedBuffer;
pub use gpu_host::Pipeline;
pub use gpu_host::streams::GpuStream;

/// Returns the path to the workspace `models/` directory.
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
```

---

## 5. Feature Flag Mapping

| async-gpu feature | gpu-host feature | What it enables |
|-------------------|-----------------|-----------------|
| (none/default) | (none) | `gpu::*`, error types, `GpuRuntime`, `HostcallSession`, `MappedBuffer`, `Pipeline`, `GpuStream` |
| `nn` | `nn` | `async_gpu::nn` — tensors, layers, ops, autograd, models |
| `async` | `async` | `async_gpu::async_rt` — `AsyncGpuRuntime`, `GpuTask`, `AsyncHostcallSession` |

Features NOT forwarded (internal to gpu-host):
- `gpt2` — model loading internals (safetensors, tiktoken)
- `onnx` — ONNX proto/executor internals
- `cublas` — cuBLAS acceleration for nn ops (could be forwarded later if needed)

---

## 6. Workspace Integration

The facade crate should be added to the root workspace:

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/async-gpu",          # <-- NEW
    "crates/core/gpu-host",
    "crates/core/gpu-protocol",
    "crates/macro/warp-macro",
    "crates/test/gpu-test-harness",
]
```

Crate location: `crates/async-gpu/` (top-level under crates/, not under core/).
This reflects its role as the user-facing facade, not an internal implementation crate.

---

## 7. User-Facing Import Patterns

### Tier 1: Simple usage
```rust
use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    gpu::run("hello_world")?;
    Ok(())
}
```

### Tier 2: Compute with output
```rust
use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    let result: Vec<f32> = gpu::launch("matmul", 1024, 256)?;
    println!("result[0] = {}", result[0]);
    Ok(())
}
```

### Tier 3: Builder API
```rust
use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    let ctx = gpu::custom("my_kernel")
        .ptx(include_str!("my_kernel.ptx"))
        .threads(256)
        .elements(4096)
        .hostcall()
        .prepare()?;

    let input = ctx.upload(&data)?;
    let mut output = ctx.alloc_zeros::<f32>(4096)?;
    let result = unsafe { ctx.launch((ctx.hostcall_ptr(), &input, &mut output)) }?;
    let host_data = result.download(&output)?;
    Ok(())
}
```

### Tier 4: Advanced (full control)
```rust
use async_gpu::{GpuRuntime, HostcallSession, MappedBuffer};

fn main() -> async_gpu::Result<()> {
    let rt = GpuRuntime::new(0)?;
    rt.load_ptx(include_str!("kernel.ptx"), "mod", &["my_kernel"])?;
    let session = HostcallSession::start(64)?;
    // ... manual launch ...
    Ok(())
}
```

### Tier 5: nn module
```rust
use async_gpu::nn::{GpuTensor, KernelRegistry, Module};

fn main() -> async_gpu::Result<()> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let tensor = GpuTensor::from_host(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &dev)?;
    // ...
    Ok(())
}
```

### Tier 6: Tokio async
```rust
use async_gpu::async_rt::{AsyncGpuRuntime, GpuTask};

#[tokio::main]
async fn main() -> async_gpu::Result<()> {
    let rt = AsyncGpuRuntime::new(0)?;
    // ...
    Ok(())
}
```

---

## 8. Open Questions for lib-facade.2 (Implementation)

1. **gpu-runtime re-export**: Should the facade also re-export `gpu-runtime` for kernel-side
   code? Decision: NO. The kernel-side crate (`gpu-runtime`) targets `nvptx64` and cannot
   be compiled for the host. Users depend on `gpu-runtime` separately in their kernel crates.
   The facade is host-side only.

2. **cublas feature forwarding**: Currently not forwarded. If users need cuBLAS-accelerated
   nn ops, they'd need `gpu-host/cublas` directly. Consider forwarding in a future iteration
   if user demand arises.

3. **Example migration**: lib-facade.2 should create at least one example that depends on
   `async-gpu` instead of `gpu-host`, proving the facade works end-to-end.
