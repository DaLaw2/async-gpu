# showcase-api.2 — Add missing types to async-gpu facade re-exports

## Changes Made

### 1. Facade re-exports added (`crates/async-gpu/src/lib.rs`)

**Priority 1 — user-facing, needed now:**
- `pub use gpu_host::HostcallBuffer;` — advanced hostcall buffer management
- `pub use gpu_host::hostcall::HostcallError;` — error type from buffer/session/pipeline constructors
- `#[doc(hidden)] pub use gpu_host::ptx;` — needed by `run_zero_param()` callers and examples

**Priority 2 — convenience re-exports:**
- `pub use gpu_host::error::error_category_name;` — helper for interpreting GPU error categories
- `pub use gpu_host::hostcall::FlightRecorder;` — post-mortem GPU trace debugging tool
- Feature-gated root re-exports for async types (when `async` feature is enabled):
  - `AsyncGpuRuntime`
  - `AsyncHostcallSession`
  - `GpuTask`
  - `HostcallEvent`

### 2. Examples migrated from `gpu_host` to `async_gpu`

| Example | Changes |
|---------|---------|
| `examples/hostcall/vector-math/host` | `gpu_host::gpu` → `async_gpu::gpu`, `gpu_host::Result` → `async_gpu::Result`, Cargo.toml dep updated |
| `examples/hostcall/parallel-search/host` | `gpu_host::gpu` → `async_gpu::gpu`, Cargo.toml dep updated |
| `examples/hostcall/tokio-offload` | `gpu_host::async_rt::*` → `async_gpu::*`, `gpu_host::memory::MappedBuffer` → `async_gpu::MappedBuffer`, `gpu_host::runtime::GpuRuntime` → `async_gpu::GpuRuntime`, `gpu_host::ptx::KERNEL` → `async_gpu::ptx::KERNEL`, Cargo.toml dep updated |

### 3. What was NOT re-exported (by design)

- `CommandBuffer` / `Command` — internal/advanced host-to-GPU command channel; defer until users request
- GPU-side types (BlockScope, channels, par_iter, etc.) — these are `no_std` kernel-side types from `gpu-runtime`, not host-side

## Verification

- `cargo doc --no-deps -p async-gpu` — clean, no warnings
- `cargo doc --no-deps -p async-gpu --features async` — clean, no warnings
- All 3 migrated examples compile cleanly (`cargo check`)
- `bash scripts/ci-lint.sh` — all checks passed

## Remaining Migration Candidates

These examples still use `gpu_host` directly and could be migrated in future:
- `examples/hostcall/async-io/host` — uses `gpu_host::gpu`
- `examples/hostcall/async-pipeline/host` — uses `gpu_host::gpu`
- `examples/hostcall/tcp-echo/host` — uses `gpu_host::gpu`
- `examples/hostcall/warp-cooperative/host` — uses `gpu_host::gpu`
- `examples/std/benchmark` — uses `gpu_host::nn::*`
- `examples/std/cifar-train` — uses `gpu_host::nn::*`
- `examples/std/gpt2-lora` — uses `gpu_host::nn::*`
- `examples/std/thread-demo` — uses `gpu_host::gpu`
