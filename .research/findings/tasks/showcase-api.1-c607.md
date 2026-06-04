# showcase-api.1 — Audit: async-gpu facade missing re-exports

## Architecture Clarification

The task brief lists GPU-side types (BlockScope, block_channel, par_iter, executor,
unified_channel, grid_work, etc.) as candidates for re-export through the `async-gpu`
facade. However, **these are GPU-side constructs defined in `gpu-runtime` (a `no_std`
crate)**. The facade crate `async-gpu` re-exports from `gpu-host` (host-side), not
from `gpu-runtime`.

GPU kernel authors use `gpu-runtime` directly in their kernel crates (which target
`nvptx64`). The `async-gpu` facade is for **host-side** code that launches kernels,
manages sessions, and reads results. Therefore, the GPU-side types listed in the brief
(scopes, block channels, unified channels, grid_work, par_iter, executor) are
**correctly absent** from the facade — they belong in kernel crates, not host code.

## Current Facade Exports (async-gpu/src/lib.rs)

| Export | Source | Kind |
|--------|--------|------|
| `gpu` (module) | `gpu_host::gpu` | Module (run, launch, custom, etc.) |
| `GpuHostError` | `gpu_host::error` | Enum |
| `GpuKernelErrorInfo` | `gpu_host::error` | Struct |
| `Result` | `gpu_host::error` | Type alias |
| `GpuRuntime` | `gpu_host::runtime` | Struct |
| `HostcallSession` | `gpu_host::hostcall` | Struct |
| `MappedBuffer` | `gpu_host::memory` | Struct |
| `Pipeline` | `gpu_host::hostcall` | Struct |
| `GpuStream` | `gpu_host::streams` | Struct |
| `model_dir` | `gpu_host` | Function (doc-hidden) |
| `nn` (module) | `gpu_host::nn` | Feature-gated (nn) |
| `async_rt` (module) | `gpu_host::async_rt` | Feature-gated (async) |

## Missing Host-Side Re-exports

These are host-side types from `gpu-host` that users may need but must currently
import via `gpu_host::` instead of `async_gpu::`:

### 1. From `gpu::` module (accessible via `async_gpu::gpu::*` already)

These are already accessible through the `pub use gpu_host::gpu` re-export:
- `GpuStdModule` — via `async_gpu::gpu::GpuStdModule`
- `GpuContext` — via `async_gpu::gpu::GpuContext`
- `GpuResult` — via `async_gpu::gpu::GpuResult`
- `CustomLaunchBuilder` — via `async_gpu::gpu::CustomLaunchBuilder`
- `custom()` — via `async_gpu::gpu::custom()`
- `run_zero_param()` — via `async_gpu::gpu::run_zero_param()`
- `run_zero_param_with_config()` — via `async_gpu::gpu::run_zero_param_with_config()`

**Status: OK** — all accessible through the module re-export.

### 2. Missing from root: `HostcallBuffer`

`gpu-host` exports `HostcallBuffer` at its root (`pub use hostcall::HostcallBuffer`),
but `async-gpu` does not re-export it. Advanced users creating custom hostcall
setups need this.

**Recommendation**: Add `pub use gpu_host::HostcallBuffer;`

### 3. Missing: `HostcallError`

The error type for hostcall buffer allocation (`HostcallError`) is not re-exported.
Users calling `HostcallSession::start()` or `Pipeline::new()` get this error type
but can't name it from `async-gpu`.

**Recommendation**: Add `pub use gpu_host::hostcall::HostcallError;`

### 4. Missing: `CommandBuffer` and `Command`

Host-to-GPU command channel types. These are for persistent GPU kernels that
receive commands from the host.

**Recommendation**: Add re-exports if this is a user-facing API. Currently seems
more internal/advanced — defer unless users request it.

### 5. Missing: `FlightRecorder`

Post-mortem GPU tracing. Advanced debugging tool.

**Recommendation**: Add `pub use gpu_host::hostcall::FlightRecorder;` — useful
for debugging GPU kernel crashes.

### 6. Missing: `error_category_name`

Helper function for interpreting GPU kernel error categories.

**Recommendation**: Available through `async_gpu::error` if we re-export the module,
but currently only `GpuHostError` and `GpuKernelErrorInfo` are cherry-picked. Consider
adding `pub use gpu_host::error::error_category_name;`.

### 7. Missing async_rt types at root level

When `async` feature is enabled, users must write `async_gpu::async_rt::AsyncGpuRuntime`.
The key types could be re-exported at root for convenience:
- `AsyncGpuRuntime`
- `AsyncHostcallSession`
- `GpuTask`
- `HostcallEvent`

**Recommendation**: Add feature-gated root re-exports for the 4 async types.

### 8. Missing: `ptx` module

The embedded PTX strings are `#[doc(hidden)]` in `gpu-host` but not re-exported
at all in `async-gpu`. Users who want `run_zero_param()` need PTX sources.

**Recommendation**: Add `#[doc(hidden)] pub use gpu_host::ptx;` — needed by
`gpu::run_zero_param()` callers.

### 9. Missing: `streams` module access

Only `GpuStream` is re-exported, but `GpuRuntime::create_stream()` is the main
entry point and that's already available via the `GpuRuntime` re-export.

**Status: OK** — `GpuStream` is sufficient.

## Feature Flag Coverage

| Feature | Cargo.toml | Re-export |
|---------|-----------|-----------|
| `nn` | `gpu-host/nn` | `pub use gpu_host::nn` |
| `async` | `gpu-host/async` | `pub use gpu_host::async_rt` |
| `gpt2` | NOT forwarded | N/A (internal model) |
| `onnx` | NOT forwarded | N/A (likely internal) |

The `gpt2` and `onnx` features are not forwarded through the facade. This is
likely intentional since they're for internal model demos, not user-facing API.

## Cargo Doc Output

```
Compiling gpu-host v0.1.0
Documenting async-gpu v0.1.0
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.75s
Generated /home/dalaw2/async-gpu/target/doc/async_gpu/index.html
```

No warnings. Clean build.

## Examples Still Using `gpu_host` Directly

Several examples import from `gpu_host` instead of `async_gpu`, indicating the
facade is incomplete or examples haven't been migrated:

- `examples/hostcall/async-io/host/src/main.rs` — `use gpu_host::gpu`
- `examples/hostcall/async-pipeline/host/src/main.rs` — `use gpu_host::gpu`
- `examples/hostcall/parallel-search/host/src/main.rs` — `use gpu_host::gpu`
- `examples/hostcall/tcp-echo/host/src/main.rs` — `use gpu_host::gpu`
- `examples/hostcall/tokio-offload/src/main.rs` — `use gpu_host::async_rt::*`
- `examples/hostcall/vector-math/host/src/main.rs` — `use gpu_host::gpu`
- `examples/hostcall/warp-cooperative/host/src/main.rs` — `use gpu_host::gpu`
- `examples/std/benchmark/src/main.rs` — `use gpu_host::nn::*`
- `examples/std/cifar-train/src/main.rs` — `use gpu_host::nn::*`
- `examples/std/gpt2-lora/src/main.rs` — `use gpu_host::nn::*`

Only `examples/hostcall/hello-gpu/host/src/main.rs` uses `async_gpu`.

## Summary of Recommended Actions

### Priority 1 — Add to facade (user-facing, needed now)
1. `pub use gpu_host::HostcallBuffer;`
2. `pub use gpu_host::hostcall::HostcallError;`
3. `#[doc(hidden)] pub use gpu_host::ptx;` (needed for run_zero_param callers)

### Priority 2 — Convenience re-exports (nice to have)
4. Feature-gated root re-exports for async types (AsyncGpuRuntime, GpuTask, etc.)
5. `pub use gpu_host::hostcall::FlightRecorder;`
6. `pub use gpu_host::error::error_category_name;`

### Priority 3 — Example migration
7. Migrate all examples from `gpu_host::` imports to `async_gpu::` imports

### NOT needed (GPU-side types)
The following are GPU-side types used in kernel crates (targeting nvptx64) and
should NOT be re-exported through the host-side facade:
- BlockScope, GridScope, block_scope, grid_scope, ScopeJoinHandle
- BlockOneshotSlot, BlockMpscChannel, block_oneshot, block_mpsc
- ScopedOneshotSender, ScopedMpscSender, etc.
- BlockWorkSlot, grid_worker_loop, grid_worker_loop_continuous
- GpuParallelIterator, GpuParIter, GpuSlice, GpuSliceMut, par_iter
- GPU executor types
