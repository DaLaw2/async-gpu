# api-ergonomics.1: Audit API pain points from all 7 examples
**Cycle**: 312 | **Theme**: api-ergonomics | **Kind**: investigation | **Status**: done

## Summary
Audited all 7 host-side examples to identify repeated boilerplate, clunky error handling, and opportunities for API improvement. The biggest pain points are: (1) the manual `thread::scope` + `spawn` + `listen` + `signal_shutdown` dance for hostcall, which consumes 8-12 lines in every example and is error-prone; (2) raw pointer casting (`dev_ptr as u64`, `sideband_dev_ptr as u64`) leaked into user code; and (3) the `get_func().ok_or(GpuHostError::KernelNotFound(...))` pattern repeated verbatim dozens of times. A `HostcallSession` already exists in the SDK but none of the 7 examples use it, suggesting it was added later and the examples were never updated.

## Findings

### Q: What boilerplate is repeated across examples?
A: Five patterns appear in nearly every example:

1. **Hostcall listener ceremony** (6/7 examples): `HostcallBuffer::new(8)` + `thread::scope` + `scope.spawn(|| hcbuf.listen(...))` + `hcbuf.signal_shutdown()` + `let _ = listener;`. This is 8-12 lines of identical scaffolding.

2. **Kernel lookup with error wrapping** (all 7): `rt.get_func("mod", "name").ok_or(GpuHostError::KernelNotFound("name"))` — the function name is written twice, and the user must import `GpuHostError` just for this.

3. **Raw pointer casting in launch args** (6/7): `hcbuf.dev_ptr as u64`, `hcbuf.sideband_dev_ptr as u64`, `result_buf.dev_ptr() as u64`. Users handle raw CUDA device pointers directly.

4. **Sleep-after-synchronize** (6/7): `rt.synchronize()` followed by `thread::sleep(Duration::from_millis(50))` — a 50ms sleep to let the hostcall listener drain. This is a workaround for a race condition between kernel completion and hostcall processing.

5. **MappedBuffer read/write ceremony** (5/7): `unsafe { result_buf.write(0, 0) }` to zero a result slot, then `unsafe { result_buf.read(0) }` to extract the value. Every read/write is `unsafe` even for simple u32 access.

**Confidence**: high

### Q: What error handling patterns are clunky?
A: Several issues:

1. **Inconsistent error types across examples**: 3 examples use `gpu_host::Result<()>`, 3 use `Result<(), Box<dyn std::error::Error>>`, 1 uses raw `CudaDevice` directly. The SDK's `Result` type isn't ergonomic enough to be the universal choice.

2. **KernelNotFound wrapping**: `get_func()` returns `Option`, forcing every callsite to manually convert to an error with `.ok_or(GpuHostError::KernelNotFound("name"))`. A `require_func()` method returning `Result` would eliminate this.

3. **Magic sentinel values**: `result >= 0xE000` checked manually in async-pipeline and parallel-search. There's a `check_kernel_result()` in error.rs that maps structured results, but examples don't use it — they parse raw u32 sentinels instead.

4. **unsafe for trivial operations**: `MappedBuffer::read(0)` and `write(0, val)` are always unsafe, even though bounds-checked access to a u32 result slot is a safe operation conceptually.

**Confidence**: high

### Q: What would a minimal "hello world" look like with ideal API?
A: Current minimal hello-world (hostcall-based) requires ~25 lines of setup. An ideal API:

```rust
use gpu_host::{GpuRuntime, HostcallSession};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    let rt = GpuRuntime::new(0)?;
    rt.load_ptx(PTX, "hello", &["hello_gpu"])?;

    let session = HostcallSession::start(8)?;
    let result = rt.launch_kernel("hello", "hello_gpu", (1,1,1), (32,1,1), (
        session.dev_ptr(),
    ))?;

    session.shutdown();
    Ok(())
}
```

This reduces ~40 lines to ~12. Key savings: no manual `thread::scope`, no raw pointer casting, no `get_func` + `ok_or`, no manual `synchronize` + sleep.

**Confidence**: medium

### Q: Where can builder/helpers reduce friction?
A: Five concrete opportunities:

1. **`GpuRuntime::require_func(&self, module, name) -> Result<CudaFunction>`** — eliminates `get_func().ok_or(KernelNotFound(...))` pattern (saves 2-3 lines per kernel call, ~20 instances across examples).

2. **`HostcallSession` adoption** — already exists in the SDK (`hostcall.rs:1849`) with `start()`, `shutdown()`, auto-`Drop`, but zero examples use it. It wraps the entire `thread::scope` + `spawn` + `listen` + `signal_shutdown` pattern.

3. **`MappedBuffer<T>::zeroed_slot() -> MappedSlot<T>`** — a safe, single-element wrapper with `fn get(&self) -> T` and `fn set(&self, val: T)` instead of `unsafe read(0)` / `unsafe write(0, val)`.

4. **`GpuRuntime::launch_kernel(module, func, grid, block, args) -> Result<()>`** — combines `get_func` + `ok_or` + `launch` + `synchronize` into one call. The `Pipeline` struct already does multi-stage version of this.

5. **Hostcall drain-after-sync** — the 50ms `thread::sleep` after `synchronize` should be handled internally by `HostcallSession::wait_idle()` or automatically during `shutdown()`. The current approach is fragile.

**Confidence**: medium

## Boilerplate Analysis (per example)

| Example | Setup Lines | Hostcall Lines | Teardown Lines | Total Overhead | Error Type |
|---------|------------|----------------|----------------|---------------|------------|
| hello-gpu | 13 (rt+ptx+hcbuf+mapbuf) | 8 (scope+spawn+listen+shutdown) | 3 (cleanup files) | 24 | `gpu_host::Result` |
| async-io | 11 | 8 | 7 (cleanup 4 files) | 26 | `gpu_host::Result` |
| async-pipeline | 8 | 8 (x2 demos) | 3 (per demo) | 22 | `Box<dyn Error>` |
| vector-math | 8 (rt+ptx) | 0 (no hostcall) | 0 | 8 | `gpu_host::Result` |
| parallel-search | 10 | 8 | 3 | 21 | `Box<dyn Error>` |
| tcp-echo | 10 | 8 | 1 | 19 | `gpu_host::Result` |
| warp-cooperative | 3 (raw CudaDevice) | 0 (no hostcall) | 0 | 3 | `Box<dyn Error>` |

**Key observation**: Hostcall-based examples have 19-26 lines of pure overhead. Non-hostcall examples (vector-math, warp-cooperative) are already lean.

## Proposed API Improvements

1. **`require_func()` method** on GpuRuntime — returns `Result<CudaFunction>` instead of `Option`. Eliminates the single most repeated pattern across all examples. Estimated savings: 2 lines per kernel invocation site.

2. **Promote `HostcallSession` to primary API** — update all examples to use `HostcallSession::start()` instead of manual `thread::scope` + `listen`. It already exists and handles spawn/shutdown/Drop. Saves 6-10 lines per example.

3. **Safe `MappedSlot<T>` wrapper** — single-element mapped buffer with safe `get()`/`set()` API for the common "one result value" pattern. Eliminates all `unsafe { result_buf.read(0) }` / `unsafe { result_buf.write(0, 0) }` calls.

4. **Type-safe launch args wrapper** — hide `dev_ptr as u64` casting behind a trait or extension. `HostcallSession` could implement `Into<u64>` or provide typed launch helpers.

5. **Drain-on-sync integration** — `HostcallSession` should automatically drain pending hostcall messages after `synchronize()`, eliminating the manual 50ms sleep hack.

6. **Unified `Result` type** — investigate why 3/7 examples choose `Box<dyn Error>` over `gpu_host::Result`. Root cause: `gpu_host::Result` doesn't implement `From<cudarc::driver::DriverError>` for launch errors returned through `LaunchAsync` (it does via `GpuHostError::Cudarc`, but the `?` chain breaks when mixed with `cudarc` types used directly).

## Unexpected Discoveries

1. **`HostcallSession` is dead code** — marked `#[allow(dead_code)]`, has full implementation including `Drop`, but is used by zero examples and zero tests (only internal tests). This is the exact abstraction the examples need but don't use.

2. **`Pipeline` struct exists but unused** — a multi-stage kernel pipeline with auto packet reinit and session management (`hostcall.rs:1947`). None of the examples use it despite async-pipeline being a perfect candidate.

3. **warp-cooperative example bypasses gpu-host entirely** — uses raw `CudaDevice` + `cudarc` directly, not `GpuRuntime`. This suggests `GpuRuntime` doesn't add enough value for pure-compute kernels.

4. **`CommandBuffer` and `FlightRecorder`** — additional high-level abstractions exist in the hostcall module that examples never showcase.

## Open Questions

1. Should `GpuRuntime` wrap kernel launch entirely (hiding `unsafe { f.launch(...) }`)? This would be a significant API change but would make the common case safe.
2. Should `HostcallSession` be re-exported at the crate root like `HostcallBuffer` is?
3. Is the 50ms sleep actually sufficient in all cases, or should there be a proper synchronization mechanism (e.g., the listener signals when it has drained all pending packets)?
4. Should `MappedBuffer` provide a `From<cudarc::driver::sys::CUdeviceptr>` impl or a method that returns the pointer as u64 directly for launch args?

## Impact on Downstream Tasks

- **api-ergonomics.2**: Implement `require_func()` on GpuRuntime — low-hanging fruit, high impact
- **api-ergonomics.3**: Update all 6 hostcall examples to use `HostcallSession` instead of manual ceremony
- **api-ergonomics.4**: Design and implement `MappedSlot<T>` safe wrapper
- **api-ergonomics.5**: Investigate drain-on-sync mechanism to replace 50ms sleep hack
- **api-ergonomics.6**: Ensure `gpu_host::Result` works with `?` in all examples (fix `From` impls)
