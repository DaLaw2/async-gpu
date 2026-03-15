# tokio-investigate.1: Current host threading model and tokio integration points
**Cycle**: 337 | **Theme**: tokio-investigate | **Kind**: investigation | **Status**: done

## Summary
Investigated the host-side threading architecture and tokio integration status. Key discovery: tokio is already an optional dependency with `async_rt.rs` providing `AsyncGpuRuntime` and `AsyncHostcallSession`. The foundation exists but the listener loop still uses `std::thread` with spin+sleep polling. Full tokio integration requires converting the listener loop to async and using `tokio::sync::mpsc` for I/O dispatch.

## Findings

### Q: How does GpuRuntime currently manage threads?
A: GpuRuntime itself is **single-threaded and synchronous**. It wraps `CudaDevice` (Arc for sharing). No thread spawning — threading is delegated to `HostcallSession`. Kernel launch (`f.launch()`) is blocking. `dev.synchronize()` blocks with `cuCtxSynchronize()`.
**Confidence**: high

### Q: What threading model does HostcallSession use?
A: **Dual-thread architecture** in `listen_unified()`:
- **Listener thread**: `std::thread::spawn` — polls GPU doorbell atomically, handles fast services inline (NOP, PRINT, TIME), uses adaptive spin→sleep (100μs) polling
- **I/O thread**: `std::thread::scope` — handles blocking FILE/STDIN/TCP via `std::sync::mpsc` channel
- Shutdown via atomic flag, graceful thread join
- File handle persistence (fd_table as HashMap) across kernel launches
**Confidence**: high

### Q: Does gpu-host already depend on tokio?
A: **Yes, as optional dependency**: `tokio = { version = "1", features = ["rt", "sync", "macros"], optional = true }` behind `async = ["dep:tokio"]` feature gate.
**Existing async layer** in `src/async_rt.rs` (182 lines):
- `AsyncGpuRuntime`: wraps `GpuRuntime`, `synchronize()` via `spawn_blocking`
- `AsyncHostcallSession`: wraps `HostcallSession`, returns `tokio::sync::mpsc::Receiver<HostcallEvent>` for print events
**Confidence**: high

### Q: How should gpu_spawn() work?
A: The `AsyncGpuRuntime` already has the foundation. `gpu_spawn()` should:
1. Launch kernel via `spawn_blocking` (since `cuLaunchKernel` is a blocking FFI call)
2. Return a `JoinHandle`-like future that resolves when `synchronize()` completes
3. Coordinate with `AsyncHostcallSession` event stream
The existing `AsyncHostcallSession::start()` already returns a `(session, Receiver<HostcallEvent>)` pair.
**Confidence**: high

### Q: What would need to change for full tokio integration?
A: Changes needed in `hostcall.rs::listen_unified()`:
1. **Listener loop**: Replace `std::thread::sleep` with `tokio::time::sleep().await` (or `tokio::time::interval`)
2. **I/O dispatch**: Replace `std::sync::mpsc` with `tokio::sync::mpsc`
3. **I/O thread**: Keep as `spawn_blocking` (blocking file I/O can't be made async without OS-level async I/O)
4. **Shutdown**: Replace `thread.join()` with `task.await`
5. **HostcallSession**: New `start_async()` that returns tokio JoinHandle instead of std JoinHandle
**Confidence**: high

## Unexpected Discoveries
- The async layer (`async_rt.rs`) already exists and is more complete than expected — `AsyncGpuRuntime` and `AsyncHostcallSession` are already implemented.
- The listener's adaptive spin→sleep polling is well-designed with `SPIN_PHASE_LIMIT` tuning.
- `HostcallEvent` enum already defined for the event stream.

## Open Questions
- Should the listener loop become fully async (yield to tokio scheduler during idle) or stay as a dedicated thread? Fully async is cleaner but adds latency from tokio task scheduling vs. dedicated thread's spin polling.
- Is there a way to use CUDA events or interrupts instead of polling the doorbell? This would eliminate spin-polling entirely.

## Impact on Downstream Tasks
- **tokio-investigate.2**: Can now design the bridge architecture with full knowledge of existing code.
- Key insight: Much of the work is already done — focus should be on `gpu_spawn()` API design and converting the listener loop.
