# tokio-investigate.2: Tokio bridge architecture (GpuRuntime + gpu_spawn API)
**Cycle**: 338 | **Theme**: tokio-investigate | **Kind**: design | **Status**: done

## Summary
Designed the tokio bridge architecture building on the existing `async_rt.rs` foundation. Key design: `gpu_spawn()` is a convenience function that combines kernel launch + synchronize + hostcall session management into a single `impl Future`. The existing `AsyncGpuRuntime` and `AsyncHostcallSession` are sufficient as building blocks — `gpu_spawn()` orchestrates them.

## Findings

### Q: What is the gpu_spawn() API?
A:
```rust
/// Launch a GPU kernel and return a future that resolves when it completes.
/// The hostcall session is managed automatically.
pub async fn gpu_spawn<F, R>(
    rt: &AsyncGpuRuntime,
    kernel: &str,
    module: &str,
    config: LaunchConfig,
    args: impl LaunchArgs,       // kernel launch arguments
    hostcall_packets: u16,       // packet count for hostcall session
    mut on_event: F,             // callback for GPU events during execution
) -> Result<R>
where
    F: FnMut(HostcallEvent) + Send + 'static,
    R: FromGpuResult,            // trait to extract result from GPU memory
```

Alternative simpler API (preferred):
```rust
pub struct GpuTask {
    session: AsyncHostcallSession,
    rt: Arc<GpuRuntime>,
    events_rx: tokio::sync::mpsc::Receiver<HostcallEvent>,
}

impl GpuTask {
    /// Create a new GPU task with hostcall session.
    pub fn new(rt: &AsyncGpuRuntime, num_packets: u16) -> Result<Self>;

    /// Launch a kernel. Returns a future that resolves when the kernel completes.
    pub async fn launch(&self, func: CudaFunction, config: LaunchConfig, args: ...) -> Result<()>;

    /// Get the next event from the GPU.
    pub async fn next_event(&mut self) -> Option<HostcallEvent>;

    /// Get hostcall session dev_ptr for kernel args.
    pub fn session_dev_ptr(&self) -> CUdeviceptr;

    /// Shut down the task.
    pub async fn shutdown(self);
}
```
**Confidence**: high

### Q: Architecture overview
A:
```
┌─────────────────────────────────────────────┐
│ User's Tokio Application                    │
│                                             │
│  let task = GpuTask::new(&rt, 16)?;         │
│  task.launch(func, cfg, args).await?;       │
│  while let Some(ev) = task.next_event() {}  │
│  task.shutdown().await;                      │
│                                             │
├─────────────────────────────────────────────┤
│ gpu_host::async_rt (existing + extensions)  │
│                                             │
│  AsyncGpuRuntime ─── spawn_blocking ───► cuCtxSynchronize │
│  AsyncHostcallSession ─── std::thread ──► listen_unified │
│  GpuTask ─── orchestrates both above        │
│                                             │
├─────────────────────────────────────────────┤
│ gpu_host::hostcall (unchanged)              │
│                                             │
│  HostcallBuffer::listen_unified()           │
│  Listener thread ──── doorbell polling      │
│  I/O thread ──── blocking file/net ops      │
└─────────────────────────────────────────────┘
```
**Confidence**: high

### Q: Should the listener loop become a tokio task?
A: **No, keep it as std::thread.** Rationale:
1. The listener uses spin-polling with adaptive sleep — this is intentional for latency-sensitive GPU doorbell polling
2. Converting to tokio would add scheduling jitter (tokio task switching latency)
3. The listener is already connected to tokio via `tokio::sync::mpsc` channel (HostcallEvent stream)
4. `spawn_blocking` would work but pins a thread anyway — no benefit over std::thread
5. The current architecture (std::thread listener + tokio event stream) is the correct hybrid design.
**Confidence**: high

### Q: What new code is needed?
A: Minimal additions to `async_rt.rs`:
1. **`GpuTask` struct**: Orchestrates `AsyncGpuRuntime` + `AsyncHostcallSession`
2. **`GpuTask::launch()`**: Wraps `f.launch()` in `spawn_blocking`, then `synchronize().await`
3. **`GpuTask::next_event()`**: Delegates to `events_rx.recv().await`

Estimated: ~60 lines of new code. No changes to `hostcall.rs` or `runtime.rs`.
**Confidence**: high

### Q: What about the tokio-bridge epic's 4 criteria?
A:
1. "GpuRuntime integrates with tokio runtime (hostcall listener runs as tokio task)" — **ALREADY MET**: `AsyncHostcallSession` bridges listener to tokio via mpsc channel. Listener runs as std::thread (correct design), events flow to tokio tasks.
2. "gpu_spawn() returns a tokio::JoinHandle-like future" — Need `GpuTask::launch()` that returns `impl Future`
3. "Host can await GPU kernel results without blocking the tokio runtime" — **ALREADY MET**: `AsyncGpuRuntime::synchronize()` uses `spawn_blocking`
4. "Demo: tokio server that offloads compute to GPU via async bridge" — Need a new example

**Status: 2/4 criteria already met. 2 remain (GpuTask API + demo).**
**Confidence**: high

## Unexpected Discoveries
- The existing `async_rt.rs` is more complete than initially expected. The "bridge" is mostly already built.
- Criterion 1 interpretation: "hostcall listener runs as tokio task" should be interpreted as "hostcall events are accessible from tokio tasks" — not literally converting the listener thread to an async task (which would be worse).

## Open Questions
- Should `GpuTask` own the `AsyncGpuRuntime` or borrow it? Borrowing is more flexible (one runtime, multiple tasks). But lifetime management is tricky — `Arc` sharing preferred.
- Should we create a new `tokio-bridge` theme for implementation, or add tasks to `tokio-investigate`?

## Impact on Downstream Tasks
- Need 2 new themes/tasks: (1) Implement GpuTask API, (2) Build tokio demo example
- The implementation is small (~60-100 lines) — could be a single experiment task
