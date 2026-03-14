# host-async.1: Design async API surface — launch().await, Stream<HostcallEvent>, cancellation
**Cycle**: 248 | **Theme**: host-async | **Kind**: design | **Status**: done

## Summary
Design for tokio-compatible async GPU API wrapping the existing blocking gpu-host SDK. Three tiers: basic `spawn_blocking` wrapper, CUDA event-based notification, and hostcall event stream.

## Current Architecture (Baseline)

### Threading Model
1. **Main thread**: calls `dev.launch()` (async enqueue) + `dev.synchronize()` (blocking)
2. **Listener thread**: spawned by HostcallBuffer/Session, polls doorbell with adaptive spin+sleep
3. **I/O thread**: spawned by listener for blocking FILE/STDIN operations

### Key Blocking Points
- `GpuDevice::synchronize()` → `cuCtxSynchronize` (full device sync, blocks until all kernels complete)
- No CUDA stream or event API exposed — cudarc 0.12 hides this
- HostcallBuffer::listen() is a blocking loop (spin+sleep)
- CommandBuffer::submit() busy-waits if ring buffer full

## Design: Three Tiers

### Tier 1: spawn_blocking Wrapper (Minimum Viable)

Wrap the blocking synchronize call in `tokio::task::spawn_blocking`:

```rust
pub struct AsyncGpuRuntime {
    inner: Arc<GpuRuntime>,
}

impl AsyncGpuRuntime {
    /// Launch kernel and await completion.
    /// The kernel is enqueued immediately; .await blocks on synchronize.
    pub async fn launch_kernel(
        &self,
        module: &str,
        func: &str,
        config: LaunchConfig,
        args: KernelArgs,
    ) -> Result<(), GpuHostError> {
        let dev = self.inner.device_arc();
        let f = self.inner.get_func(module, func)
            .ok_or(GpuHostError::KernelNotFound)?;

        // Enqueue kernel on current thread (non-blocking CUDA call)
        unsafe { f.launch(config, args)? }

        // Move synchronize to blocking thread pool
        tokio::task::spawn_blocking(move || {
            dev.synchronize().map_err(GpuHostError::from)
        }).await?
    }
}
```

**Pros**: Zero new dependencies, works today, no CUDA event API needed.
**Cons**: Consumes a blocking thread per kernel launch. Full device sync (not per-stream).

### Tier 2: CUDA Event Polling (Better)

Use CUDA events for per-stream completion notification without blocking a thread:

```rust
pub struct GpuLaunchFuture {
    event: CudaEvent,       // Created via cuEventCreate
    waker: Option<Waker>,
}

impl Future for GpuLaunchFuture {
    type Output = Result<(), GpuHostError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match unsafe { cuEventQuery(self.event.0) } {
            CUDA_SUCCESS => Poll::Ready(Ok(())),
            CUDA_ERROR_NOT_READY => {
                self.get_mut().waker = Some(cx.waker().clone());
                // Register waker to be called later (via timer or dedicated poll thread)
                Poll::Pending
            }
            err => Poll::Ready(Err(cuda_error(err))),
        }
    }
}
```

**Challenge**: CUDA events are not self-notifying — we need a dedicated poll thread or timer to re-poll. Options:
1. **Timer-based**: `tokio::time::interval(Duration::from_micros(50))` polls event
2. **Dedicated GPU poll thread**: Single thread polls all pending events, wakes futures

**Recommendation**: Option 2 — single GPU-event-poll thread scales to N concurrent kernels.

**Blockers**: cudarc 0.12 does not expose `cuEventCreate`/`cuEventQuery`. Would need to use raw CUDA driver API via `cudarc::driver::sys`.

### Tier 3: Hostcall Event Stream

Expose hostcall events (print, file I/O, trace) as a tokio-compatible Stream:

```rust
pub struct HostcallEventStream {
    rx: tokio::sync::mpsc::Receiver<HostcallEvent>,
}

pub enum HostcallEvent {
    Print(Vec<u8>),
    FileOpen { fd: u64, path: String },
    FileClose { fd: u64 },
    TraceEvent { tid: u32, bid: u32, data: Vec<u8> },
    AssertFailure { tid: u32, bid: u32, msg: String },
    Shutdown,
}

impl Stream for HostcallEventStream {
    type Item = HostcallEvent;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().rx.poll_recv(cx)
    }
}
```

**Implementation**: Modify the existing listener thread to send events through a `tokio::sync::mpsc::Sender` in addition to (or instead of) the current `on_print: F` callback.

```rust
pub struct AsyncHostcallSession {
    inner: HostcallSession,
    event_rx: tokio::sync::mpsc::Receiver<HostcallEvent>,
}

impl AsyncHostcallSession {
    pub fn start(num_packets: u16) -> Result<(Self, HostcallEventStream)> {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        let session = HostcallSession::start_with_print(num_packets, move |msg| {
            let _ = tx.blocking_send(HostcallEvent::Print(msg.to_vec()));
        })?;
        Ok((Self { inner: session, event_rx: rx },
            HostcallEventStream { rx }))
    }
}
```

**Note**: This is a Tier 3 feature because the listener thread is already running — we just need to bridge it to tokio channels. No fundamental architecture change.

## Cancellation

### Kernel Cancellation
CUDA does not support cooperative kernel cancellation. Options:
1. **Flag-based**: Set a mapped-memory flag that the kernel checks periodically (best effort)
2. **cuStreamAbort**: Not available in cudarc
3. **Context destruction**: Nuclear option, destroys all GPU state

**Recommendation**: Flag-based cancellation via mapped memory. The kernel must opt in by checking the flag. This is already possible with the existing CommandBuffer EXIT command.

### Session Cancellation
`HostcallSession::shutdown()` already supports graceful shutdown via atomic flag. For async:

```rust
impl AsyncHostcallSession {
    pub async fn shutdown(self) {
        self.inner.signal_shutdown();
        // Wait a tick for listener to drain
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Drop inner session (joins listener thread)
        drop(self.inner);
    }
}
```

## Recommended Implementation Plan

### Phase 1: Tier 1 (spawn_blocking) — host-async.2a
- Create `AsyncGpuRuntime` wrapping `GpuRuntime`
- `launch_kernel().await` via `spawn_blocking`
- No new dependencies except `tokio`
- Test: launch vector_add kernel inside `#[tokio::test]`

### Phase 2: Tier 3 (event stream) — host-async.2b
- Add `tokio::sync::mpsc` channel to HostcallSession
- Create `HostcallEventStream` implementing `Stream`
- Test: consume print events from GPU kernel as async stream

### Phase 3: Tier 2 (CUDA events) — DEFERRED
- Requires raw CUDA driver API access
- cudarc 0.12 may add event support in future versions
- Only worth doing if spawn_blocking becomes a bottleneck (unlikely for most workloads)

## API Surface Summary

```rust
// Tier 1: Basic async wrapper
pub struct AsyncGpuRuntime { inner: Arc<GpuRuntime> }
impl AsyncGpuRuntime {
    pub fn new(device: usize) -> Result<Self>;
    pub async fn launch_kernel(...) -> Result<()>;
    pub async fn synchronize(&self) -> Result<()>;
    pub fn load_ptx(&self, ...) -> Result<()>;  // sync, fast
}

// Tier 3: Hostcall event stream
pub struct AsyncHostcallSession { ... }
impl AsyncHostcallSession {
    pub fn start(num_packets: u16) -> Result<(Self, HostcallEventStream)>;
    pub fn dev_ptr(&self) -> CUdeviceptr;
    pub fn reinit_packets(&self);
    pub async fn shutdown(self);
}

pub struct HostcallEventStream { ... }
impl Stream for HostcallEventStream {
    type Item = HostcallEvent;
}
```

## Decision: ADR-018

**Title**: Host-side async/await via spawn_blocking + hostcall event stream
**Status**: PROPOSED
**Context**: Host SDK is fully blocking. tokio integration enables composable GPU workflows.
**Decision**: Tier 1 (spawn_blocking) + Tier 3 (event stream) first. Tier 2 (CUDA events) deferred.
**Rationale**: spawn_blocking is zero-risk, works with cudarc 0.12. Event stream bridges existing listener thread to async. CUDA events need raw driver API — premature complexity.

## Open Questions
1. Should `tokio` be a required dependency or feature-gated (`feature = "async"`)?
2. Should `AsyncGpuRuntime` own `GpuRuntime` or share via `Arc`?
3. Event stream backpressure: what happens if consumer is slow? (bounded channel, drop events?)

## Impact on Downstream Tasks
- host-async.2 should implement Tier 1 + Tier 3 as described
- No GPU-side changes needed
- tokio dependency should be feature-gated to keep the sync API clean
