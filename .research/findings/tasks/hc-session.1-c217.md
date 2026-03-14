# hc-session.1: Design HostcallSession API
**Cycle**: 217 | **Theme**: hc-session | **Kind**: design | **Status**: done

## Summary
Design a HostcallSession that wraps HostcallBuffer + listener thread lifecycle, allowing multiple kernel launches to share the same hostcall infrastructure without teardown/rebuild.

## Design

### API Surface

```rust
/// A persistent hostcall session that survives across kernel launches.
///
/// Lifecycle: start → (launch kernel → synchronize → reinit)* → shutdown
pub struct HostcallSession {
    buf: Arc<HostcallBuffer>,
    listener_handle: Option<JoinHandle<()>>,
    // I/O thread + fd_table live inside the listener scope
}

impl HostcallSession {
    /// Start a new session with the given packet count.
    /// Spawns listener + I/O threads immediately.
    pub fn start(num_packets: u16) -> Result<Self>;

    /// Start a sharded session.
    pub fn start_sharded(num_shards: u32, pkts_per_shard: u32) -> Result<Self>;

    /// Get the device pointer for kernel launch args.
    pub fn dev_ptr(&self) -> CUdeviceptr;

    /// Get the sideband device pointer for bulk transfer args.
    pub fn sideband_dev_ptr(&self) -> CUdeviceptr;

    /// Reinitialize packet pool between kernel launches.
    /// MUST be called after dev.synchronize() and before next kernel launch.
    /// - Drains any stale packets from ready stacks
    /// - Returns all packets to free stacks
    /// - Resets sideband bump allocator
    /// - Does NOT reset fd_table (file handles persist)
    /// - Does NOT reset doorbell (monotonic)
    pub fn reinit_packets(&self);

    /// Shut down the session. Stops listener + I/O threads.
    /// All open file handles are closed (dropped).
    pub fn shutdown(self);
}
```

### Packet Reinit Protocol

The critical operation is `reinit_packets()` — resetting the packet pool to a clean state between kernel launches without reallocating memory.

**Problem**: When a kernel exits, packets can be in 3 states:
1. **FREE** (on free stack) — safe, ready for reuse
2. **FILLED** (submitted, not yet processed by host) — host will process these eventually
3. **READY** (host processed, waiting for GPU release) — leaked, GPU exited before releasing

**Solution**: After `dev.synchronize()` returns:
1. Host waits a short drain period (e.g., 50ms) for listener to finish processing any in-flight FILLED packets
2. Host reinitializes all free stacks and ready stacks to the same initial state as `HostcallBuffer::init()`
3. All packets' control fields zeroed
4. Sideband alloc_offset reset to 0

This is safe because:
- `dev.synchronize()` guarantees the GPU is idle — no GPU thread is accessing the buffer
- The listener thread is NOT stopped — it continues polling, but the ready stacks are now empty
- The listener's adaptive polling will naturally idle (no doorbell changes)

**Implementation detail**: The reinit must be done atomically with respect to the listener thread. Since the listener reads the ready stack with `swap(null, AcqRel)`, and we're writing the ready stack to NULL, there's no race — the listener will see NULL and skip. For free stacks, only the GPU pops from them, and the GPU is idle.

### Listener Thread Changes

Current `listen_unified()` runs in a `thread::scope` and exits when shutdown flag is set. For session mode:

**Option A: Persistent scope** — Use `std::thread::spawn` (not scoped) so the listener outlives any single kernel launch. The fd_table and I/O thread live inside the listener closure.

**Option B: Keep scoped, add session wrapper** — The `HostcallSession::start()` spawns a non-scoped thread that runs the scoped listener internally. The session holds the JoinHandle.

**Decision: Option A** — simpler. The listener thread runs `listen_session()` which is like `listen_unified()` but:
- Does NOT return on kernel exit (no implicit synchronization with kernel lifecycle)
- Only returns when `signal_shutdown()` is called
- The listener naturally handles gaps between kernels (doorbell stays constant, polling idles)

### File Handle Persistence

The `fd_table: HashMap<u64, File>` inside `io_thread_loop()` naturally persists because the I/O thread stays alive. Kernel A opens fd=1, Kernel B can write to fd=1.

**Caveat**: The GPU kernel must somehow know the fd value. Options:
1. Pass fd via mapped memory parameter (host writes fd, kernel reads)
2. Use a convention (e.g., fd=1 is always the first opened file)
3. The kernel opens its own files (most realistic — each kernel manages its own fds)

For the demo (hc-session.3), Kernel A opens a file and writes the fd to a mapped u64. Kernel B reads the fd and uses it.

### Sideband Reset

Between launches, the sideband bump allocator offset must be reset to 0. This is a single atomic store to `sideband_host_ptr + SIDEBAND_OFF_ALLOC_OFFSET`. Safe because GPU is idle after synchronize.

### Error Handling

- If `reinit_packets()` is called while a kernel is still running → undefined behavior (data race on packet pool). Document this as a precondition.
- If the session is dropped without shutdown → the listener thread is detached (runs until process exit). Add `Drop` impl that calls `signal_shutdown()`.

## ADR: HostcallSession Lifecycle

**Decision**: HostcallSession wraps HostcallBuffer with persistent listener thread. Packets reinitialized between launches via `reinit_packets()`. File handles persist across launches.

**Rationale**: Avoids the overhead of creating/destroying listener threads and reallocating mapped memory for each kernel launch. Enables cross-launch file handle sharing.

**Tradeoffs**:
- Pro: Zero allocation overhead between launches
- Pro: fd_table persists — Kernel B can use Kernel A's files
- Con: `reinit_packets()` must be called manually (forgotten → stale packets → pool exhaustion)
- Con: Listener thread stays alive between launches (wastes CPU polling, mitigated by adaptive sleep)

## Impact on Downstream Tasks
- hc-session.2: Implement this API in gpu-host/src/hostcall.rs (or new file session.rs)
- hc-session.3: Test with two kernels sharing fd
- cmd-buffer.2: CommandBuffer can be a field of HostcallSession
- cross-pipeline.1: Pipeline uses HostcallSession for multi-launch
