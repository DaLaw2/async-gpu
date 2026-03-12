# host-scaling.2: Design scaled listener architecture
**Cycle**: 64 | **Theme**: host-scaling | **Kind**: design | **Status**: done

## Summary

Designed a two-phase listener refactor based on host-scaling.1 profiling data. Phase A unifies
the duplicated listener code into a single generic implementation with a `StdinSource` trait.
Phase B adds a dedicated I/O thread for blocking FILE/STDIN operations, keeping the listener
lock-free and responsive for fast services.

## Design

### Phase A: Unified Listener

**Problem**: Two nearly-identical 100-line listener loops (`listen()` and `listen_with_stdin()`)
duplicate polling, dispatch, and packet processing. Any change must be applied twice.

**Solution**: Extract a `StdinSource` trait and make the listener generic over it.

```rust
/// Trait for providing stdin data to the listener.
pub trait StdinSource {
    /// Read a line of stdin. Returns bytes read, or empty for EOF.
    fn read_line(&mut self, buf: &mut [u8]) -> usize;
}

/// Real stdin — reads from std::io::stdin().
pub struct RealStdin;

impl StdinSource for RealStdin {
    fn read_line(&mut self, buf: &mut [u8]) -> usize {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => 0,
            Ok(n) => {
                let copy = n.min(buf.len());
                buf[..copy].copy_from_slice(&line.as_bytes()[..copy]);
                copy
            }
        }
    }
}

/// Canned stdin — returns pre-loaded data once, then EOF.
pub struct CannedStdin {
    data: Vec<u8>,
    consumed: bool,
}

impl CannedStdin {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, consumed: false }
    }
}

impl StdinSource for CannedStdin {
    fn read_line(&mut self, buf: &mut [u8]) -> usize {
        if self.consumed { return 0; }
        self.consumed = true;
        let copy = self.data.len().min(buf.len());
        buf[..copy].copy_from_slice(&self.data[..copy]);
        copy
    }
}
```

**Unified listener signature:**
```rust
pub fn listen_unified<F, S>(&self, on_print: F, stdin: S)
where
    F: FnMut(&[u8]),
    S: StdinSource,
```

**Migration**:
- `listen(on_print)` → `listen_unified(on_print, RealStdin)`
- `listen_with_stdin(on_print, data)` → `listen_unified(on_print, CannedStdin::new(data))`
- Keep old signatures as thin wrappers for backward compatibility during transition.

### Phase B: I/O Thread Separation

**Problem**: Blocking FILE I/O handlers (OPEN: 50-500µs, WRITE: 10-100µs, STDIN: unbounded)
stall all pending packets in the same batch. A STDIN `read_line()` blocks the entire listener.

**Solution**: Offload slow services to a dedicated I/O thread via a channel.

```
┌─────────────────────────────────────────────────────────┐
│ Listener Thread (fast path)                             │
│                                                         │
│ loop {                                                  │
│   poll doorbell → swap ready stack                      │
│   for each packet:                                      │
│     match service {                                     │
│       NOP/PRINT/TIME/PANIC → handle inline              │
│                               → control.store(READY)    │
│       OPEN/WRITE/READ/CLOSE/STDIN                       │
│         → io_tx.send(IoRequest { pkt_idx, service })    │
│     }                                                   │
│ }                                                       │
│                                                         │
└──────────────┬──────────────────────────────────────────┘
               │ mpsc channel
┌──────────────▼──────────────────────────────────────────┐
│ I/O Thread (blocking ops)                               │
│                                                         │
│ loop {                                                  │
│   let req = io_rx.recv()                                │
│   match req.service {                                   │
│     OPEN  → handle_open(pkt)                            │
│     WRITE → handle_write(pkt)                           │
│     READ  → handle_read(pkt)                            │
│     CLOSE → handle_close(pkt)                           │
│     STDIN → handle_stdin(pkt)                           │
│   }                                                     │
│   control.store(READY/ERROR)                            │
│ }                                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Channel type**: `std::sync::mpsc::channel()` (unbounded). The listener never blocks on send.
The I/O thread processes requests FIFO. This is sufficient because:
- FILE I/O is sequential by nature (can't parallelize writes to same fd)
- The I/O thread drains faster than the GPU produces (GPU limited to ~28K calls/s)

**IoRequest struct:**
```rust
struct IoRequest {
    pkt_idx: u16,    // packet index to write response to
    service: u32,    // SERVICE_OPEN | SERVICE_WRITE | etc.
}
```

The I/O thread accesses `self.host_ptr` via a shared reference (already `Sync`). It reads
the packet payload, performs the operation, and writes the response + sets CONTROL_READY.
This is safe because:
- The listener thread does NOT touch the packet after sending it to I/O thread
- The GPU thread is spin-waiting on CONTROL_READY and won't modify the packet
- Only the I/O thread writes to the packet after the listener hands it off

**Shared state**: The `fd_table` HashMap and `next_fd` counter move to the I/O thread
(they're only accessed by FILE handlers). The listener thread no longer needs them.

**Shutdown**: The listener thread drops `io_tx` when exiting the loop. The I/O thread
detects the closed channel via `recv() → Err(RecvError)` and exits.

### What we're NOT doing (and why)

1. **Multi-threaded dispatch**: The ready stack is atomically swapped as a whole — you can't
   split it across threads without protocol changes. Not worth the complexity.

2. **Async runtime (tokio)**: Adds a heavy dependency, requires async-ifying all handlers.
   The listener is CPU-bound for 90% of services. Overkill.

3. **Per-warp packet pools**: Requires GPU-side protocol changes (partitioned free stacks).
   Major effort, deferred to future theme if contention proves to be the bottleneck in
   real workloads.

4. **Increasing packet pool**: Easy but doesn't fix the fundamental throughput limit.
   Could add as a configurable parameter later.

## Findings

### Q: Multi-threaded vs async I/O vs hybrid — which approach based on profiling?

**Hybrid: fast inline + I/O thread**. This is the minimum-complexity solution that addresses
the actual bottleneck (blocking I/O stalls) without over-engineering.

- Fast services (NOP/PRINT/TIME/PANIC) stay inline — no overhead added
- Slow services (FILE/STDIN) go to I/O thread — no blocking
- Channel overhead (~100ns per send) is negligible vs FILE I/O cost (10-500µs)

**Confidence**: high

### Q: How to partition or share the ready stack among multiple consumers?

**Don't partition**. The current atomic-swap design is optimal for single-consumer:
one swap grabs all pending work. Partitioning would require N separate ready queues
and N doorbell checks — more overhead, not less.

Instead, keep single-consumer ready stack but offload slow work via channel. The
listener thread remains the sole consumer of the ready stack.

**Confidence**: high

### Q: What protocol changes (if any) are needed?

**None**. The design works entirely within the existing protocol:
- Buffer layout unchanged
- Packet format unchanged
- GPU-side code unchanged
- Only host-side dispatch logic changes

The I/O thread writes CONTROL_READY directly to the packet — same mechanism as the
listener thread. The GPU doesn't know or care which host thread responded.

**Confidence**: high

## ADR

**ADR-6: Host listener I/O thread separation**

- **Status**: proposed
- **Context**: Blocking FILE I/O handlers stall the listener thread, preventing timely
  processing of fast services (PRINT, PANIC). STDIN can block indefinitely.
- **Decision**: Split listener into fast-path (inline) and slow-path (I/O thread via channel).
  Fast services: NOP, PRINT, TIME, PANIC. Slow services: OPEN, WRITE, READ, CLOSE, STDIN.
- **Consequences**: Listener responsiveness guaranteed regardless of FILE I/O load.
  Small channel overhead (~100ns) for slow-path services. No protocol changes.

## Impact on Downstream Tasks

- **host-scaling.3 (implement)**: Clear implementation plan:
  1. Add `StdinSource` trait + `RealStdin` + `CannedStdin`
  2. Create `listen_unified()` replacing both existing listeners
  3. Add `IoRequest` struct + channel + I/O thread spawn
  4. Move FILE/STDIN handlers to I/O thread
  5. Update `listen()` and `listen_with_stdin()` as wrappers
- **host-scaling.4 (benchmark)**: Test FILE I/O workload with I/O thread to measure
  improvement in PRINT latency when FILE ops are concurrent.
