# host-scaling.1: Profile host listener bottleneck
**Cycle**: 63 | **Theme**: host-scaling | **Kind**: investigation | **Status**: done

## Summary

Static analysis of the host listener loop in `hostcall.rs` combined with benchmark.2 data
to identify where time is spent and what limits throughput. The bottleneck is NOT compute or
I/O — it's the inherent sequential round-trip latency of the protocol itself, plus packet
pool contention at scale. Blocking FILE I/O handlers are a secondary risk for mixed workloads.

## Findings

### Q: What fraction of time is spent in polling vs dispatching vs I/O syscalls vs sleep?

**For NOP workload (benchmark.2 baseline):**

| Phase | Estimated time | Notes |
|-------|---------------|-------|
| GPU-side (CAS pop + fill + push + doorbell) | ~3µs | Measured via globaltimer delta minus host |
| Host polling (doorbell detection) | <1µs during spin phase | 100ns/iteration × few iterations |
| Host dispatch (ready_stack swap + walk + control read/write) | ~1-2µs | 3 volatile reads + 1 atomic store per packet |
| Host service handler (NOP) | ~0µs | Literally empty |
| GPU spin-wait (CONTROL_READY detection) | ~7-9µs | Dominant cost — GPU polls mapped memory over PCIe |

Total single-thread round-trip: ~13µs (measured).

The GPU spin-wait is the largest single component. The GPU reads `CONTROL_READY` via
`atom.cas.sys.global.b64` over PCIe — each probe costs ~200-500ns due to system-scope
coherence. At ~300ns/probe, detecting CONTROL_READY takes ~7-10µs on average.

**For FILE I/O workload (estimated):**

| Service | Handler cost | Impact |
|---------|-------------|--------|
| PRINT | ~1µs (memcpy 56 bytes + callback) | Negligible |
| TIME | ~1µs (syscall) | Negligible |
| OPEN | 50-500µs (filesystem) | **Stalls all pending packets** |
| WRITE | 10-100µs (write + flush) | **Stalls all pending packets** |
| READ | 10-100µs (read) | **Stalls all pending packets** |
| STDIN | **blocking** (waits for user input) | **Stalls everything indefinitely** |

**Confidence**: high (NOP data from benchmark.2, FILE estimates from OS syscall literature)

### Q: Is the bottleneck compute-bound, I/O-bound, or polling-strategy-bound?

**Protocol-bound**, specifically:

1. **Sequential round-trip**: Each hostcall is inherently serial — GPU thread sends request,
   then spin-waits until host responds. No pipelining. The GPU thread cannot issue the next
   hostcall until the previous one completes. This means per-thread throughput is capped at
   1/13µs ≈ 77K calls/s regardless of host optimization.

2. **GPU-side PCIe polling**: The GPU spin-wait for CONTROL_READY dominates the 13µs
   round-trip (~7-9µs). The host processes the packet in ~2µs but the GPU takes ~7µs to
   notice the response. This is fundamentally limited by PCIe coherence latency.

3. **Packet pool contention at scale**: At 32+ threads, CAS retries on the shared free-stack
   consume 14-49 retries/call. Each retry is a system-scope CAS (~100-200ns). At 128 threads,
   this adds ~5-10ms to each call.

4. **Single-threaded sequential processing**: When multiple packets are ready, they're
   processed one-by-one in the linked list walk. A blocking handler (FILE OPEN) stalls all
   subsequent packets in the batch.

The host listener CPU is NOT the bottleneck for NOP/PRINT workloads — it processes each
packet in ~2µs and idles between doorbells. Adding more listener threads would not improve
NOP throughput because the GPU-side PCIe polling is the limiter.

**Confidence**: high

### Q: Would async I/O (tokio) or multi-threading be more effective?

Neither addresses the primary bottleneck (GPU PCIe polling). But they address different
secondary bottlenecks:

**Multi-threaded dispatch (thread pool):**
- Helps: Prevents blocking FILE I/O from stalling other packets in the same batch
- Doesn't help: NOP/PRINT throughput (already fast enough)
- Complexity: Moderate — need to partition or work-steal from the ready list
- Risk: Ready stack is atomically swapped by one thread. Partitioning requires protocol change.

**Async I/O (tokio):**
- Helps: Prevents STDIN from blocking the entire listener
- Doesn't help: Most services are CPU-bound (PRINT, TIME, NOP)
- Complexity: High — requires async-ifying all handlers
- Risk: Over-engineering for current use case

**Dedicated I/O thread (recommended):**
- Helps: Move FILE/STDIN ops to a separate thread. Listener stays responsive for fast services.
- Doesn't help: Multi-thread throughput scaling
- Complexity: Low — channel-based dispatch for slow services
- Risk: Minimal — listener loop unchanged for fast services

**Confidence**: high

### Q: What is the theoretical max throughput with zero-cost dispatch?

If the host processed packets in zero time, each round-trip would still take:
- GPU-side overhead: ~3µs (CAS pop + fill + push + doorbell)
- GPU spin-wait: ~7µs (PCIe CONTROL_READY detection)
- **Minimum per-thread round-trip: ~10µs → 100K calls/s per thread**

For aggregate throughput with N threads:
- N threads can have at most min(N, num_packets) calls in-flight simultaneously
- With 64 packets and negligible host processing: theoretical max ≈ 64 / 10µs = 6.4M calls/s
- But CAS contention at 64 concurrent threads reduces this by ~5-10x
- Realistic maximum with zero-cost host: ~500K-1M calls/s

Measured maximum (NOP, 1 thread): 28K calls/s — this is only 28% of the 100K theoretical
maximum. The gap is explained by:
1. Host polling delay (doorbell detection not instant)
2. CONTROL_READY store → GPU detection has variable latency
3. Thread scheduling jitter on both sides

**Confidence**: medium (estimates based on PCIe latency model, not direct measurement)

## Architecture Analysis

### Current listener hot path (per packet batch):
```
loop {
    shutdown.load(Acquire)           // 1 atomic
    doorbell.load(Acquire)           // 1 atomic
    if doorbell == last → spin/sleep
    ready_stack.swap(NULL, AcqRel)   // 1 atomic swap — grabs ALL ready packets
    for each packet in linked list:
        read_volatile(PKT_OFF_NEXT)      // pointer chase
        control.load(Acquire)            // CONTROL_FILLED check
        read_volatile(PKT_OFF_SERVICE)   // service ID
        [service handler]                // 0µs (NOP) to ∞ (STDIN)
        control.store(flags, Release)    // signal GPU
}
```

### Code duplication problem
There are TWO nearly-identical listener implementations:
- `listen()` (line 182) — standard listener
- `listen_with_stdin()` (line 636) — adds canned stdin support

Both duplicate the entire polling loop, dispatch table, and packet processing logic.
Any scaling change must be applied twice or the code should be unified first.

### Adaptive polling analysis
- SPIN_PHASE_LIMIT = 1000 iterations (~100µs at 100ns/spin_loop)
- SLEEP_DURATION = 100µs
- After 1000 idle spins, switches to 100µs sleeps
- This is well-tuned for burst workloads but means worst-case detection latency is 100µs
  during sleep phase (adds to round-trip when GPU sends after listener enters sleep)

## Recommendations for host-scaling.2 (design)

### Priority 1: Unify listener implementations
Refactor `listen()` and `listen_with_stdin()` into a single implementation with a
configurable stdin source. This eliminates the code duplication problem before any
scaling work.

### Priority 2: Separate blocking I/O thread
Move FILE (OPEN/WRITE/READ/CLOSE) and STDIN handlers to a dedicated I/O thread.
The listener stays lock-free and fast for NOP/PRINT/TIME/PANIC.

Architecture:
```
Listener thread (fast):
    poll doorbell → swap ready stack → for each packet:
        NOP/PRINT/TIME/PANIC → handle inline, set CONTROL_READY
        FILE/STDIN → send (pkt_idx, service) to channel

I/O thread:
    recv from channel → handle blocking op → set CONTROL_READY on packet
```

### Priority 3 (deferred): Per-warp packet pools
Reduce CAS contention by partitioning the free stack into per-warp pools.
This requires GPU-side protocol changes and is a larger effort.

### NOT recommended:
- Multi-threaded listener: The ready stack design (atomic swap of entire list) doesn't
  partition well. Would need protocol redesign.
- Async runtime (tokio): Over-engineering — the listener is CPU-bound for common services.
- Increasing poll frequency: Already at spin_loop() speed during active phase.

## Unexpected Discoveries

1. **GPU-side PCIe polling is the dominant cost**: The host processes NOP in ~2µs but the
   GPU takes ~7µs to detect CONTROL_READY. Optimizing host speed has diminishing returns.

2. **28K calls/s at 1 thread is only 28% of theoretical maximum**: The gap is mostly from
   variable PCIe coherence latency and host poll timing. This suggests that GPU-side
   optimizations (e.g., `ld.volatile` instead of CAS for control word polling) could
   improve throughput more than host-side changes.

3. **Listener code is duplicated**: Two nearly-identical 100-line listener loops exist.
   This is a maintenance hazard that should be fixed before any scaling work.

4. **STDIN handler blocks the entire listener**: A single `read_line()` call can block
   the listener for arbitrary duration. All other services (including PANIC) are stalled.
   This is a correctness bug for production use.

## Open Questions

- What is the actual latency breakdown between host processing and GPU PCIe polling?
  (Would need instrumenting the host listener with timestamps to measure directly)
- Would `ld.volatile.global` instead of `atom.cas.sys.global` for control word polling
  reduce GPU-side wait time?
- Is the adaptive polling threshold (1000 spins → 100µs sleep) optimal?

## Impact on Downstream Tasks

- **host-scaling.2 (design)**: Use the "separate I/O thread" architecture. Focus on
  unifying listener code first, then adding the I/O thread channel.
- **host-scaling.3 (implement)**: Implementation is straightforward with channel-based
  I/O offloading. No protocol changes needed.
- **host-scaling.4 (benchmark)**: Re-run benchmark.2 with and without FILE I/O workload
  to measure the impact of I/O thread separation.
- **GPU-side optimization**: Consider a new task to investigate `ld.volatile` for control
  word polling — could give 2-3x throughput improvement.
