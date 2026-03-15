# bench-suite.4: Consolidated benchmark results and analysis
**Cycle**: 324 | **Theme**: bench-suite | **Kind**: design | **Status**: done

## Summary
Consolidated analysis of all hostcall benchmark results: NOP latency, throughput scaling,
file I/O per-phase breakdown, and scalability curves. Identifies bottlenecks and provides
guidance for users.

## Hostcall Protocol Performance

### Architecture
GPU threads write hostcall requests into pinned mapped memory visible to both GPU and host.
A host-side polling thread detects requests, executes them, and writes responses. The GPU
thread spin-waits for the response. The protocol uses a lock-free packet pool with CAS-based
allocation (free stack pop/push) and per-block sharding to reduce contention.

### Baseline: NOP Hostcall Round-Trip
| Metric | Value |
|--------|-------|
| Single-thread p50 | 13 μs |
| Single-thread throughput | 66,015 calls/s |
| Coefficient of variation | 96% |

The 13 μs baseline includes: GPU → mapped memory write (system-scope store) + PCIe coherence
propagation + host polling detection + host response write + PCIe propagation + GPU spin-load
acquire. The high CV (96%) reflects PCIe latency jitter.

### Throughput vs Thread Count
| Threads | Throughput (calls/s) | p50 Latency | CAS Retries/call |
|---------|---------------------|-------------|------------------|
| 1 | 66,015 | 13 μs | 0.00 |
| 2 | 4,021 | 524 μs | 0.50 |
| 4 | 7,921 | 523 μs | 2.59 |
| 8 | 15,455 | 520 μs | 6.69 |
| 16 | **21,837** | 529 μs | 9.96 |
| 32 | 19,147 | 1.57 ms | 28.65 |
| 64 | 18,513 | 1.03 ms | 33.25 |
| 128 | 15,486 | 1.25 ms | 30.15 |
| 256 | 15,084 | 1.41 ms | 19.64 |
| 512 | 155 | 1.07 ms | 18.79 |
| 1024 | 215 | 1.26 ms | 30.93 |

**Key observations:**
1. **Peak aggregate throughput at 16 threads** (~22K/s), then degrades
2. **40x latency cliff at 2 threads** (13μs → 524μs) — PCIe coherence cost
3. **CAS retries explode beyond 32 threads** — free stack contention
4. **Pool exhaustion at 512+ threads** — only 64 packets for 512+ threads

### File I/O Per-Phase Latency (Single Thread, 48-byte Payload)
| Phase | p50 (μs) | Mean (μs) | Comment |
|-------|----------|-----------|---------|
| open (write) | 911 | 943 | Host syscall overhead |
| write | 175 | 243 | OS buffered — just memcpy |
| close (write) | 732 | 765 | Flush + fd cleanup |
| open (read) | 535 | 526 | Faster than create |
| read | 346 | 362 | Seek + read + copy to packet |
| close (read) | 502 | 536 | fd cleanup |
| **Total** | **3,201** | **3,375** | **296 round-trips/s** |

**Key observations:**
1. **File open is most expensive** (911 μs) — host kernel-mode syscall
2. **Write is cheapest** (175 μs) — OS buffer layer absorbs it
3. **Total overhead vs native**: ~38-190x over direct host file I/O
4. **Each operation is a separate hostcall**: 6 round-trips per file cycle

## Bottleneck Analysis

### Primary: CAS Contention on Free Stack
The global free packet stack uses tagged CAS for ABA prevention. At 32+ threads, CAS
retry rates hit 28-33x. This is the #1 scalability bottleneck.

**Mitigation (already implemented):** Per-block sharding divides the packet pool into
`num_blocks` independent shards. Each block operates on its own shard, eliminating
cross-block CAS contention. With 4 blocks, CAS contention drops to ~4-thread levels.

### Secondary: Packet Pool Size (64 packets)
At 128+ threads, the pool exhausts. Threads spin-wait for free packets, creating
priority inversion.

**Mitigation:** Increase `MAX_PACKETS` or use backpressure (yield to executor when
no packets available).

### Tertiary: PCIe Latency Floor
The 13 μs minimum is dominated by PCIe round-trip (~5-10 μs) plus coherence overhead.
This is a hardware constraint that cannot be optimized away.

## Guidance for Users
1. **Use sharding** for multi-block kernels — enables near-linear scaling up to packet pool size
2. **Batch operations** — minimize individual hostcall count (e.g., write larger chunks)
3. **16 threads per block** is the sweet spot for hostcall-heavy workloads without sharding
4. **File I/O: minimize open/close** — keep file descriptors open across operations
5. **Expect 13 μs minimum** per hostcall — design around this latency floor
