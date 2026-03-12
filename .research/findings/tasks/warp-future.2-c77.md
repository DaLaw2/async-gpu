# warp-future.2: Measure Per-Thread Async SIMT Efficiency Across Full 32-Thread Warp
**Cycle**: 77 | **Theme**: warp-future | **Kind**: experiment | **Status**: done

## Summary
Measured hostcall NOP round-trip performance with 32 threads in two configurations: 1 block x 32 threads (one warp, intra-warp divergence) vs 32 blocks x 1 thread (no intra-warp divergence). Surprisingly, the single-warp configuration is **27% FASTER** than the 32-block configuration. The bottleneck is CAS contention on the global free/ready stacks, not warp divergence. This significantly weakens the case for WarpFuture as a throughput optimization for hostcall-bound workloads.

## Findings

### Q: Does per-thread async across a full 32-thread warp actually work?
A: **Yes.** All 32 threads in one warp successfully completed all 20 NOP hostcall iterations (640/640). No deadlocks, no timeouts.
**Confidence**: high

### Q: What is the throughput vs 32 blocks x 1 thread (no intra-warp divergence)?
A: The single-warp config (1x32) achieves **1.27x higher throughput** than the 32-block config (32x1):
- Config A (1x32): 7720 calls/s, mean=1040µs, p50=979µs, CAS/call=3.31
- Config B (32x1): 6068 calls/s, mean=1584µs, p50=1593µs, CAS/call=13.61

The single-warp config is faster because intra-warp threads naturally serialize their CAS operations (hardware SIMT scheduling), reducing CAS retry rates by ~4x compared to cross-block threads that all hit the global stack simultaneously.
**Confidence**: high

### Q: How severe is warp divergence with per-thread futures in practice?
A: **Not measurably severe for this workload.** The single-warp config showed no throughput penalty from divergence. The hostcall pattern (pop free → fill → push ready → spin-wait) spends >99% of time in the convergent spin-wait loop. The brief divergence during CAS and state transitions is negligible.
**Confidence**: high

### Q: Is the divergence penalty significant compared to hostcall latency?
A: **No.** Per-call latency is ~1ms (dominated by host response time). Even if divergence caused 32x overhead on the Future::poll match statement (~100ns), it would be <0.01% of total latency. The divergence window is a few hundred nanoseconds at most; the hostcall round-trip is 1,000,000+ nanoseconds.
**Confidence**: high

## Unexpected Discoveries
1. **Single-warp is FASTER than multi-block** for hostcall workloads. The natural SIMT serialization of CAS operations within a warp reduces contention vs independent blocks racing on the same stack.
2. **CAS contention is the real bottleneck**, not SIMT divergence. Config B shows 13.61 CAS retries/call vs Config A's 3.31 — a 4.1x difference that directly explains the throughput gap.
3. **The skeptic was right**: the divergence problem is minimal for hostcall-bound workloads because spin-wait dominates execution time and is fully convergent.

## Key Insight for WarpFuture Epic
WarpFuture's value proposition is NOT throughput for hostcall-bound workloads. Instead, its value is:
1. **CAS contention reduction**: Warp-cooperative packet allocation (1 CAS per warp vs 32) — but this can be achieved without WarpFuture via warp-cooperative CAS alone.
2. **SIMT efficiency for compute-heavy async**: If futures contain significant computation between yield points, divergence during compute could matter. But our current workloads are hostcall-dominated.
3. **Architectural correctness**: WarpFuture is the "right" abstraction for GPU async, even if current workloads don't show the divergence penalty.

## Open Questions
1. What about compute-heavy futures (not hostcall-bound)? Does divergence matter when poll() contains significant computation?
2. Would warp-cooperative CAS alone (without full WarpFuture) capture most of the practical benefit?
3. Should WarpFuture be deprioritized in favor of warp-cooperative CAS as a simpler optimization?

## Impact on Downstream Tasks
- **warp-future.3** (intrinsics): Still valuable — shfl_sync and syncwarp are needed for warp-cooperative CAS too.
- **warp-future.4** (WarpFuture PoC): May be deprioritized. The empirical case for WarpFuture as a throughput optimization is weak for hostcall-bound workloads.
- **ADR-9**: Evidence now supports the skeptic's position. Consider adding warp-cooperative CAS as a simpler alternative.
