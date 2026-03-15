# bench-suite.1: Design benchmark suite
**Cycle**: 312 | **Theme**: bench-suite | **Kind**: investigation | **Status**: done

## Summary
Analyzed the existing benchmark infrastructure (3 benchmark functions in `tests_benchmark.rs`, 2 GPU kernel variants) and researched GPU microbenchmarking best practices. Designed a comprehensive 7-scenario benchmark suite covering latency, throughput, scalability, I/O services, bulk transfer, and end-to-end pipeline performance, with specific metrics, measurement methodology, and implementation plan.

## Findings

### Q: What metrics matter most?
A: For a hostcall RPC system, the critical metrics fall into three tiers:

**Tier 1 — Must have:**
- **Round-trip latency percentiles** (p50, p95, p99, max): The single most important metric. p50 shows typical behavior; p95/p99 reveal tail latency from CAS contention and host listener scheduling. The existing code already captures these well.
- **Aggregate throughput** (hostcalls/sec): Measures the system's capacity under load. Existing code computes this from wall time but could be more rigorous with sustained-load measurement.
- **CAS retry rate** (retries/completed call): Direct proxy for contention in the lock-free packet pool. Already measured.

**Tier 2 — Important for optimization:**
- **Scalability curve**: Throughput as a function of thread count (1, 32, 128, 512, 1024). Shows where the system saturates and what the bottleneck is (host CPU, PCIe, CAS contention).
- **Per-iteration latency distribution**: Instead of per-thread averages, record individual iteration latencies (requires larger result buffers) to get true p99 across all calls, not p99 of per-thread means.
- **Host listener service time**: Time from doorbell detection to response write. Isolates host-side overhead from GPU spin-wait time.

**Tier 3 — Nice to have:**
- **PCIe bandwidth utilization**: For bulk transfer benchmarks (sideband buffer).
- **Warp occupancy efficiency**: How many lanes are active during hostcalls.
- **Thermal throttle detection**: Monitor GPU clock via `nvmlDeviceGetClockInfo` to flag runs where throttling occurred.

**Confidence**: high

### Q: How to measure hostcall overhead accurately?
A: The existing approach using `%globaltimer` is sound but has nuances:

**Current approach (good):**
- Uses `gpu_instant_nanos()` which reads `%globaltimer` — a system-wide monotonic nanosecond counter. This is the correct choice for cross-SM timing (unlike `clock64()` which is per-SM and not synchronized across SMs).
- Wraps the entire iteration loop with start/end timestamps, computing average latency per thread.

**Improvements needed:**

1. **Per-iteration timestamps**: The current kernel records `(t_end - t_start) / completed` which gives mean latency per thread but loses the distribution. A more informative kernel would record each iteration's latency individually:
   ```
   results[tid * num_iters + iter] = t_after_response - t_before_push
   ```
   This requires more memory but gives true latency histograms.

2. **Breakdown timing**: Instrument the hostcall phases separately:
   - `t_pop_free`: Time to acquire a packet from the free pool (CAS contention)
   - `t_fill_push`: Time to fill packet + push to ready stack
   - `t_host_service`: Time from push to response (includes PCIe + host processing + PCIe back)
   - `t_release`: Time to return packet to free pool
   This pinpoints where time is spent.

3. **Warmup**: The existing warp-divergence benchmark does a warmup run (1x1, 5 iters) which is correct. The latency benchmark does NOT warm up. All benchmarks should do at least one warmup kernel launch to ensure PTX JIT compilation and memory allocation are not included.

4. **Clock stability**: GPU boost clocks are stochastic. For reproducible results:
   - Lock GPU clocks with `nvidia-smi -lgc <freq>` before benchmarking
   - Record the actual GPU clock frequency in output
   - Run enough iterations (>100) to amortize any clock transitions

5. **Host listener warm-up**: The listener thread may be in the sleep phase of its adaptive polling (after SPIN_PHASE_LIMIT=1000 idle spins, it sleeps). A warmup run also ensures the listener is in the fast spin-polling mode.

6. **Memory fence artifacts**: The `%globaltimer` read itself is essentially free (~4 cycles), but on NVIDIA GPUs, inline asm acts as an optimization barrier. This is actually desirable — it prevents the compiler from reordering timing calls past the code being measured.

**Confidence**: medium (per-iteration timing needs validation that the larger result buffer doesn't cause TLB pressure)

### Q: What baselines to compare against?
A: Three categories of baselines:

**1. Theoretical lower bounds:**
- **PCIe round-trip latency**: ~2-5 us for a mapped memory write + read on PCIe Gen3/4. This is the absolute floor for any hostcall.
- **CAS instruction latency**: ~100-400 cycles per atomic on global memory (GPU-side). With 32 threads contending on one CAS target, expect ~1-10 us just for allocation.
- **Host-side service dispatch**: ~0.1-1 us for a NOP handler in a tight poll loop.

**2. Internal baselines (before/after optimization):**
- **Unsharded vs sharded**: Already benchmarked. Keep as a regression test.
- **Single-thread vs full-warp vs multi-block**: Already benchmarked (warp divergence measurement).
- **Before/after any protocol change**: Archive current numbers as the baseline.

**3. External comparisons (informational, not apples-to-apples):**
- **CUDA Dynamic Parallelism**: Kernel launch from device — ~10-50 us overhead. Hostcall is competitive.
- **AMD GPU hostcall (ROCm)**: AMD's hostcall mechanism has ~15-30 us reported latency. Direct comparison possible.
- **CPU IPC mechanisms**: Unix domain sockets (~2-5 us), shared memory + futex (~0.5-1 us). Hostcall over PCIe is in the same ballpark as socket IPC.
- **NVIDIA CUDA Graphs**: Not a direct comparison but useful context for "how fast can GPU-host coordination be."

**Confidence**: medium (external numbers are approximate and hardware-dependent)

### Q: What benchmark scenarios are most informative?
A: Seven scenarios, ordered by priority:

1. **NOP Latency Sweep** — Core metric. Measures pure protocol overhead.
2. **Sustained Throughput** — Maximum hostcalls/sec under continuous load.
3. **Scalability Curve** — How throughput/latency change with thread count.
4. **Sharding A/B** — Validates per-block sharding benefit (already exists, formalize).
5. **File I/O Round-trip** — Real service latency (open + write + read + close).
6. **Bulk Transfer Bandwidth** — Sideband buffer throughput at various sizes.
7. **Mixed Workload** — Concurrent NOP + PRINT + FILE_IO from different blocks.

**Confidence**: high

## Existing Benchmark Analysis

### What exists (in `tests_benchmark.rs`, ~550 lines):

| Benchmark | Function | GPU Kernel | What it measures |
|-----------|----------|------------|-----------------|
| Latency sweep | `run_hostcall_latency_benchmark` | `hostcall_latency_bench` | NOP round-trip at [1,32,128,512] threads × [2x,4x] packet multipliers |
| Warp divergence | `run_warp_divergence_measurement` | `hostcall_latency_bench` | 1x32 vs 32x1 layout comparison, 3 runs averaged |
| Sharding A/B | `run_sharding_benchmark` | `hostcall_latency_bench_v2` | Global vs sharded pool at [1,4,16] blocks × 32 threads |

### Strengths:
- Clean per-thread result collection via mapped memory (3 u64s per thread: elapsed_ns, retries, completed)
- Percentile computation (p50/p95/p99) with proper nearest-rank method
- CAS retry counting on the GPU side (instrumented `hc_pop_free_counted`)
- Multiple configuration matrix (thread counts × packet counts)
- Warmup run in warp divergence test

### Gaps:
1. **No per-iteration latency**: Only per-thread averages. Loses tail latency information.
2. **No warmup in latency sweep**: The main benchmark (`run_hostcall_latency_benchmark`) has no warmup run.
3. **No host-side timing breakdown**: Cannot distinguish GPU spin-wait from host service time.
4. **No throughput saturation test**: All tests use only 10-20 iterations per thread. Need sustained load (1000+ iterations) to find steady-state throughput.
5. **No file I/O or TCP benchmarks**: Only NOP service is benchmarked.
6. **No bulk transfer benchmark**: Sideband buffer performance is untested.
7. **No regression tracking**: Results are printed to stdout only, not stored in machine-readable format.
8. **Fixed packet pool sizes**: Packets capped at 64 in the sweep. Should test larger pools.
9. **No standard deviation / coefficient of variation**: Cannot tell if results are stable.
10. **50ms sleep before shutdown**: This delay is included in wall time for throughput calculation in some paths. Should be excluded.

## Proposed Benchmark Suite

### Scenario 1: NOP Latency (Micro)
- **Metric**: Per-iteration round-trip latency histogram (p50, p95, p99, p999, max, stddev)
- **Method**: New kernel that writes per-iteration timestamps (not just per-thread totals). Launch 1 thread, 1000 iterations. Then 32 threads (1 warp), 100 iterations. Result buffer: `u64[num_threads * num_iters]` for individual latencies.
- **Warmup**: 1 run of (1 thread, 10 iters) discarded before measurement.
- **Output**: Latency histogram with bucket widths of 5 us.
- **Why**: Gives true tail latency. Current per-thread average masks worst-case calls.

### Scenario 2: Sustained Throughput
- **Metric**: Hostcalls/sec at steady state, measured over 5+ seconds
- **Method**: Launch 512 threads (16 blocks × 32), 2000 iterations each. Measure wall time for all completions. Subtract warmup. Report throughput = total_completed / wall_seconds.
- **Variants**: (a) All threads doing NOP, (b) Half NOP + half PRINT (mixed service)
- **Why**: Current benchmarks use only 10 iterations — too short to reach steady state. Listener adaptive polling may not even leave sleep phase.

### Scenario 3: Scalability Curve
- **Metric**: Throughput and mean latency as a function of active thread count
- **Method**: Sweep thread counts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]. Fixed 100 iters per thread, fixed packet pool (2x thread count, capped at 128). Plot throughput vs threads and latency vs threads.
- **Why**: Reveals the saturation point. Is it host CPU (single listener thread), PCIe bandwidth, or CAS contention?

### Scenario 4: Sharding Benefit (formalized)
- **Metric**: CAS retry reduction and throughput improvement, sharded vs unsharded
- **Method**: Same as existing `run_sharding_benchmark` but with more data points: [2, 4, 8, 16, 32] blocks × 32 threads. Fixed 100 iters. Report speedup ratio and CAS reduction percentage.
- **Why**: Existing benchmark only tests 3 configs. More data points show the scaling curve.

### Scenario 5: File I/O Round-trip
- **Metric**: End-to-end latency for open→write→read→close sequence
- **Method**: Single-thread kernel that opens a temp file, writes 48 bytes, reads them back, closes. Time each phase separately using `%globaltimer`. Repeat 50 times. Report per-phase and total latency.
- **Why**: File I/O is the primary use case. Need to know how much overhead hostcall adds vs native file I/O.

### Scenario 6: Bulk Transfer Bandwidth
- **Metric**: MB/s for sideband buffer reads/writes at various sizes
- **Method**: Single-thread kernel. Write [1KB, 4KB, 16KB, 64KB, 256KB, 1MB] to sideband, issue BULK_WRITE. Time from push to response. Report bandwidth = size / time.
- **Why**: Sideband is the path for large data. Need to know where PCIe bandwidth saturates.

### Scenario 7: Contention Breakdown
- **Metric**: Time spent in each hostcall phase (pop_free, fill+push, host_service, release)
- **Method**: Instrumented kernel with 4 timestamp captures per iteration. 32 threads, 100 iters. Report phase breakdown as percentage of total latency.
- **Why**: Identifies the bottleneck. Is it CAS contention on pop (allocation)? Host processing? PCIe latency? GPU spin-wait?

## Implementation Plan

### Phase 1: Enhanced kernel + harness (week 1)
1. **New benchmark kernel** `hostcall_latency_bench_v3` in `hostcall_kernels.rs`:
   - Per-iteration timestamp recording (not just per-thread totals)
   - Phase breakdown timestamps (4 timestamps per iteration)
   - Configurable via kernel args: `(buf, results, num_iters, mode)` where mode selects per-iter vs phase-breakdown
2. **Benchmark harness** in `tests_benchmark.rs`:
   - `BenchmarkResult` struct with all metrics (latencies, throughput, CAS stats, metadata)
   - `run_bench_scenario()` generic runner with warmup, multiple runs, statistical aggregation
   - JSON output option for regression tracking
3. **Warmup** added to all existing benchmarks

### Phase 2: New scenarios (week 2)
4. **File I/O benchmark kernel** `file_io_bench` with per-phase timing
5. **Bulk transfer benchmark kernel** `bulk_transfer_bench` with size sweep
6. **Scalability sweep** using existing `hostcall_latency_bench_v2` with more thread counts

### Phase 3: Reporting + CI (week 3)
7. **JSON output** for all benchmarks → `bench-results/` directory
8. **Comparison script** that reads two JSON files and reports regressions
9. **CI integration**: Run a subset (NOP latency + throughput) on every PR, flag >10% regression

### Code structure:
```
crates/core/gpu-host/src/
  tests_benchmark.rs        — existing, extend with new scenarios
  bench_harness.rs           — new: BenchmarkResult, statistical helpers, JSON output

crates/kernel/gpu-kernel/src/
  hostcall_kernels.rs        — existing, add v3 kernel with per-iter timestamps
  bench_kernels.rs           — new: file_io_bench, bulk_transfer_bench

bench-results/               — gitignored, machine-readable output
  baseline.json              — reference numbers
  latest.json                — most recent run
```

### Why not Criterion:
Criterion is designed for CPU benchmarks with nanosecond-level measurement via `rdtsc`. It assumes:
- The benchmark body can be run in a closure (GPU kernels need setup/teardown)
- Timing is done on the CPU side (we need GPU-side `%globaltimer`)
- No kernel launch overhead amortization needed

A custom harness is more appropriate because:
- GPU kernel launch has ~5-20 us overhead that must be amortized over many iterations
- Timing happens on the GPU, results come back via mapped memory
- The listener thread lifecycle is part of the benchmark setup
- Statistical analysis can reuse Criterion's approach (mean, stddev, CI) without the framework

## Unexpected Discoveries

1. **The 50ms sleep is included in throughput**: In `run_latency_bench_config`, `wall_elapsed` is measured before the 50ms sleep, so this is actually correct. But in `run_sharding_bench_config`, the `hc_buf.signal_shutdown()` is called AFTER the sleep, meaning the listener keeps running during the sleep. This is fine for correctness but the sleep could be reduced.

2. **Latency benchmark doesn't warmup**: `run_hostcall_latency_benchmark` goes straight to the benchmark matrix without a warmup. The first run (1 thread, 4 packets) may include PTX JIT time. The warp divergence test does warmup correctly.

3. **Packet pool exhaustion is silent**: If `hc_pop_free_counted` returns `NULL_INDEX`, the thread silently stops iterating. The per-thread `completed` count captures this, but the benchmark output shows `completed=X/Y` without flagging that pool exhaustion occurred. For a benchmark, pool exhaustion should be treated as a configuration error.

4. **Per-thread average masks contention**: With 512 threads doing 10 iterations each, thread 0 might average 50 us while thread 511 averages 500 us due to CAS contention. The current p99 of per-thread means captures some of this, but a true p99 across all 5120 iterations would be more informative.

5. **No nanosleep in bench kernel spin-wait**: The benchmark kernel spin-waits with a tight loop (`sys_spin_load_acquire_u32`), unlike the production code which uses `nanosleep`. This means the benchmark measures a slightly different code path than production.

## Open Questions

1. **TDR (Timeout Detection and Recovery)**: Windows has a default 2-second TDR timeout. With 512 threads × 2000 iterations at ~50 us each, total wall time is ~50 seconds. Need to either increase TDR timeout or use smaller iteration counts.

2. **Multi-GPU**: Should the benchmark suite support multi-GPU? Current code uses a single `CudaDevice`. Probably out of scope for v1.

3. **Host CPU affinity**: The listener thread's CPU core assignment affects latency. Should we pin the listener to a specific core? This would improve reproducibility but reduce portability.

4. **Async runtime benchmarks**: The project has `HostcallSession` and an async runtime (`async_rt.rs`). Should we benchmark the async runtime overhead vs raw hostcall? This would measure the Future/waker machinery cost.

## Impact on Downstream Tasks
- **bench-suite.2**: Implement the v3 kernel with per-iteration timestamps
- **bench-suite.3**: Implement the benchmark harness with JSON output
- **bench-suite.4**: Implement file I/O and bulk transfer benchmarks
- **bench-suite.5**: CI integration with regression detection
- **async-std theme**: Benchmarks will validate that `#[warp_cooperative]` async transforms don't add measurable overhead to hostcall latency
