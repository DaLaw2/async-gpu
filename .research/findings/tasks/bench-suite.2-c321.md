# bench-suite.2: Hostcall throughput + scalability benchmark
**Cycle**: 321 | **Theme**: bench-suite | **Kind**: experiment | **Status**: done

## Summary
Implemented the v3 per-iteration benchmark kernel (`hostcall_latency_bench_v3`) with individual
iteration timestamps, a benchmark harness (`bench_harness.rs`) with `BenchmarkResult`, statistical
helpers (stddev, p999), and JSON output, plus throughput and scalability benchmark runners. All code
compiles and runs successfully on GPU. Results reveal clear contention patterns.

## Changes Made

### 1. New `hostcall_latency_bench_v3` kernel (`crates/kernel/gpu-kernel/src/hostcall_kernels.rs`)
- Records per-iteration round-trip latency individually (not just per-thread totals)
- Layout: header (3 u64/thread) + per-iter (1 u64/thread/iter)
- Uses shard-aware pop/push (same as v2)
- Takes `num_threads_total` arg for header size computation

### 2. New `bench_harness.rs` (`crates/core/gpu-host/src/bench_harness.rs`)
- `BenchmarkResult` struct: all metrics, formatted summary, JSON serialization
- `LatencyStats`: mean, stddev, min, p50, p95, p99, p999, max
- `compute_stats()` / `percentile_sorted()` statistical helpers
- `write_results_json()` for regression tracking (`bench-results/` dir, gitignored)
- `print_report()` for formatted multi-result output

### 3. New benchmark functions (`crates/core/gpu-host/src/tests_benchmark.rs`)
- `run_v3_bench()`: generic runner for v3 kernel with `BenchmarkResult` output
- `run_throughput_benchmark()`: sustained load at 1/32/128/512 threads
- `run_scalability_benchmark()`: sweep 1-1024 threads, find saturation point

### 4. Integration
- `main.rs`: `ONLY_TEST=throughput|scalability|bench` shortcuts
- `main.rs`: benchmarks added to default test flow after sharding benchmark
- `.gitignore`: `bench-results/` added

## Results

### Throughput Benchmark
| Scenario | Threads | Throughput | p50 | p99 | CAS/call | Completed |
|----------|---------|------------|-----|-----|----------|-----------|
| nop_micro_1t | 1 | 66,015/s | 13 μs | 14 μs | 0.00 | 100/100 |
| nop_warp_32t | 32 | 18,509/s | 1.57 ms | 3.11 ms | 29.50 | 1600/1600 |
| nop_sustained_128t | 128 | 15,441/s | 1.17 ms | 103 ms | 25.22 | 4147/6400 |
| nop_stress_512t | 512 | 182/s | 1.21 ms | 118 ms | 27.42 | 2084/10240 |

### Scalability Curve
| Threads | Throughput | p50 | CAS/call |
|---------|------------|-----|----------|
| 1 | 36,004/s | 13 μs | 0.00 |
| 2 | 4,021/s | 524 μs | 0.50 |
| 4 | 7,921/s | 523 μs | 2.59 |
| 8 | 15,455/s | 520 μs | 6.69 |
| 16 | 21,837/s | 529 μs | 9.96 |
| 32 | 19,147/s | 1.57 ms | 28.65 |
| 64 | 18,513/s | 1.03 ms | 33.25 |
| 128 | 15,486/s | 1.25 ms | 30.15 |
| 256 | 15,084/s | 1.41 ms | 19.64 |
| 512 | 155/s | 1.07 ms | 18.79 |
| 1024 | 215/s | 1.26 ms | 30.93 |

## Key Findings

1. **Single-thread baseline**: 13 μs round-trip (66K/s). This is the pure hostcall overhead — PCIe mapped memory write + host polling + response write + GPU spin-load.

2. **Optimal concurrent throughput at 16 threads**: 21.8K/s aggregate with 529 μs p50. Beyond this, packet pool contention dominates.

3. **CAS retry explosion at 32+ threads**: 28-33 retries per successful pop. The global free stack becomes a severe contention bottleneck.

4. **Pool exhaustion at 128+ threads**: With 64 max packets and 128 threads each doing 50 iters, threads starve. Only 4147/6400 iterations completed at 128t.

5. **Catastrophic at 512+ threads**: 155-215 calls/s with massive tail latency (p99 > 100ms). The packet pool is completely saturated.

6. **Sharding is critical**: The per-block sharding feature (already implemented) would dramatically help at 4+ blocks by reducing CAS contention on the free stack.

## Unexpected Discoveries

1. **2-thread latency jump**: Going from 1→2 threads shows a 40x p50 latency increase (13μs → 524μs). This suggests the PCIe coherence protocol adds significant cost even with just 2 threads writing to mapped memory.

2. **p50 stability across 2-256 threads**: p50 stays in the 520μs-1.6ms range. The real problem is tail latency and completion rate, not median performance.

3. **Per-iteration stddev is enormous**: stddev=14,231ns at 1 thread (mean=14,776ns). Coefficient of variation ≈ 96%. Individual hostcalls have high variance even in the best case.

## Impact on Downstream Tasks
- **bench-suite.3**: File I/O benchmark can reuse `run_v3_bench()` pattern with a file I/O kernel
- **bench-suite.4**: Contention analysis (phase breakdown) is the highest-value next step
- **public-api**: Results motivate documenting the "use sharding for multi-block" guidance
- **gpu-executor**: Executor design should minimize per-task hostcall overhead
