# bench-suite.3: File I/O latency + throughput benchmark
**Cycle**: 322 | **Theme**: bench-suite | **Kind**: experiment | **Status**: done

## Summary
Implemented file I/O benchmark kernel (`file_io_bench`) with per-phase timing for
open/write/close/open/read/close cycles. Single-thread, 30 iterations with warmup.
All 30 iterations completed successfully, providing clear per-operation latency data.

## Changes Made

### 1. New `file_io_bench` kernel (`crates/kernel/gpu-kernel/src/hostcall_kernels.rs`)
- Times each phase individually: open-write, write, close-write, open-read, read, close-read
- Results layout: 2 header u64 + 6 u64 per iteration (phase latencies)
- Uses `gpu_instant_nanos()` for nanosecond timestamps

### 2. New `run_file_io_benchmark()` (`crates/core/gpu-host/src/tests_benchmark.rs`)
- Warmup run (3 iters) before measurement
- Reads per-phase timestamps, computes stats via `bench_harness::compute_stats()`
- Prints per-phase table (p50, p95, p99, mean, stddev in microseconds)
- Cleans up benchmark file after completion

### 3. Integration
- `main.rs`: `ONLY_TEST=file_io_bench` shortcut, added to `bench` composite
- Added to default test flow after throughput/scalability benchmarks

## Results

### Per-Phase Latency (30 iterations, 48-byte payload)
| Phase | p50 (μs) | p95 (μs) | p99 (μs) | mean (μs) | stddev (μs) |
|-------|----------|----------|----------|-----------|-------------|
| open-write | 911 | 1111 | 1415 | 943 | 111 |
| write | 175 | 594 | 933 | 243 | 190 |
| close-write | 732 | 972 | 1350 | 765 | 132 |
| open-read | 535 | 684 | 704 | 526 | 76 |
| read | 346 | 469 | 609 | 362 | 67 |
| close-read | 502 | 875 | 1087 | 536 | 147 |

### Summary
- **Total round-trip mean**: 3,375 μs (3.4 ms)
- **File I/O throughput**: 296 round-trips/s
- **All 30/30 iterations completed** (no pool exhaustion)
- **Wall time**: 101.3 ms

## Key Findings

1. **Open is the most expensive operation**: open-write at 943μs, open-read at 526μs.
   This makes sense — host file system operations (create, truncate, stat) are
   kernel-mode syscalls with I/O scheduler involvement.

2. **Write is surprisingly cheap**: 175μs p50 for 48 bytes. The OS buffered I/O layer
   means write() just copies to kernel buffer, doesn't hit disk.

3. **Read is moderately expensive**: 346μs p50. The host needs to seek + read + copy
   data back into the hostcall response packet.

4. **Close costs as much as open-read**: ~500-765μs. This is higher than expected —
   close() should be cheap. Possibly the host listener's fd table lookup + removal
   adds overhead, or the kernel is flushing buffers on close.

5. **Stddev is moderate**: CV = 11-78% depending on phase. write has the highest
   variability (CV=78%), likely due to OS buffer management decisions.

6. **Comparison to NOP hostcall**: NOP round-trip is ~13μs. File operations add
   12-72x overhead per hostcall, dominated by host-side syscall latency.

7. **Comparison to native file I/O**: A typical Linux file I/O round-trip
   (open+write+close) takes ~10-50μs with buffered I/O. Our ~1.9ms total for
   open+write+close means ~38-190x overhead from the PCIe + hostcall protocol.
   This is expected — each operation is a separate hostcall round-trip.

## Impact on Downstream Tasks
- **benchmarks epic**: Criteria 1 (throughput) and 2 (file I/O) are now met
- **public-api**: File I/O latency numbers useful for documentation
- Remaining: criterion 3 (scalability chart data) captured in bench-suite.2
- Remaining: criterion 4 (results documented with analysis) — need consolidated doc
