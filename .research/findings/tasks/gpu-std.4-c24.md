# gpu-std.4: GPU Instant via %globaltimer + SystemTime via hostcall
**Cycle**: 24 | **Theme**: gpu-std | **Kind**: experiment | **Status**: done

## Summary
Implemented GPU-side timing primitives: (1) GPU Instant using PTX `%globaltimer` register — a 64-bit monotonic nanosecond counter, and (2) SystemTime via hostcall to host. Both work correctly. The %globaltimer measured 54272 ns for 1000 loop iterations. SystemTime via hostcall returns correct Unix epoch seconds. stdin hostcall service was also implemented (SERVICE_STDIN) but not tested interactively (skip_stdin=1 to avoid blocking).

## Findings
### Q: Does PTX %globaltimer give accurate nanosecond timing on GPU?
A: **Yes.** The `%globaltimer` special register provides a 64-bit monotonic nanosecond counter available on SM 3.0+ (all modern GPUs). Inline asm `mov.u64 %result, %globaltimer;` successfully reads it.

Measurement: 1000 iterations of `wrapping_add` (with volatile reads to prevent optimization) takes 54272 ns = ~54 ns per iteration. This is reasonable for GPU clock speeds (~1.5 GHz = ~0.7 ns/cycle, so ~77 cycles per volatile-read iteration including memory traffic).

**Key detail**: The initial implementation had a dummy loop that LLVM optimized away entirely, giving delta=0. Fix: use `core::ptr::read_volatile` on the loop counter and `core::ptr::write_volatile` on the accumulator to prevent dead-code elimination.

**Confidence**: high (verified on hardware)

### Q: Can SystemTime be obtained via hostcall to host clock?
A: **Yes.** Added SERVICE_TIME (opcode 9) to the hostcall protocol. The GPU sends a request with no payload; the host responds with (seconds_since_epoch, nanoseconds_within_second) in slots 0 and 1. Measured: epoch_secs=1773248951, nanos=603434800 — correct Unix time for the test date.

**Confidence**: high (verified on hardware)

### Q: Can we read from stdin via a new hostcall SERVICE_STDIN opcode?
A: **Implemented but not interactively tested.** SERVICE_STDIN (opcode 8) was added to the protocol. The GPU sends max_bytes_to_read in slot 0; the host calls `stdin().read_line()` and returns bytes_read + data in the response payload (up to 56 bytes).

The test kernel has a `skip_stdin` parameter — set to 1 during automated testing to avoid blocking on stdin. Interactive testing would require piping input to the process.

**Confidence**: medium (code written and compiles, host handler implemented, but not tested end-to-end)

## Implementation Details

### GPU-side (crates/gpu-kernel)
- `gpu_instant_nanos()`: inline asm `mov.u64 {result}, %globaltimer;`
- `gpu_hostcall_stdin_read()`: SERVICE_STDIN request, copies response data to caller buffer
- `gpu_hostcall_time()`: SERVICE_TIME request, returns (secs, nanos) tuple
- `hostcall_stdin_time_test` kernel: exercises all three, writes results to u64[4]

### Protocol (crates/gpu-protocol)
- `SERVICE_STDIN = 8`, `SERVICE_TIME = 9`
- `STDIN_MAX_READ_LEN = 56` (7 slots × 8 bytes)
- Payload layout documentation for both services

### Host-side (crates/gpu-host)
- `handle_stdin()`: calls `std::io::stdin().read_line()`, copies to response payload
- `handle_time()`: calls `SystemTime::now().duration_since(UNIX_EPOCH)`, writes secs + nanos

## Test Results
### hostcall_stdin_time_test (skip_stdin=1)
- Config: 1 block × 32 threads, 4 packets, skip_stdin=1
- Result: **PASSED**
- GPU Instant delta: 54272 ns (1000 volatile-read iterations)
- Host SystemTime: epoch_secs=1773248951, nanos=603434800
- stdin: skipped (skip_stdin=1)

## Unexpected Discoveries

1. **LLVM aggressively optimizes away timer benchmarks.** The initial loop `while i < 1000 { dummy += i; i += 1; }` was completely eliminated by LLVM, giving %globaltimer delta = 0. Only volatile reads on the loop counter prevent this. This is consistent with the async-runtime.3 finding about LLVM loop unrolling.

2. **%globaltimer resolution is sub-microsecond.** 54 ns per loop iteration is well within the timer's precision. This makes %globaltimer suitable for microsecond-level performance measurement on GPU.

## Open Questions

1. **%globaltimer vs %clock64**: `%clock64` counts SM clock cycles (variable with DVFS), while `%globaltimer` counts wall-clock nanoseconds (fixed frequency). For benchmarking, `%globaltimer` is better. For cycle-accurate profiling, `%clock64` is needed.

2. **Interactive stdin test**: The stdin handler blocks the host listener thread while waiting for input. For production use, stdin should be non-blocking or handled on a separate thread.

## Impact on Downstream Tasks
- **gpu-std theme**: All 3 success criteria met (println ✓, File I/O ✓, libc facade ✓). Plus stdin + time as bonus.
- **integration.4** (benchmarks): GPU Instant can now be used for in-kernel timing measurements.
- **VectorWare parity**: stdin + time were both demonstrated by VectorWare. We now have matching capabilities (stdin via hostcall, Instant via %globaltimer).
