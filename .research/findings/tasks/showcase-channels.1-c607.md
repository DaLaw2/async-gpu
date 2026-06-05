# showcase-channels.1 — GPU Channels Example

## Status: DONE

## What was built

Standalone example at `examples/hostcall/gpu-channels/` demonstrating three GPU concurrency primitives using pre-built kernels from `gpu-kernel-std`:

### Demo 1: Oneshot Channels
- Kernel: `channel_oneshot_demo`
- 4 producer-consumer pairs, each communicating through `OneshotSlot<u32>`
- Consumers spawn first (poll Pending until slot filled), producers send on first poll
- Verifies values [42, 100, 255, 1337] received correctly

### Demo 2: MPSC Channel
- Kernel: `channel_mpsc_demo`
- 3 producers x 4 values -> 1 consumer via `MpscChannel<u32, 16>`
- Demonstrates backpressure (try_send retries on Full) and waker-based re-scheduling
- Verifies sum=312, count=12

### Demo 3: Async Executor
- Kernel: `executor_demo`
- 8 tasks: 4 WriteValueFuture (immediate) + 4 CounterFuture (multi-poll)
- All 32 lanes cooperatively drive the executor
- Verifies values + shared counter

## Files created

- `examples/hostcall/gpu-channels/Cargo.toml` — standalone crate using `async-gpu` facade
- `examples/hostcall/gpu-channels/src/main.rs` — host-side driver (~220 LOC)
- `examples/hostcall/gpu-channels/run.sh` — convenience launcher

## Files modified

- `scripts/ci-lint.sh` — added `gpu-channels` to example host checks

## Architecture

Uses `gpu::custom()` builder with `mapped_buffer()` for executor memory (256KB+).
Kernel pointers passed as `u64` device addresses. Results read from mapped memory
after kernel synchronization. No custom PTX or build.rs needed — uses built-in
kernels from the default PTX blob.

## Verification

- `cargo check --release` — clean (0 warnings)
- `bash scripts/ci-lint.sh` — all checks passed including new `gpu-channels`
