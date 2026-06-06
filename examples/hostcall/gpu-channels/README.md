# GPU Async Channels

GPU concurrency primitives — oneshot channels, MPSC channels, and an async
executor with dynamic task scheduling, all running entirely on the GPU.

## What It Demonstrates

- **Oneshot Channels** — 4 producer-consumer pairs communicate through
  `OneshotSlot<u32>`. Each producer sends a unique value; each consumer
  receives it. All 8 tasks run concurrently on the GPU executor.
- **MPSC Channel** — 3 producers send multiple values through a single
  `MpscChannel<u32, 16>`. The consumer accumulates a running sum and count.
  Demonstrates backpressure handling and waker-based re-scheduling.
- **Async Executor** — 8 async tasks (4 `WriteValueFuture` + 4
  `CounterFuture`) are spawned on `GpuExecutor`. Shows dynamic cooperative
  scheduling: single-poll completion vs yield-and-reschedule lifecycle
  (QUEUED -> RUNNING -> PARKED -> re-QUEUED -> RUNNING -> COMPLETED).

## Running

```bash
cd examples/hostcall/gpu-channels
cargo run --release
```

## How It Works

### Kernel Side

- `gpu_runtime::executor::GpuExecutor` — cooperative task executor on GPU
- `gpu_runtime::channel::OneshotSlot` — single-use channel with system-scope atomics
- `gpu_runtime::channel::MpscChannel` — lock-free ring buffer with waker integration
- Standard `core::future::Future` — tasks are polled by the executor

### Host Side

The host allocates mapped memory for the executor and result buffers,
launches the kernels via `gpu::custom()`, and verifies results.

## Expected Output

```
=== GPU Async Channels ===

--- Demo 1: Oneshot Channels ---
  Spawned 4 producer-consumer pairs (8 tasks total)
  Executor stats: spawned=8, completed=8
  Received values: [42, 100, 255, 1337]
  Verification: PASSED

--- Demo 2: MPSC Channel ---
  3 producers x 4 values each -> 1 consumer
  Executor stats: spawned=4, completed=4
  Consumer received: sum=312, count=12
  Verification: PASSED

--- Demo 3: Async Executor (Dynamic Scheduling) ---
  8 async tasks spawned on GPU executor:
    4 WriteValueFuture — complete in 1 poll (immediate)
    4 CounterFuture   — yield + re-schedule (2 polls each)
  Written values: [42, 100, 255, 1337]
  Shared counter: 4 (4 tasks each increment once)
  Dynamic scheduling: CounterFuture tasks parked and re-queued
  Verification: PASSED

=== All demos passed! ===
```
