//! GPU Async Channels — showcase example.
//!
//! Demonstrates three GPU concurrency primitives:
//!
//! 1. **Oneshot Channels** — 4 producer-consumer pairs communicate through
//!    oneshot channels. Each producer sends a unique value; each consumer
//!    receives it and writes to a result slot. All 8 tasks run concurrently
//!    on the GPU executor.
//!
//! 2. **MPSC Channel** — 3 producer tasks send multiple values through a
//!    single MPSC (multi-producer, single-consumer) channel. The consumer
//!    accumulates a running sum and count. Demonstrates backpressure
//!    handling and waker-based re-scheduling.
//!
//! 3. **Async Executor** — 8 async tasks (4 `WriteValueFuture` + 4
//!    `CounterFuture`) are spawned on the GPU executor. The executor
//!    polls all tasks to completion, demonstrating dynamic cooperative
//!    scheduling entirely on the GPU. `WriteValueFuture` tasks complete
//!    in a single poll, while `CounterFuture` tasks yield (return
//!    `Poll::Pending`) and are re-scheduled by the executor — showing
//!    multi-round task lifecycle: QUEUED → RUNNING → PARKED → re-QUEUED
//!    → RUNNING → COMPLETED.
//!
//! # How It Works
//!
//! The kernel side (in `crates/kernel/gpu-kernel-std/src/hostcall_kernels.rs`) uses:
//! - `gpu_runtime::executor::GpuExecutor` — a cooperative task executor running on GPU
//! - `gpu_runtime::channel::OneshotSlot` — single-use channel with system-scope atomics
//! - `gpu_runtime::channel::MpscChannel` — lock-free ring buffer with waker integration
//! - Standard `core::future::Future` trait — tasks are polled by the executor
//!
//! The host side (this file) allocates mapped memory for the executor and result
//! buffers, launches the kernels, and verifies results. No manual PTX loading
//! is needed — the pre-built kernels from `gpu-kernel-std` are used via `gpu::custom()`.

use async_gpu::gpu;

/// Size of mapped memory for GpuExecutor + channel slots.
///
/// GpuExecutor is ~136KB (256 TaskSlots x 528 bytes each). We allocate 256KB
/// to leave room for oneshot slots or MPSC channel placed after the executor.
const EXECUTOR_MEM_SIZE: usize = 256 * 1024 + 256;

fn main() {
    println!("=== GPU Async Channels ===\n");

    let mut all_passed = true;

    // ----------------------------------------------------------------
    // Demo 1: Oneshot Channels — 4 producer-consumer pairs
    // ----------------------------------------------------------------
    //
    // The `channel_oneshot_demo` kernel:
    //   - Thread 0 initializes the GpuExecutor and 4 OneshotSlot<u32>
    //   - Spawns 4 consumers (each polls Pending until the slot is filled)
    //   - Spawns 4 producers (each writes a value + sets SENT state)
    //   - All 32 lanes enter executor.run() cooperatively
    //   - Executor polls tasks until all 8 complete
    //
    // Expected: consumers receive [42, 100, 255, 1337]
    println!("--- Demo 1: Oneshot Channels ---");
    match run_oneshot_demo() {
        Ok((values, spawned, completed, tasks_exec, polls)) => {
            let expected = [42u32, 100, 255, 1337];
            let pass = values == expected && spawned == 8 && completed == 8;

            println!("  Spawned 4 producer-consumer pairs (8 tasks total)");
            println!("  Executor stats: spawned={spawned}, completed={completed}");
            println!("  Task executions={tasks_exec}, total polls={polls}");
            println!("  Received values: {values:?}");
            println!(
                "  Verification: {} (expected {expected:?})\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 2: MPSC Channel — 3 producers, 1 consumer
    // ----------------------------------------------------------------
    //
    // The `channel_mpsc_demo` kernel:
    //   - Thread 0 initializes the GpuExecutor and MpscChannel<u32, 16>
    //   - Spawns 1 consumer (drains values, stores waker for re-scheduling)
    //   - Spawns 3 producers, each sending 4 values:
    //       Producer 0: [10, 20, 30, 40] (sum=100)
    //       Producer 1: [11, 21, 31, 41] (sum=104)
    //       Producer 2: [12, 22, 32, 42] (sum=108)
    //   - Consumer accumulates sum and count
    //
    // Expected: sum=312, count=12
    println!("--- Demo 2: MPSC Channel ---");
    match run_mpsc_demo() {
        Ok((sum, count, spawned, completed, tasks_exec, polls)) => {
            let pass = sum == 312 && count == 12 && spawned == 4 && completed == 4;

            println!("  3 producers x 4 values each -> 1 consumer");
            println!("  Executor stats: spawned={spawned}, completed={completed}");
            println!("  Task executions={tasks_exec}, total polls={polls}");
            println!("  Consumer received: sum={sum}, count={count}");
            println!(
                "  Verification: {} (expected sum=312, count=12)\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 3: Async Executor — dynamic multi-task scheduling
    // ----------------------------------------------------------------
    //
    // The `executor_demo` kernel:
    //   - Thread 0 initializes the GpuExecutor
    //   - Spawns 4 WriteValueFuture tasks: each writes a known value
    //     to a result slot on first poll (immediate completion)
    //   - Spawns 4 CounterFuture tasks: each yields on first poll
    //     (returns Pending), then completes on second poll — the
    //     executor dynamically re-schedules them
    //   - All 32 lanes cooperatively drive the executor
    //
    // Task lifecycle demonstrates dynamic scheduling:
    //   QUEUED → RUNNING → Ready (WriteValueFuture: single-poll)
    //   QUEUED → RUNNING → PARKED → re-QUEUED → RUNNING → Ready (CounterFuture: yield + re-schedule)
    //
    // Expected: values=[42, 100, 255, 1337], counter=4
    println!("--- Demo 3: Async Executor (Dynamic Scheduling) ---");
    match run_executor_demo() {
        Ok((values, counter, spawned, completed, tasks_exec, polls)) => {
            let expected_vals = [42u32, 100, 255, 1337];
            let pass = values == expected_vals
                && counter == 4
                && spawned == 8
                && completed == 8;

            println!("  8 async tasks spawned on GPU executor:");
            println!("    4 WriteValueFuture — complete in 1 poll (immediate)");
            println!("    4 CounterFuture   — yield + re-schedule (2 polls each)");
            println!("  Executor stats: spawned={spawned}, completed={completed}");
            println!("  Task executions={tasks_exec}, total polls={polls}");
            println!("  Written values: {values:?}");
            println!("  Shared counter: {counter} (4 tasks each increment once)");
            println!("  Dynamic scheduling: CounterFuture tasks parked and re-queued");
            println!(
                "  Verification: {} (expected values={expected_vals:?}, counter=4)\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Summary
    // ----------------------------------------------------------------
    if all_passed {
        println!("=== All demos passed! ===");
    } else {
        println!("=== Some demos failed or were skipped ===");
    }
}

/// Run the oneshot channel demo kernel.
///
/// Returns (received_values, spawned, completed, tasks_executed, polls_total).
fn run_oneshot_demo() -> Result<([u32; 4], u32, u32, u32, u32), Box<dyn std::error::Error>> {
    let ctx = gpu::custom("channel_oneshot_demo")
        .threads(32)
        .prepare()?;

    // Allocate mapped memory for executor + oneshot slots
    let executor_buf = ctx.mapped_buffer::<u8>(EXECUTOR_MEM_SIZE)?;
    let results_buf = ctx.mapped_buffer::<u32>(16)?;

    let exec_ptr = executor_buf.dev_ptr() as u64;
    let res_ptr = results_buf.dev_ptr() as u64;

    let _result = unsafe { ctx.launch((exec_ptr, res_ptr))? };

    // Read results from mapped memory
    let spawned = unsafe { results_buf.read(0) };
    let completed = unsafe { results_buf.read(1) };
    let tasks_exec = unsafe { results_buf.read(2) };
    let polls = unsafe { results_buf.read(3) };
    let values = [
        unsafe { results_buf.read(4) },
        unsafe { results_buf.read(5) },
        unsafe { results_buf.read(6) },
        unsafe { results_buf.read(7) },
    ];

    Ok((values, spawned, completed, tasks_exec, polls))
}

/// Run the MPSC channel demo kernel.
///
/// Returns (sum, count, spawned, completed, tasks_executed, polls_total).
fn run_mpsc_demo() -> Result<(u32, u32, u32, u32, u32, u32), Box<dyn std::error::Error>> {
    let ctx = gpu::custom("channel_mpsc_demo")
        .threads(32)
        .prepare()?;

    // Allocate mapped memory for executor + MPSC channel
    let executor_buf = ctx.mapped_buffer::<u8>(EXECUTOR_MEM_SIZE)?;
    let results_buf = ctx.mapped_buffer::<u32>(8)?;

    let exec_ptr = executor_buf.dev_ptr() as u64;
    let res_ptr = results_buf.dev_ptr() as u64;

    let _result = unsafe { ctx.launch((exec_ptr, res_ptr))? };

    // Read results from mapped memory
    let spawned = unsafe { results_buf.read(0) };
    let completed = unsafe { results_buf.read(1) };
    let tasks_exec = unsafe { results_buf.read(2) };
    let polls = unsafe { results_buf.read(3) };
    let sum = unsafe { results_buf.read(4) };
    let count = unsafe { results_buf.read(5) };

    Ok((sum, count, spawned, completed, tasks_exec, polls))
}

/// Run the executor demo kernel.
///
/// Returns (values, counter, spawned, completed, tasks_executed, polls_total).
fn run_executor_demo() -> Result<([u32; 4], u32, u32, u32, u32, u32), Box<dyn std::error::Error>> {
    let ctx = gpu::custom("executor_demo")
        .threads(32)
        .prepare()?;

    // Allocate mapped memory for executor
    let executor_buf = ctx.mapped_buffer::<u8>(EXECUTOR_MEM_SIZE)?;
    let results_buf = ctx.mapped_buffer::<u32>(16)?;

    let exec_ptr = executor_buf.dev_ptr() as u64;
    let res_ptr = results_buf.dev_ptr() as u64;

    let _result = unsafe { ctx.launch((exec_ptr, res_ptr))? };

    // Read results from mapped memory
    let spawned = unsafe { results_buf.read(0) };
    let completed = unsafe { results_buf.read(1) };
    let tasks_exec = unsafe { results_buf.read(2) };
    let polls = unsafe { results_buf.read(3) };
    let values = [
        unsafe { results_buf.read(4) },
        unsafe { results_buf.read(5) },
        unsafe { results_buf.read(6) },
        unsafe { results_buf.read(7) },
    ];
    let counter = unsafe { results_buf.read(8) };

    Ok((values, counter, spawned, completed, tasks_exec, polls))
}
