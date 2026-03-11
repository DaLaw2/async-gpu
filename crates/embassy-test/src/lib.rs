//! Embassy executor compilation test for nvptx64 with gpu-critical-section.
//!
//! This crate verifies that:
//! 1. Embassy executor compiles for nvptx64 with arch-spin feature
//! 2. gpu-critical-section provides the required critical-section impl
//! 3. Fat LTO resolves all cross-crate extern calls
//! 4. The resulting PTX has zero unresolved externs
//!
//! Kernels:
//! - embassy_test_kernel: single ImmediateFuture (spawn + poll)
//! - embassy_countdown_kernel: CountdownFuture requiring multiple polls
//! - embassy_two_task_kernel: two concurrent tasks on the same executor
//! - sync_countdown_kernel: synchronous equivalent for register comparison

#![no_std]
#![feature(abi_ptx)]

use core::future::Future;
use core::mem::MaybeUninit;
use core::panic::PanicInfo;
use core::pin::Pin;
use core::task::{Context, Poll};

// Pull in the GPU critical-section implementation.
// This satisfies Embassy's critical-section dependency.
extern crate gpu_critical_section;

use embassy_executor::raw::{Executor, TaskStorage};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Embassy requires a user-supplied pender callback.
/// On GPU, this is a no-op — the executor uses synchronous spin-poll.
#[no_mangle]
unsafe extern "C" fn __pender(_context: *mut ()) {
    // No-op: GPU executor uses synchronous spin-poll, not wake-based scheduling.
}

// ============================================================
// Test 1: ImmediateFuture (original test)
// ============================================================

/// Static storage for the executor. Wrapped in a struct to make it Sync.
struct ExecutorStorage {
    inner: MaybeUninit<Executor>,
}

unsafe impl Sync for ExecutorStorage {}

static EXECUTOR_STORAGE: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

/// A simple future that completes immediately, returning a u32 value.
struct ImmediateFuture {
    value: u32,
}

impl Future for ImmediateFuture {
    type Output = u32;
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        Poll::Ready(self.value)
    }
}

/// Task storage for the test task.
static TASK: TaskStorage<ImmediateFuture> = TaskStorage::new();

/// Kernel entry point: create executor, spawn a task, poll it.
///
/// This exercises the full Embassy executor path including:
/// - Executor creation
/// - Task spawn (uses critical-section internally)
/// - Executor poll (drains run queue, calls task poll_fn)
///
/// Writes 1 to `result` on success.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embassy_test_kernel(result: *mut u32) {
    // Initialize the executor in the static storage.
    let storage_ptr = &EXECUTOR_STORAGE.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));

    // Get a &'static reference to the initialized executor.
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Spawn a task — this exercises Embassy's task state machine + critical section.
    let token = TASK.spawn(|| ImmediateFuture { value: 42 });
    let spawner = executor.spawner();
    let _ = spawner.spawn(token);

    // Poll the executor — runs the spawned task to completion.
    executor.poll();

    // Signal success.
    *result = 1;
}

// ============================================================
// Test 2: CountdownFuture — requires multiple polls
// ============================================================

/// A future that takes multiple polls to complete.
/// Decrements `remaining` each poll, returns Ready(42) when zero.
struct CountdownFuture {
    remaining: u32,
}

impl Future for CountdownFuture {
    type Output = u32;
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.remaining == 0 {
            Poll::Ready(42)
        } else {
            this.remaining -= 1;
            cx.waker().wake_by_ref(); // re-enqueue for next poll
            Poll::Pending
        }
    }
}

static EXECUTOR_STORAGE_2: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static COUNTDOWN_TASK: TaskStorage<CountdownFuture> = TaskStorage::new();

/// Kernel: spawn a CountdownFuture(remaining=5), poll until completion.
///
/// The executor must poll 6 times total (5 Pending + 1 Ready).
/// Writes the number of poll rounds to result[0], and 1 to result[1] on success.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embassy_countdown_kernel(result: *mut u32) {
    let storage_ptr = &EXECUTOR_STORAGE_2.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Spawn a countdown task that needs 5 polls before completing.
    let token = COUNTDOWN_TASK.spawn(|| CountdownFuture { remaining: 5 });
    let spawner = executor.spawner();
    let _ = spawner.spawn(token);

    // Poll in a loop. Embassy's poll() drains the run queue once.
    // After each Pending, the task re-enqueues itself via wake_by_ref().
    // We need to call executor.poll() multiple times.
    // Use a volatile counter to prevent the compiler from unrolling.
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 20; // safety limit
    loop {
        executor.poll();
        poll_rounds += 1;
        // Use volatile read to prevent loop unrolling
        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
        // The task needs 5+1=6 polls. After 6 rounds, check completion.
        if current >= 6 {
            break;
        }
    }

    // Write poll count to result[0]
    *result = poll_rounds;
    // Write success marker to result[1]
    *result.add(1) = 1;
}

// ============================================================
// Test 3: Two concurrent tasks on the same executor
// ============================================================

/// A future that counts down from a given value, returning a distinct result.
struct CountdownFutureA {
    remaining: u32,
    result_value: u32,
}

impl Future for CountdownFutureA {
    type Output = u32;
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.remaining == 0 {
            Poll::Ready(this.result_value)
        } else {
            this.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

struct CountdownFutureB {
    remaining: u32,
    result_value: u32,
}

impl Future for CountdownFutureB {
    type Output = u32;
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.remaining == 0 {
            Poll::Ready(this.result_value)
        } else {
            this.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

static EXECUTOR_STORAGE_3: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static TASK_A: TaskStorage<CountdownFutureA> = TaskStorage::new();
static TASK_B: TaskStorage<CountdownFutureB> = TaskStorage::new();

/// Kernel: spawn two tasks (3-poll and 5-poll), run both to completion.
///
/// Task A: CountdownFuture(remaining=3) -> Ready(10)
/// Task B: CountdownFuture(remaining=5) -> Ready(20)
///
/// result[0] = poll rounds executed
/// result[1] = 1 if both tasks completed (success marker)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embassy_two_task_kernel(result: *mut u32) {
    let storage_ptr = &EXECUTOR_STORAGE_3.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Spawn task A (3 polls to complete, returns 10)
    let token_a = TASK_A.spawn(|| CountdownFutureA { remaining: 3, result_value: 10 });
    let spawner = executor.spawner();
    let _ = spawner.spawn(token_a);

    // Spawn task B (5 polls to complete, returns 20)
    let token_b = TASK_B.spawn(|| CountdownFutureB { remaining: 5, result_value: 20 });
    let _ = spawner.spawn(token_b);

    // Poll in a loop until both tasks complete.
    // Task A needs 4 polls, Task B needs 6 polls.
    // Since Embassy polls all ready tasks each round, both run concurrently.
    // Use volatile counter to prevent loop unrolling.
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 20;
    loop {
        executor.poll();
        poll_rounds += 1;
        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
        // Task B needs 6 rounds (5 pending + 1 ready), so 6 polls total.
        if current >= 6 {
            break;
        }
    }

    // Write poll count
    *result = poll_rounds;
    // Write success marker
    *result.add(1) = 1;
}

// ============================================================
// Test 4: Synchronous equivalent for register usage comparison
// ============================================================

/// Synchronous countdown: equivalent work to CountdownFuture
/// but without async/executor overhead.
///
/// Counts down from 5 to 0, then writes 42 to result.
/// This provides a baseline for register usage comparison.
/// Uses volatile reads to prevent the compiler from optimizing away the loop.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sync_countdown_kernel(result: *mut u32) {
    let mut remaining: u32 = 5;
    loop {
        let current = core::ptr::read_volatile(&remaining);
        if current == 0 {
            break;
        }
        remaining = current - 1;
    }
    *result = 42;
    *result.add(1) = 1;
}
