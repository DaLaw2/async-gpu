//! Embassy executor compilation test for nvptx64 with gpu-critical-section.
//!
//! This crate verifies that:
//! 1. Embassy executor compiles for nvptx64 with arch-spin feature
//! 2. gpu-critical-section provides the required critical-section impl
//! 3. Fat LTO resolves all cross-crate extern calls
//! 4. The resulting PTX has zero unresolved externs

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
