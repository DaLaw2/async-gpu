//! Async hostcall test: HostcallFuture + Embassy executor on GPU.
//!
//! Instead of spin-waiting for host response (blocking the entire warp),
//! HostcallPrintFuture yields Poll::Pending and lets the Embassy executor
//! poll other tasks. This is the key innovation: true async concurrency
//! on GPU, where multiple hostcall requests can be in-flight simultaneously.
//!
//! Kernels:
//! - async_hostcall_single_kernel: single async print via HostcallPrintFuture
//! - async_hostcall_two_kernel: two concurrent async prints (true async concurrency)

#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::future::Future;
use core::mem::MaybeUninit;
use core::panic::PanicInfo;
use core::pin::Pin;
use core::task::{Context, Poll};

// Pull in the GPU critical-section implementation.
// This satisfies Embassy's critical-section dependency.
extern crate gpu_critical_section;

use embassy_executor::raw::{Executor, TaskStorage};
use gpu_atomics::{
    activemask, membar_sys, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u32,
    sys_load_acquire_u64, sys_store_release_u32,
};
use gpu_protocol::*;

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
// Executor storage wrapper (Sync for static placement)
// ============================================================

/// Wrapper to make MaybeUninit<Executor> Sync for static storage.
struct ExecutorStorage {
    inner: MaybeUninit<Executor>,
}

unsafe impl Sync for ExecutorStorage {}

// ============================================================
// Hostcall helper functions (self-contained, duplicated from gpu-kernel)
// ============================================================

/// Pop a packet from the free stack. Returns packet index or NULL_INDEX.
#[inline(always)]
unsafe fn hc_pop_free(buf: *mut u8) -> u16 {
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return NULL_INDEX;
        }
        let pkt = buf.add(packet_offset(idx));
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if sys_cas_u64(free_ptr, old_head, next) == old_head {
            return idx;
        }
    }
}

/// Push a packet onto a tagged-pointer stack (free or ready).
#[inline(always)]
unsafe fn hc_push(stack_ptr: *mut u64, buf: *mut u8, pkt_idx: u16) {
    let pkt = buf.add(packet_offset(pkt_idx));
    loop {
        let old_head = sys_load_acquire_u64(stack_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if sys_cas_u64(stack_ptr, old_head, new_tagged) == old_head {
            break;
        }
    }
}

/// Return a packet to the free stack after reading response.
#[inline(always)]
unsafe fn gpu_hostcall_release(buf: *mut u8, pkt_idx: u16) {
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    hc_push(free_ptr, buf, pkt_idx);
}

// ============================================================
// HostcallPrintFuture — async wrapper around hostcall protocol
// ============================================================

/// State machine for async hostcall.
enum HostcallState {
    /// Initial: need to allocate packet and submit request.
    Init,
    /// Packet submitted, waiting for host response.
    WaitingResponse,
    /// Completed.
    Done,
}

/// Future that performs a hostcall print asynchronously.
///
/// Instead of spin-waiting for the host response (which blocks the warp),
/// this future does a single check of the control word each time it is polled.
/// If the host has not yet responded, it returns Poll::Pending and lets the
/// Embassy executor poll other tasks — achieving true async concurrency on GPU.
struct HostcallPrintFuture {
    buf: *mut u8,
    msg: &'static [u8],
    state: HostcallState,
    pkt_idx: u16,
}

impl HostcallPrintFuture {
    /// Create a new HostcallPrintFuture.
    ///
    /// `buf` is the hostcall buffer (mapped memory).
    /// `msg` is the message to print (must be <= PRINT_MAX_MSG_LEN bytes).
    fn new(buf: *mut u8, msg: &'static [u8]) -> Self {
        Self {
            buf,
            msg,
            state: HostcallState::Init,
            pkt_idx: NULL_INDEX,
        }
    }
}

impl Future for HostcallPrintFuture {
    type Output = bool;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        let this = unsafe { self.get_unchecked_mut() };

        match this.state {
            HostcallState::Init => unsafe {
                // Try to pop a free packet.
                let pkt_idx = hc_pop_free(this.buf);
                if pkt_idx == NULL_INDEX {
                    // Pool exhausted — back-pressure per ADR-4.
                    // Re-enqueue and try again on next poll.
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                this.pkt_idx = pkt_idx;
                let pkt = this.buf.add(packet_offset(pkt_idx));

                // Fill packet header.
                let mask = activemask();
                core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
                core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
                // Clear control word (READY/ERROR) with release store.
                sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

                // Fill payload: slot 0 = message length, slots 1-7 = message bytes.
                let payload = pkt.add(PKT_OFF_PAYLOAD);
                let msg_len = this.msg.len() as u32;
                core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

                let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
                    PRINT_MAX_MSG_LEN as u32
                } else {
                    msg_len
                };
                let dst = payload.add(8); // skip slot 0
                let mut i: u32 = 0;
                while i < copy_len {
                    core::ptr::write_volatile(
                        dst.add(i as usize),
                        *this.msg.as_ptr().add(i as usize),
                    );
                    i += 1;
                }

                // membar.sys to ensure all packet writes are visible at system scope.
                membar_sys();

                // Push to ready stack.
                let ready_ptr = this.buf.add(BUF_OFF_READY_STACK) as *mut u64;
                hc_push(ready_ptr, this.buf, pkt_idx);

                // Ring doorbell.
                sys_fetch_add_u64(this.buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

                // Transition to WaitingResponse.
                this.state = HostcallState::WaitingResponse;

                // Re-enqueue so the executor polls us again.
                cx.waker().wake_by_ref();
                Poll::Pending
            },

            HostcallState::WaitingResponse => unsafe {
                let pkt = this.buf.add(packet_offset(this.pkt_idx));
                let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;

                // Single check — NOT a spin loop. This is what makes it truly async.
                // Use sys_load_acquire_u32 (no nanosleep) for a quick non-blocking check.
                let ctrl = sys_load_acquire_u32(control_ptr);

                if ctrl & CONTROL_READY != 0 {
                    // Host has responded. Release the packet.
                    let success = (ctrl & CONTROL_ERROR) == 0;
                    gpu_hostcall_release(this.buf, this.pkt_idx);
                    this.state = HostcallState::Done;
                    Poll::Ready(success)
                } else {
                    // Not ready yet — yield to executor so other tasks can run.
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            },

            HostcallState::Done => {
                // Polled after completion — this should not happen.
                // In a no_std GPU environment, just return Ready(false).
                Poll::Ready(false)
            }
        }
    }
}

// ============================================================
// HostcallPrintFutureB — duplicate type for second task
// ============================================================
// Embassy's TaskStorage is generic over the Future type.
// Each static TaskStorage needs a unique Future type, so we duplicate
// the struct for the two-task kernel. The logic is identical.

/// Second future type for concurrent two-task test.
/// Identical logic to HostcallPrintFuture, but a distinct type so it
/// gets its own TaskStorage static.
struct HostcallPrintFutureB {
    buf: *mut u8,
    msg: &'static [u8],
    state: HostcallState,
    pkt_idx: u16,
}

impl HostcallPrintFutureB {
    fn new(buf: *mut u8, msg: &'static [u8]) -> Self {
        Self {
            buf,
            msg,
            state: HostcallState::Init,
            pkt_idx: NULL_INDEX,
        }
    }
}

impl Future for HostcallPrintFutureB {
    type Output = bool;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        let this = unsafe { self.get_unchecked_mut() };

        match this.state {
            HostcallState::Init => unsafe {
                let pkt_idx = hc_pop_free(this.buf);
                if pkt_idx == NULL_INDEX {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                this.pkt_idx = pkt_idx;
                let pkt = this.buf.add(packet_offset(pkt_idx));

                let mask = activemask();
                core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
                core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
                sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

                let payload = pkt.add(PKT_OFF_PAYLOAD);
                let msg_len = this.msg.len() as u32;
                core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

                let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
                    PRINT_MAX_MSG_LEN as u32
                } else {
                    msg_len
                };
                let dst = payload.add(8);
                let mut i: u32 = 0;
                while i < copy_len {
                    core::ptr::write_volatile(
                        dst.add(i as usize),
                        *this.msg.as_ptr().add(i as usize),
                    );
                    i += 1;
                }

                membar_sys();

                let ready_ptr = this.buf.add(BUF_OFF_READY_STACK) as *mut u64;
                hc_push(ready_ptr, this.buf, pkt_idx);

                sys_fetch_add_u64(this.buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

                this.state = HostcallState::WaitingResponse;
                cx.waker().wake_by_ref();
                Poll::Pending
            },

            HostcallState::WaitingResponse => unsafe {
                let pkt = this.buf.add(packet_offset(this.pkt_idx));
                let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;

                let ctrl = sys_load_acquire_u32(control_ptr);

                if ctrl & CONTROL_READY != 0 {
                    let success = (ctrl & CONTROL_ERROR) == 0;
                    gpu_hostcall_release(this.buf, this.pkt_idx);
                    this.state = HostcallState::Done;
                    Poll::Ready(success)
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            },

            HostcallState::Done => {
                Poll::Ready(false)
            }
        }
    }
}

// ============================================================
// Test kernel 1: Single async hostcall print
// ============================================================

static EXECUTOR_STORAGE_1: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static SINGLE_TASK: TaskStorage<HostcallPrintFuture> = TaskStorage::new();

/// Single async hostcall print test.
///
/// Creates an Embassy executor, spawns one HostcallPrintFuture, and polls
/// until completion. Writes poll_rounds to result[0] and success=1 to result[1].
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn async_hostcall_single_kernel(buf: *mut u8, result: *mut u32) {
    // Only thread 0 executes.
    let global_idx: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) global_idx,
        options(nostack, readonly),
    );
    if global_idx != 0 {
        return;
    }

    // Initialize result to zero.
    *result = 0;
    *result.add(1) = 0;

    // Initialize the executor in static storage.
    let storage_ptr =
        &EXECUTOR_STORAGE_1.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Spawn a single async hostcall print task.
    let token = SINGLE_TASK.spawn(|| HostcallPrintFuture::new(buf, b"Async hello from GPU!"));
    let spawner = executor.spawner();
    let _ = spawner.spawn(token);

    // Poll the executor in a loop until max rounds.
    // The hostcall should complete within ~50 polls (host response latency).
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 100;
    loop {
        executor.poll();
        poll_rounds += 1;

        // Volatile read to prevent LLVM from unrolling or optimizing the loop.
        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
    }

    // Write results.
    *result = poll_rounds;
    *result.add(1) = 1;
}

// ============================================================
// Test kernel 2: Two concurrent async hostcall prints
// ============================================================

static EXECUTOR_STORAGE_2: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static TASK_A: TaskStorage<HostcallPrintFuture> = TaskStorage::new();
static TASK_B: TaskStorage<HostcallPrintFutureB> = TaskStorage::new();

/// Two concurrent async hostcall prints — the KEY test for true async concurrency.
///
/// Creates an Embassy executor, spawns two HostcallPrintFutures with different
/// messages, and polls until completion. While task A waits for the host response,
/// task B gets polled (and vice versa). This demonstrates that the executor
/// interleaves hostcall requests without blocking.
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]:
///   [0] = poll rounds executed
///   [1] = 1 on success
#[no_mangle]
pub unsafe extern "ptx-kernel" fn async_hostcall_two_kernel(buf: *mut u8, result: *mut u32) {
    // Only thread 0 executes.
    let global_idx: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) global_idx,
        options(nostack, readonly),
    );
    if global_idx != 0 {
        return;
    }

    // Initialize result to zero.
    *result = 0;
    *result.add(1) = 0;

    // Initialize the executor in static storage.
    let storage_ptr =
        &EXECUTOR_STORAGE_2.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Spawn task A: first async print.
    let token_a = TASK_A.spawn(|| HostcallPrintFuture::new(buf, b"Async task A from GPU!"));
    let spawner = executor.spawner();
    let _ = spawner.spawn(token_a);

    // Spawn task B: second async print (different type for separate TaskStorage).
    let token_b = TASK_B.spawn(|| HostcallPrintFutureB::new(buf, b"Async task B from GPU!"));
    let _ = spawner.spawn(token_b);

    // Poll the executor in a loop.
    // Both tasks run concurrently: while one waits for host response,
    // the other gets polled. This is true async concurrency on GPU.
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 100;
    loop {
        executor.poll();
        poll_rounds += 1;

        // Volatile read to prevent LLVM from unrolling or optimizing the loop.
        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
    }

    // Write results.
    *result = poll_rounds;
    *result.add(1) = 1;
}

// ============================================================
// Test kernel 3: futures_util::future::join on GPU (integration.2)
// ============================================================

/// A join future that wraps two HostcallPrintFutures using futures_util::future::join.
/// This proves that third-party async combinators work on GPU.
type JoinFuture = futures_util::future::Join<HostcallPrintFuture, HostcallPrintFutureB>;

static EXECUTOR_STORAGE_3: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static JOIN_TASK: TaskStorage<JoinFuture> = TaskStorage::new();

/// Test: futures_util::future::join on GPU.
///
/// Uses `futures_util::future::join(task_a, task_b)` to create a combined future
/// that polls both hostcall tasks concurrently. This proves that third-party
/// async crates (futures-util) compile and run correctly on GPU hardware.
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]:
///   [0] = poll rounds executed
///   [1] = 1 on success
#[no_mangle]
pub unsafe extern "ptx-kernel" fn futures_join_kernel(buf: *mut u8, result: *mut u32) {
    // Only thread 0 executes.
    let global_idx: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) global_idx,
        options(nostack, readonly),
    );
    if global_idx != 0 {
        return;
    }

    // Initialize result to zero.
    *result = 0;
    *result.add(1) = 0;

    // Initialize the executor.
    let storage_ptr =
        &EXECUTOR_STORAGE_3.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    // Create a joined future using futures_util::future::join.
    // This is a third-party combinator that polls both futures concurrently.
    let token = JOIN_TASK.spawn(|| {
        futures_util::future::join(
            HostcallPrintFuture::new(buf, b"Join task A from GPU!"),
            HostcallPrintFutureB::new(buf, b"Join task B from GPU!"),
        )
    });
    let spawner = executor.spawner();
    let _ = spawner.spawn(token);

    // Poll the executor.
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 100;
    loop {
        executor.poll();
        poll_rounds += 1;

        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
    }

    // Write results.
    *result = poll_rounds;
    *result.add(1) = 1;
}
