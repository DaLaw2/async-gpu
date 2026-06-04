//! Async pipeline test: 4-step sequential hostcall pipeline on GPU.
//!
//! Demonstrates chaining multiple async hostcall operations in sequence:
//! READ → PROCESS → WRITE → PRINT. Each step is a full hostcall round-trip
//! (allocate packet → submit → wait for response → release).
//!
//! Kernels:
//! - pipeline_kernel: 4-step sequential async pipeline

#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(asm_experimental_arch)]

use core::future::Future;
use core::mem::MaybeUninit;
use core::panic::PanicInfo;
use core::pin::Pin;
use core::task::{Context, Poll};

extern crate gpu_critical_section;

use embassy_executor::raw::{Executor, TaskStorage};
use gpu_atomics::{
    activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u32,
    sys_load_acquire_u64, sys_store_release_u32,
};
use gpu_protocol::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
unsafe extern "C" fn __pender(_context: *mut ()) {}

// ============================================================
// Executor storage wrapper
// ============================================================

struct ExecutorStorage {
    inner: MaybeUninit<Executor>,
}

unsafe impl Sync for ExecutorStorage {}

// ============================================================
// Hostcall helpers (duplicated — required per-crate for Fat LTO)
// ============================================================

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

#[inline(always)]
unsafe fn gpu_hostcall_release(buf: *mut u8, pkt_idx: u16) {
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    hc_push(free_ptr, buf, pkt_idx);
}

/// Submit a hostcall PRINT packet with the given message.
/// Returns the packet index for waiting on the response.
#[inline(always)]
unsafe fn submit_print(buf: *mut u8, pkt_idx: u16, msg: &[u8]) {
    let pkt = buf.add(packet_offset(pkt_idx));

    let mask = activemask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    let payload = pkt.add(PKT_OFF_PAYLOAD);
    let msg_len = msg.len() as u32;
    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

    let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
        PRINT_MAX_MSG_LEN as u32
    } else {
        msg_len
    };
    let dst = payload.add(8);
    let mut i: u32 = 0;
    while i < copy_len {
        core::ptr::write_volatile(dst.add(i as usize), *msg.as_ptr().add(i as usize));
        i += 1;
    }

    // Mark packet as filled (release store ensures all prior writes visible).
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

    let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
    hc_push(ready_ptr, buf, pkt_idx);
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);
}

// ============================================================
// PipelineFuture — 4-step sequential async hostcall pipeline
// ============================================================

/// Messages for each pipeline step.
const STEP_MSGS: [&[u8]; 4] = [
    b"Pipeline step 1: READ",
    b"Pipeline step 2: PROCESS",
    b"Pipeline step 3: WRITE",
    b"Pipeline step 4: PRINT",
];

/// Future that performs 4 sequential hostcall prints.
/// Models a read→process→write→print pipeline where each step
/// is an async hostcall operation.
struct PipelineFuture {
    buf: *mut u8,
    /// Current step (0-3), or 4 when done.
    step: u8,
    /// true = waiting for host response, false = need to submit.
    waiting: bool,
    pkt_idx: u16,
}

impl PipelineFuture {
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            step: 0,
            waiting: false,
            pkt_idx: NULL_INDEX,
        }
    }
}

impl Future for PipelineFuture {
    type Output = u32; // number of steps completed

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            if this.step >= 4 {
                return Poll::Ready(this.step as u32);
            }

            if !this.waiting {
                // Submit phase: allocate packet and send.
                let pkt_idx = unsafe { hc_pop_free(this.buf) };
                if pkt_idx == NULL_INDEX {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                this.pkt_idx = pkt_idx;
                unsafe { submit_print(this.buf, pkt_idx, STEP_MSGS[this.step as usize]) };

                this.waiting = true;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            } else {
                // Wait phase: check if host responded.
                let pkt = unsafe { this.buf.add(packet_offset(this.pkt_idx)) };
                let control_ptr = unsafe { pkt.add(PKT_OFF_CONTROL) as *const u32 };
                let ctrl = unsafe { sys_load_acquire_u32(control_ptr) };

                if ctrl & CONTROL_READY != 0 {
                    // Host responded. Release and advance.
                    unsafe { gpu_hostcall_release(this.buf, this.pkt_idx) };
                    this.step += 1;
                    this.waiting = false;
                    // Continue loop to immediately start next step.
                    continue;
                } else {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
            }
        }
    }
}

// ============================================================
// Kernel: 4-step sequential async pipeline
// ============================================================

static EXECUTOR_STORAGE: ExecutorStorage = ExecutorStorage {
    inner: MaybeUninit::uninit(),
};

static PIPELINE_TASK: TaskStorage<PipelineFuture> = TaskStorage::new();

/// 4-step sequential async pipeline test.
///
/// Creates an Embassy executor, spawns one PipelineFuture that chains
/// 4 sequential hostcall print operations: READ → PROCESS → WRITE → PRINT.
/// Each step is a full hostcall round-trip (allocate → submit → wait → release).
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]:
///   [0] = poll rounds executed
///   [1] = 1 on success (host verifies 4 messages received)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn pipeline_kernel(buf: *mut u8, result: *mut u32) {
    let global_idx: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) global_idx,
        options(nostack, readonly),
    );
    if global_idx != 0 {
        return;
    }

    *result = 0;
    *result.add(1) = 0;

    let storage_ptr =
        &EXECUTOR_STORAGE.inner as *const MaybeUninit<Executor> as *mut MaybeUninit<Executor>;
    (*storage_ptr).write(Executor::new(core::ptr::null_mut()));
    let executor: &'static Executor = (*storage_ptr).assume_init_ref();

    let token = PIPELINE_TASK.spawn(|| PipelineFuture::new(buf));
    let spawner = executor.spawner();
    let _ = spawner.spawn(token);

    // Poll the executor. 4 hostcall round-trips need more polls.
    let mut poll_rounds: u32 = 0;
    let max_rounds: u32 = 500;
    loop {
        executor.poll();
        poll_rounds += 1;

        let current = core::ptr::read_volatile(&poll_rounds);
        if current >= max_rounds {
            break;
        }
    }

    *result = poll_rounds;
    *result.add(1) = 1;
}
