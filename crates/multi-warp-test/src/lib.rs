//! Multi-warp and multi-block scaling tests.
//!
//! Tests verify that GPU threads can independently issue hostcall requests
//! through the lock-free protocol concurrently.
//!
//! - multi_warp_sync_kernel: 1 block × 32 threads (single warp)
//! - multi_block_sync_kernel: N blocks × 32 threads (multi-block)
//!
//! Uses synchronous (spin-wait) hostcall rather than async Embassy executor.
//! The key insight being tested is multi-thread concurrent access to the
//! hostcall packet pool under varying levels of contention.

#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::panic::PanicInfo;

use gpu_atomics::{
    activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u64,
    sys_spin_load_acquire_u32, sys_store_release_u32,
};
use gpu_protocol::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// ============================================================
// Hostcall helpers (duplicated — required per-crate for Fat LTO)
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
// Synchronous hostcall print (spin-wait)
// ============================================================

/// Perform a synchronous hostcall print: allocate packet, submit, spin-wait
/// for host response, release packet. Returns true on success.
#[inline(never)]
unsafe fn hostcall_print_sync(buf: *mut u8, msg: &[u8]) -> bool {
    // Spin to acquire a free packet (contention possible with 32 threads).
    let mut pkt_idx: u16;
    let mut spin: u32 = 0;
    loop {
        pkt_idx = hc_pop_free(buf);
        if pkt_idx != NULL_INDEX {
            break;
        }
        spin += 1;
        if spin > GPU_MAX_SPIN {
            return false;
        }
    }

    let pkt = buf.add(packet_offset(pkt_idx));

    // Fill packet header.
    let mask = activemask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Fill payload: slot 0 = message length, slots 1-7 = message bytes.
    let payload = pkt.add(PKT_OFF_PAYLOAD);
    let msg_len = msg.len() as u32;
    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

    let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
        PRINT_MAX_MSG_LEN as u32
    } else {
        msg_len
    };
    let dst = payload.add(8); // skip slot 0
    let mut i: u32 = 0;
    while i < copy_len {
        core::ptr::write_volatile(dst.add(i as usize), *msg.as_ptr().add(i as usize));
        i += 1;
    }

    // Mark packet as filled (release store ensures all prior writes visible).
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

    // Push to ready stack.
    let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
    hc_push(ready_ptr, buf, pkt_idx);

    // Ring doorbell.
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    // Spin-wait for host response.
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
    spin = 0;
    loop {
        let ctrl = sys_spin_load_acquire_u32(control_ptr);
        if ctrl & CONTROL_READY != 0 {
            let success = (ctrl & CONTROL_ERROR) == 0;
            gpu_hostcall_release(buf, pkt_idx);
            return success;
        }
        spin += 1;
        if spin > GPU_MAX_SPIN {
            gpu_hostcall_release(buf, pkt_idx);
            return false;
        }
    }
}

// ============================================================
// Kernel: 32-thread synchronous hostcall scaling test
// ============================================================

/// Multi-warp synchronous hostcall scaling test.
///
/// Launches with block_dim=(32, 1, 1) — a full warp.
/// Each of the 32 threads independently:
/// 1. Computes its thread ID
/// 2. Builds a unique message "Thread NN hello!"
/// 3. Performs a synchronous hostcall print (spin-wait)
///
/// Thread 0 writes results:
///   result[0] = 1 if thread 0's hostcall succeeded
///   result[1] = 32 (thread count)
///
/// `buf` = hostcall buffer (mapped memory, must have >= 64 packets)
/// `result` = output array of u32[2]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_warp_sync_kernel(buf: *mut u8, result: *mut u32) {
    // Get thread ID within the block.
    let tid: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) tid,
        options(nostack, readonly),
    );

    // Build per-thread message: "Thread NN hello!"
    // 16 bytes, with NN replaced by the two-digit thread ID.
    let mut msg = *b"Thread 00 hello!";
    msg[7] = b'0' + (tid / 10) as u8;
    msg[8] = b'0' + (tid % 10) as u8;

    // Each thread performs its own synchronous hostcall.
    let ok = hostcall_print_sync(buf, &msg);

    // Thread 0 writes result markers.
    if tid == 0 {
        *result = if ok { 1 } else { 0 };
        *result.add(1) = 32; // thread count
    }
}

// ============================================================
// Kernel: multi-block synchronous hostcall scaling test
// ============================================================

/// Multi-block synchronous hostcall scaling test.
///
/// Launches with grid_dim=(N, 1, 1), block_dim=(M, 1, 1).
/// Each thread computes a global thread ID = blockIdx.x * blockDim.x + threadIdx.x,
/// builds a unique message "Thread NNN hello!" (3-digit ID), and performs
/// a synchronous hostcall print.
///
/// Thread 0 (global) writes results:
///   result[0] = 1 if thread 0's hostcall succeeded
///   result[1] = total thread count (num_blocks * block_dim)
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_block_sync_kernel(buf: *mut u8, result: *mut u32) {
    // Get thread ID within the block.
    let tid: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %tid.x;",
        idx = out(reg32) tid,
        options(nostack, readonly),
    );

    // Get block ID.
    let bid: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %ctaid.x;",
        idx = out(reg32) bid,
        options(nostack, readonly),
    );

    // Get block dim (threads per block).
    let block_dim: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %ntid.x;",
        idx = out(reg32) block_dim,
        options(nostack, readonly),
    );

    // Get grid dim (number of blocks).
    let num_blocks: u32;
    core::arch::asm!(
        "mov.u32 {idx}, %nctaid.x;",
        idx = out(reg32) num_blocks,
        options(nostack, readonly),
    );

    // Global thread ID.
    let global_tid = bid * block_dim + tid;
    let total_threads = num_blocks * block_dim;

    // Build per-thread message: "Thread NNN hello!"
    // 18 bytes, with NNN replaced by the three-digit global thread ID.
    let mut msg = *b"Thread 000 hello!";
    msg[7] = b'0' + (global_tid / 100) as u8;
    msg[8] = b'0' + ((global_tid / 10) % 10) as u8;
    msg[9] = b'0' + (global_tid % 10) as u8;

    // Each thread performs its own synchronous hostcall.
    let ok = hostcall_print_sync(buf, &msg);

    // Global thread 0 writes result markers.
    if global_tid == 0 {
        *result = if ok { 1 } else { 0 };
        *result.add(1) = total_threads;
    }
}
