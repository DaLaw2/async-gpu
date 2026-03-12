//! GPU Runtime — facade crate for writing GPU kernels with async hostcall support.
//!
//! Re-exports all necessary GPU-side APIs so kernel authors only need
//! one dependency instead of four.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! gpu-runtime = { path = "../gpu-runtime" }
//! ```
//!
//! ```rust,ignore
//! #![no_std]
//! #![feature(abi_ptx)]
//!
//! use gpu_runtime::prelude::*;
//!
//! #[no_mangle]
//! pub unsafe extern "ptx-kernel" fn my_kernel(buf: *mut u8, result: *mut u32) {
//!     let msg = b"Hello from GPU!";
//!     gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
//!     core::ptr::write_volatile(result, 1);
//! }
//! ```

#![no_std]

// Re-export sub-crates
pub use gpu_protocol;
pub use gpu_atomics;

// Ensure critical-section is linked (needed for Embassy executor)
extern crate gpu_critical_section;

/// Hostcall helpers for GPU-side hostcall protocol operations.
///
/// These functions implement the lock-free two-stack hostcall protocol
/// (pop from free stack, fill packet, push to ready stack, spin-wait for response).
pub mod hostcall {
    use gpu_atomics::{
        activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u64,
        sys_spin_load_acquire_u32, sys_store_release_u32,
    };
    use gpu_protocol::*;

    /// Pop a packet from the free stack. Returns packet index or NULL_INDEX.
    #[inline(always)]
    pub unsafe fn hc_pop_free(buf: *mut u8) -> u16 {
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
    pub unsafe fn hc_push(stack_ptr: *mut u64, buf: *mut u8, pkt_idx: u16) {
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

    /// Release a packet back to the free stack.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_release(buf: *mut u8, pkt: *mut u8) {
        let pkt_offset_bytes = (pkt as usize) - (buf as usize);
        let idx = ((pkt_offset_bytes - BUFFER_HEADER_SIZE) / PACKET_SIZE) as u16;
        let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
        hc_push(free_ptr, buf, idx);
    }

    /// Maximum spin iterations before declaring timeout.
    pub const GPU_MAX_SPIN: u32 = 10_000_000;

    /// Submit a hostcall request: pop packet, fill header + payload, push to ready
    /// stack, ring doorbell, spin-wait for response.
    ///
    /// Returns `(pkt_ptr, success)`. On success, the payload contains the host's response.
    /// On failure (pool exhaustion or timeout), returns `(null, false)`.
    /// Caller must call `gpu_hostcall_release(buf, pkt)` after reading the response.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_request(
        buf: *mut u8,
        service: u32,
        fill_payload: impl FnOnce(*mut u8),
    ) -> (*mut u8, bool) {
        // Step 1: Pop free packet
        let pkt_idx = hc_pop_free(buf);
        if pkt_idx == NULL_INDEX {
            return (core::ptr::null_mut(), false);
        }

        let pkt = buf.add(packet_offset(pkt_idx));

        // Step 2: Fill packet header
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Step 3: Fill payload
        fill_payload(pkt.add(PKT_OFF_PAYLOAD));

        // Step 4: Mark packet as filled (release store ensures all prior writes visible)
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Step 5: Push to ready stack
        let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
        hc_push(ready_ptr, buf, pkt_idx);

        // Step 6: Ring doorbell
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Step 7: Spin-wait for host response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        let success;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                success = (ctrl & CONTROL_ERROR) == 0;
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                // Timeout — return packet to free stack
                let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
                hc_push(free_ptr, buf, pkt_idx);
                return (core::ptr::null_mut(), false);
            }
        }

        (pkt, success)
    }

    /// Send a PRINT hostcall with a short message (max 56 bytes).
    /// Returns true on success.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_print(buf: *mut u8, msg: *const u8, msg_len: u32) -> bool {
        let pkt_idx = hc_pop_free(buf);
        if pkt_idx == NULL_INDEX {
            return false;
        }

        let pkt = buf.add(packet_offset(pkt_idx));

        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        let payload = pkt.add(PKT_OFF_PAYLOAD);
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

        let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
            PRINT_MAX_MSG_LEN as u32
        } else {
            msg_len
        };
        let dst = payload.add(8);
        let mut i: u32 = 0;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
            i += 1;
        }

        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
        hc_push(ready_ptr, buf, pkt_idx);

        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        let success;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                success = true;
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                success = false;
                break;
            }
        }

        let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
        hc_push(free_ptr, buf, pkt_idx);

        success
    }
}

/// Prelude — import everything you need for a basic GPU kernel.
///
/// ```rust,ignore
/// use gpu_runtime::prelude::*;
/// ```
pub mod prelude {
    pub use gpu_atomics::{
        activemask, lane_id, membar_sys, st_global_u32, sys_cas_u32, sys_cas_u64,
        sys_exchange_u64, sys_fetch_add_u32, sys_fetch_add_u64, sys_load_acquire_u32,
        sys_load_acquire_u64, sys_spin_load_acquire_u32, sys_spin_load_acquire_u64,
        sys_store_release_u32, sys_store_release_u64,
    };
    pub use gpu_protocol::*;

    pub use crate::hostcall::{
        gpu_hostcall_print, gpu_hostcall_release, gpu_hostcall_request, hc_pop_free, hc_push,
        GPU_MAX_SPIN,
    };
}
