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
#![allow(clippy::missing_safety_doc)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))]

// GPU intrinsic wrappers — stubs on non-nvptx targets for doc builds.
#[cfg(target_arch = "nvptx64")]
mod nvptx_shim {
    #[inline(always)]
    pub fn block_idx_x() -> u32 {
        unsafe { core::arch::nvptx::_block_idx_x() }
    }
    #[inline(always)]
    pub fn thread_idx_x() -> u32 {
        unsafe { core::arch::nvptx::_thread_idx_x() }
    }
}
#[cfg(not(target_arch = "nvptx64"))]
mod nvptx_shim {
    #[inline(always)]
    pub fn block_idx_x() -> u32 {
        0
    }
    #[inline(always)]
    pub fn thread_idx_x() -> u32 {
        0
    }
}

// Re-export sub-crates
pub use gpu_atomics;
pub use gpu_protocol;

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

    // ================================================================
    // Sharding helpers — compute shard index and resolve packet offsets
    // ================================================================

    /// Read sharding metadata from buffer header. Returns (num_shards, shard_array_offset, pkts_per_shard).
    /// If num_shards == 0, this is a legacy (unsharded) buffer.
    #[inline(always)]
    pub unsafe fn read_shard_info(buf: *const u8) -> (u32, u32, u32) {
        let num_shards = core::ptr::read_volatile(buf.add(BUF_OFF_NUM_SHARDS) as *const u32);
        if num_shards == 0 {
            return (0, BUFFER_HEADER_SIZE as u32, 0);
        }
        let pkts_per_shard =
            core::ptr::read_volatile(buf.add(BUF_OFF_PKTS_PER_SHARD) as *const u32);
        let shard_array_off =
            core::ptr::read_volatile(buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *const u32);
        (num_shards, shard_array_off, pkts_per_shard)
    }

    /// Compute the byte offset of a packet from buf base, handling both legacy and sharded layouts.
    #[inline(always)]
    pub unsafe fn pkt_offset(buf: *const u8, idx: u16) -> usize {
        let (num_shards, shard_array_off, _) = read_shard_info(buf);
        if num_shards == 0 {
            packet_offset(idx)
        } else {
            packet_offset_sharded(idx, shard_array_off as usize, num_shards)
        }
    }

    /// Get the free stack pointer for the current block's shard (or global if unsharded).
    #[inline(always)]
    pub unsafe fn get_free_stack_ptr(
        buf: *mut u8,
        num_shards: u32,
        shard_array_off: u32,
    ) -> *mut u64 {
        if num_shards == 0 {
            buf.add(BUF_OFF_FREE_STACK) as *mut u64
        } else {
            let shard_idx = crate::nvptx_shim::block_idx_x() % num_shards;
            let entry_off = shard_entry_offset(shard_array_off as usize, shard_idx);
            buf.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64
        }
    }

    /// Get the ready stack pointer for the current block's shard (or global if unsharded).
    #[inline(always)]
    pub unsafe fn get_ready_stack_ptr(
        buf: *mut u8,
        num_shards: u32,
        shard_array_off: u32,
    ) -> *mut u64 {
        if num_shards == 0 {
            buf.add(BUF_OFF_READY_STACK) as *mut u64
        } else {
            let shard_idx = crate::nvptx_shim::block_idx_x() % num_shards;
            let entry_off = shard_entry_offset(shard_array_off as usize, shard_idx);
            buf.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64
        }
    }

    // ================================================================
    // Core stack operations
    // ================================================================

    /// Pop a packet from the free stack. Returns packet index or NULL_INDEX.
    #[inline(always)]
    pub unsafe fn hc_pop_free(buf: *mut u8) -> u16 {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off)
    }

    /// Pop a packet from a specific free stack pointer.
    #[inline(always)]
    unsafe fn hc_pop_free_from(
        buf: *mut u8,
        free_ptr: *mut u64,
        num_shards: u32,
        shard_array_off: u32,
    ) -> u16 {
        loop {
            let old_head = sys_load_acquire_u64(free_ptr as *const u64);
            let idx = tagged_index(old_head);
            if idx == NULL_INDEX {
                return NULL_INDEX;
            }
            let pkt_off = if num_shards == 0 {
                packet_offset(idx)
            } else {
                packet_offset_sharded(idx, shard_array_off as usize, num_shards)
            };
            let pkt = buf.add(pkt_off);
            let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
            if sys_cas_u64(free_ptr, old_head, next) == old_head {
                return idx;
            }
        }
    }

    /// Push a packet onto a tagged-pointer stack (free or ready).
    #[inline(always)]
    pub unsafe fn hc_push(stack_ptr: *mut u64, buf: *mut u8, pkt_idx: u16) {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        hc_push_with(stack_ptr, buf, pkt_idx, num_shards, shard_array_off);
    }

    /// Push with pre-computed sharding info.
    #[inline(always)]
    unsafe fn hc_push_with(
        stack_ptr: *mut u64,
        buf: *mut u8,
        pkt_idx: u16,
        num_shards: u32,
        shard_array_off: u32,
    ) {
        let pkt_off = if num_shards == 0 {
            packet_offset(pkt_idx)
        } else {
            packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);
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
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let pkt_offset_bytes = (pkt as usize) - (buf as usize);
        let packet_base = if num_shards == 0 {
            BUFFER_HEADER_SIZE
        } else {
            shard_array_off as usize + (num_shards as usize) * SHARD_ENTRY_SIZE
        };
        let idx = ((pkt_offset_bytes - packet_base) / PACKET_SIZE) as u16;
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        hc_push_with(free_ptr, buf, idx, num_shards, shard_array_off);
    }

    /// Maximum spin iterations before declaring timeout.
    pub const GPU_MAX_SPIN: u32 = 10_000_000;

    /// Submit a hostcall request: pop packet, fill header + payload, push to ready
    /// stack, ring doorbell, spin-wait for response.
    ///
    /// Returns `Ok(pkt_ptr)` on success — the payload contains the host's response.
    /// Returns `Err(GpuError)` on failure: pool exhaustion, timeout, or host-side error
    /// (decoded from CONTROL_ERROR + payload slot 0).
    /// Caller must call `gpu_hostcall_release(buf, pkt)` after reading the response.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_request(
        buf: *mut u8,
        service: u32,
        fill_payload: impl FnOnce(*mut u8),
    ) -> Result<*mut u8, GpuError> {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        let ready_ptr = get_ready_stack_ptr(buf, num_shards, shard_array_off);

        // Step 1: Pop free packet
        let pkt_idx = hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            return Err(GpuError::pool_exhausted());
        }

        let pkt_off = if num_shards == 0 {
            packet_offset(pkt_idx)
        } else {
            packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);

        // Step 2: Fill packet header
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Step 3: Fill payload
        fill_payload(pkt.add(PKT_OFF_PAYLOAD));

        // Step 4: Mark packet as filled (release store ensures all prior writes visible)
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Step 5: Push to ready stack (shard-local or global)
        hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);

        // Step 6: Ring doorbell (always global)
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Step 7: Spin-wait for host response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                if ctrl & CONTROL_ERROR != 0 {
                    // Host reported an error — decode from payload slot 0
                    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                    // Release packet before returning error
                    hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);
                    return Err(GpuError::from_encoded(slot0));
                }
                return Ok(pkt);
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                // Timeout — return packet to free stack
                hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);
                return Err(GpuError::timeout());
            }
        }
    }

    /// Submit a hostcall request with a longer spin-wait timeout.
    ///
    /// Identical to [`gpu_hostcall_request`] but uses `max_spin` iterations instead
    /// of the default `GPU_MAX_SPIN`. Useful for blocking host operations like stdin
    /// that may take longer to complete due to I/O thread routing.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_request_with_timeout(
        buf: *mut u8,
        service: u32,
        max_spin: u32,
        fill_payload: impl FnOnce(*mut u8),
    ) -> Result<*mut u8, GpuError> {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        let ready_ptr = get_ready_stack_ptr(buf, num_shards, shard_array_off);

        let pkt_idx = hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            return Err(GpuError::pool_exhausted());
        }

        let pkt_off = if num_shards == 0 {
            packet_offset(pkt_idx)
        } else {
            packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);

        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        fill_payload(pkt.add(PKT_OFF_PAYLOAD));

        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
        hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                if ctrl & CONTROL_ERROR != 0 {
                    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                    hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);
                    return Err(GpuError::from_encoded(slot0));
                }
                return Ok(pkt);
            }
            spins += 1;
            if spins >= max_spin {
                hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);
                return Err(GpuError::timeout());
            }
        }
    }

    /// Send a PRINT hostcall with a short message (max 56 bytes).
    /// Returns `Ok(())` on success, `Err(GpuError)` on pool exhaustion or timeout.
    #[inline(always)]
    pub unsafe fn gpu_hostcall_print(
        buf: *mut u8,
        msg: *const u8,
        msg_len: u32,
    ) -> Result<(), GpuError> {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        let ready_ptr = get_ready_stack_ptr(buf, num_shards, shard_array_off);

        let pkt_idx = hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            return Err(GpuError::pool_exhausted());
        }

        let pkt_off = if num_shards == 0 {
            packet_offset(pkt_idx)
        } else {
            packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);

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

        // Write thread/block metadata at payload+64 (lane 1 area, unused by PRINT message)
        let block_idx = crate::nvptx_shim::block_idx_x();
        let thread_idx = crate::nvptx_shim::thread_idx_x();
        core::ptr::write_volatile(payload.add(64) as *mut u32, block_idx);
        core::ptr::write_volatile(payload.add(68) as *mut u32, thread_idx);

        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);

        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                // Don't release packet on timeout — it may still be in-flight
                return Err(GpuError::timeout());
            }
        }

        hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);

        Ok(())
    }
}

/// Sideband buffer helpers for bulk data transfer (>56 bytes).
///
/// The sideband buffer is a separate CUDA mapped allocation with a bump allocator.
/// GPU threads allocate regions via atomic fetch_add, write/read data, and use
/// hostcall packets to coordinate with the host.
pub mod sideband {
    use gpu_atomics::{sys_fetch_add_u64, sys_store_release_u64};
    use gpu_protocol::*;

    use crate::hostcall::{gpu_hostcall_release, gpu_hostcall_request};

    /// Allocate `size` bytes from the sideband bump allocator.
    /// Returns offset from data region start, or u64::MAX if insufficient space.
    #[inline(always)]
    pub unsafe fn sideband_alloc(sideband: *mut u8, size: u64) -> u64 {
        let alloc_ptr = sideband.add(SIDEBAND_OFF_ALLOC) as *mut u64;
        let capacity = core::ptr::read_volatile(sideband.add(SIDEBAND_OFF_CAPACITY) as *const u64);
        let old_offset = sys_fetch_add_u64(alloc_ptr, size);
        if old_offset + size > capacity {
            return u64::MAX;
        }
        old_offset
    }

    /// Reset the sideband bump allocator to zero.
    /// Call at kernel start or after all pending bulk operations complete.
    #[inline(always)]
    pub unsafe fn sideband_reset(sideband: *mut u8) {
        let alloc_ptr = sideband.add(SIDEBAND_OFF_ALLOC) as *mut u64;
        sys_store_release_u64(alloc_ptr, 0);
    }

    /// Write `len` bytes from `src` to file `fd` via sideband bulk transfer.
    /// Returns bytes written, or 0 on error.
    #[inline(always)]
    pub unsafe fn gpu_bulk_write(
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        src: *const u8,
        len: usize,
    ) -> usize {
        if len == 0 {
            return 0;
        }

        // Allocate space in sideband
        let offset = sideband_alloc(sideband, len as u64);
        if offset == u64::MAX {
            return 0;
        }

        // Copy data to sideband
        let dst = sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
        let mut i = 0;
        while i < len {
            core::ptr::write_volatile(dst.add(i), *src.add(i));
            i += 1;
        }

        // Send hostcall with sideband metadata
        let pkt = match gpu_hostcall_request(buf, SERVICE_BULK_WRITE, |payload| {
            core::ptr::write_volatile(payload as *mut u64, fd);
            core::ptr::write_volatile(payload.add(8) as *mut u64, offset);
            core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
        }) {
            Ok(p) => p,
            Err(_) => return 0,
        };

        let written = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        gpu_hostcall_release(buf, pkt);
        if written == FILE_ERROR_SENTINEL {
            0
        } else {
            written as usize
        }
    }

    /// Read up to `max_len` bytes from file `fd` into `dst` via sideband bulk transfer.
    /// Returns bytes read, or 0 on error/EOF.
    #[inline(always)]
    pub unsafe fn gpu_bulk_read(
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        dst: *mut u8,
        max_len: usize,
    ) -> usize {
        if max_len == 0 {
            return 0;
        }

        // Allocate space in sideband for response data
        let offset = sideband_alloc(sideband, max_len as u64);
        if offset == u64::MAX {
            return 0;
        }

        // Send hostcall requesting read
        let pkt = match gpu_hostcall_request(buf, SERVICE_BULK_READ, |payload| {
            core::ptr::write_volatile(payload as *mut u64, fd);
            core::ptr::write_volatile(payload.add(8) as *mut u64, offset);
            core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
        }) {
            Ok(p) => p,
            Err(_) => return 0,
        };

        let bytes_read = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        gpu_hostcall_release(buf, pkt);

        if bytes_read == FILE_ERROR_SENTINEL || bytes_read == 0 {
            return 0;
        }

        // Copy data from sideband to destination
        let src = sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
        let n = bytes_read as usize;
        let mut i = 0;
        while i < n {
            core::ptr::write_volatile(dst.add(i), core::ptr::read_volatile(src.add(i)));
            i += 1;
        }

        n
    }
}

/// GPU panic handler that sends panic messages via hostcall before trapping.
///
/// # Usage
///
/// 1. Call `gpu_panic_init(buf)` at the start of your kernel to register the
///    hostcall buffer pointer. If not called, panics will trap without sending
///    a message.
///
/// 2. Add `gpu_runtime::panic_handler!();` in your kernel crate to install the
///    panic handler (replaces the `loop {}` handler).
pub mod panic {
    use gpu_atomics::{
        activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u64,
        sys_spin_load_acquire_u32, sys_store_release_u32,
    };
    use gpu_protocol::*;

    /// Global hostcall buffer pointer. Set by `gpu_panic_init()`.
    /// Each GPU thread reads this when a panic occurs.
    static mut PANIC_BUF: *mut u8 = core::ptr::null_mut();

    /// Global kernel result buffer pointer. Set by `gpu_result_init()`.
    /// The panic handler writes error info here before trapping.
    static mut RESULT_BUF: *mut GpuKernelResult = core::ptr::null_mut();

    /// Initialize the panic handler with the hostcall buffer pointer.
    /// Must be called at the start of every kernel that might panic.
    #[inline(always)]
    pub unsafe fn gpu_panic_init(buf: *mut u8) {
        PANIC_BUF = buf;
    }

    /// Register the kernel result buffer for panic reporting.
    /// When set, the panic handler writes error info here before trapping.
    /// Call at kernel entry alongside `gpu_panic_init()`.
    #[inline(always)]
    pub unsafe fn gpu_result_init(result: *mut GpuKernelResult) {
        RESULT_BUF = result;
    }

    /// Get the current hostcall buffer pointer (for use by panic handler).
    #[inline(always)]
    pub unsafe fn panic_buf() -> *mut u8 {
        PANIC_BUF
    }

    /// Get the current result buffer pointer (for use by panic handler).
    #[inline(always)]
    pub unsafe fn result_buf() -> *mut GpuKernelResult {
        RESULT_BUF
    }

    /// Write a GpuError to the kernel result buffer with panic message.
    /// Called by the panic handler before trapping.
    #[inline(always)]
    pub unsafe fn write_panic_to_result(msg: &[u8]) {
        let result = RESULT_BUF;
        if result.is_null() {
            return;
        }
        let thread_idx = crate::nvptx_shim::thread_idx_x() as u16;
        let block_idx = crate::nvptx_shim::block_idx_x() as u16;
        let err = GpuError::new(ERR_OTHER, 0);
        (*result).set_err(err, thread_idx, block_idx, msg);
    }

    /// Fixed-size buffer for formatting panic messages on GPU (no allocator needed).
    pub struct PanicBuf {
        pub buf: [u8; PANIC_MAX_MSG_LEN],
        pub pos: usize,
    }

    impl Default for PanicBuf {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PanicBuf {
        #[inline(always)]
        pub const fn new() -> Self {
            Self {
                buf: [0u8; PANIC_MAX_MSG_LEN],
                pos: 0,
            }
        }

        #[inline(always)]
        pub fn as_slice(&self) -> &[u8] {
            // Safety: pos <= PANIC_MAX_MSG_LEN guaranteed by write_str
            unsafe { core::slice::from_raw_parts(self.buf.as_ptr(), self.pos) }
        }
    }

    impl core::fmt::Write for PanicBuf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let remaining = PANIC_MAX_MSG_LEN - self.pos;
            let copy_len = if bytes.len() < remaining {
                bytes.len()
            } else {
                remaining
            };
            let mut i = 0;
            while i < copy_len {
                self.buf[self.pos + i] = bytes[i];
                i += 1;
            }
            self.pos += copy_len;
            Ok(()) // Always succeed — silently truncate
        }
    }

    /// Send a panic message via hostcall. Best-effort: if pool is exhausted or
    /// timeout occurs, returns without sending. Never panics itself.
    /// Supports both legacy (unsharded) and per-block sharded buffers.
    #[inline(never)]
    pub unsafe fn send_panic_hostcall(buf: *mut u8, msg: &[u8]) {
        // Read sharding info
        let num_shards = core::ptr::read_volatile(buf.add(BUF_OFF_NUM_SHARDS) as *const u32);
        let shard_array_off;
        let free_ptr: *mut u64;
        let ready_ptr: *mut u64;

        if num_shards == 0 {
            free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
            ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
            shard_array_off = BUFFER_HEADER_SIZE as u32;
        } else {
            shard_array_off =
                core::ptr::read_volatile(buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *const u32);
            let shard_idx = crate::nvptx_shim::block_idx_x() % num_shards;
            let entry_off = shard_entry_offset(shard_array_off as usize, shard_idx);
            free_ptr = buf.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64;
            ready_ptr = buf.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64;
        }

        /// Inline helper — compute packet byte offset supporting both layouts.
        #[inline(always)]
        unsafe fn panic_pkt_off(
            _buf: *const u8,
            idx: u16,
            num_shards: u32,
            shard_array_off: u32,
        ) -> usize {
            if num_shards == 0 {
                packet_offset(idx)
            } else {
                packet_offset_sharded(idx, shard_array_off as usize, num_shards)
            }
        }

        // Pop a free packet
        let pkt_idx;
        loop {
            let old_head = sys_load_acquire_u64(free_ptr as *const u64);
            let idx = tagged_index(old_head);
            if idx == NULL_INDEX {
                return; // Pool exhausted — can't send panic message
            }
            let pkt = buf.add(panic_pkt_off(buf, idx, num_shards, shard_array_off));
            let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
            if sys_cas_u64(free_ptr, old_head, next) == old_head {
                pkt_idx = idx;
                break;
            }
        }

        let pkt = buf.add(panic_pkt_off(buf, pkt_idx, num_shards, shard_array_off));

        // Fill packet header
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PANIC);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Fill payload: metadata in slot 0, message in slots 1-7
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let thread_x = crate::nvptx_shim::thread_idx_x() as u16;
        let block_x = crate::nvptx_shim::block_idx_x() as u16;
        let msg_len = if msg.len() > PANIC_MAX_MSG_LEN {
            PANIC_MAX_MSG_LEN as u16
        } else {
            msg.len() as u16
        };
        let meta = encode_panic_metadata(thread_x, block_x, msg_len);
        core::ptr::write_volatile(payload as *mut u64, meta);

        // Copy message bytes
        let dst = payload.add(8);
        let mut i: u16 = 0;
        while i < msg_len {
            core::ptr::write_volatile(dst.add(i as usize), msg[i as usize]);
            i += 1;
        }

        // Mark filled and push to ready stack
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Push inline to ready stack
        loop {
            let old_head = sys_load_acquire_u64(ready_ptr as *const u64);
            core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
            let new_tag = tagged_tag(old_head).wrapping_add(1);
            let new_tagged = make_tagged(new_tag, pkt_idx);
            if sys_cas_u64(ready_ptr, old_head, new_tagged) == old_head {
                break;
            }
        }

        // Ring doorbell (always global)
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response (with timeout)
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                break; // Timeout — proceed to trap anyway
            }
        }

        // Release packet back to free stack
        loop {
            let old_head = sys_load_acquire_u64(free_ptr as *const u64);
            core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
            let new_tag = tagged_tag(old_head).wrapping_add(1);
            let new_tagged = make_tagged(new_tag, pkt_idx);
            if sys_cas_u64(free_ptr, old_head, new_tagged) == old_head {
                break;
            }
        }
    }
}

/// Macro to install the GPU panic handler provided by gpu-runtime.
///
/// Place `gpu_runtime::panic_handler!();` at the top level of your kernel crate.
/// This replaces the `loop {}` panic handler with one that sends the panic message
/// via hostcall before trapping. Call `gpu_panic_init(buf)` at kernel entry.
#[macro_export]
macro_rules! panic_handler {
    () => {
        #[panic_handler]
        fn _gpu_panic_handler(info: &core::panic::PanicInfo) -> ! {
            unsafe {
                // Format the panic message into a fixed-size buffer
                let mut pbuf = $crate::panic::PanicBuf::new();
                use core::fmt::Write;
                let _ = write!(pbuf, "{}", info);
                let msg = pbuf.as_slice();

                // Write to kernel result buffer (if registered)
                $crate::panic::write_panic_to_result(msg);

                // Send via hostcall (if registered)
                let buf = $crate::panic::panic_buf();
                if !buf.is_null() {
                    $crate::panic::send_panic_hostcall(buf, msg);
                }

                // Terminate this GPU thread
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("trap;", options(noreturn));
                #[cfg(not(target_arch = "nvptx64"))]
                panic!("GPU trap");
            }
        }
    };
}

/// Warp-level Future — SIMT-convergent async on GPU.
///
/// A `WarpFuture` represents an entire warp (32 lanes) executing in lockstep.
/// Unlike `core::future::Future` where each thread has its own state machine,
/// a WarpFuture has ONE state discriminant shared across all 32 lanes via
/// `shfl.sync.idx.b32`. All lanes enter the same match arm on every poll;
/// only per-lane data differs (SIMD semantics).
///
/// # Key Concepts
///
/// - **WarpPoll**: `Ready(T)` or `Pending`, analogous to `core::task::Poll`.
/// - **WarpContext**: Provides `lane_id` and `active_mask`. No `Waker` —
///   warp futures use synchronous spin-poll.
/// - **WarpFuture trait**: `poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<T>`.
/// - **WarpExecutor**: Simple loop that polls a WarpFuture until Ready.
///
/// # Example
///
/// ```rust,ignore
/// // All 32 lanes call this simultaneously
/// let mut future = MyWarpFuture::new(buf);
/// let result = WarpExecutor::run(&mut future);
/// ```
pub mod warp_future {
    use gpu_atomics::{
        activemask, lane_id, shfl_sync_idx_u32, syncwarp, sys_fetch_add_u64,
        sys_spin_load_acquire_u32, sys_store_release_u32,
    };
    use gpu_protocol::*;

    /// Result of polling a warp-level future.
    pub enum WarpPoll<T> {
        /// All lanes completed. Output is per-lane.
        Ready(T),
        /// Warp yielded — will be re-polled.
        Pending,
    }

    /// Context passed to WarpFuture::poll_warp.
    ///
    /// Contains warp metadata needed during polling. Unlike `core::task::Context`,
    /// there is no Waker — warp futures use synchronous spin-poll driven by
    /// the WarpExecutor.
    pub struct WarpContext {
        /// Active lane mask (from `activemask.b32`)
        pub active_mask: u32,
        /// This lane's ID (0..31)
        pub lane_id: u32,
    }

    impl WarpContext {
        /// Create a new WarpContext by reading hardware registers.
        #[inline(always)]
        pub unsafe fn new() -> Self {
            Self {
                active_mask: activemask(),
                lane_id: lane_id(),
            }
        }

        /// Returns true if this is lane 0 (the "leader" lane).
        #[inline(always)]
        pub fn is_leader(&self) -> bool {
            self.lane_id == 0
        }
    }

    /// A future representing an entire warp (32 lanes) in SIMT lockstep.
    ///
    /// # Contract
    /// - All active lanes must call `poll_warp()` simultaneously.
    /// - The state discriminant must be uniform across all lanes
    ///   (broadcast via `shfl.sync.idx.b32` from lane 0).
    /// - Divergent control flow within `poll_warp()` is forbidden —
    ///   all lanes must execute the same code path with different data.
    ///
    /// # Safety
    /// Implementing this trait requires maintaining warp convergence.
    /// Breaking convergence causes deadlock or incorrect results.
    pub unsafe trait WarpFuture {
        /// Per-lane output type.
        type Output;

        /// Poll the warp future. Called by all active lanes simultaneously.
        fn poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<Self::Output>;
    }

    /// Minimal warp-level executor.
    ///
    /// Polls a single WarpFuture in a loop until completion. All active lanes
    /// participate in every poll. No run queue, no waker — just spin-poll
    /// with `nanosleep` yield between iterations.
    pub struct WarpExecutor;

    impl WarpExecutor {
        /// Run a WarpFuture to completion. All active lanes must call this.
        ///
        /// Returns the per-lane output value.
        ///
        /// # Safety
        /// Must be called by all active lanes of a warp simultaneously.
        #[inline(always)]
        pub unsafe fn run<F: WarpFuture>(future: &mut F) -> F::Output {
            let mut wcx = WarpContext::new();
            let mut polls: u32 = 0;
            const MAX_POLLS: u32 = 10_000_000;

            loop {
                match future.poll_warp(&mut wcx) {
                    WarpPoll::Ready(output) => return output,
                    WarpPoll::Pending => {
                        polls += 1;
                        if polls >= MAX_POLLS {
                            // Timeout — trap to avoid infinite loop
                            #[cfg(target_arch = "nvptx64")]
                            core::arch::asm!("trap;", options(noreturn));
                            #[cfg(not(target_arch = "nvptx64"))]
                            panic!("WarpExecutor timeout");
                        }
                        // Yield warp scheduler slot
                        #[cfg(target_arch = "nvptx64")]
                        core::arch::asm!("nanosleep.u32 64;", options(nostack));
                    }
                }
                // Ensure convergence before next poll
                syncwarp(wcx.active_mask);
            }
        }
    }

    /// Broadcast a u32 from lane 0 to all lanes. Convenience wrapper.
    #[inline(always)]
    pub unsafe fn broadcast_u32(mask: u32, val: u32) -> u32 {
        shfl_sync_idx_u32(mask, val, 0)
    }

    /// Warp-cooperative hostcall submit: pop a free packet, fill payload, push to
    /// ready stack, and ring the doorbell. Only lane 0 performs actual memory ops;
    /// all lanes participate in broadcasts to maintain warp convergence.
    ///
    /// Returns `WarpPoll::Pending` always — the caller must transition to a WAIT
    /// state to collect the response.
    ///
    /// # Arguments
    /// * `buf` — hostcall buffer base pointer
    /// * `wcx` — warp context (active mask + lane ID)
    /// * `service` — service ID (e.g., `SERVICE_OPEN`, `SERVICE_PRINT`)
    /// * `fill_payload` — closure called on lane 0 to fill the packet payload
    /// * `next_state` — state value to transition to after submit
    /// * `state_cell` — mutable reference to the state machine's state field
    /// * `pkt_idx_cell` — mutable reference to store the allocated packet index
    #[inline(always)]
    pub unsafe fn warp_hostcall_submit(
        buf: *mut u8,
        wcx: &mut WarpContext,
        service: u32,
        fill_payload: impl FnOnce(*mut u8),
        next_state: u32,
        state_cell: &mut u32,
        pkt_idx_cell: &mut u16,
    ) -> WarpPoll<bool> {
        let mut idx_raw: u32 = NULL_INDEX as u32;
        if wcx.is_leader() {
            idx_raw = crate::hostcall::hc_pop_free(buf) as u32;
        }
        let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
        if idx == NULL_INDEX {
            return WarpPoll::Pending;
        }
        *pkt_idx_cell = idx;

        let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, idx);
        let pkt = buf.add(pkt_off);
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Only lane 0 fills the payload
        if wcx.is_leader() {
            fill_payload(payload);
        }

        syncwarp(wcx.active_mask);

        if wcx.is_leader() {
            core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, wcx.active_mask);
            core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
            sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
            let (num_shards, shard_off, _) = crate::hostcall::read_shard_info(buf as *const u8);
            let ready_ptr = crate::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
            crate::hostcall::hc_push(ready_ptr, buf, idx);
            sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);
            *state_cell = next_state;
        }

        syncwarp(wcx.active_mask);
        WarpPoll::Pending
    }

    /// Warp-cooperative hostcall wait: poll the control word of a previously
    /// submitted packet. Returns `Some(u64)` with the first payload slot when
    /// the host has responded, or `None` if still pending.
    ///
    /// On completion, releases the packet back to the free pool and transitions
    /// to `next_state`. The u64 return value is broadcast to all lanes via
    /// two `shfl.sync.idx.b32` operations (low + high halves).
    ///
    /// # Arguments
    /// * `buf` — hostcall buffer base pointer
    /// * `wcx` — warp context
    /// * `pkt_idx` — packet index from a prior `warp_hostcall_submit` call
    /// * `next_state` — state to transition to on completion
    /// * `state_cell` — mutable reference to the state machine's state field
    #[inline(always)]
    pub unsafe fn warp_hostcall_wait_u64(
        buf: *mut u8,
        wcx: &mut WarpContext,
        pkt_idx: u16,
        next_state: u32,
        state_cell: &mut u32,
    ) -> Option<u64> {
        let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
        let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, idx);
        let pkt = buf.add(pkt_off);
        let ctrl = sys_spin_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

        if ctrl & CONTROL_READY != 0 {
            let mut val: u64 = 0;
            if wcx.is_leader() {
                val = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                crate::hostcall::gpu_hostcall_release(buf, pkt);
                *state_cell = next_state;
            }
            // Broadcast u64 as two u32 halves
            let lo = broadcast_u32(wcx.active_mask, val as u32) as u64;
            let hi = broadcast_u32(wcx.active_mask, (val >> 32) as u32) as u64;
            syncwarp(wcx.active_mask);
            Some(lo | (hi << 32))
        } else {
            None
        }
    }
}

/// Prelude — import everything you need for a basic GPU kernel.
///
/// The prelude exports high-level APIs for common tasks. For low-level
/// access (atomics, protocol constants, packet layout), use the module
/// paths directly: `gpu_runtime::hostcall::*`, `gpu_atomics::*`,
/// `gpu_protocol::*`.
///
/// ```rust,ignore
/// use gpu_runtime::prelude::*;
/// ```
pub mod prelude {
    // --- High-level hostcall API ---
    pub use crate::hostcall::{gpu_hostcall_print, gpu_hostcall_release, gpu_hostcall_request};
    pub use crate::panic::{gpu_panic_init, gpu_result_init};
    pub use crate::sideband::{gpu_bulk_read, gpu_bulk_write, sideband_alloc, sideband_reset};

    // --- WarpFuture API ---
    pub use crate::warp_future::{
        broadcast_u32, warp_hostcall_submit, warp_hostcall_wait_u64, WarpContext, WarpExecutor,
        WarpFuture, WarpPoll,
    };

    // --- Error types ---
    pub use gpu_protocol::{GpuError, GpuKernelResult, TAG_ERR, TAG_OK, TAG_UNINIT};

    // --- Commonly needed protocol constants ---
    pub use gpu_protocol::{
        CONTROL_ERROR, CONTROL_FILLED, CONTROL_READY, FILE_ERROR_SENTINEL, FILE_MAX_PATH_LEN,
        FILE_MAX_READ_LEN, FILE_MAX_WRITE_LEN, FILE_OPEN_APPEND, FILE_OPEN_READ,
        FILE_OPEN_WRITE_CREATE, NULL_INDEX, PACKET_SIZE, PKT_OFF_ACTIVE_MASK, PKT_OFF_CONTROL,
        PKT_OFF_PAYLOAD, PKT_OFF_SERVICE, PRINT_MAX_MSG_LEN, SERVICE_BULK_READ, SERVICE_BULK_WRITE,
        SERVICE_CLOSE, SERVICE_OPEN, SERVICE_PANIC, SERVICE_PRINT, SERVICE_READ, SERVICE_STDIN,
        SERVICE_TIME, SERVICE_WRITE,
    };

    // --- Warp intrinsics ---
    pub use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

    // --- Commonly needed atomics ---
    pub use gpu_atomics::{sys_load_acquire_u32, sys_store_release_u32};
}
