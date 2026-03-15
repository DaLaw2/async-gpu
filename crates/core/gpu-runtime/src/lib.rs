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
    #[inline(always)]
    pub fn thread_idx_y() -> u32 {
        unsafe { core::arch::nvptx::_thread_idx_y() }
    }
    #[inline(always)]
    pub fn thread_idx_z() -> u32 {
        unsafe { core::arch::nvptx::_thread_idx_z() }
    }
    #[inline(always)]
    pub fn block_dim_x() -> u32 {
        unsafe { core::arch::nvptx::_block_dim_x() }
    }
    #[inline(always)]
    pub fn block_dim_y() -> u32 {
        unsafe { core::arch::nvptx::_block_dim_y() }
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
    #[inline(always)]
    pub fn thread_idx_y() -> u32 {
        0
    }
    #[inline(always)]
    pub fn thread_idx_z() -> u32 {
        0
    }
    #[inline(always)]
    pub fn block_dim_x() -> u32 {
        1
    }
    #[inline(always)]
    pub fn block_dim_y() -> u32 {
        1
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
    pub unsafe fn hc_pop_free_from(
        buf: *mut u8,
        free_ptr: *mut u64,
        num_shards: u32,
        shard_array_off: u32,
    ) -> u16 {
        // SAFETY: Lock-free Treiber stack pop with ABA prevention.
        //
        // The stack head is a tagged pointer: bits 48-63 = epoch tag, bits 0-15
        // = packet index (NULL_INDEX 0xFFFF = empty). The CAS compares the full
        // 64-bit tagged value, so even if a packet is popped and re-pushed at
        // the same index, the epoch tag will differ, preventing ABA.
        //
        // Acquire load on the head ensures we see the `next` pointer written by
        // the most recent push. System-scope CAS (sys_cas_u64) provides
        // visibility across all GPU SMs and the host CPU.
        //
        // Ownership protocol: a successful CAS transfers exclusive ownership of
        // the packet to the popping thread. The packet is not on any stack until
        // explicitly pushed back.
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
    pub(crate) unsafe fn hc_push_with(
        stack_ptr: *mut u64,
        buf: *mut u8,
        pkt_idx: u16,
        num_shards: u32,
        shard_array_off: u32,
    ) {
        // SAFETY: Lock-free Treiber stack push with ABA prevention.
        //
        // We write the current head into the packet's `next` field, then CAS
        // the stack head from old_head to a new tagged value containing our
        // packet index and an incremented epoch tag. The epoch tag (bits 48-63)
        // is incremented on every push to prevent ABA: even if a concurrent
        // pop+push restores the same index, the tag will differ.
        //
        // The caller must have exclusive ownership of the packet (obtained via
        // hc_pop_free_from or initial allocation). After a successful CAS, the
        // packet is visible to other threads via the stack.
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
        // SAFETY: Hostcall request protocol — packet ownership lifecycle:
        //
        // 1. Pop from free stack: CAS grants exclusive ownership of the packet.
        // 2. Fill header + payload: only the owning thread writes to the packet.
        // 3. Release-store CONTROL_FILLED: makes all prior writes visible to host.
        // 4. Push to ready stack: transfers packet to the host for processing.
        // 5. Doorbell fetch_add: wakes the host poller (system-scope atomic).
        // 6. Spin-wait on CONTROL_READY: acquire-load ensures host's response
        //    writes are visible before we read the payload.
        // 7. On success, caller owns the packet and must call gpu_hostcall_release.
        //    On error/timeout, packet is pushed back to free stack here.
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
        // SAFETY: Same packet ownership protocol as gpu_hostcall_request,
        // but with a caller-specified spin limit instead of GPU_MAX_SPIN.
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

    /// Send a TRACE hostcall with a structured trace event.
    ///
    /// Emits a trace event with thread/block/warp metadata and GPU timestamp.
    /// Fire-and-forget: the host acknowledges but no response data is used.
    ///
    /// # Arguments
    /// - `buf`: hostcall buffer pointer
    /// - `level`: trace level (TRACE_LEVEL_DEBUG/INFO/WARN/ERROR)
    /// - `msg`: message bytes (max 48 bytes, truncated if longer)
    /// - `msg_len`: message length
    #[inline(always)]
    pub unsafe fn gpu_hostcall_trace(
        buf: *mut u8,
        level: u8,
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
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_TRACE);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Slot 0: trace metadata
        let thread_idx = crate::nvptx_shim::thread_idx_x() as u16;
        let block_idx = crate::nvptx_shim::block_idx_x() as u16;
        let lane = gpu_atomics::lane_id() as u16;
        let copy_len = if msg_len > TRACE_MAX_MSG_LEN as u32 {
            TRACE_MAX_MSG_LEN as u32
        } else {
            msg_len
        };
        let meta = encode_trace_metadata(thread_idx, block_idx, level, copy_len as u8, lane);
        core::ptr::write_volatile(payload as *mut u64, meta);

        // Slot 1: GPU timestamp
        let timestamp: u64;
        #[cfg(target_arch = "nvptx64")]
        {
            core::arch::asm!("mov.u64 {}, %clock64;", out(reg64) timestamp);
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            timestamp = 0;
        }
        core::ptr::write_volatile(payload.add(8) as *mut u64, timestamp);

        // Slots 2-7: message bytes (up to 48 bytes)
        let dst = payload.add(16);
        let mut i: u32 = 0;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
            i += 1;
        }

        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);

        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Wait for host acknowledgment
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                return Err(GpuError::timeout());
            }
        }

        hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);

        Ok(())
    }

    /// Send a GPU assert diagnostic via hostcall, then trap.
    ///
    /// Sends the assertion message to the host (SERVICE_ASSERT), waits for
    /// acknowledgment, then executes PTX `trap` to halt the kernel.
    /// The host can display the assertion failure with thread coordinates.
    ///
    /// # Arguments
    /// - `buf`: hostcall buffer pointer
    /// - `msg`: assertion message bytes (max 56 bytes)
    /// - `msg_len`: message length
    #[inline(always)]
    pub unsafe fn gpu_hostcall_assert(buf: *mut u8, msg: *const u8, msg_len: u32) -> ! {
        let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
        let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
        let ready_ptr = get_ready_stack_ptr(buf, num_shards, shard_array_off);

        let pkt_idx = hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx != NULL_INDEX {
            let pkt_off = if num_shards == 0 {
                packet_offset(pkt_idx)
            } else {
                packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
            };
            let pkt = buf.add(pkt_off);

            let mask = activemask();
            core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
            core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_ASSERT);
            sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

            let payload = pkt.add(PKT_OFF_PAYLOAD);

            // Slot 0: metadata (same format as PANIC)
            let thread_idx = crate::nvptx_shim::thread_idx_x() as u16;
            let block_idx = crate::nvptx_shim::block_idx_x() as u16;
            let copy_len = if msg_len > ASSERT_MAX_MSG_LEN as u32 {
                ASSERT_MAX_MSG_LEN as u32
            } else {
                msg_len
            };
            let meta = encode_panic_metadata(thread_idx, block_idx, copy_len as u16);
            core::ptr::write_volatile(payload as *mut u64, meta);

            // Slots 1-7: message bytes
            let dst = payload.add(8);
            let mut i: u32 = 0;
            while i < copy_len {
                core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
                i += 1;
            }

            sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

            hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);

            sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

            // Wait for acknowledgment before trap
            let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
            let mut spins: u32 = 0;
            loop {
                let ctrl = sys_spin_load_acquire_u32(control_ptr);
                if ctrl & CONTROL_READY != 0 {
                    break;
                }
                spins += 1;
                if spins >= GPU_MAX_SPIN {
                    break; // Give up waiting, trap anyway
                }
            }
        }

        // Trap — halts the entire device
        #[cfg(target_arch = "nvptx64")]
        core::arch::asm!("trap;", options(noreturn));
        #[cfg(not(target_arch = "nvptx64"))]
        core::hint::unreachable_unchecked();
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
        // SAFETY: System-scope atomic fetch_add on the bump pointer ensures each
        // thread gets a unique, non-overlapping offset. The returned old_offset
        // is exclusive to this thread because fetch_add is atomic. The capacity
        // check after fetch_add may over-commit (the pointer advances even if
        // we return u64::MAX), but this is acceptable because sideband_reset()
        // is called between operations.
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

/// GPU-side buffered print — accumulate messages and flush via sideband.
///
/// Instead of one hostcall per `gpu_hostcall_print()` (~20-100us each),
/// messages are buffered in a per-thread sideband slot and flushed in
/// a single `SERVICE_BULK_PRINT` round-trip.
///
/// # Usage
///
/// ```ignore
/// // At kernel start:
/// print_buffer::init(sideband, thread_count);
///
/// // During kernel:
/// print_buffer::print(buf, sideband, msg.as_ptr(), msg.len() as u32);
///
/// // At kernel end (required!):
/// print_buffer::flush(buf, sideband);
/// ```
pub mod print_buffer {
    use gpu_atomics::{
        activemask, sys_fetch_add_u64, sys_spin_load_acquire_u32, sys_store_release_u32,
    };
    use gpu_protocol::*;

    use crate::hostcall::{
        get_free_stack_ptr, get_ready_stack_ptr, hc_pop_free_from, hc_push_with, read_shard_info,
    };

    /// Per-thread buffer slot size in bytes (header + data).
    const SLOT_SIZE: usize = 512;
    /// Header size within each slot (write_offset u32 + msg_count u32).
    const SLOT_HEADER: usize = 8;
    /// Maximum message data per slot.
    const SLOT_DATA_SIZE: usize = SLOT_SIZE - SLOT_HEADER;

    /// Get the flat thread ID for buffer indexing.
    #[inline(always)]
    fn tid() -> u32 {
        let x = crate::nvptx_shim::thread_idx_x();
        let y = crate::nvptx_shim::thread_idx_y();
        let z = crate::nvptx_shim::thread_idx_z();
        let dx = crate::nvptx_shim::block_dim_x();
        let dy = crate::nvptx_shim::block_dim_y();
        x + y * dx + z * dx * dy
    }

    /// Get pointer to this thread's buffer slot in sideband.
    /// The print buffer starts at a fixed offset after the sideband header.
    #[inline(always)]
    unsafe fn slot_ptr(sideband: *mut u8, thread_id: u32) -> *mut u8 {
        // Print buffer lives at a reserved region: sideband data offset + 0
        // (We reserve the first `max_threads * SLOT_SIZE` bytes of sideband data)
        sideband
            .add(SIDEBAND_DATA_OFFSET)
            .add(thread_id as usize * SLOT_SIZE)
    }

    /// Initialize this thread's print buffer slot.
    /// Call once at kernel start.
    ///
    /// # Safety
    /// `sideband` must be a valid sideband buffer pointer.
    /// `max_threads` must match the launch configuration.
    #[inline(always)]
    pub unsafe fn init(sideband: *mut u8, _max_threads: u32) {
        let t = tid();
        let slot = slot_ptr(sideband, t);
        // Zero write_offset and msg_count
        core::ptr::write_volatile(slot as *mut u32, 0u32);
        core::ptr::write_volatile(slot.add(4) as *mut u32, 0u32);
    }

    /// Buffer a print message without issuing a hostcall.
    /// If the buffer is full, auto-flushes first.
    ///
    /// Messages are stored as `[u16 len][len bytes data]`.
    ///
    /// # Safety
    /// `buf` must be a valid hostcall buffer pointer.
    /// `sideband` must be a valid sideband buffer pointer.
    #[inline(always)]
    pub unsafe fn print(
        buf: *mut u8,
        sideband: *mut u8,
        msg: *const u8,
        msg_len: u32,
    ) -> Result<(), GpuError> {
        let t = tid();
        let slot = slot_ptr(sideband, t);
        let write_off = core::ptr::read_volatile(slot as *const u32) as usize;
        let framed_len = 2 + msg_len as usize; // u16 length prefix + data

        // Auto-flush if message won't fit
        if write_off + framed_len > SLOT_DATA_SIZE {
            flush(buf, sideband)?;
            // Re-read (flush resets)
        }

        let write_off = core::ptr::read_volatile(slot as *const u32) as usize;
        // If message is too large even for empty buffer, fall back to direct print
        if framed_len > SLOT_DATA_SIZE {
            return crate::hostcall::gpu_hostcall_print(buf, msg, msg_len);
        }

        let data = slot.add(SLOT_HEADER + write_off);
        // Write length prefix (u16 LE)
        core::ptr::write_volatile(data as *mut u16, msg_len as u16);
        // Copy message bytes
        let dst = data.add(2);
        let mut i: u32 = 0;
        while i < msg_len {
            core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
            i += 1;
        }

        // Update write offset and message count
        let new_off = (write_off + framed_len) as u32;
        core::ptr::write_volatile(slot as *mut u32, new_off);
        let count = core::ptr::read_volatile(slot.add(4) as *const u32);
        core::ptr::write_volatile(slot.add(4) as *mut u32, count + 1);

        Ok(())
    }

    /// Flush all buffered print messages via SERVICE_BULK_PRINT.
    /// Resets the buffer after successful flush.
    ///
    /// # Safety
    /// Must be called before kernel exit.
    #[inline(always)]
    pub unsafe fn flush(buf: *mut u8, sideband: *mut u8) -> Result<(), GpuError> {
        let t = tid();
        let slot = slot_ptr(sideband, t);
        let write_off = core::ptr::read_volatile(slot as *const u32) as usize;

        if write_off == 0 {
            return Ok(()); // Nothing to flush
        }

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
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_BULK_PRINT);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Payload: sideband offset of this thread's data, data length, thread metadata
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let sideband_offset = (t as usize * SLOT_SIZE + SLOT_HEADER) as u64;
        core::ptr::write_volatile(payload as *mut u64, sideband_offset);
        core::ptr::write_volatile(payload.add(8) as *mut u64, write_off as u64);
        let block_idx = crate::nvptx_shim::block_idx_x();
        let thread_idx = crate::nvptx_shim::thread_idx_x();
        let metadata = (block_idx as u64) << 32 | (thread_idx as u64);
        core::ptr::write_volatile(payload.add(16) as *mut u64, metadata);

        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);

        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= crate::hostcall::GPU_MAX_SPIN {
                return Err(GpuError::timeout());
            }
        }

        hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);

        // Reset buffer
        core::ptr::write_volatile(slot as *mut u32, 0u32);
        core::ptr::write_volatile(slot.add(4) as *mut u32, 0u32);

        Ok(())
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
    ///
    /// SAFETY invariant: written exactly once at kernel entry via gpu_panic_init(),
    /// then read-only by all threads. The single-writer-then-read-only pattern
    /// prevents data races. No synchronization is needed because the kernel entry
    /// point completes init before any thread can panic.
    static mut PANIC_BUF: *mut u8 = core::ptr::null_mut();

    /// Global kernel result buffer pointer. Set by `gpu_result_init()`.
    /// The panic handler writes error info here before trapping.
    ///
    /// SAFETY invariant: same single-writer-then-read-only pattern as PANIC_BUF.
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

        // SAFETY: Inline Treiber stack pop — same ABA-tagged CAS protocol as
        // hc_pop_free_from. Duplicated here because the panic path must be
        // self-contained (cannot rely on other functions being reachable).
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

        // SAFETY: Inline Treiber stack push — same ABA-tagged CAS protocol as
        // hc_push_with. Epoch tag incremented to prevent ABA.
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

        // SAFETY: Inline Treiber stack push to free stack — returns packet
        // after host acknowledgment. Same ABA-tagged CAS as above.
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

/// Emit a structured trace event from GPU to host.
///
/// Usage:
/// ```rust,ignore
/// gpu_trace!(buf, INFO, "processing item {}", idx);
/// gpu_trace!(buf, DEBUG, "loop iteration");
/// gpu_trace!(buf, WARN, "buffer nearly full");
/// gpu_trace!(buf, ERROR, "unexpected value");
/// ```
///
/// `buf` is the hostcall buffer pointer. Level is one of DEBUG, INFO, WARN, ERROR.
/// The message is formatted into a fixed-size buffer (max 48 bytes) and sent via
/// SERVICE_TRACE hostcall with thread/block/warp metadata and GPU timestamp.
///
/// When the `gpu-trace` feature is disabled, this macro compiles to a no-op
/// for zero overhead in release builds.
#[cfg(feature = "gpu-trace")]
#[macro_export]
macro_rules! gpu_trace {
    ($buf:expr, DEBUG, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_DEBUG, $($arg)*)
    };
    ($buf:expr, INFO, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_INFO, $($arg)*)
    };
    ($buf:expr, WARN, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_WARN, $($arg)*)
    };
    ($buf:expr, ERROR, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_ERROR, $($arg)*)
    };
}

/// No-op version of `gpu_trace!` when `gpu-trace` feature is disabled.
#[cfg(not(feature = "gpu-trace"))]
#[macro_export]
macro_rules! gpu_trace {
    ($buf:expr, $level:ident, $($arg:tt)*) => {
        // Compiled out — zero overhead
        let _ = &$buf;
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _gpu_trace_impl {
    ($buf:expr, $level:expr, $($arg:tt)*) => {{
        let mut tbuf = $crate::panic::PanicBuf::new();
        {
            use core::fmt::Write;
            let _ = write!(tbuf, $($arg)*);
        }
        let msg = tbuf.as_slice();
        let _ = unsafe {
            $crate::hostcall::gpu_hostcall_trace($buf, $level, msg.as_ptr(), msg.len() as u32)
        };
    }};
}

/// Assert a condition on GPU, sending diagnostic info to host before trapping.
///
/// Usage:
/// ```rust,ignore
/// gpu_assert!(buf, x > 0, "x must be positive, got {}", x);
/// gpu_assert!(buf, ptr != core::ptr::null(), "null pointer");
/// ```
///
/// When `gpu-trace` feature is enabled: sends diagnostic message (with
/// thread/block coordinates) to host via SERVICE_ASSERT, then traps.
/// When disabled: traps without sending diagnostics (still catches the bug).
#[cfg(feature = "gpu-trace")]
#[macro_export]
macro_rules! gpu_assert {
    ($buf:expr, $cond:expr, $($arg:tt)*) => {
        if !($cond) {
            let mut tbuf = $crate::panic::PanicBuf::new();
            {
                use core::fmt::Write;
                let _ = write!(tbuf, "assertion failed: {}", stringify!($cond));
                let _ = write!(tbuf, " — ");
                let _ = write!(tbuf, $($arg)*);
            }
            let msg = tbuf.as_slice();
            unsafe {
                $crate::hostcall::gpu_hostcall_assert($buf, msg.as_ptr(), msg.len() as u32);
            }
        }
    };
    ($buf:expr, $cond:expr) => {
        if !($cond) {
            let msg = concat!("assertion failed: ", stringify!($cond));
            unsafe {
                $crate::hostcall::gpu_hostcall_assert($buf, msg.as_ptr(), msg.len() as u32);
            }
        }
    };
}

/// Minimal version of `gpu_assert!` when `gpu-trace` feature is disabled.
/// Still checks the condition and traps, but without sending diagnostics.
#[cfg(not(feature = "gpu-trace"))]
#[macro_export]
macro_rules! gpu_assert {
    ($buf:expr, $cond:expr $(, $($arg:tt)*)?) => {
        if !($cond) {
            let _ = &$buf;
            #[cfg(target_arch = "nvptx64")]
            unsafe {
                core::arch::asm!("trap;", options(noreturn));
            }
            #[cfg(not(target_arch = "nvptx64"))]
            panic!("GPU assertion failed");
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

/// Standard `core::future::Future` wrappers for hostcall operations.
///
/// These types implement `core::future::Future` — they are per-thread,
/// independent, and have NO warp awareness. They can be polled by any
/// single-threaded executor (Embassy, manual spin-poll, etc.).
///
/// The key design insight: inner futures are standard per-thread futures.
/// Warp cooperation is added by the CALLER's state machine (either
/// `#[warp_async]` proc macro or a future `#[warp_cooperative]` rustc pass).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::std_future::GpuPrintFuture;
/// use core::future::Future;
///
/// // Poll manually (single-thread):
/// let mut future = GpuPrintFuture::new(buf, b"Hello!");
/// loop {
///     match Pin::new_unchecked(&mut future).poll(&mut cx) {
///         Poll::Ready(ok) => break,
///         Poll::Pending => { /* yield / nanosleep */ }
///     }
/// }
/// ```
pub mod std_future {
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    use gpu_atomics::{activemask, sys_fetch_add_u64, sys_load_acquire_u32, sys_store_release_u32};
    use gpu_protocol::*;

    /// State machine for async hostcall print.
    enum PrintState {
        /// Initial: need to allocate packet and submit request.
        Init,
        /// Packet submitted, waiting for host response.
        Waiting { pkt_idx: u16 },
        /// Completed.
        Done,
    }

    /// A `core::future::Future` that performs a hostcall print asynchronously.
    ///
    /// On first poll: allocates a packet, fills the PRINT payload, submits
    /// to the ready stack, and rings the doorbell. Returns `Poll::Pending`.
    ///
    /// On subsequent polls: checks the control word for `CONTROL_READY`.
    /// Returns `Poll::Ready(true)` on success, `Poll::Ready(false)` on error,
    /// or `Poll::Pending` if the host hasn't responded yet.
    ///
    /// This is a standard per-thread future — no warp cooperation.
    pub struct GpuPrintFuture {
        buf: *mut u8,
        msg: *const u8,
        msg_len: u32,
        state: PrintState,
    }

    // SAFETY: On GPU, all threads access the same global memory.
    // The future is only used by one thread at a time.
    unsafe impl Send for GpuPrintFuture {}

    impl GpuPrintFuture {
        /// Create a new GpuPrintFuture.
        ///
        /// `buf` is the hostcall buffer (mapped memory).
        /// `msg` is the message to print (max 56 bytes, truncated if longer).
        #[inline(always)]
        pub fn new(buf: *mut u8, msg: &[u8]) -> Self {
            Self {
                buf,
                msg: msg.as_ptr(),
                msg_len: msg.len() as u32,
                state: PrintState::Init,
            }
        }
    }

    impl Future for GpuPrintFuture {
        type Output = bool;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<bool> {
            // SAFETY: This future has no self-referential fields (all fields are
            // raw pointers, integers, or a plain enum), so unpinning is safe.
            // The future is never moved after first poll — the executor (block_on
            // / SpinExecutor) pins it in place and polls via Pin::as_mut().
            //
            // All subsequent GPU futures (GpuOpenFuture, GpuReadFuture, etc.)
            // use the same get_unchecked_mut() pattern with identical structural
            // pinning justification and are not re-documented individually.
            let this = unsafe { self.get_unchecked_mut() };

            match this.state {
                PrintState::Init => unsafe {
                    // Pop a free packet using the hostcall module's helpers
                    let (num_shards, shard_array_off, _) =
                        crate::hostcall::read_shard_info(this.buf as *const u8);
                    let free_ptr =
                        crate::hostcall::get_free_stack_ptr(this.buf, num_shards, shard_array_off);

                    let pkt_idx = crate::hostcall::hc_pop_free_from(
                        this.buf,
                        free_ptr,
                        num_shards,
                        shard_array_off,
                    );
                    if pkt_idx == NULL_INDEX {
                        // Pool exhausted — backpressure, retry on next poll
                        return Poll::Pending;
                    }

                    let pkt_off = crate::hostcall::pkt_offset(this.buf as *const u8, pkt_idx);
                    let pkt = this.buf.add(pkt_off);

                    // Fill packet header
                    let mask = activemask();
                    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
                    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
                    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

                    // Fill payload: slot 0 = message length, then message bytes
                    let payload = pkt.add(PKT_OFF_PAYLOAD);
                    core::ptr::write_volatile(payload as *mut u64, this.msg_len as u64);

                    let copy_len = if this.msg_len > PRINT_MAX_MSG_LEN as u32 {
                        PRINT_MAX_MSG_LEN as u32
                    } else {
                        this.msg_len
                    };
                    let dst = payload.add(8);
                    let mut i: u32 = 0;
                    while i < copy_len {
                        core::ptr::write_volatile(dst.add(i as usize), *this.msg.add(i as usize));
                        i += 1;
                    }

                    // Thread/block metadata
                    let block_idx = crate::nvptx_shim::block_idx_x();
                    let thread_idx = crate::nvptx_shim::thread_idx_x();
                    core::ptr::write_volatile(payload.add(64) as *mut u32, block_idx);
                    core::ptr::write_volatile(payload.add(68) as *mut u32, thread_idx);

                    // Mark filled, push to ready stack, ring doorbell
                    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

                    let ready_ptr =
                        crate::hostcall::get_ready_stack_ptr(this.buf, num_shards, shard_array_off);
                    crate::hostcall::hc_push(ready_ptr, this.buf, pkt_idx);
                    sys_fetch_add_u64(this.buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

                    this.state = PrintState::Waiting { pkt_idx };
                    Poll::Pending
                },

                PrintState::Waiting { pkt_idx } => unsafe {
                    let pkt_off = crate::hostcall::pkt_offset(this.buf as *const u8, pkt_idx);
                    let pkt = this.buf.add(pkt_off);
                    let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

                    if ctrl & CONTROL_READY != 0 {
                        let success = (ctrl & CONTROL_ERROR) == 0;
                        // Release packet back to free stack
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = PrintState::Done;
                        Poll::Ready(success)
                    } else {
                        Poll::Pending
                    }
                },

                PrintState::Done => Poll::Ready(false),
            }
        }
    }

    /// Wrapper around `GpuPrintFuture` that returns `Result<bool, u32>`.
    ///
    /// Maps `true` → `Ok(true)`, `false` → `Err(1)` (print failure).
    /// Used for testing warp-cooperative `?` operator broadcasting.
    pub struct GpuPrintResultFuture {
        inner: GpuPrintFuture,
    }

    impl GpuPrintResultFuture {
        /// Create a new Result-returning print future.
        #[inline(always)]
        pub fn new(buf: *mut u8, msg: &[u8]) -> Self {
            Self {
                inner: GpuPrintFuture::new(buf, msg),
            }
        }
    }

    impl Future for GpuPrintResultFuture {
        type Output = Result<bool, u32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<bool, u32>> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let inner = unsafe { &mut self.get_unchecked_mut().inner };
            match unsafe { Pin::new_unchecked(inner) }.poll(cx) {
                Poll::Ready(true) => Poll::Ready(Ok(true)),
                Poll::Ready(false) => Poll::Ready(Err(1)), // print failure → error
                Poll::Pending => Poll::Pending,
            }
        }
    }

    // ================================================================
    // Async I/O Futures — generic hostcall futures for file operations
    // ================================================================

    /// Internal state for all hostcall futures.
    enum HostcallState {
        /// Initial: need to allocate packet and submit request.
        Init,
        /// Packet submitted, waiting for host response.
        Waiting { pkt_idx: u16 },
        /// Completed.
        Done,
    }

    /// Submit a hostcall packet and transition to Waiting state.
    ///
    /// Returns `Poll::Pending` on success, `Poll::Pending` on pool exhaustion (retry),
    /// or the new state.
    #[inline(always)]
    unsafe fn submit_hostcall(
        buf: *mut u8,
        service: u32,
        fill_payload: impl FnOnce(*mut u8),
    ) -> Result<u16, ()> {
        let (num_shards, shard_array_off, _) = crate::hostcall::read_shard_info(buf as *const u8);
        let free_ptr = crate::hostcall::get_free_stack_ptr(buf, num_shards, shard_array_off);

        let pkt_idx = crate::hostcall::hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            return Err(()); // Pool exhausted — retry on next poll
        }

        let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, pkt_idx);
        let pkt = buf.add(pkt_off);

        // Fill header
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Fill payload
        fill_payload(pkt.add(PKT_OFF_PAYLOAD));

        // Mark filled, push to ready stack, ring doorbell
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
        let ready_ptr = crate::hostcall::get_ready_stack_ptr(buf, num_shards, shard_array_off);
        crate::hostcall::hc_push(ready_ptr, buf, pkt_idx);
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        Ok(pkt_idx)
    }

    /// Check if a hostcall packet has a response ready.
    /// Returns `Some(pkt_ptr)` if ready, `None` if still pending.
    #[inline(always)]
    unsafe fn check_response(buf: *mut u8, pkt_idx: u16) -> Option<*mut u8> {
        let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, pkt_idx);
        let pkt = buf.add(pkt_off);
        let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);
        if ctrl & CONTROL_READY != 0 {
            Some(pkt)
        } else {
            None
        }
    }

    /// A `Future` that opens a file via hostcall.
    ///
    /// On first poll: submits SERVICE_OPEN request with path and flags.
    /// On subsequent polls: checks for response.
    /// Returns `Ok(fd)` on success, `Err(errno)` on failure.
    pub struct GpuOpenFuture {
        buf: *mut u8,
        path: *const u8,
        path_len: u32,
        flags: u32,
        state: HostcallState,
    }

    unsafe impl Send for GpuOpenFuture {}

    impl GpuOpenFuture {
        /// Create a new open future.
        ///
        /// `flags` uses gpu-protocol FILE_OPEN_* constants.
        #[inline(always)]
        pub fn new(buf: *mut u8, path: &[u8], flags: u32) -> Self {
            Self {
                buf,
                path: path.as_ptr(),
                path_len: path.len() as u32,
                flags,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuOpenFuture {
        type Output = Result<i32, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let path_ptr = this.path;
                    let path_len = this.path_len;
                    let flags = this.flags;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_OPEN, |payload| {
                            let slot0 = (path_len as u64) | ((flags as u64) << 32);
                            core::ptr::write_volatile(payload as *mut u64, slot0);
                            let dst = payload.add(8);
                            let mut i: u32 = 0;
                            while i < path_len && i < FILE_MAX_PATH_LEN as u32 {
                                core::ptr::write_volatile(
                                    dst.add(i as usize),
                                    *path_ptr.add(i as usize),
                                );
                                i += 1;
                            }
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending, // pool exhausted, retry
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let fd =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(fd as i32))
                            } else {
                                Poll::Ready(Ok(fd as i32))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that writes data to a file descriptor via hostcall.
    ///
    /// Returns `Ok(bytes_written)` on success, `Err(errno)` on failure.
    pub struct GpuWriteFuture {
        buf: *mut u8,
        fd: i32,
        data: *const u8,
        data_len: u32,
        state: HostcallState,
    }

    unsafe impl Send for GpuWriteFuture {}

    impl GpuWriteFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: i32, data: &[u8]) -> Self {
            Self {
                buf,
                fd,
                data: data.as_ptr(),
                data_len: data.len() as u32,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuWriteFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    let data_ptr = this.data;
                    let write_len = if this.data_len as usize > FILE_MAX_WRITE_LEN {
                        FILE_MAX_WRITE_LEN as u32
                    } else {
                        this.data_len
                    };
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_WRITE, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd as u64);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
                            let dst = payload.add(16);
                            let mut i: u32 = 0;
                            while i < write_len {
                                core::ptr::write_volatile(
                                    dst.add(i as usize),
                                    *data_ptr.add(i as usize),
                                );
                                i += 1;
                            }
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let written =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(written as i32))
                            } else {
                                Poll::Ready(Ok(written as usize))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that reads data from a file descriptor via hostcall.
    ///
    /// Returns `Ok((bytes_read, data))` on success. The data is copied into
    /// the caller-provided buffer.
    pub struct GpuReadFuture {
        buf: *mut u8,
        fd: i32,
        out_buf: *mut u8,
        max_len: u32,
        state: HostcallState,
    }

    unsafe impl Send for GpuReadFuture {}

    impl GpuReadFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: i32, out_buf: &mut [u8]) -> Self {
            Self {
                buf,
                fd,
                out_buf: out_buf.as_mut_ptr(),
                max_len: out_buf.len() as u32,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuReadFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    let max_len = if this.max_len as usize > FILE_MAX_READ_LEN {
                        FILE_MAX_READ_LEN as u32
                    } else {
                        this.max_len
                    };
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_READ, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd as u64);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let bytes_read =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            if ctrl & CONTROL_ERROR != 0 {
                                crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                                this.state = HostcallState::Done;
                                Poll::Ready(Err(bytes_read as i32))
                            } else {
                                // Copy data from payload to output buffer
                                let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
                                let n = core::cmp::min(bytes_read as usize, this.max_len as usize);
                                let mut i = 0;
                                while i < n {
                                    core::ptr::write_volatile(
                                        this.out_buf.add(i),
                                        core::ptr::read_volatile(src.add(i)),
                                    );
                                    i += 1;
                                }
                                crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                                this.state = HostcallState::Done;
                                Poll::Ready(Ok(n))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that closes a file descriptor via hostcall.
    ///
    /// Returns `Ok(())` on success, `Err(errno)` on failure.
    pub struct GpuCloseFuture {
        buf: *mut u8,
        fd: i32,
        state: HostcallState,
    }

    unsafe impl Send for GpuCloseFuture {}

    impl GpuCloseFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: i32) -> Self {
            Self {
                buf,
                fd,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuCloseFuture {
        type Output = Result<(), i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_CLOSE, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(-1))
                            } else {
                                Poll::Ready(Ok(()))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    // ================================================================
    // Async Bulk I/O Futures — sideband-based large data transfers
    // ================================================================

    /// A `Future` that writes data to a file via sideband bulk transfer.
    ///
    /// On first poll: allocates sideband space, copies data, submits SERVICE_BULK_WRITE.
    /// On subsequent polls: checks for response.
    /// Returns `Ok(bytes_written)` on success, `Err(-1)` on failure.
    pub struct GpuBulkWriteFuture {
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        src: *const u8,
        len: usize,
        sideband_offset: u64,
        state: HostcallState,
    }

    unsafe impl Send for GpuBulkWriteFuture {}

    impl GpuBulkWriteFuture {
        /// Create a new bulk write future.
        ///
        /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
        /// `fd` is the file descriptor, `src`/`len` describe the data to write.
        #[inline(always)]
        pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, src: *const u8, len: usize) -> Self {
            Self {
                buf,
                sideband,
                fd,
                src,
                len,
                sideband_offset: 0,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuBulkWriteFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    if this.len == 0 {
                        this.state = HostcallState::Done;
                        return Poll::Ready(Ok(0));
                    }

                    // Allocate sideband space
                    let offset =
                        unsafe { crate::sideband::sideband_alloc(this.sideband, this.len as u64) };
                    if offset == u64::MAX {
                        return Poll::Pending; // Retry on next poll
                    }
                    this.sideband_offset = offset;

                    // Copy data to sideband
                    unsafe {
                        let dst = this.sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
                        let mut i = 0;
                        while i < this.len {
                            core::ptr::write_volatile(dst.add(i), *this.src.add(i));
                            i += 1;
                        }
                    }

                    // Submit hostcall
                    let fd = this.fd;
                    let sb_offset = this.sideband_offset;
                    let len = this.len;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_BULK_WRITE, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                            core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let written =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if written == FILE_ERROR_SENTINEL {
                                Poll::Ready(Err(-1))
                            } else {
                                Poll::Ready(Ok(written as usize))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that reads data from a file via sideband bulk transfer.
    ///
    /// On first poll: allocates sideband space, submits SERVICE_BULK_READ.
    /// On subsequent polls: checks for response, copies data from sideband.
    /// Returns `Ok(bytes_read)` on success, `Err(-1)` on failure.
    pub struct GpuBulkReadFuture {
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        dst: *mut u8,
        max_len: usize,
        sideband_offset: u64,
        state: HostcallState,
    }

    unsafe impl Send for GpuBulkReadFuture {}

    impl GpuBulkReadFuture {
        /// Create a new bulk read future.
        ///
        /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
        /// `fd` is the file descriptor, `dst`/`max_len` describe the output buffer.
        #[inline(always)]
        pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, dst: *mut u8, max_len: usize) -> Self {
            Self {
                buf,
                sideband,
                fd,
                dst,
                max_len,
                sideband_offset: 0,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuBulkReadFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    if this.max_len == 0 {
                        this.state = HostcallState::Done;
                        return Poll::Ready(Ok(0));
                    }

                    // Allocate sideband space for response data
                    let offset = unsafe {
                        crate::sideband::sideband_alloc(this.sideband, this.max_len as u64)
                    };
                    if offset == u64::MAX {
                        return Poll::Pending; // Retry on next poll
                    }
                    this.sideband_offset = offset;

                    // Submit hostcall
                    let fd = this.fd;
                    let sb_offset = this.sideband_offset;
                    let max_len = this.max_len;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_BULK_READ, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                            core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let bytes_read =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;

                            if bytes_read == FILE_ERROR_SENTINEL || bytes_read == 0 {
                                if bytes_read == 0 {
                                    return Poll::Ready(Ok(0)); // EOF
                                }
                                return Poll::Ready(Err(-1));
                            }

                            // Copy data from sideband to destination
                            let src = this
                                .sideband
                                .add(SIDEBAND_DATA_OFFSET + this.sideband_offset as usize);
                            let n = bytes_read as usize;
                            let mut i = 0;
                            while i < n {
                                core::ptr::write_volatile(
                                    this.dst.add(i),
                                    core::ptr::read_volatile(src.add(i)),
                                );
                                i += 1;
                            }

                            Poll::Ready(Ok(n))
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    // ================================================================
    // Async TCP Futures — inline data and sideband-based transfers
    // ================================================================

    /// A `Future` that connects to a remote TCP address:port via hostcall.
    ///
    /// On first poll: submits SERVICE_TCP_CONNECT with address and port.
    /// On subsequent polls: checks for response.
    /// Returns `Ok(fd)` on success, `Err(errno)` on failure.
    pub struct GpuTcpConnectFuture {
        buf: *mut u8,
        addr: *const u8,
        addr_len: u32,
        port: u32,
        state: HostcallState,
    }

    // SAFETY: On GPU, all threads access the same global memory.
    // The future is only used by one thread at a time.
    unsafe impl Send for GpuTcpConnectFuture {}

    impl GpuTcpConnectFuture {
        /// Create a new TCP connect future.
        ///
        /// `addr` is the address string (e.g. "127.0.0.1"), max 56 bytes.
        /// `port` is the TCP port number.
        #[inline(always)]
        pub fn new(buf: *mut u8, addr: &[u8], port: u32) -> Self {
            Self {
                buf,
                addr: addr.as_ptr(),
                addr_len: addr.len() as u32,
                port,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpConnectFuture {
        type Output = Result<u64, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let addr_ptr = this.addr;
                    let addr_len = this.addr_len;
                    let port = this.port;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_CONNECT, |payload| {
                            // slot0 = port (low 32) | addr_len (high 32)
                            let slot0 = (port as u64) | ((addr_len as u64) << 32);
                            core::ptr::write_volatile(payload as *mut u64, slot0);
                            // slots 1-7: address string (max 56 bytes)
                            let dst = payload.add(8);
                            let mut i: u32 = 0;
                            while i < addr_len && i < TCP_MAX_ADDR_LEN as u32 {
                                core::ptr::write_volatile(
                                    dst.add(i as usize),
                                    *addr_ptr.add(i as usize),
                                );
                                i += 1;
                            }
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending, // pool exhausted, retry
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let fd =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(fd as i32))
                            } else {
                                Poll::Ready(Ok(fd))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that writes inline data to a TCP socket via hostcall.
    ///
    /// Returns `Ok(bytes_written)` on success, `Err(errno)` on failure.
    /// Maximum inline write is 48 bytes (6 slots).
    pub struct GpuTcpWriteFuture {
        buf: *mut u8,
        fd: u64,
        src: *const u8,
        len: u32,
        state: HostcallState,
    }

    unsafe impl Send for GpuTcpWriteFuture {}

    impl GpuTcpWriteFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: u64, data: &[u8]) -> Self {
            Self {
                buf,
                fd,
                src: data.as_ptr(),
                len: data.len() as u32,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpWriteFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    let src_ptr = this.src;
                    let write_len = if this.len as usize > TCP_MAX_WRITE_LEN {
                        TCP_MAX_WRITE_LEN as u32
                    } else {
                        this.len
                    };
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_WRITE, |payload| {
                            // slot0 = fd, slot1 = len
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
                            // slots 2-7: data bytes (max 48 bytes)
                            let dst = payload.add(16);
                            let mut i: u32 = 0;
                            while i < write_len {
                                core::ptr::write_volatile(
                                    dst.add(i as usize),
                                    *src_ptr.add(i as usize),
                                );
                                i += 1;
                            }
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let written =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(written as i32))
                            } else {
                                Poll::Ready(Ok(written as usize))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that reads inline data from a TCP socket via hostcall.
    ///
    /// Returns `Ok(bytes_read)` on success. Data is copied into the
    /// caller-provided buffer. Maximum inline read is 56 bytes (7 slots).
    pub struct GpuTcpReadFuture {
        buf: *mut u8,
        fd: u64,
        dst: *mut u8,
        max_len: u32,
        state: HostcallState,
    }

    unsafe impl Send for GpuTcpReadFuture {}

    impl GpuTcpReadFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: u64, out_buf: &mut [u8]) -> Self {
            Self {
                buf,
                fd,
                dst: out_buf.as_mut_ptr(),
                max_len: out_buf.len() as u32,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpReadFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    let max_len = if this.max_len as usize > TCP_MAX_READ_LEN {
                        TCP_MAX_READ_LEN as u32
                    } else {
                        this.max_len
                    };
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_READ, |payload| {
                            // slot0 = fd, slot1 = max_len
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            let bytes_read =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            if ctrl & CONTROL_ERROR != 0 {
                                crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                                this.state = HostcallState::Done;
                                Poll::Ready(Err(bytes_read as i32))
                            } else {
                                // Copy data from payload slots 1-7 to destination buffer
                                let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
                                let n = core::cmp::min(bytes_read as usize, this.max_len as usize);
                                let mut i = 0;
                                while i < n {
                                    core::ptr::write_volatile(
                                        this.dst.add(i),
                                        core::ptr::read_volatile(src.add(i)),
                                    );
                                    i += 1;
                                }
                                crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                                this.state = HostcallState::Done;
                                Poll::Ready(Ok(n))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that closes a TCP socket via hostcall.
    ///
    /// Returns `Ok(())` on success, `Err(errno)` on failure.
    pub struct GpuTcpCloseFuture {
        buf: *mut u8,
        fd: u64,
        state: HostcallState,
    }

    unsafe impl Send for GpuTcpCloseFuture {}

    impl GpuTcpCloseFuture {
        #[inline(always)]
        pub fn new(buf: *mut u8, fd: u64) -> Self {
            Self {
                buf,
                fd,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpCloseFuture {
        type Output = Result<(), i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    let fd = this.fd;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_CLOSE, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let ctrl =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if ctrl & CONTROL_ERROR != 0 {
                                Poll::Ready(Err(-1))
                            } else {
                                Poll::Ready(Ok(()))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that writes data to a TCP socket via sideband bulk transfer.
    ///
    /// On first poll: allocates sideband space, copies data, submits SERVICE_TCP_BULK_WRITE.
    /// On subsequent polls: checks for response.
    /// Returns `Ok(bytes_written)` on success, `Err(-1)` on failure.
    pub struct GpuTcpBulkWriteFuture {
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        src: *const u8,
        len: usize,
        sideband_offset: u64,
        state: HostcallState,
    }

    unsafe impl Send for GpuTcpBulkWriteFuture {}

    impl GpuTcpBulkWriteFuture {
        /// Create a new TCP bulk write future.
        ///
        /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
        /// `fd` is the socket fd, `src`/`len` describe the data to write.
        #[inline(always)]
        pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, src: *const u8, len: usize) -> Self {
            Self {
                buf,
                sideband,
                fd,
                src,
                len,
                sideband_offset: 0,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpBulkWriteFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    if this.len == 0 {
                        this.state = HostcallState::Done;
                        return Poll::Ready(Ok(0));
                    }

                    // Allocate sideband space
                    let offset =
                        unsafe { crate::sideband::sideband_alloc(this.sideband, this.len as u64) };
                    if offset == u64::MAX {
                        return Poll::Pending; // Retry on next poll
                    }
                    this.sideband_offset = offset;

                    // Copy data to sideband
                    unsafe {
                        let dst = this.sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
                        let mut i = 0;
                        while i < this.len {
                            core::ptr::write_volatile(dst.add(i), *this.src.add(i));
                            i += 1;
                        }
                    }

                    // Submit hostcall
                    let fd = this.fd;
                    let sb_offset = this.sideband_offset;
                    let len = this.len;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_BULK_WRITE, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                            core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let written =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            if written == FILE_ERROR_SENTINEL {
                                Poll::Ready(Err(-1))
                            } else {
                                Poll::Ready(Ok(written as usize))
                            }
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// A `Future` that reads data from a TCP socket via sideband bulk transfer.
    ///
    /// On first poll: allocates sideband space, submits SERVICE_TCP_BULK_READ.
    /// On subsequent polls: checks for response, copies data from sideband.
    /// Returns `Ok(bytes_read)` on success, `Err(-1)` on failure.
    pub struct GpuTcpBulkReadFuture {
        buf: *mut u8,
        sideband: *mut u8,
        fd: u64,
        dst: *mut u8,
        max_len: usize,
        sideband_offset: u64,
        state: HostcallState,
    }

    unsafe impl Send for GpuTcpBulkReadFuture {}

    impl GpuTcpBulkReadFuture {
        /// Create a new TCP bulk read future.
        ///
        /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
        /// `fd` is the socket fd, `dst`/`max_len` describe the output buffer.
        #[inline(always)]
        pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, dst: *mut u8, max_len: usize) -> Self {
            Self {
                buf,
                sideband,
                fd,
                dst,
                max_len,
                sideband_offset: 0,
                state: HostcallState::Init,
            }
        }
    }

    impl Future for GpuTcpBulkReadFuture {
        type Output = Result<usize, i32>;

        #[inline(always)]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
            let this = unsafe { self.get_unchecked_mut() };
            match this.state {
                HostcallState::Init => {
                    if this.max_len == 0 {
                        this.state = HostcallState::Done;
                        return Poll::Ready(Ok(0));
                    }

                    // Allocate sideband space for response data
                    let offset = unsafe {
                        crate::sideband::sideband_alloc(this.sideband, this.max_len as u64)
                    };
                    if offset == u64::MAX {
                        return Poll::Pending; // Retry on next poll
                    }
                    this.sideband_offset = offset;

                    // Submit hostcall
                    let fd = this.fd;
                    let sb_offset = this.sideband_offset;
                    let max_len = this.max_len;
                    match unsafe {
                        submit_hostcall(this.buf, SERVICE_TCP_BULK_READ, |payload| {
                            core::ptr::write_volatile(payload as *mut u64, fd);
                            core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                            core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
                        })
                    } {
                        Ok(idx) => {
                            this.state = HostcallState::Waiting { pkt_idx: idx };
                            Poll::Pending
                        }
                        Err(()) => Poll::Pending,
                    }
                }
                HostcallState::Waiting { pkt_idx } => unsafe {
                    match check_response(this.buf, pkt_idx) {
                        Some(pkt) => {
                            let bytes_read =
                                core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;

                            if bytes_read == FILE_ERROR_SENTINEL || bytes_read == 0 {
                                if bytes_read == 0 {
                                    return Poll::Ready(Ok(0)); // Connection closed / EOF
                                }
                                return Poll::Ready(Err(-1));
                            }

                            // Copy data from sideband to destination
                            let src = this
                                .sideband
                                .add(SIDEBAND_DATA_OFFSET + this.sideband_offset as usize);
                            let n = bytes_read as usize;
                            let mut i = 0;
                            while i < n {
                                core::ptr::write_volatile(
                                    this.dst.add(i),
                                    core::ptr::read_volatile(src.add(i)),
                                );
                                i += 1;
                            }

                            Poll::Ready(Ok(n))
                        }
                        None => Poll::Pending,
                    }
                },
                HostcallState::Done => Poll::Ready(Err(-1)),
            }
        }
    }

    /// Default maximum poll iterations before timeout.
    const DEFAULT_MAX_POLLS: u32 = 10_000_000;

    /// Default nanosleep duration between polls (nanoseconds).
    const DEFAULT_NANOSLEEP_NS: u32 = 1000;

    /// No-op waker vtable for GPU (no real wake mechanism).
    const NOOP_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
        |_| core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    /// Create a no-op waker suitable for GPU spin-polling.
    #[inline(always)]
    fn noop_waker() -> core::task::Waker {
        unsafe {
            core::task::Waker::from_raw(core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE))
        }
    }

    /// Run a future to completion by spin-polling with nanosleep yield.
    ///
    /// This is the primary way to drive async code on GPU. Replaces the
    /// manual waker/context/pin/poll boilerplate that every kernel previously
    /// needed.
    ///
    /// Returns `Some(output)` on completion, `None` on timeout (10M polls).
    ///
    /// # Safety
    /// The caller must ensure the future is safe to poll on the current thread
    /// and that any raw pointers captured by the future remain valid.
    ///
    /// # Example
    /// ```no_run
    /// let result = unsafe { block_on(data_pipeline(buf)) };
    /// ```
    #[inline(always)]
    pub unsafe fn block_on<F: Future>(future: F) -> Option<F::Output> {
        block_on_with(future, DEFAULT_MAX_POLLS, DEFAULT_NANOSLEEP_NS)
    }

    /// Run a future with custom poll limit and nanosleep duration.
    ///
    /// # Safety
    /// Same as [`block_on`].
    #[inline(always)]
    pub unsafe fn block_on_with<F: Future>(
        future: F,
        max_polls: u32,
        #[allow(unused_variables)] nanosleep_ns: u32,
    ) -> Option<F::Output> {
        let mut future = future;
        // SAFETY: The future is stack-pinned here and never moved — we poll it
        // in a loop via Pin::as_mut() until completion or timeout.
        let mut future = Pin::new_unchecked(&mut future);
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut polls: u32 = 0;
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(output) => return Some(output),
                Poll::Pending => {
                    polls += 1;
                    if polls >= max_polls {
                        return None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    {
                        // Yield SM scheduler slot — gives host time to respond
                        // and allows other warps to execute.
                        match nanosleep_ns {
                            64 => core::arch::asm!("nanosleep.u32 64;", options(nostack)),
                            1000 => core::arch::asm!("nanosleep.u32 1000;", options(nostack)),
                            _ => core::arch::asm!("nanosleep.u32 1000;", options(nostack)),
                        }
                    }
                }
            }
        }
    }

    /// Minimal spin-poll executor for a single `Future`.
    ///
    /// No waker, no task queue — just polls in a loop with nanosleep yield.
    /// Prefer the free function [`block_on`] for simpler usage.
    pub struct SpinExecutor;

    impl SpinExecutor {
        /// Run a future to completion by spin-polling.
        ///
        /// Returns the future's output, or `None` if MAX_POLLS exceeded.
        ///
        /// # Safety
        /// The future must be safe to poll repeatedly on the current thread.
        #[inline(always)]
        pub unsafe fn run<F: Future>(future: &mut F) -> Option<F::Output> {
            // SAFETY: The caller provides a mutable reference that we pin in place.
            // We only poll via Pin::as_mut() and never move the future.
            let mut future = Pin::new_unchecked(future);
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);

            let mut polls: u32 = 0;
            loop {
                match future.as_mut().poll(&mut cx) {
                    Poll::Ready(output) => return Some(output),
                    Poll::Pending => {
                        polls += 1;
                        if polls >= DEFAULT_MAX_POLLS {
                            return None;
                        }
                        #[cfg(target_arch = "nvptx64")]
                        core::arch::asm!("nanosleep.u32 1000;", options(nostack));
                    }
                }
            }
        }
    }
}

/// Warp-cooperative wrapper for standard `core::future::Future`.
///
/// This is the key proof for warp-future-bridge: standard per-thread futures
/// can be polled warp-cooperatively. Lane 0 polls the inner future, broadcasts
/// the `Poll` discriminant via `shfl.sync`, and all lanes converge.
///
/// This bridges the gap between `core::future::Future` (per-thread) and
/// warp-cooperative execution (SIMT lockstep).
pub mod warp_cooperative {
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

    // Poll result encoding for broadcast:
    // 0 = Pending, 1 = Ready(true), 2 = Ready(false)
    const POLL_PENDING: u32 = 0;
    const POLL_READY_TRUE: u32 = 1;
    const POLL_READY_FALSE: u32 = 2;

    /// Warp-cooperatively poll a standard `impl Future<Output = bool>`.
    ///
    /// All 32 lanes must call this simultaneously. Lane 0 actually polls
    /// the future; the result is broadcast via `shfl.sync.idx.b32` so all
    /// lanes observe the same `Poll` value.
    ///
    /// Returns the same `Poll<bool>` to all lanes.
    ///
    /// # Safety
    /// - Must be called by all active lanes simultaneously
    /// - `future` must be safe to poll on lane 0
    #[inline(always)]
    pub unsafe fn warp_poll_future(
        future: Pin<&mut impl Future<Output = bool>>,
        cx: &mut Context<'_>,
    ) -> Poll<bool> {
        let mask = activemask();
        let lid = lane_id();

        // Lane 0 polls the actual future
        let mut result_code: u32 = POLL_PENDING;
        if lid == 0 {
            match future.poll(cx) {
                Poll::Ready(true) => result_code = POLL_READY_TRUE,
                Poll::Ready(false) => result_code = POLL_READY_FALSE,
                Poll::Pending => result_code = POLL_PENDING,
            }
        }

        // Broadcast poll result to all lanes
        let broadcast_result = shfl_sync_idx_u32(mask, result_code, 0);

        // All lanes see the same result
        syncwarp(mask);

        match broadcast_result {
            POLL_READY_TRUE => Poll::Ready(true),
            POLL_READY_FALSE => Poll::Ready(false),
            _ => Poll::Pending,
        }
    }

    /// Warp-cooperative spin executor for a standard `impl Future<Output = bool>`.
    ///
    /// All 32 lanes call this together. Lane 0 polls the future; result is
    /// broadcast to all lanes. Returns when the future completes.
    ///
    /// # Safety
    /// - Must be called by all active lanes simultaneously
    /// - The future must be safe to poll on lane 0
    #[inline(always)]
    pub unsafe fn warp_run_future(future: &mut impl Future<Output = bool>) -> Option<bool> {
        const MAX_POLLS: u32 = 10_000_000;

        let mask = activemask();
        let lid = lane_id();

        let mut future = Pin::new_unchecked(future);

        // Create a no-op waker (only lane 0 uses it, but all lanes must have it
        // for convergence)
        const VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
            |_| core::task::RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        let raw_waker = core::task::RawWaker::new(core::ptr::null(), &VTABLE);
        let waker = core::task::Waker::from_raw(raw_waker);
        let mut cx = Context::from_waker(&waker);

        let mut polls: u32 = 0;
        loop {
            // Lane 0 polls, broadcasts result
            let mut result_code: u32 = POLL_PENDING;
            if lid == 0 {
                match future.as_mut().poll(&mut cx) {
                    Poll::Ready(true) => result_code = POLL_READY_TRUE,
                    Poll::Ready(false) => result_code = POLL_READY_FALSE,
                    Poll::Pending => result_code = POLL_PENDING,
                }
            }

            let broadcast_result = shfl_sync_idx_u32(mask, result_code, 0);
            syncwarp(mask);

            match broadcast_result {
                POLL_READY_TRUE => return Some(true),
                POLL_READY_FALSE => return Some(false),
                _ => {
                    polls += 1;
                    if polls >= MAX_POLLS {
                        return None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
        }
    }
}

/// Warp-cooperative sequential executor for two standard futures.
///
/// Simulates two sequential `.await` points: runs F1 to completion
/// (warp-cooperatively), then runs F2 to completion. All lanes
/// stay converged throughout.
///
/// This is the manual proof of what `#[warp_cooperative] async fn` will
/// eventually generate automatically.
pub mod warp_sequential {
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

    const POLL_PENDING: u32 = 0;
    const POLL_READY_TRUE: u32 = 1;
    const POLL_READY_FALSE: u32 = 2;

    /// Run two futures sequentially with warp convergence.
    ///
    /// All lanes participate. Lane 0 polls each future; result is broadcast.
    /// Returns `(ok1, ok2)` — the results of both futures.
    ///
    /// # Safety
    /// Must be called by all active lanes simultaneously.
    #[inline(always)]
    pub unsafe fn warp_run_two_futures(
        f1: &mut impl Future<Output = bool>,
        f2: &mut impl Future<Output = bool>,
    ) -> (Option<bool>, Option<bool>) {
        const MAX_POLLS: u32 = 10_000_000;

        let mask = activemask();
        let lid = lane_id();

        const VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
            |_| core::task::RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        let raw_waker = core::task::RawWaker::new(core::ptr::null(), &VTABLE);
        let waker = core::task::Waker::from_raw(raw_waker);
        let mut cx = Context::from_waker(&waker);

        // Phase 1: poll f1 to completion
        let mut f1 = Pin::new_unchecked(f1);
        let mut polls: u32 = 0;
        let ok1 = loop {
            let mut result_code: u32 = POLL_PENDING;
            if lid == 0 {
                match f1.as_mut().poll(&mut cx) {
                    Poll::Ready(true) => result_code = POLL_READY_TRUE,
                    Poll::Ready(false) => result_code = POLL_READY_FALSE,
                    Poll::Pending => result_code = POLL_PENDING,
                }
            }
            let broadcast = shfl_sync_idx_u32(mask, result_code, 0);
            syncwarp(mask);

            match broadcast {
                POLL_READY_TRUE => break Some(true),
                POLL_READY_FALSE => break Some(false),
                _ => {
                    polls += 1;
                    if polls >= MAX_POLLS {
                        break None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
        };

        // Convergence barrier between the two "await" points
        syncwarp(mask);

        // Phase 2: poll f2 to completion
        let mut f2 = Pin::new_unchecked(f2);
        polls = 0;
        let ok2 = loop {
            let mut result_code: u32 = POLL_PENDING;
            if lid == 0 {
                match f2.as_mut().poll(&mut cx) {
                    Poll::Ready(true) => result_code = POLL_READY_TRUE,
                    Poll::Ready(false) => result_code = POLL_READY_FALSE,
                    Poll::Pending => result_code = POLL_PENDING,
                }
            }
            let broadcast = shfl_sync_idx_u32(mask, result_code, 0);
            syncwarp(mask);

            match broadcast {
                POLL_READY_TRUE => break Some(true),
                POLL_READY_FALSE => break Some(false),
                _ => {
                    polls += 1;
                    if polls >= MAX_POLLS {
                        break None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
        };

        (ok1, ok2)
    }
}

/// Warp-cooperative Result broadcasting for `? operator` across .await boundaries.
///
/// Extends the warp-cooperative model to handle `Result<T, E>` — when lane 0
/// polls a future that returns `Result`, the discriminant (Ok/Err) and error
/// code are broadcast to all lanes. If Err, all lanes can early-return together.
pub mod warp_result {
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

    // Result encoding for broadcast:
    // 0 = Pending, 1 = Ready(Ok(true)), 2 = Ready(Ok(false)), 3 = Ready(Err(code))
    const POLL_PENDING: u32 = 0;
    const POLL_OK_TRUE: u32 = 1;
    const POLL_OK_FALSE: u32 = 2;
    const POLL_ERR: u32 = 3;

    /// Warp-cooperative poll result with error support.
    pub enum WarpResult {
        /// Future still pending
        Pending,
        /// Future completed with Ok(true)
        OkTrue,
        /// Future completed with Ok(false)
        OkFalse,
        /// Future completed with Err(error_code)
        Err(u32),
    }

    /// Run a `Future<Output = Result<bool, u32>>` warp-cooperatively.
    ///
    /// Lane 0 polls; broadcasts both discriminant and error code (if any).
    /// All lanes see the same `WarpResult`.
    ///
    /// # Safety
    /// Must be called by all active lanes simultaneously.
    #[inline(always)]
    pub unsafe fn warp_run_result_future(
        future: &mut impl Future<Output = Result<bool, u32>>,
    ) -> WarpResult {
        const MAX_POLLS: u32 = 10_000_000;

        let mask = activemask();
        let lid = lane_id();

        let mut future = Pin::new_unchecked(future);

        const VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
            |_| core::task::RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        let raw_waker = core::task::RawWaker::new(core::ptr::null(), &VTABLE);
        let waker = core::task::Waker::from_raw(raw_waker);
        let mut cx = Context::from_waker(&waker);

        let mut polls: u32 = 0;
        loop {
            let mut result_code: u32 = POLL_PENDING;
            let mut error_code: u32 = 0;
            if lid == 0 {
                match future.as_mut().poll(&mut cx) {
                    Poll::Ready(Ok(true)) => result_code = POLL_OK_TRUE,
                    Poll::Ready(Ok(false)) => result_code = POLL_OK_FALSE,
                    Poll::Ready(Err(e)) => {
                        result_code = POLL_ERR;
                        error_code = e;
                    }
                    Poll::Pending => result_code = POLL_PENDING,
                }
            }

            // Broadcast both discriminant and error code
            let bc_result = shfl_sync_idx_u32(mask, result_code, 0);
            let bc_error = shfl_sync_idx_u32(mask, error_code, 0);
            syncwarp(mask);

            match bc_result {
                POLL_OK_TRUE => return WarpResult::OkTrue,
                POLL_OK_FALSE => return WarpResult::OkFalse,
                POLL_ERR => return WarpResult::Err(bc_error),
                _ => {
                    polls += 1;
                    if polls >= MAX_POLLS {
                        return WarpResult::Err(0xDEAD); // timeout
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
        }
    }

    /// Run two Result futures sequentially with ? semantics.
    ///
    /// If f1 returns Err, f2 is skipped (all lanes return Err together).
    /// This is the warp-cooperative equivalent of:
    ///   f1.await?;
    ///   f2.await?;
    ///
    /// # Safety
    /// Must be called by all active lanes simultaneously.
    #[inline(always)]
    pub unsafe fn warp_run_two_result_futures(
        f1: &mut impl Future<Output = Result<bool, u32>>,
        f2: &mut impl Future<Output = Result<bool, u32>>,
    ) -> Result<u32, u32> {
        let mask = activemask();

        // First .await?
        match warp_run_result_future(f1) {
            WarpResult::OkTrue | WarpResult::OkFalse => {} // continue
            WarpResult::Err(e) => return Err(e),           // all lanes early-return
            WarpResult::Pending => return Err(0xDEAD),     // unreachable
        }

        syncwarp(mask);

        // Second .await?
        match warp_run_result_future(f2) {
            WarpResult::OkTrue | WarpResult::OkFalse => {} // continue
            WarpResult::Err(e) => return Err(e),
            WarpResult::Pending => return Err(0xDEAD),
        }

        Ok(2) // both succeeded
    }
}

/// Command buffer polling — GPU-side API for host→GPU command channel.
///
/// The host submits commands to a mapped-memory ring buffer; the GPU kernel
/// polls with `cmd_poll()` and acknowledges with `cmd_ack()`.
pub mod cmd {
    use gpu_atomics::{sys_load_acquire_u64, sys_store_release_u64};
    use gpu_protocol::{
        CMD_BUF_HEADER_SIZE, CMD_OFF_CAPACITY, CMD_OFF_READ_IDX, CMD_OFF_WRITE_IDX,
        CMD_SLOT_OFF_PAYLOAD, CMD_SLOT_OFF_TYPE, CMD_SLOT_SIZE,
    };

    /// Poll the command buffer for the next command.
    ///
    /// Returns `Some((cmd_type, payload_ptr))` if a command is available,
    /// or `None` if no commands are pending (caller should retry).
    ///
    /// # Safety
    /// `cmd_buf` must point to a valid mapped-memory command buffer.
    #[inline(always)]
    pub unsafe fn cmd_poll(cmd_buf: *const u8) -> Option<(u32, *const u8)> {
        // read_idx is GPU-local — only we write it
        let read_idx = core::ptr::read_volatile(cmd_buf.add(CMD_OFF_READ_IDX) as *const u64);
        // write_idx is written by host — acquire to see payload
        let write_idx = sys_load_acquire_u64(cmd_buf.add(CMD_OFF_WRITE_IDX) as *const u64);
        if read_idx >= write_idx {
            return None;
        }
        let capacity = core::ptr::read_volatile(cmd_buf.add(CMD_OFF_CAPACITY) as *const u32);
        let slot_idx = (read_idx % capacity as u64) as usize;
        let slot_ptr = cmd_buf.add(CMD_BUF_HEADER_SIZE + slot_idx * CMD_SLOT_SIZE);
        let cmd_type = core::ptr::read_volatile(slot_ptr.add(CMD_SLOT_OFF_TYPE) as *const u32);
        Some((cmd_type, slot_ptr.add(CMD_SLOT_OFF_PAYLOAD)))
    }

    /// Acknowledge that the current command has been processed.
    ///
    /// Increments `read_idx` with release semantics so the host can
    /// observe that a slot has been freed (for backpressure).
    ///
    /// # Safety
    /// `cmd_buf` must point to a valid mapped-memory command buffer.
    /// Must be called exactly once after each successful `cmd_poll()`.
    #[inline(always)]
    pub unsafe fn cmd_ack(cmd_buf: *mut u8) {
        let read_idx = core::ptr::read_volatile(cmd_buf.add(CMD_OFF_READ_IDX) as *const u64);
        sys_store_release_u64(cmd_buf.add(CMD_OFF_READ_IDX) as *mut u64, read_idx + 1);
    }

    /// Sleep briefly to yield the warp scheduler slot while polling.
    #[inline(always)]
    pub unsafe fn cmd_yield() {
        #[cfg(target_arch = "nvptx64")]
        core::arch::asm!("nanosleep.u32 1000;", options(nostack));
    }
}

/// Flight recorder — GPU-side ring buffer for post-mortem trace events.
///
/// Unlike `gpu_trace!()` which sends events to the host via hostcall,
/// the flight recorder writes directly to mapped memory with no round-trip.
/// On kernel crash, the host can dump the last N events for post-mortem analysis.
pub mod flight_recorder {
    use gpu_atomics::sys_fetch_add_u64;
    use gpu_protocol::{
        encode_trace_metadata, FR_HEADER_SIZE, FR_MAX_MSG_LEN, FR_OFF_CAPACITY, FR_OFF_FLAGS,
        FR_OFF_WRITE_IDX, FR_SLOT_OFF_META, FR_SLOT_OFF_MSG, FR_SLOT_OFF_TIMESTAMP, FR_SLOT_SIZE,
    };

    /// Record a trace event to the flight recorder ring buffer.
    ///
    /// This is a fire-and-forget write — no hostcall needed. Multiple GPU
    /// threads can write concurrently using atomic fetch_add on write_idx.
    ///
    /// # Safety
    /// `fr_buf` must point to a valid flight recorder buffer (mapped memory).
    #[inline(always)]
    pub unsafe fn fr_record(fr_buf: *mut u8, level: u8, msg: *const u8, msg_len: u32) {
        let capacity = core::ptr::read_volatile(fr_buf.add(FR_OFF_CAPACITY) as *const u32) as u64;
        if capacity == 0 {
            return;
        }

        // Atomically claim a slot
        let write_idx = sys_fetch_add_u64(fr_buf.add(FR_OFF_WRITE_IDX) as *mut u64, 1);
        let slot_idx = (write_idx % capacity) as usize;
        let slot = fr_buf.add(FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE);

        // Build metadata (reuse trace protocol encoding)
        let thread_idx;
        let block_idx;
        let lane;
        #[cfg(target_arch = "nvptx64")]
        {
            thread_idx = crate::nvptx_shim::thread_idx_x() as u16;
            block_idx = crate::nvptx_shim::block_idx_x() as u16;
            lane = gpu_atomics::lane_id() as u16;
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            thread_idx = 0u16;
            block_idx = 0u16;
            lane = 0u16;
        }

        let copy_len = if msg_len > FR_MAX_MSG_LEN as u32 {
            FR_MAX_MSG_LEN as u32
        } else {
            msg_len
        };
        let meta = encode_trace_metadata(thread_idx, block_idx, level, copy_len as u8, lane);
        core::ptr::write_volatile(slot.add(FR_SLOT_OFF_META) as *mut u64, meta);

        // Timestamp
        let timestamp: u64;
        #[cfg(target_arch = "nvptx64")]
        {
            core::arch::asm!("mov.u64 {}, %clock64;", out(reg64) timestamp);
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            timestamp = 0;
        }
        core::ptr::write_volatile(slot.add(FR_SLOT_OFF_TIMESTAMP) as *mut u64, timestamp);

        // Copy message bytes
        let msg_dst = slot.add(FR_SLOT_OFF_MSG);
        for i in 0..copy_len as usize {
            core::ptr::write_volatile(msg_dst.add(i), core::ptr::read_volatile(msg.add(i)));
        }
        // Zero-pad remaining
        for i in copy_len as usize..FR_MAX_MSG_LEN {
            core::ptr::write_volatile(msg_dst.add(i), 0);
        }
    }

    /// Set the crashed flag in the flight recorder buffer.
    ///
    /// Call this before `trap` so the host knows to dump the recorder.
    #[inline(always)]
    pub unsafe fn fr_set_crashed(fr_buf: *mut u8) {
        core::ptr::write_volatile(
            fr_buf.add(FR_OFF_FLAGS) as *mut u32,
            core::ptr::read_volatile(fr_buf.add(FR_OFF_FLAGS) as *const u32)
                | gpu_protocol::FR_FLAG_CRASHED,
        );
    }
}

/// GPU-side async task executor with dynamic spawning.
///
/// Provides a work-stealing executor where warps dequeue tasks from a lock-free
/// MPMC queue, poll type-erased futures, and recycle slots on completion. Tasks
/// can spawn new tasks dynamically via `GpuExecutor::spawn()`.
///
/// # Architecture
///
/// - **WorkQueue**: Bounded FIFO using tagged CAS (same pattern as hostcall protocol)
/// - **TaskSlot**: Fixed-size arena slot with type-erased `poll_fn` + inline future bytes
/// - **Free list**: Tagged CAS stack for slot recycling (no general allocator needed)
/// - **Scheduling**: Lane 0 dequeues, broadcasts task ID via `shfl.sync` to all 32 lanes
///
/// # Safety
///
/// The executor must reside in GPU global memory (device or mapped). All warps
/// entering `run()` must have all 32 lanes active.
pub mod executor {
    use core::cell::UnsafeCell;
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};
    use gpu_atomics::{
        activemask, lane_id, shfl_sync_idx_u32, syncwarp, sys_cas_u64, sys_load_acquire_u64,
        sys_spin_load_acquire_u32, sys_store_release_u32,
    };

    /// Maximum number of tasks the executor can hold.
    pub const MAX_TASKS: usize = 256;

    /// Maximum size of a spawned future in bytes.
    pub const TASK_FUTURE_MAX_SIZE: usize = 512;

    /// Sentinel value for empty queue/slot entries.
    pub const EMPTY_SENTINEL: u32 = 0xFFFF_FFFF;

    /// Maximum polls before a task is considered stuck.
    const MAX_POLLS_PER_TASK: u32 = 10_000_000;

    // Task slot states
    const SLOT_FREE: u32 = 0;
    const SLOT_QUEUED: u32 = 1;
    const SLOT_RUNNING: u32 = 2;

    /// Error type for executor operations.
    #[derive(Debug, Clone, Copy)]
    pub enum ExecutorError {
        /// Work queue is full.
        QueueFull,
        /// No free task slots available.
        NoFreeSlots,
        /// Future exceeds `TASK_FUTURE_MAX_SIZE` bytes.
        FutureTooLarge,
    }

    /// Handle to a spawned task.
    #[derive(Clone, Copy, Debug)]
    pub struct TaskId(pub u32);

    /// Statistics returned when a warp exits the executor loop.
    #[derive(Clone, Copy, Debug)]
    pub struct ExecutorStats {
        /// Number of tasks this warp executed to completion.
        pub tasks_executed: u32,
        /// Total number of poll calls this warp made.
        pub polls_total: u32,
    }

    // ================================================================
    // Tagged pointer helpers (same pattern as hostcall protocol)
    // ================================================================

    #[inline(always)]
    const fn tagged_value(tag: u32, index: u32) -> u64 {
        ((tag as u64) << 32) | (index as u64)
    }

    #[inline(always)]
    fn tagged_tag(v: u64) -> u32 {
        (v >> 32) as u32
    }

    #[inline(always)]
    fn tagged_index(v: u64) -> u32 {
        v as u32
    }

    // ================================================================
    // WorkQueue — bounded lock-free MPMC FIFO
    // ================================================================

    /// Bounded lock-free MPMC FIFO queue.
    ///
    /// Uses tagged CAS on head and tail to prevent ABA. The buffer is a
    /// circular array of task slot indices.
    #[repr(C)]
    pub struct WorkQueue {
        /// Consumer index (dequeue here). Tagged u64 for ABA prevention.
        head: UnsafeCell<u64>,
        /// Producer index (enqueue here). Tagged u64 for ABA prevention.
        tail: UnsafeCell<u64>,
        /// Circular buffer of task slot indices. EMPTY_SENTINEL = unoccupied.
        buffer: [UnsafeCell<u32>; MAX_TASKS],
    }

    #[allow(clippy::new_without_default)]
    impl WorkQueue {
        /// Create a new empty work queue.
        pub const fn new() -> Self {
            #[allow(clippy::declare_interior_mutable_const)]
            const EMPTY: UnsafeCell<u32> = UnsafeCell::new(EMPTY_SENTINEL);
            Self {
                head: UnsafeCell::new(tagged_value(0, 0)),
                tail: UnsafeCell::new(tagged_value(0, 0)),
                buffer: [EMPTY; MAX_TASKS],
            }
        }

        /// Enqueue a task index. Returns Err if the queue is full.
        ///
        /// # Safety
        /// Must be called from a single lane (typically lane 0).
        #[inline(always)]
        pub unsafe fn enqueue(&self, task_id: u32) -> Result<(), ExecutorError> {
            let head_ptr = self.head.get();
            let tail_ptr = self.tail.get();

            loop {
                let old_tail = sys_load_acquire_u64(tail_ptr as *const _);
                let old_head = sys_load_acquire_u64(head_ptr as *const _);

                let tail_idx = tagged_index(old_tail);
                let head_idx = tagged_index(old_head);

                // Check if full (allow wraparound comparison)
                if tail_idx.wrapping_sub(head_idx) >= MAX_TASKS as u32 {
                    return Err(ExecutorError::QueueFull);
                }

                let slot = tail_idx & (MAX_TASKS as u32 - 1);
                let new_tag = tagged_tag(old_tail).wrapping_add(1);
                let new_tail = tagged_value(new_tag, tail_idx.wrapping_add(1));

                if sys_cas_u64(tail_ptr, old_tail, new_tail) == old_tail {
                    // Write the task ID into the buffer slot
                    sys_store_release_u32(self.buffer[slot as usize].get(), task_id);
                    return Ok(());
                }
                // CAS failed — retry
            }
        }

        /// Dequeue a task index. Returns EMPTY_SENTINEL if the queue is empty.
        ///
        /// # Safety
        /// Must be called from a single lane (typically lane 0).
        #[inline(always)]
        pub unsafe fn dequeue(&self) -> u32 {
            let head_ptr = self.head.get();
            let tail_ptr = self.tail.get();

            loop {
                let old_head = sys_load_acquire_u64(head_ptr as *const _);
                let old_tail = sys_load_acquire_u64(tail_ptr as *const _);

                let head_idx = tagged_index(old_head);
                let tail_idx = tagged_index(old_tail);

                // Empty check
                if head_idx == tail_idx {
                    return EMPTY_SENTINEL;
                }

                let slot = head_idx & (MAX_TASKS as u32 - 1);

                // Read the task ID
                let task_id =
                    sys_spin_load_acquire_u32(self.buffer[slot as usize].get() as *const _);
                if task_id == EMPTY_SENTINEL {
                    // Producer hasn't written yet — retry
                    continue;
                }

                let new_tag = tagged_tag(old_head).wrapping_add(1);
                let new_head = tagged_value(new_tag, head_idx.wrapping_add(1));

                if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                    // Clear the buffer slot
                    sys_store_release_u32(self.buffer[slot as usize].get(), EMPTY_SENTINEL);
                    return task_id;
                }
                // CAS failed — retry
            }
        }
    }

    // ================================================================
    // TaskSlot — type-erased future storage
    // ================================================================

    /// Type alias for a type-erased poll function pointer.
    type PollFn = unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>;

    /// A fixed-size slot for storing a type-erased future.
    ///
    /// The future bytes are stored inline. The `poll_fn` pointer provides
    /// type-erased access to `Future::poll()`.
    #[repr(C)]
    pub struct TaskSlot {
        /// Slot state: FREE / QUEUED / RUNNING
        state: UnsafeCell<u32>,
        /// Type-erased poll function.
        poll_fn: UnsafeCell<Option<PollFn>>,
        /// Size of the stored future in bytes (for debugging).
        future_size: UnsafeCell<u32>,
        /// Inline storage for the future.
        future_bytes: UnsafeCell<[u8; TASK_FUTURE_MAX_SIZE]>,
    }

    #[allow(clippy::new_without_default)]
    impl TaskSlot {
        /// Create a new free task slot.
        pub const fn new() -> Self {
            Self {
                state: UnsafeCell::new(SLOT_FREE),
                poll_fn: UnsafeCell::new(None),
                future_size: UnsafeCell::new(0),
                future_bytes: UnsafeCell::new([0u8; TASK_FUTURE_MAX_SIZE]),
            }
        }
    }

    /// Type-erased poll trampoline. Casts the raw bytes back to `F` and polls.
    ///
    /// # Safety
    /// `ptr` must point to a valid `F` that was previously copied into the slot.
    #[inline(always)]
    unsafe fn erased_poll<F: Future<Output = ()>>(ptr: *mut u8, cx: &mut Context<'_>) -> Poll<()> {
        let future = &mut *(ptr as *mut F);
        Pin::new_unchecked(future).poll(cx)
    }

    // ================================================================
    // Free slot stack (tagged CAS, same as hostcall free packets)
    // ================================================================

    /// Tagged free-slot stack head.
    /// Bits 63-48: epoch tag, Bits 15-0: slot index (0xFFFF = empty)
    #[repr(C)]
    pub struct FreeSlotStack {
        head: UnsafeCell<u64>,
    }

    /// Encode a free-stack tagged pointer.
    #[inline(always)]
    const fn free_tagged(tag: u16, index: u16) -> u64 {
        ((tag as u64) << 48) | (index as u64)
    }

    #[inline(always)]
    fn free_tag(v: u64) -> u16 {
        (v >> 48) as u16
    }

    #[inline(always)]
    fn free_index(v: u64) -> u16 {
        v as u16
    }

    const FREE_NULL: u16 = 0xFFFF;

    impl FreeSlotStack {
        /// Create a stack pre-populated with all slot indices [0..count).
        ///
        /// The `next` links are stored in the task slots' `future_size` field
        /// (reused as a next pointer when the slot is FREE).
        pub const fn empty() -> Self {
            Self {
                head: UnsafeCell::new(free_tagged(0, FREE_NULL)),
            }
        }

        /// Initialize the stack with slots [0..count). Must be called once
        /// before any pop/push operations.
        ///
        /// # Safety
        /// `slots` must point to a valid TaskSlot array of at least `count` elements.
        pub unsafe fn init(&self, slots: *mut TaskSlot, count: usize) {
            // Build a linked list: slot[0] -> slot[1] -> ... -> slot[count-1] -> NULL
            // We store the "next" pointer in the slot's `future_size` field (reused).
            for i in 0..count {
                let slot = &*slots.add(i);
                let next = if i + 1 < count {
                    (i + 1) as u32
                } else {
                    FREE_NULL as u32
                };
                core::ptr::write_volatile(slot.future_size.get(), next);
                core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
            }
            // Head points to slot 0
            core::ptr::write_volatile(self.head.get(), free_tagged(0, 0));
        }

        /// Pop a free slot index. Returns FREE_NULL if none available.
        ///
        /// # Safety
        /// `slots` must point to the TaskSlot array.
        #[inline(always)]
        pub unsafe fn pop(&self, slots: *const TaskSlot) -> u16 {
            let head_ptr = self.head.get();
            loop {
                let old_head = sys_load_acquire_u64(head_ptr as *const _);
                let idx = free_index(old_head);
                if idx == FREE_NULL {
                    return FREE_NULL;
                }
                // Read next pointer from the slot's future_size field
                let slot = &*slots.add(idx as usize);
                let next_idx = core::ptr::read_volatile(slot.future_size.get()) as u16;
                let new_tag = free_tag(old_head).wrapping_add(1);
                let new_head = free_tagged(new_tag, next_idx);
                if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                    return idx;
                }
            }
        }

        /// Push a slot back onto the free stack.
        ///
        /// # Safety
        /// `slots` must point to the TaskSlot array. The slot must not be in use.
        #[inline(always)]
        pub unsafe fn push(&self, slots: *mut TaskSlot, slot_idx: u16) {
            let head_ptr = self.head.get();
            let slot = &*slots.add(slot_idx as usize);
            loop {
                let old_head = sys_load_acquire_u64(head_ptr as *const _);
                // Store current head as our "next"
                core::ptr::write_volatile(slot.future_size.get(), free_index(old_head) as u32);
                let new_tag = free_tag(old_head).wrapping_add(1);
                let new_head = free_tagged(new_tag, slot_idx);
                if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                    return;
                }
            }
        }
    }

    // ================================================================
    // GpuExecutor — the main executor struct
    // ================================================================

    /// GPU-side async task executor with work-stealing.
    ///
    /// Allocated in global memory. Host initializes, kernel warps call `run()`.
    #[repr(C)]
    pub struct GpuExecutor {
        /// Lock-free MPMC work queue.
        pub work_queue: WorkQueue,
        /// Free slot recycling stack.
        free_slots: FreeSlotStack,
        /// Number of active tasks (for shutdown detection).
        tasks_active: UnsafeCell<u32>,
        /// Shutdown flag (0 = running, 1 = shutting down).
        shutdown: UnsafeCell<u32>,
        /// Total tasks spawned (diagnostic counter).
        tasks_spawned: UnsafeCell<u32>,
        /// Total tasks completed (diagnostic counter).
        tasks_completed: UnsafeCell<u32>,
        /// Task slot arena.
        slots: [TaskSlot; MAX_TASKS],
    }

    // SAFETY: GpuExecutor is designed for concurrent access across warps/blocks.
    // All mutable state is protected by atomic CAS operations.
    unsafe impl Send for GpuExecutor {}
    unsafe impl Sync for GpuExecutor {}
    unsafe impl Send for WorkQueue {}
    unsafe impl Sync for WorkQueue {}
    unsafe impl Send for TaskSlot {}
    unsafe impl Sync for TaskSlot {}
    unsafe impl Send for FreeSlotStack {}
    unsafe impl Sync for FreeSlotStack {}

    /// No-op waker vtable for GPU (no real wake mechanism).
    const NOOP_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
        |_| core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    #[inline(always)]
    fn noop_waker() -> core::task::Waker {
        unsafe {
            core::task::Waker::from_raw(core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE))
        }
    }

    #[allow(clippy::new_without_default)]
    impl GpuExecutor {
        /// Create a new executor with all slots free.
        ///
        /// After construction, call `init()` to set up the free slot linked list.
        pub const fn new() -> Self {
            #[allow(clippy::declare_interior_mutable_const)]
            const SLOT: TaskSlot = TaskSlot::new();
            Self {
                work_queue: WorkQueue::new(),
                free_slots: FreeSlotStack::empty(),
                tasks_active: UnsafeCell::new(0),
                shutdown: UnsafeCell::new(0),
                tasks_spawned: UnsafeCell::new(0),
                tasks_completed: UnsafeCell::new(0),
                slots: [SLOT; MAX_TASKS],
            }
        }

        /// Initialize the executor. Must be called once before `spawn()` or `run()`.
        ///
        /// Sets up the free slot linked list.
        ///
        /// # Safety
        /// Must be called by exactly one thread (e.g., lane 0 of the first warp).
        pub unsafe fn init(&self) {
            self.free_slots
                .init(self.slots.as_ptr() as *mut TaskSlot, MAX_TASKS);
            core::ptr::write_volatile(self.tasks_active.get(), 0);
            core::ptr::write_volatile(self.shutdown.get(), 0);
            core::ptr::write_volatile(self.tasks_spawned.get(), 0);
            core::ptr::write_volatile(self.tasks_completed.get(), 0);
        }

        /// Spawn a new async task onto the executor.
        ///
        /// The future is copied into a task slot and enqueued for execution.
        /// Any warp currently in `run()` may pick it up.
        ///
        /// # Safety
        /// - `self` must point to valid executor memory in global/mapped space
        /// - The future must be safe to poll from any warp
        /// - Should be called from lane 0 only (single-lane operation)
        #[inline(always)]
        pub unsafe fn spawn<F: Future<Output = ()>>(
            &self,
            future: F,
        ) -> Result<TaskId, ExecutorError> {
            let size = core::mem::size_of::<F>();
            if size > TASK_FUTURE_MAX_SIZE {
                return Err(ExecutorError::FutureTooLarge);
            }

            // Allocate a free slot
            let slot_idx = self.free_slots.pop(self.slots.as_ptr());
            if slot_idx == FREE_NULL {
                return Err(ExecutorError::NoFreeSlots);
            }

            let slot = &self.slots[slot_idx as usize];

            // Copy future bytes into the slot
            core::ptr::copy_nonoverlapping(
                &future as *const F as *const u8,
                (*slot.future_bytes.get()).as_mut_ptr(),
                size,
            );
            core::mem::forget(future); // ownership transferred to slot

            // Set the type-erased poll function
            core::ptr::write(slot.poll_fn.get(), Some(erased_poll::<F> as _));
            core::ptr::write_volatile(slot.future_size.get(), size as u32);

            // Mark as queued (release store ensures all prior writes visible)
            sys_store_release_u32(slot.state.get(), SLOT_QUEUED);

            // Enqueue into work queue
            self.work_queue.enqueue(slot_idx as u32)?;

            // Increment spawn counter
            let old = core::ptr::read_volatile(self.tasks_spawned.get());
            core::ptr::write_volatile(self.tasks_spawned.get(), old.wrapping_add(1));

            Ok(TaskId(slot_idx as u32))
        }

        /// Enter the executor loop (ExitOnEmpty policy).
        ///
        /// The calling warp dequeues and executes tasks until the queue is empty
        /// and no more tasks are active. All 32 lanes of the warp must call this.
        ///
        /// # Safety
        /// - Must be called by all active lanes of a warp simultaneously
        /// - `self` must point to valid executor memory
        #[inline(always)]
        pub unsafe fn run(&self) -> ExecutorStats {
            let mask = activemask();
            let lid = lane_id();
            let mut stats = ExecutorStats {
                tasks_executed: 0,
                polls_total: 0,
            };

            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);

            loop {
                // Lane 0 dequeues, broadcasts to all lanes
                let mut task_id: u32 = EMPTY_SENTINEL;
                if lid == 0 {
                    task_id = self.work_queue.dequeue();
                }
                let task_id = shfl_sync_idx_u32(mask, task_id, 0);
                syncwarp(mask);

                if task_id == EMPTY_SENTINEL {
                    // Check shutdown or no more active tasks
                    let shutdown = core::ptr::read_volatile(self.shutdown.get() as *const u32);
                    if shutdown != 0 {
                        break;
                    }
                    // ExitOnEmpty: if nothing in queue, exit
                    break;
                }

                // Mark slot as RUNNING
                let slot = &self.slots[task_id as usize];
                if lid == 0 {
                    sys_store_release_u32(slot.state.get(), SLOT_RUNNING);
                }
                syncwarp(mask);

                // Get the poll function
                let poll_fn = core::ptr::read_volatile(slot.poll_fn.get());
                let poll_fn = match poll_fn {
                    Some(f) => f,
                    None => {
                        // Invalid slot — skip (shouldn't happen)
                        if lid == 0 {
                            self.recycle_slot(task_id);
                        }
                        syncwarp(mask);
                        continue;
                    }
                };

                let future_ptr = (*slot.future_bytes.get()).as_mut_ptr();

                // Spin-poll the task to completion
                let mut polls: u32 = 0;
                loop {
                    let result = poll_fn(future_ptr, &mut cx);
                    stats.polls_total += 1;
                    polls += 1;

                    match result {
                        Poll::Ready(()) => {
                            // Task complete — recycle (lane 0 only)
                            if lid == 0 {
                                self.recycle_slot(task_id);
                                let old = core::ptr::read_volatile(self.tasks_completed.get());
                                core::ptr::write_volatile(
                                    self.tasks_completed.get(),
                                    old.wrapping_add(1),
                                );
                            }
                            syncwarp(mask);
                            stats.tasks_executed += 1;
                            break;
                        }
                        Poll::Pending => {
                            if polls >= MAX_POLLS_PER_TASK {
                                // Task stuck — recycle and move on
                                if lid == 0 {
                                    self.recycle_slot(task_id);
                                }
                                syncwarp(mask);
                                break;
                            }
                            #[cfg(target_arch = "nvptx64")]
                            core::arch::asm!("nanosleep.u32 1000;", options(nostack));
                        }
                    }
                }
            }

            stats
        }

        /// Recycle a task slot back to the free list.
        ///
        /// # Safety
        /// Must be called by lane 0 only. The task must be complete.
        #[inline(always)]
        unsafe fn recycle_slot(&self, task_id: u32) {
            let slot = &self.slots[task_id as usize];
            core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
            core::ptr::write(slot.poll_fn.get(), None);
            self.free_slots
                .push(self.slots.as_ptr() as *mut TaskSlot, task_id as u16);
        }

        /// Signal shutdown. Warps in `run()` will exit after draining the queue.
        ///
        /// # Safety
        /// Must be called by lane 0 of exactly one warp.
        #[inline(always)]
        pub unsafe fn shutdown(&self) {
            sys_store_release_u32(self.shutdown.get(), 1);
        }

        /// Get the number of tasks spawned (diagnostic).
        pub unsafe fn spawned_count(&self) -> u32 {
            core::ptr::read_volatile(self.tasks_spawned.get() as *const u32)
        }

        /// Get the number of tasks completed (diagnostic).
        pub unsafe fn completed_count(&self) -> u32 {
            core::ptr::read_volatile(self.tasks_completed.get() as *const u32)
        }
    }
}

/// GPU synchronization primitives — Mutex, MutexGuard.
///
/// Provides cross-warp/cross-block mutual exclusion on GPU using system-scope
/// atomic CAS spin-locks. Designed for protecting shared data structures in
/// global (mapped) memory.
///
/// # Design Notes
///
/// - Uses `sys_cas_u32` for lock acquisition (system-scope CAS)
/// - Uses `sys_store_release_u32` for unlock (release semantics)
/// - Spin-loop includes `nanosleep` yield via `sys_spin_load_acquire_u32`
/// - Safe across warps and blocks (different warps have independent PCs)
/// - **Not recommended for intra-warp use** — warp-cooperative patterns
///   (lane 0 acts, `shfl.sync` broadcasts) are strictly superior
/// - No poisoning semantics (GPU panics trap the thread)
pub mod sync {
    use core::cell::UnsafeCell;
    use core::ops::{Deref, DerefMut};
    use gpu_atomics::{sys_cas_u32, sys_spin_load_acquire_u32, sys_store_release_u32};

    /// Maximum spin iterations before `lock()` panics with a timeout.
    /// Same as the hostcall protocol's GPU_MAX_SPIN (10M iterations).
    pub const MUTEX_MAX_SPIN: u32 = 10_000_000;

    /// Lock states.
    const UNLOCKED: u32 = 0;
    const LOCKED: u32 = 1;

    /// A mutual exclusion primitive for GPU global memory.
    ///
    /// Protects shared data with a spin-lock using system-scope atomic CAS.
    /// Works correctly across warps and blocks. The lock word and data must
    /// reside in global memory (device or mapped) — not shared or local memory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gpu_runtime::sync::Mutex;
    ///
    /// // In global memory (e.g., passed as kernel argument pointer)
    /// let mutex: &Mutex<u32> = unsafe { &*(ptr as *const Mutex<u32>) };
    ///
    /// // Lock, modify, auto-unlock via Drop
    /// {
    ///     let mut guard = unsafe { mutex.lock() };
    ///     *guard += 1;
    /// } // guard dropped here → unlock
    /// ```
    #[repr(C)]
    pub struct Mutex<T> {
        lock_word: UnsafeCell<u32>,
        data: UnsafeCell<T>,
    }

    // SAFETY: GPU threads are not OS threads. The Mutex provides the necessary
    // synchronization for cross-warp/cross-block access. Marking as Send+Sync
    // allows Mutex to be used in static/global contexts on GPU.
    unsafe impl<T: Send> Send for Mutex<T> {}
    unsafe impl<T: Send> Sync for Mutex<T> {}

    impl<T> Mutex<T> {
        /// Create a new unlocked Mutex wrapping the given value.
        ///
        /// The Mutex must reside in global memory for cross-warp/cross-block use.
        /// Typically you'll initialize a Mutex in mapped memory from the host side
        /// by zeroing the lock word and writing the initial value.
        pub const fn new(value: T) -> Self {
            Self {
                lock_word: UnsafeCell::new(UNLOCKED),
                data: UnsafeCell::new(value),
            }
        }

        /// Acquire the lock, spinning until it becomes available.
        ///
        /// Returns a `MutexGuard` that automatically releases the lock on drop.
        /// Panics (traps) if the lock is not acquired within `MUTEX_MAX_SPIN`
        /// iterations, indicating likely deadlock.
        ///
        /// # Safety
        ///
        /// - The Mutex must reside in global memory (device or mapped).
        /// - Must not be called from within the same warp that already holds
        ///   the lock (will deadlock on pre-Volta GPUs, may stall on Volta+).
        #[inline(always)]
        pub unsafe fn lock(&self) -> MutexGuard<'_, T> {
            let lock_ptr = self.lock_word.get();
            let mut spins: u32 = 0;
            loop {
                // Try to acquire: CAS(ptr, UNLOCKED, LOCKED)
                // If returns UNLOCKED, we won the lock.
                let old = sys_cas_u32(lock_ptr, UNLOCKED, LOCKED);
                if old == UNLOCKED {
                    return MutexGuard { mutex: self };
                }
                // Spin with nanosleep yield (prevents warp starvation)
                let _ = sys_spin_load_acquire_u32(lock_ptr as *const u32);
                spins += 1;
                if spins >= MUTEX_MAX_SPIN {
                    // Likely deadlock — trap
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("trap;", options(noreturn, nostack));
                    #[cfg(not(target_arch = "nvptx64"))]
                    panic!("GPU Mutex: spin timeout (likely deadlock)");
                }
            }
        }

        /// Try to acquire the lock without spinning.
        ///
        /// Returns `Some(MutexGuard)` if the lock was acquired, `None` if
        /// it's currently held by another thread.
        ///
        /// # Safety
        ///
        /// Same requirements as `lock()`.
        #[inline(always)]
        pub unsafe fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
            let lock_ptr = self.lock_word.get();
            let old = sys_cas_u32(lock_ptr, UNLOCKED, LOCKED);
            if old == UNLOCKED {
                Some(MutexGuard { mutex: self })
            } else {
                None
            }
        }

        /// Release the lock.
        ///
        /// Normally called automatically via `MutexGuard::drop()`. Only call
        /// this directly if you need to release without a guard.
        ///
        /// # Safety
        ///
        /// Caller must hold the lock.
        #[inline(always)]
        unsafe fn unlock(&self) {
            sys_store_release_u32(self.lock_word.get(), UNLOCKED);
        }
    }

    /// RAII guard for a locked Mutex. Releases the lock on drop.
    pub struct MutexGuard<'a, T> {
        mutex: &'a Mutex<T>,
    }

    impl<'a, T> Deref for MutexGuard<'a, T> {
        type Target = T;
        #[inline(always)]
        fn deref(&self) -> &T {
            // SAFETY: We hold the lock, so exclusive access is guaranteed.
            unsafe { &*self.mutex.data.get() }
        }
    }

    impl<'a, T> DerefMut for MutexGuard<'a, T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut T {
            // SAFETY: We hold the lock, so exclusive access is guaranteed.
            unsafe { &mut *self.mutex.data.get() }
        }
    }

    impl<'a, T> Drop for MutexGuard<'a, T> {
        #[inline(always)]
        fn drop(&mut self) {
            // SAFETY: Guard exists only when lock is held.
            unsafe { self.mutex.unlock() };
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
    pub use crate::hostcall::{
        gpu_hostcall_assert, gpu_hostcall_print, gpu_hostcall_release, gpu_hostcall_request,
        gpu_hostcall_trace,
    };
    pub use crate::panic::{gpu_panic_init, gpu_result_init};
    pub use crate::print_buffer;
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
        PKT_OFF_PAYLOAD, PKT_OFF_SERVICE, PRINT_MAX_MSG_LEN, SERVICE_ASSERT, SERVICE_BULK_PRINT,
        SERVICE_BULK_READ, SERVICE_BULK_WRITE, SERVICE_CLOSE, SERVICE_OPEN, SERVICE_PANIC,
        SERVICE_PRINT, SERVICE_READ, SERVICE_STDIN, SERVICE_TCP_BULK_READ, SERVICE_TCP_BULK_WRITE,
        SERVICE_TCP_CLOSE, SERVICE_TCP_CONNECT, SERVICE_TCP_READ, SERVICE_TCP_WRITE, SERVICE_TIME,
        SERVICE_TRACE, SERVICE_WRITE, TCP_MAX_ADDR_LEN, TCP_MAX_READ_LEN, TCP_MAX_WRITE_LEN,
        TRACE_LEVEL_DEBUG, TRACE_LEVEL_ERROR, TRACE_LEVEL_INFO, TRACE_LEVEL_WARN,
    };

    // --- Warp intrinsics ---
    pub use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

    // --- Command buffer polling ---
    pub use crate::cmd::{cmd_ack, cmd_poll, cmd_yield};

    // --- Command buffer constants ---
    pub use gpu_protocol::{CMD_COMPUTE, CMD_EXIT, CMD_NOP, CMD_PRINT};

    // --- Commonly needed atomics ---
    pub use gpu_atomics::{sys_load_acquire_u32, sys_store_release_u32};

    // --- Async executor + futures ---
    pub use crate::std_future::{
        block_on, GpuBulkReadFuture, GpuBulkWriteFuture, GpuCloseFuture, GpuOpenFuture,
        GpuReadFuture, GpuTcpBulkReadFuture, GpuTcpBulkWriteFuture, GpuTcpCloseFuture,
        GpuTcpConnectFuture, GpuTcpReadFuture, GpuTcpWriteFuture, GpuWriteFuture,
    };

    // --- Sync primitives ---
    pub use crate::sync::{Mutex, MutexGuard};

    // --- Executor ---
    pub use crate::executor::{ExecutorError, ExecutorStats, GpuExecutor, TaskId};
}
