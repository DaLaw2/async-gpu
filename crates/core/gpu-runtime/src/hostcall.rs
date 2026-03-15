use gpu_atomics::{
    activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u64, sys_spin_load_acquire_u32,
    sys_store_release_u32,
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
    let pkts_per_shard = core::ptr::read_volatile(buf.add(BUF_OFF_PKTS_PER_SHARD) as *const u32);
    let shard_array_off = core::ptr::read_volatile(buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *const u32);
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
pub unsafe fn get_free_stack_ptr(buf: *mut u8, num_shards: u32, shard_array_off: u32) -> *mut u64 {
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
pub unsafe fn get_ready_stack_ptr(buf: *mut u8, num_shards: u32, shard_array_off: u32) -> *mut u64 {
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
