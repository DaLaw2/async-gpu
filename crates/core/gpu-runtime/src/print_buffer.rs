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
