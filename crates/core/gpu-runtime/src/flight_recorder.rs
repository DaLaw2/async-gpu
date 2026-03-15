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
