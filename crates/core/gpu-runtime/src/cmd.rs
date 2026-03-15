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
