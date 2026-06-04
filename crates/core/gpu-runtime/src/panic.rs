use gpu_atomics::{
    activemask, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u64, sys_spin_load_acquire_u32,
    sys_store_release_u32,
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

/// Set this warp's status to `STATUS_TRAPPED` before calling `trap;`.
///
/// This allows `BlockScope::join_all()` to detect the dead warp instead of
/// spinning forever. Only lane 0 of the current warp performs the store.
#[inline(always)]
pub unsafe fn set_warp_trapped() {
    let lid = crate::index::thread_idx_x() % 32;
    if lid == 0 {
        let wid = (crate::index::thread_idx_x() / 32) as usize;
        use core::sync::atomic::Ordering;
        crate::thread::WARP_STATUS[wid].store(crate::thread::STATUS_TRAPPED, Ordering::Release);
    }
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
        shard_array_off = core::ptr::read_volatile(buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *const u32);
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
