// Shared helper functions used by multiple kernel modules.
//
// All functions here are `pub(crate)` — they are internal helpers, not kernel entry points.

use gpu_atomics::{sys_cas_u64, sys_load_acquire_u64};
use gpu_protocol::*;

// ============================================================
// File I/O hostcall helpers (gpu-std.3)
// ============================================================

/// GPU-side hostcall: open a file.
/// Returns `(fd, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_open(
    buf: *mut u8,
    path: *const u8,
    path_len: u32,
    flags: u32,
) -> (u64, u16) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        // Slot 0: low 32 bits = path_len, high 32 bits = flags
        let slot0_val = (path_len as u64) | ((flags as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0_val);

        // Slots 1-7: path bytes
        let copy_len = if path_len > FILE_MAX_PATH_LEN as u32 {
            FILE_MAX_PATH_LEN as u32
        } else {
            path_len
        };
        let dst = payload.add(8);
        let mut i: u32 = 0;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *path.add(i as usize));
            i += 1;
        }
    }) {
        Ok(p) => p,
        Err(e) => return (0, e.category),
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

/// GPU-side hostcall: write data to a file.
/// Returns `(bytes_written, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_write(
    buf: *mut u8,
    fd: u64,
    data: *const u8,
    data_len: u32,
) -> (u64, u16) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
        // Slot 1: data length
        core::ptr::write_volatile(payload.add(8) as *mut u64, data_len as u64);
        // Slots 2-7: data bytes (up to 48 bytes)
        let copy_len = if data_len > FILE_MAX_WRITE_LEN as u32 {
            FILE_MAX_WRITE_LEN as u32
        } else {
            data_len
        };
        let dst = payload.add(16); // skip slots 0 and 1
        let mut i: u32 = 0;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *data.add(i as usize));
            i += 1;
        }
    }) {
        Ok(p) => p,
        Err(e) => return (0, e.category),
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

/// GPU-side hostcall: close a file.
/// Returns `(0, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_close(buf: *mut u8, fd: u64) -> (u64, u16) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
    }) {
        Ok(p) => p,
        Err(e) => return (0, e.category),
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

/// GPU-side hostcall: read data from a file.
/// Returns `(bytes_read, 0)` on success (data copied to out_buf), `(0, error_category)` on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_read(
    buf: *mut u8,
    fd: u64,
    out_buf: *mut u8,
    max_len: u32,
) -> (u64, u16) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_READ, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
        // Slot 1: max bytes to read
        core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
    }) {
        Ok(p) => p,
        Err(e) => return (0, e.category),
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);

    // Success — copy data from slots 1-7
    let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
    let copy_len = if slot0 > max_len as u64 {
        max_len
    } else {
        slot0 as u32
    };
    let mut i: u32 = 0;
    while i < copy_len {
        *out_buf.add(i as usize) = core::ptr::read_volatile(src.add(i as usize));
        i += 1;
    }
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

// ============================================================
// GPU Instant + stdin + time hostcall helpers (gpu-std.4)
// ============================================================

/// Read the GPU's %globaltimer register (64-bit nanosecond counter).
/// Available on SM 3.0+ (all modern GPUs).
/// Returns a monotonic nanosecond timestamp.
#[inline(always)]
pub(crate) unsafe fn gpu_instant_nanos() -> u64 {
    let result: u64;
    core::arch::asm!(
        "mov.u64 {result}, %globaltimer;",
        result = out(reg64) result,
    );
    result
}

/// GPU-side hostcall: read a line from stdin.
/// Returns `(bytes_read, 0)` on success (data copied to out_buf), `(0, error_category)` on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_stdin_read(
    buf: *mut u8,
    out_buf: *mut u8,
    max_len: u32,
) -> (u64, u16) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_STDIN, |payload| {
        // Slot 0: max bytes to read
        core::ptr::write_volatile(payload as *mut u64, max_len as u64);
    }) {
        Ok(p) => p,
        Err(e) => return (0, e.category),
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);

    // Success — copy data from slots 1-7
    let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
    let copy_len = if slot0 > max_len as u64 {
        max_len
    } else {
        slot0 as u32
    };
    let mut i: u32 = 0;
    while i < copy_len {
        *out_buf.add(i as usize) = core::ptr::read_volatile(src.add(i as usize));
        i += 1;
    }
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

/// GPU-side hostcall: get wall-clock time from host.
/// Returns (seconds_since_epoch, nanoseconds) on success, (0, 0) on failure.
#[inline(always)]
pub(crate) unsafe fn gpu_hostcall_time(buf: *mut u8) -> (u64, u64) {
    let pkt = match gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_TIME, |_payload| {
        // No request payload needed
    }) {
        Ok(p) => p,
        Err(_) => return (0, 0),
    };

    let secs = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    let nanos = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD).add(8) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (secs, nanos)
}

// ============================================================
// Instrumented hc_pop_free variants (benchmark.2, per-block-sharding.3)
// ============================================================

/// Instrumented hc_pop_free: returns (packet_index, cas_retry_count).
#[inline(always)]
pub(crate) unsafe fn hc_pop_free_counted(buf: *mut u8) -> (u16, u32) {
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    let mut retries: u32 = 0;
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return (NULL_INDEX, retries);
        }
        let pkt = buf.add(packet_offset(idx));
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if sys_cas_u64(free_ptr, old_head, next) == old_head {
            return (idx, retries);
        }
        retries += 1;
    }
}

/// Instrumented shard-aware hc_pop_free: returns (packet_index, cas_retry_count).
/// Uses shard-local free stack when num_shards > 0, global free stack otherwise.
#[inline(always)]
pub(crate) unsafe fn hc_pop_free_counted_v2(
    buf: *mut u8,
    free_ptr: *mut u64,
    num_shards: u32,
    shard_array_off: u32,
) -> (u16, u32) {
    let mut retries: u32 = 0;
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return (NULL_INDEX, retries);
        }
        let pkt_off = if num_shards == 0 {
            packet_offset(idx)
        } else {
            packet_offset_sharded(idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if sys_cas_u64(free_ptr, old_head, next) == old_head {
            return (idx, retries);
        }
        retries += 1;
    }
}

// ============================================================
// GPU compute helpers
// ============================================================

/// Get the base address of dynamic shared memory.
///
/// Returns a generic-address-space pointer to the dynamically-allocated
/// shared memory region. The host must set `shared_mem_bytes > 0` in
/// the launch config.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(crate) unsafe fn get_dynamic_smem_ptr() -> *mut u8 {
    let ptr: u64;
    core::arch::asm!(
        "cvta.shared.u64 {out}, dynamic_smem;",
        out = out(reg64) ptr,
    );
    ptr as *mut u8
}

/// Block-level barrier synchronization.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(crate) unsafe fn bar_sync() {
    core::arch::asm!("bar.sync 0;");
}

/// Fast f32 exponential using PTX ex2.approx + multiplication.
/// exp(x) = 2^(x * log2(e)) where log2(e) ≈ 1.4426950408889634
#[inline(always)]
pub(crate) unsafe fn gpu_exp_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        let log2_e: f32 = 1.442695;
        let t = x * log2_e;
        core::arch::asm!(
            "ex2.approx.f32 {out}, {inp};",
            out = out(reg32) result,
            inp = in(reg32) t,
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate square root via inline PTX. Uses `sqrt.approx.f32` (1 ULP precision).
#[inline(always)]
pub(crate) unsafe fn gpu_sqrtf(x: f32) -> f32 {
    let result: f32;
    core::arch::asm!("sqrt.approx.f32 {out}, {inp};", out = out(reg32) result, inp = in(reg32) x);
    result
}

/// Search a byte buffer for lines containing a pattern.
#[inline(always)]
pub(crate) unsafe fn grep_buffer(
    buf: *mut u8,
    data: *const u8,
    data_len: usize,
    pattern: &[u8],
    thread_id: u32,
) -> u32 {
    let mut matches: u32 = 0;
    let mut line_start: usize = 0;

    let mut i: usize = 0;
    while i <= data_len {
        let is_eol = i == data_len || *data.add(i) == b'\n';
        if is_eol {
            let line_len = i - line_start;
            if line_len >= pattern.len() && line_len > 0 {
                let mut found = false;
                let mut j: usize = 0;
                while j + pattern.len() <= line_len {
                    let mut match_ok = true;
                    let mut k: usize = 0;
                    while k < pattern.len() {
                        if *data.add(line_start + j + k) != pattern[k] {
                            match_ok = false;
                            break;
                        }
                        k += 1;
                    }
                    if match_ok {
                        found = true;
                        break;
                    }
                    j += 1;
                }

                if found {
                    let mut msg = [0u8; 56];
                    let mut pos: usize = 0;
                    msg[pos] = b'T';
                    pos += 1;
                    if thread_id >= 10 {
                        msg[pos] = b'0' + (thread_id / 10) as u8;
                        pos += 1;
                    }
                    msg[pos] = b'0' + (thread_id % 10) as u8;
                    pos += 1;
                    msg[pos] = b':';
                    pos += 1;
                    msg[pos] = b' ';
                    pos += 1;
                    let copy_len = line_len.min(56 - pos);
                    let mut c: usize = 0;
                    while c < copy_len {
                        msg[pos] = *data.add(line_start + c);
                        pos += 1;
                        c += 1;
                    }
                    let _ = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), pos as u32);
                    matches += 1;
                }
            }
            line_start = i + 1;
        }
        i += 1;
    }
    matches
}

