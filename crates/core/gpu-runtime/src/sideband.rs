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
