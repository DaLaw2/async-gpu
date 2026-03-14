//! Hostcall-based I/O for GPU libc (open, read, write, close).
//!
//! Replaces the ENOSYS stubs with real implementations that route
//! through the gpu-runtime hostcall protocol. Requires `gpu_libc_io_init(buf)`
//! to be called at kernel entry before any I/O operations.

use crate::errno::*;
use crate::types::*;
use gpu_protocol::*;

/// Global hostcall buffer pointer for libc I/O.
static mut HC_BUF: *mut u8 = core::ptr::null_mut();

/// Initialize the hostcall I/O subsystem.
/// Must be called at kernel entry before any file I/O operations.
#[inline(always)]
pub unsafe fn gpu_libc_io_init(buf: *mut u8) {
    // SAFETY: HC_BUF is a static mut written once at kernel entry before any I/O.
    // After init, all threads only read this pointer. The single-writer-then-
    // read-only pattern prevents data races.
    HC_BUF = buf;
}

/// Map gpu-protocol error category to libc errno.
#[inline(always)]
fn error_to_errno(category: u16, raw_errno: u16) -> c_int {
    // If host provided a raw errno, use it directly
    if raw_errno != 0 {
        return raw_errno as c_int;
    }
    // Otherwise map from category
    match category {
        ERR_NOT_FOUND => ENOENT,
        ERR_PERMISSION_DENIED => EACCES,
        ERR_ALREADY_EXISTS => EEXIST,
        ERR_INVALID_INPUT => EINVAL,
        ERR_IO_ERROR => EIO,
        ERR_TIMED_OUT => EIO,
        ERR_RESOURCE_BUSY => ENOSYS,
        ERR_OUT_OF_MEMORY => ENOMEM,
        ERR_INVALID_FD => EBADF,
        ERR_HOST_TIMEOUT => EIO,
        ERR_UNSUPPORTED => ENOSYS,
        _ => EIO,
    }
}

/// Map libc open flags to gpu-protocol FILE_OPEN_* constants.
#[inline(always)]
fn map_open_flags(flags: c_int) -> u32 {
    if flags & O_APPEND != 0 {
        FILE_OPEN_APPEND
    } else if flags & (O_WRONLY | O_RDWR) != 0 {
        FILE_OPEN_WRITE_CREATE
    } else {
        FILE_OPEN_READ
    }
}

/// Open a file via hostcall. Returns fd on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn open(pathname: *const c_char, flags: c_int, _mode: mode_t) -> c_int {
    let buf = HC_BUF;
    if buf.is_null() {
        set_errno(ENOSYS);
        return -1;
    }

    // Compute path length
    let mut path_len: u32 = 0;
    let mut p = pathname;
    while !p.is_null() && *p != 0 {
        path_len += 1;
        p = p.add(1);
        if path_len >= FILE_MAX_PATH_LEN as u32 {
            break;
        }
    }

    let proto_flags = map_open_flags(flags);

    // Use gpu_hostcall_request for SERVICE_OPEN
    let result = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        // Slot 0: low 32 bits = path_len, high 32 bits = flags
        let slot0 = (path_len as u64) | ((proto_flags as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        // Slots 1-7: path bytes
        let dst = payload.add(8);
        let mut i: u32 = 0;
        while i < path_len {
            core::ptr::write_volatile(
                dst.add(i as usize),
                *(pathname as *const u8).add(i as usize),
            );
            i += 1;
        }
    });

    match result {
        Ok(pkt) => {
            let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            fd as c_int
        }
        Err(e) => {
            set_errno(error_to_errno(e.category, e.raw_errno));
            -1
        }
    }
}

/// Write to a file descriptor via hostcall.
#[no_mangle]
pub unsafe extern "C" fn write(fd: c_int, data: *const c_void, count: size_t) -> ssize_t {
    let buf = HC_BUF;
    if buf.is_null() {
        set_errno(ENOSYS);
        return -1;
    }
    if count == 0 {
        return 0;
    }

    let src = data as *const u8;
    let write_len = if count > FILE_MAX_WRITE_LEN {
        FILE_MAX_WRITE_LEN
    } else {
        count
    };

    let result = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd as u64);
        core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
        let dst = payload.add(16);
        let mut i = 0;
        while i < write_len {
            core::ptr::write_volatile(dst.add(i), *src.add(i));
            i += 1;
        }
    });

    match result {
        Ok(pkt) => {
            let written = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            written as ssize_t
        }
        Err(e) => {
            set_errno(error_to_errno(e.category, e.raw_errno));
            -1
        }
    }
}

/// Read from a file descriptor via hostcall.
#[no_mangle]
pub unsafe extern "C" fn read(fd: c_int, out_buf: *mut c_void, count: size_t) -> ssize_t {
    let buf = HC_BUF;
    if buf.is_null() {
        set_errno(ENOSYS);
        return -1;
    }
    if count == 0 {
        return 0;
    }

    let max_len = if count > FILE_MAX_READ_LEN {
        FILE_MAX_READ_LEN
    } else {
        count
    };

    let result = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_READ, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd as u64);
        core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
    });

    match result {
        Ok(pkt) => {
            let bytes_read = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
            // Copy data from payload slots 1-7
            let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
            let dst = out_buf as *mut u8;
            let n = if bytes_read as usize > max_len {
                max_len
            } else {
                bytes_read as usize
            };
            let mut i = 0;
            while i < n {
                core::ptr::write_volatile(dst.add(i), core::ptr::read_volatile(src.add(i)));
                i += 1;
            }
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            bytes_read as ssize_t
        }
        Err(e) => {
            set_errno(error_to_errno(e.category, e.raw_errno));
            -1
        }
    }
}

/// Close a file descriptor via hostcall.
#[no_mangle]
pub unsafe extern "C" fn close(fd: c_int) -> c_int {
    let buf = HC_BUF;
    if buf.is_null() {
        set_errno(ENOSYS);
        return -1;
    }

    let result = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd as u64);
    });

    match result {
        Ok(pkt) => {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            0
        }
        Err(e) => {
            set_errno(error_to_errno(e.category, e.raw_errno));
            -1
        }
    }
}
