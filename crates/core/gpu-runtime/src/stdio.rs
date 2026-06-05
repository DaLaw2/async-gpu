//! GPU stdio infrastructure — hostcall-backed stdout/stdin for patched std.
//!
//! This module provides the `#[no_mangle]` functions that the patched Rust std
//! PAL (platform abstraction layer) calls for stdout writes and stdin reads on
//! `nvptx64-nvidia-cuda`. It also manages the hostcall buffer pointer and
//! print-buffer sideband state.
//!
//! # Architecture
//!
//! The patched std's `sys::stdio::cuda` module calls:
//! - `gpu_stdout_write(ptr, len) -> usize` for `Stdout::write()`
//! - `gpu_stdin_read(ptr, len) -> usize` for `Stdin::read()`
//!
//! These are resolved at link time via `#[no_mangle]` + LTO. The kernel crate
//! must force-link them (e.g. `#[used]` array) since they have no direct Rust
//! callers — only the PAL's `extern "C"` block references them.
//!
//! # Initialization
//!
//! Call `stdio_init(buf)` at kernel entry to set the hostcall buffer pointer.
//! For buffered printing, call `stdio_print_buffer_init(buf, sideband, n)`
//! before any `println!()` and `gpu_print_buffer_flush()` before kernel exit.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};

/// Global hostcall buffer pointer for stdio. Set by kernel at entry.
static STDIO_HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);

/// Global sideband buffer pointer for buffered printing. Set by `stdio_print_buffer_init`.
static STDIO_SIDEBAND_PTR: AtomicU64 = AtomicU64::new(0);

/// Flag: 1 if print buffer is initialized and ready for use.
static STDIO_PRINT_BUF_READY: AtomicU32 = AtomicU32::new(0);

/// Set the hostcall buffer pointer for stdio. Must be called at kernel entry.
pub fn stdio_init(buf: *mut u8) {
    STDIO_HOSTCALL_BUF.store(buf as u64, AtomicOrdering::Relaxed);
}

/// External function called by std's CUDA PAL Stdout::write().
/// Routes through gpu-runtime's print_buffer (if initialized) or direct hostcall.
///
/// When print buffer is active, messages are accumulated locally and flushed
/// via a single SERVICE_BULK_PRINT hostcall, reducing overhead from O(N) to O(1)
/// per flush.
///
/// # Safety
///
/// `buf` must point to `len` valid bytes (or be null, in which case the call is a no-op).
#[unsafe(no_mangle)]
pub unsafe fn gpu_stdout_write(buf: *const u8, len: usize) -> usize {
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || buf.is_null() || len == 0 {
        return len; // silently discard if no hostcall buffer set
    }

    // Fast path: use print_buffer if initialized (auto-flush when full)
    if STDIO_PRINT_BUF_READY.load(AtomicOrdering::Relaxed) != 0 {
        let sideband = STDIO_SIDEBAND_PTR.load(AtomicOrdering::Relaxed) as *mut u8;
        if !sideband.is_null() {
            let result = crate::print_buffer::print(hc_buf, sideband, buf, len as u32);
            if result.is_ok() {
                return len;
            }
            // Fall through to direct hostcall on error
        }
    }

    // Slow path: direct hostcall (56-byte chunks, one hostcall per chunk)
    const MAX_CHUNK: usize = 56;
    let mut offset = 0usize;
    while offset < len {
        let chunk_len = core::cmp::min(len - offset, MAX_CHUNK);
        let result = crate::hostcall::gpu_hostcall_print(hc_buf, buf.add(offset), chunk_len as u32);
        if result.is_err() {
            return offset; // partial write on failure
        }
        offset += chunk_len;
    }
    len
}

/// External function called by std's CUDA PAL Stdin::read().
/// Routes through gpu-runtime's hostcall SERVICE_STDIN implementation.
///
/// # Safety
///
/// `out_buf` must point to at least `max_len` writable bytes (or be null, no-op).
#[unsafe(no_mangle)]
pub unsafe fn gpu_stdin_read(out_buf: *mut u8, max_len: usize) -> usize {
    use crate::prelude::{PKT_OFF_PAYLOAD, SERVICE_STDIN};

    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || out_buf.is_null() || max_len == 0 {
        return 0;
    }
    // SERVICE_STDIN payload slots 1-7 = 56 bytes max
    const STDIN_MAX: usize = 56;
    let request_len = core::cmp::min(max_len, STDIN_MAX) as u32;

    // Stdin is blocking on host — use extended timeout (100M spins vs default 10M)
    const STDIN_MAX_SPIN: u32 = 100_000_000;
    let pkt = match crate::hostcall::gpu_hostcall_request_with_timeout(
        hc_buf,
        SERVICE_STDIN,
        STDIN_MAX_SPIN,
        |payload| {
            // Slot 0: max bytes to read
            core::ptr::write_volatile(payload as *mut u64, request_len as u64);
        },
    ) {
        Ok(p) => p,
        Err(_) => return 0,
    };

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    let src = pkt.add(PKT_OFF_PAYLOAD).add(8); // slots 1-7
    let copy_len = core::cmp::min(slot0, request_len as u64) as usize;
    let mut i = 0usize;
    while i < copy_len {
        *out_buf.add(i) = core::ptr::read_volatile(src.add(i));
        i += 1;
    }
    crate::hostcall::gpu_hostcall_release(hc_buf, pkt);
    copy_len
}

/// Initialize buffered printing for `println!()`.
///
/// After this call, `gpu_stdout_write()` routes through `print_buffer` instead
/// of issuing one hostcall per chunk. The caller MUST call
/// `gpu_print_buffer_flush()` before kernel exit.
///
/// # Safety
///
/// `buf` and `sideband` must be valid mapped device pointers.
#[unsafe(no_mangle)]
pub unsafe fn stdio_print_buffer_init(buf: *mut u8, sideband: *mut u8, thread_count: u32) {
    STDIO_HOSTCALL_BUF.store(buf as u64, AtomicOrdering::Relaxed);
    STDIO_SIDEBAND_PTR.store(sideband as u64, AtomicOrdering::Relaxed);
    crate::print_buffer::init(sideband, thread_count);
    STDIO_PRINT_BUF_READY.store(1, AtomicOrdering::Release);
}

/// Flush the print buffer for the calling thread and send all buffered
/// messages to the host via a single SERVICE_BULK_PRINT hostcall.
///
/// Must be called before kernel exit when buffered printing is active.
/// Safe to call even if the buffer was never initialized (no-op).
#[unsafe(no_mangle)]
pub fn gpu_print_buffer_flush() {
    if STDIO_PRINT_BUF_READY.load(AtomicOrdering::Relaxed) == 0 {
        return;
    }
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    let sideband = STDIO_SIDEBAND_PTR.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || sideband.is_null() {
        return;
    }
    unsafe {
        let _ = crate::print_buffer::flush(hc_buf, sideband);
    }
}
