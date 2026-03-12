//! Hello GPU — GPU kernels demonstrating the async_gpu hostcall stack.
//!
//! Four kernels of increasing complexity:
//! 1. `vector_add` — pure compute (no hostcall)
//! 2. `hello_gpu` — PRINT hostcall (send a message to host stdout)
//! 3. `file_io_demo` — OPEN + WRITE + CLOSE (create a file from GPU)
//! 4. `bulk_read_demo` — OPEN + BULK_READ + CLOSE (read a file via sideband)

#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use gpu_runtime::prelude::*;

// Install the gpu-runtime panic handler (sends panic message via hostcall)
gpu_runtime::panic_handler!();

// ================================================================
// Kernel 1: Vector addition — pure compute, no hostcall
// ================================================================

/// Each thread computes `c[i] = a[i] + b[i]`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vector_add(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: u32,
) {
    let idx = core::arch::nvptx::_block_idx_x() as u32
        * core::arch::nvptx::_block_dim_x() as u32
        + core::arch::nvptx::_thread_idx_x() as u32;
    if idx < len {
        *c.add(idx as usize) = *a.add(idx as usize) + *b.add(idx as usize);
    }
}

// ================================================================
// Kernel 2: PRINT hostcall — "Hello from GPU!"
// ================================================================

/// Send a greeting to the host via the PRINT hostcall service.
/// Only thread 0 executes the hostcall.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hello_gpu(buf: *mut u8, result: *mut u32) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);

    let msg = b"Hello from GPU via gpu-runtime!";
    let ok = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

// ================================================================
// Kernel 3: File I/O — create and write a file from GPU
// ================================================================

/// Demonstrates OPEN → WRITE → CLOSE hostcall sequence.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn file_io_demo(buf: *mut u8, result: *mut u32) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);

    // Step 1: Open file for writing
    let path = b"gpu_output.txt";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        let slot0: u64 = (path.len() as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        let dst = payload.add(8);
        let mut i = 0;
        while i < path.len() {
            core::ptr::write_volatile(dst.add(i), path[i]);
            i += 1;
        }
    });

    if pkt.is_null() || !ok {
        sys_store_release_u32(result, 0);
        return;
    }

    let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

    if fd == FILE_ERROR_SENTINEL {
        sys_store_release_u32(result, 0);
        return;
    }

    // Step 2: Write data to file
    let data = b"Written by GPU kernel!\n";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
        core::ptr::write_volatile(payload.add(8) as *mut u64, data.len() as u64);
        let dst = payload.add(16);
        let mut i = 0;
        while i < data.len() {
            core::ptr::write_volatile(dst.add(i), data[i]);
            i += 1;
        }
    });

    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Step 3: Close the file
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

// ================================================================
// Kernel 4: Bulk read — read a file via sideband buffer
// ================================================================

/// Demonstrates OPEN → BULK_READ → CLOSE using the sideband buffer
/// for data larger than the 56-byte packet payload limit.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bulk_read_demo(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
    bytes_read_out: *mut u32,
) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);
    sideband_reset(sideband);

    // Step 1: Open the file written by file_io_demo
    let path = b"gpu_output.txt";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        let slot0: u64 = (path.len() as u64) | ((FILE_OPEN_READ as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        let dst = payload.add(8);
        let mut i = 0;
        while i < path.len() {
            core::ptr::write_volatile(dst.add(i), path[i]);
            i += 1;
        }
    });

    if pkt.is_null() || !ok {
        sys_store_release_u32(result, 0);
        return;
    }

    let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

    if fd == FILE_ERROR_SENTINEL {
        sys_store_release_u32(result, 0);
        return;
    }

    // Step 2: Bulk read up to 256 bytes via sideband
    let mut read_buf = [0u8; 256];
    let n = gpu_bulk_read(buf, sideband, fd, read_buf.as_mut_ptr(), 256);

    // Step 3: Close the file
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Step 4: Print what we read back (truncate to 56 bytes for PRINT)
    if n > 0 {
        let print_len = if n > 56 { 56 } else { n };
        gpu_hostcall_print(buf, read_buf.as_ptr(), print_len as u32);
    }

    sys_store_release_u32(bytes_read_out, n as u32);
    sys_store_release_u32(result, if n > 0 { 1 } else { 0 });
}
