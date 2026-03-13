//! Async I/O — GPU kernels demonstrating multi-step file I/O pipelines.
//!
//! Two kernels:
//! 1. `write_pipeline` — write multiple files from GPU in sequence
//! 2. `transform_pipeline` — read file → transform data → write result

#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use gpu_runtime::prelude::*;

gpu_runtime::panic_handler!();

// ================================================================
// Kernel 1: Write pipeline — write 3 numbered files from GPU
// ================================================================

/// Demonstrates sequential multi-file I/O from a single GPU kernel.
/// Creates files "gpu_file_0.txt", "gpu_file_1.txt", "gpu_file_2.txt".
#[no_mangle]
pub unsafe extern "ptx-kernel" fn write_pipeline(buf: *mut u8, result: *mut u32) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);

    let files: [&[u8]; 3] = [
        b"gpu_file_0.txt",
        b"gpu_file_1.txt",
        b"gpu_file_2.txt",
    ];
    let contents: [&[u8]; 3] = [
        b"GPU wrote file 0\n",
        b"GPU wrote file 1\n",
        b"GPU wrote file 2\n",
    ];

    let mut ok_count = 0u32;
    let mut i = 0;
    while i < 3 {
        let path = files[i];
        let data = contents[i];

        // Open for writing
        let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
            let slot0: u64 = (path.len() as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
            core::ptr::write_volatile(payload as *mut u64, slot0);
            let dst = payload.add(8);
            let mut j = 0;
            while j < path.len() {
                core::ptr::write_volatile(dst.add(j), path[j]);
                j += 1;
            }
        });

        if pkt.is_null() || !ok {
            i += 1;
            continue;
        }
        let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        gpu_hostcall_release(buf, pkt);

        if fd == FILE_ERROR_SENTINEL {
            i += 1;
            continue;
        }

        // Write data
        let (pkt, wrote_ok) = gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
            core::ptr::write_volatile(payload as *mut u64, fd);
            core::ptr::write_volatile(payload.add(8) as *mut u64, data.len() as u64);
            let dst = payload.add(16);
            let mut j = 0;
            while j < data.len() {
                core::ptr::write_volatile(dst.add(j), data[j]);
                j += 1;
            }
        });
        if !pkt.is_null() {
            gpu_hostcall_release(buf, pkt);
        }

        // Close
        let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
            core::ptr::write_volatile(payload as *mut u64, fd);
        });
        if !pkt.is_null() {
            gpu_hostcall_release(buf, pkt);
        }

        if wrote_ok {
            ok_count += 1;
        }
        i += 1;
    }

    // Report: print summary
    let msg = b"Write pipeline done";
    gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
    sys_store_release_u32(result, ok_count);
}

// ================================================================
// Kernel 2: Transform pipeline — read file, transform, write result
// ================================================================

/// Demonstrates read → transform → write pipeline.
/// Reads "gpu_file_0.txt", converts to uppercase, writes "gpu_upper.txt".
#[no_mangle]
pub unsafe extern "ptx-kernel" fn transform_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);
    sideband_reset(sideband);

    // Step 1: Read source file via sideband
    let path = b"gpu_file_0.txt";
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

    let mut read_buf = [0u8; 256];
    let n = gpu_bulk_read(buf, sideband, fd, read_buf.as_mut_ptr(), 256);

    // Close source
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    if n == 0 {
        sys_store_release_u32(result, 0);
        return;
    }

    // Step 2: Transform to uppercase on GPU
    let mut upper_buf = [0u8; 256];
    let mut j = 0;
    while j < n {
        let ch = read_buf[j];
        upper_buf[j] = if ch >= b'a' && ch <= b'z' {
            ch - 32
        } else {
            ch
        };
        j += 1;
    }

    // Step 3: Write transformed data
    let out_path = b"gpu_upper.txt";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        let slot0: u64 = (out_path.len() as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        let dst = payload.add(8);
        let mut i = 0;
        while i < out_path.len() {
            core::ptr::write_volatile(dst.add(i), out_path[i]);
            i += 1;
        }
    });

    if pkt.is_null() || !ok {
        sys_store_release_u32(result, 0);
        return;
    }
    let out_fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);
    if out_fd == FILE_ERROR_SENTINEL {
        sys_store_release_u32(result, 0);
        return;
    }

    let write_len = if n > 48 { 48 } else { n };
    let (pkt, wrote_ok) = gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, out_fd);
        core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
        let dst = payload.add(16);
        let mut i = 0;
        while i < write_len {
            core::ptr::write_volatile(dst.add(i), upper_buf[i]);
            i += 1;
        }
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Close output
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, out_fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Print confirmation
    let msg = b"Transform pipeline done";
    gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
    sys_store_release_u32(result, if wrote_ok { 1 } else { 0 });
}
