//! Parallel Search — bulk I/O (thread 0) + parallel pattern matching (all 32 lanes).
//!
//! Demonstrates genuine GPU parallelism:
//! 1. Thread 0 reads file via sideband bulk I/O (async, warp-cooperative)
//! 2. All 32 lanes search different 1/32 chunks for a pattern
//! 3. Lane 0 gathers per-lane match counts via `shfl.sync`
//! 4. Thread 0 writes result via hostcall
//!
//! This is the first example that uses ALL 32 lanes for compute.

#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use gpu_runtime::prelude::*;

gpu_runtime::panic_handler!();

/// Maximum file size to search (bytes).
const MAX_DATA: usize = 4096;

// ---------------------------------------------------------------------------
// GPU Kernel: parallel search with full warp
// ---------------------------------------------------------------------------

/// Entry point: thread 0 does I/O, ALL 32 threads do parallel search.
///
/// The I/O phase runs only on thread 0 (warp-cooperative I/O is future work).
/// The search phase uses all 32 lanes — each lane searches 1/32 of the data.
/// Results are gathered via `shfl.sync.idx` and summed by lane 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn parallel_search(
    buf: *mut u8,
    sideband: *mut u8,
    pattern: *const u8,
    pattern_len: u32,
    data_buf: *mut u8,
    output: *mut u32,
) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);

    // ========================================
    // Phase 1: I/O (thread 0 only)
    // ========================================
    let mut file_size: u32 = 0;

    if tid == 0 {
        gpu_panic_init(buf);
        sideband_reset(sideband);

        // Open input file
        let pkt = match gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
            let path = b"search_input.txt";
            let slot0: u64 =
                (path.len() as u64) | ((FILE_OPEN_READ as u64) << 32);
            core::ptr::write_volatile(payload as *mut u64, slot0);
            let dst = payload.add(8);
            let mut i = 0;
            while i < path.len() {
                core::ptr::write_volatile(dst.add(i), path[i]);
                i += 1;
            }
        }) {
            Ok(p) => p,
            Err(_) => {
                *output = 0xE001;
                return;
            }
        };

        let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        gpu_hostcall_release(buf, pkt);
        if fd == FILE_ERROR_SENTINEL {
            *output = 0xE001;
            return;
        }

        // Bulk read
        let n = gpu_bulk_read(buf, sideband, fd, data_buf, MAX_DATA);

        // Close
        if let Ok(p) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
            core::ptr::write_volatile(payload as *mut u64, fd);
        }) {
            gpu_hostcall_release(buf, p);
        }

        file_size = n as u32;
    }

    // Broadcast file_size from lane 0 to all lanes
    syncwarp(0xFFFF_FFFF);
    file_size = shfl_sync_idx_u32(0xFFFF_FFFF, file_size, 0);

    if file_size == 0 || pattern_len == 0 || pattern_len > file_size {
        if tid == 0 {
            *output = 0;
        }
        return;
    }

    // ========================================
    // Phase 2: Parallel search (all 32 lanes)
    // ========================================
    let n = file_size as usize;
    let plen = pattern_len as usize;
    let chunk_size = n / 32;
    let lane = tid;

    let start = (lane as usize) * chunk_size;
    let end = if lane == 31 {
        n
    } else {
        let raw_end = start + chunk_size + plen - 1;
        if raw_end > n { n } else { raw_end }
    };

    let mut local_count: u32 = 0;
    if chunk_size > 0 && start < n {
        let search_end = if end >= plen { end - plen + 1 } else { 0 };
        let mut i = start;
        while i < search_end {
            let mut matched = true;
            let mut j = 0;
            while j < plen {
                let data_byte = core::ptr::read_volatile(data_buf.add(i + j));
                let pat_byte = core::ptr::read_volatile(pattern.add(j));
                if data_byte != pat_byte {
                    matched = false;
                    break;
                }
                j += 1;
            }
            if matched {
                local_count += 1;
            }
            i += 1;
        }
    }

    // ========================================
    // Phase 3: Warp reduction via shfl.sync
    // ========================================
    let mask = 0xFFFF_FFFFu32;
    syncwarp(mask);

    // All lanes participate in shfl (required by CUDA).
    // Lane 0 accumulates the total.
    let mut total: u32 = 0;
    let mut src = 0u32;
    while src < 32 {
        // Every lane calls shfl with its local_count.
        // Each lane receives the value from lane `src`.
        let val = shfl_sync_idx_u32(mask, local_count, src);
        if lane == 0 {
            total += val;
        }
        src += 1;
    }

    // ========================================
    // Phase 4: Write result (thread 0 only)
    // ========================================
    if tid == 0 {
        // Write result count as ASCII to file
        let mut result_buf = [0u8; 32];
        let result_len = u32_to_ascii(total, &mut result_buf);

        sideband_reset(sideband);

        let out_path = b"search_result.txt";
        let pkt = match gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
            let slot0: u64 =
                (out_path.len() as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
            core::ptr::write_volatile(payload as *mut u64, slot0);
            let dst = payload.add(8);
            let mut k = 0;
            while k < out_path.len() {
                core::ptr::write_volatile(dst.add(k), out_path[k]);
                k += 1;
            }
        }) {
            Ok(p) => p,
            Err(_) => {
                *output = total;
                return;
            }
        };

        let out_fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        gpu_hostcall_release(buf, pkt);

        if out_fd != FILE_ERROR_SENTINEL {
            gpu_bulk_write(buf, sideband, out_fd, result_buf.as_ptr(), result_len);

            if let Ok(p) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
                core::ptr::write_volatile(payload as *mut u64, out_fd);
            }) {
                gpu_hostcall_release(buf, p);
            }
        }

        let msg = b"parallel search done";
        let _ = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);

        *output = total;
    }
}

/// Convert u32 to decimal ASCII. Returns number of bytes written.
fn u32_to_ascii(mut val: u32, buf: &mut [u8; 32]) -> usize {
    if val == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut digits = [0u8; 10];
    let mut len = 0;
    while val > 0 {
        digits[len] = b'0' + (val % 10) as u8;
        val /= 10;
        len += 1;
    }
    let mut i = 0;
    while i < len {
        buf[i] = digits[len - 1 - i];
        i += 1;
    }
    len
}
