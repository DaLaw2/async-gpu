#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]
use core::arch::nvptx;
use gpu_atomics::{
    membar_sys, sys_store_release_u32, sys_load_acquire_u32, sys_cas_u32, st_global_u32,
    sys_cas_u64, sys_fetch_add_u64, sys_exchange_u64,
    sys_load_acquire_u64, sys_spin_load_acquire_u32, activemask, lane_id,
};
use gpu_protocol::*;

// Install the gpu-runtime panic handler (sends panic message via hostcall)
gpu_runtime::panic_handler!();

// ============================================================
// Step 1: Inline PTX asm test — uses gpu-atomics crate
// ============================================================

/// Test: membar.sys via gpu-atomics crate
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_membar_sys(output: *mut u32, len: u32) {
    membar_sys();
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        *output.add(idx as usize) = 0xDEAD_BEEFu32;
    }
}

/// Test: st.release.sys.global.u32 via gpu-atomics crate
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_st_release_sys(ptr: *mut u32, val: u32) {
    sys_store_release_u32(ptr, val);
}

/// Test: ld.acquire.sys.global.u32 via gpu-atomics crate
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_ld_acquire_sys(ptr: *const u32, output: *mut u32) {
    let result = sys_load_acquire_u32(ptr);
    st_global_u32(output, result);
}

/// Test: atom.cas.sys.global.b32 via gpu-atomics crate
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_cas_sys(
    ptr: *mut u32,
    expected: u32,
    desired: u32,
    output: *mut u32,
) {
    let result = sys_cas_u32(ptr, expected, desired);
    st_global_u32(output, result);
}

// ============================================================
// Step 4: Volatile semantics test
// ============================================================

/// Test: read_volatile — does it emit ld.volatile in PTX?
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_read_volatile(ptr: *const u32, output: *mut u32) {
    let val = core::ptr::read_volatile(ptr);
    *output = val;
}

/// Test: write_volatile — does it emit st.volatile in PTX?
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_write_volatile(ptr: *mut u32, val: u32) {
    core::ptr::write_volatile(ptr, val);
}

// ============================================================
// Step 5: Integration kernel using gpu-atomics crate
// ============================================================

/// Integration test kernel (producer side of GPU-CPU protocol).
///
/// Thread 0 writes `value` to `data_ptr` with a system-scope release store,
/// then sets `flag_ptr = 1` with a system-scope release store. The host can
/// poll `flag_ptr` (with an acquire load) and when it sees 1, `data_ptr`
/// is guaranteed to be visible.
///
/// The release on the flag store is the architectural guarantee that the
/// data write is ordered before it. No additional `membar.sys` is needed
/// between two `st.release.sys` instructions.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn integration_sys_store(
    data_ptr: *mut u32,
    flag_ptr: *mut u32,
    value: u32,
) {
    // Only thread 0 writes
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let idx = block_x * block_dim_x + thread_x;
    if idx == 0 {
        // Write data with system-scope release
        sys_store_release_u32(data_ptr, value);
        // Signal CPU: flag = 1, system-scope release
        // (release semantics guarantee data store is visible before flag)
        sys_store_release_u32(flag_ptr, 1u32);
    }
}

// ============================================================
// Original kernels (preserved)
// ============================================================

/// A simple kernel that writes the global thread index into an output buffer.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vector_add(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;

    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        let val = *a.add(idx as usize) + *b.add(idx as usize);
        *c.add(idx as usize) = val;
    }
}

/// A simpler kernel: write the thread index into an output buffer.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn write_thread_idx(output: *mut u32, len: u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;

    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        *output.add(idx as usize) = idx;
    }
}

// ============================================================
// Step 6: u64 atomics tests (atomics.4)
// ============================================================

/// Test: atom.cas.sys.global.b64 via gpu-atomics crate.
///
/// Thread 0 attempts CAS on a u64: if *ptr == expected, set *ptr = desired.
/// Returns the old value in output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_u64_cas(
    ptr: *mut u64,
    expected_lo: u32,
    expected_hi: u32,
    desired_lo: u32,
    desired_hi: u32,
    output: *mut u64,
) {
    let expected = (expected_hi as u64) << 32 | expected_lo as u64;
    let desired = (desired_hi as u64) << 32 | desired_lo as u64;
    let result = sys_cas_u64(ptr, expected, desired);
    // Store result to output using a plain store (single thread, no race)
    *output = result;
}

/// Test: atom.add.sys.global.u64 via gpu-atomics crate.
///
/// Thread 0 atomically adds val to *ptr, returns old value in output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_u64_fetch_add(
    ptr: *mut u64,
    val_lo: u32,
    val_hi: u32,
    output: *mut u64,
) {
    let val = (val_hi as u64) << 32 | val_lo as u64;
    let result = sys_fetch_add_u64(ptr, val);
    *output = result;
}

/// Test: atom.exch.sys.global.b64 via gpu-atomics crate.
///
/// Thread 0 atomically exchanges *ptr with val, returns old value in output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_u64_exchange(
    ptr: *mut u64,
    val_lo: u32,
    val_hi: u32,
    output: *mut u64,
) {
    let val = (val_hi as u64) << 32 | val_lo as u64;
    let result = sys_exchange_u64(ptr, val);
    *output = result;
}

// ============================================================
// Step 7: Spin-load + warp intrinsic tests (atomics.4)
// ============================================================

/// Test: spin-load acquire u32.
///
/// Reads *ptr using the spin-safe acquire load and writes to output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_spin_load_u32(ptr: *const u32, output: *mut u32) {
    let val = sys_spin_load_acquire_u32(ptr);
    st_global_u32(output, val);
}

/// Test: activemask.b32 instruction.
///
/// Each thread writes the active lane mask to output[idx].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_activemask(output: *mut u32, len: u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        let mask = activemask();
        *output.add(idx as usize) = mask;
    }
}

/// Test: lane_id intrinsic.
///
/// Each thread writes its lane ID to output[idx].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_lane_id(output: *mut u32, len: u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        let lid = lane_id();
        *output.add(idx as usize) = lid;
    }
}

// ============================================================
// Hostcall protocol — uses gpu-runtime's consolidated API (api-cleanup.1)
// ============================================================
//
// Core protocol functions (hc_pop_free, hc_push, gpu_hostcall_print,
// gpu_hostcall_request, gpu_hostcall_release) are now provided by
// gpu_runtime::hostcall. This eliminates duplicated code and gains
// automatic sharding support.

/// Hostcall kernel: print "Hello from GPU!" via the hostcall protocol.
///
/// Thread 0 of block 0 issues a single PRINT hostcall. The host listener
/// reads the message from the packet payload and prints it to stdout.
///
/// `buf` is the device-side pointer to the hostcall buffer (mapped memory).
/// `result` is a device pointer where thread 0 writes 1 (success) or 0 (failure).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_print_hello(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Hardcoded message — the bytes live in GPU .const memory
    let msg: &[u8; 15] = b"Hello from GPU!";
    let ok = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), 15);
    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

/// Hostcall kernel: multiple warps each print a message.
///
/// Each block's thread 0 issues a PRINT hostcall with the block index.
/// Tests concurrent multi-warp hostcall.
///
/// `buf` is the hostcall buffer, `num_msgs` is total number of messages to print.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_print_multi(
    buf: *mut u8,
    success_count: *mut u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;

    // Only thread 0 of each block does the hostcall
    if thread_x != 0 {
        return;
    }

    // Format: "Block NNN\n" — we write the block index as decimal digits
    // Simple manual formatting since we don't have std::fmt
    let mut msg_buf: [u8; 16] = [0u8; 16];
    // "Block "
    msg_buf[0] = b'B';
    msg_buf[1] = b'l';
    msg_buf[2] = b'o';
    msg_buf[3] = b'c';
    msg_buf[4] = b'k';
    msg_buf[5] = b' ';
    // Format block_x as decimal (max 3 digits for our test)
    let mut n = block_x;
    let mut pos = 6;
    if n >= 100 {
        msg_buf[pos] = b'0' + (n / 100) as u8;
        pos += 1;
        n %= 100;
    }
    if block_x >= 10 {
        msg_buf[pos] = b'0' + (n / 10) as u8;
        pos += 1;
        n %= 10;
    }
    msg_buf[pos] = b'0' + n as u8;
    pos += 1;

    let ok = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg_buf.as_ptr(), pos as u32);
    if ok {
        gpu_atomics::sys_fetch_add_u32(success_count, 1);
    }
}

// ============================================================
// File I/O hostcall helpers (gpu-std.3)
// ============================================================

/// GPU-side hostcall: open a file.
/// Returns `(fd, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
unsafe fn gpu_hostcall_open(buf: *mut u8, path: *const u8, path_len: u32, flags: u32) -> (u64, u16) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
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
    });

    if pkt.is_null() {
        // Timeout — no packet returned
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);

    if !success {
        // Host returned CONTROL_ERROR — slot0 contains encoded error
        (0, error_category(slot0))
    } else {
        (slot0, 0)
    }
}

/// GPU-side hostcall: write data to a file.
/// Returns `(bytes_written, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
unsafe fn gpu_hostcall_write(buf: *mut u8, fd: u64, data: *const u8, data_len: u32) -> (u64, u16) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
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
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);

    if !success {
        (0, error_category(slot0))
    } else {
        (slot0, 0)
    }
}

/// GPU-side hostcall: close a file.
/// Returns `(0, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
unsafe fn gpu_hostcall_close(buf: *mut u8, fd: u64) -> (u64, u16) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);

    if !success {
        (0, error_category(slot0))
    } else {
        (slot0, 0)
    }
}

/// GPU-side hostcall: read data from a file.
/// Returns `(bytes_read, 0)` on success (data copied to out_buf), `(0, error_category)` on failure.
#[inline(always)]
unsafe fn gpu_hostcall_read(buf: *mut u8, fd: u64, out_buf: *mut u8, max_len: u32) -> (u64, u16) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_READ, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
        // Slot 1: max bytes to read
        core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);

    if !success {
        gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
        return (0, error_category(slot0));
    }

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
// File I/O test kernel (gpu-std.3)
// ============================================================

/// Hostcall kernel: create a file, write content, close, reopen, read back, verify.
///
/// Thread 0 of block 0 performs a full file I/O round-trip:
/// 1. Open "gpu_test_output.txt" for writing
/// 2. Write "Hello from GPU file I/O!\n"
/// 3. Close the file
/// 4. Open "gpu_test_output.txt" for reading
/// 5. Read back the content
/// 6. Close the file
/// 7. Write result codes to output
///
/// `buf` = hostcall buffer, `result` = output array of u32[4]:
///   [0] = overall success (1) or failure (0)
///   [1] = open-write fd (or 0xFFFFFFFF on error)
///   [2] = bytes written (or 0xFFFFFFFF on error)
///   [3] = bytes read back (or 0xFFFFFFFF on error)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_file_test(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Initialize result to all-fail
    *result.add(0) = 0;
    *result.add(1) = 0xFFFF_FFFF;
    *result.add(2) = 0xFFFF_FFFF;
    *result.add(3) = 0xFFFF_FFFF;

    // File path
    let path: &[u8; 20] = b"gpu_test_output.txt\0";
    let path_len: u32 = 19; // excluding null terminator

    // Message to write
    let msg: &[u8; 25] = b"Hello from GPU file I/O!\n";
    let msg_len: u32 = 25;

    // Step 1: Open file for writing (create)
    let (fd, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_WRITE_CREATE);
    if err != 0 {
        return;
    }
    *result.add(1) = fd as u32;

    // Step 2: Write message
    let (written, err) = gpu_hostcall_write(buf, fd, msg.as_ptr(), msg_len);
    if err != 0 {
        // Still try to close
        gpu_hostcall_close(buf, fd);
        return;
    }
    *result.add(2) = written as u32;

    // Step 3: Close file
    let (_, err) = gpu_hostcall_close(buf, fd);
    if err != 0 {
        return;
    }

    // Step 4: Reopen for reading
    let (fd2, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_READ);
    if err != 0 {
        return;
    }

    // Step 5: Read back content
    let mut read_buf: [u8; 48] = [0u8; 48];
    let (bytes_read, err) = gpu_hostcall_read(buf, fd2, read_buf.as_mut_ptr(), 48);
    if err != 0 {
        gpu_hostcall_close(buf, fd2);
        return;
    }
    *result.add(3) = bytes_read as u32;

    // Step 6: Close read file
    gpu_hostcall_close(buf, fd2);

    // Step 7: Verify content matches
    if bytes_read == msg_len as u64 {
        let mut match_ok: bool = true;
        let mut i: u32 = 0;
        while i < msg_len {
            if read_buf[i as usize] != *msg.as_ptr().add(i as usize) {
                match_ok = false;
            }
            i += 1;
        }
        if match_ok {
            *result.add(0) = 1; // Overall success
        }
    }
}

// ============================================================
// GPU Instant + stdin + time hostcall helpers (gpu-std.4)
// ============================================================

/// Read the GPU's %globaltimer register (64-bit nanosecond counter).
/// Available on SM 3.0+ (all modern GPUs).
/// Returns a monotonic nanosecond timestamp.
#[inline(always)]
unsafe fn gpu_instant_nanos() -> u64 {
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
unsafe fn gpu_hostcall_stdin_read(buf: *mut u8, out_buf: *mut u8, max_len: u32) -> (u64, u16) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_STDIN, |payload| {
        // Slot 0: max bytes to read
        core::ptr::write_volatile(payload as *mut u64, max_len as u64);
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);

    if !success {
        gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
        return (0, error_category(slot0));
    }

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
unsafe fn gpu_hostcall_time(buf: *mut u8) -> (u64, u64) {
    let (pkt, success) = gpu_runtime::hostcall::gpu_hostcall_request(buf, SERVICE_TIME, |_payload| {
        // No request payload needed
    });

    if pkt.is_null() || !success {
        if !pkt.is_null() {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
        }
        return (0, 0);
    }

    let secs = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    let nanos = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD).add(8) as *const u64);
    gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
    (secs, nanos)
}

// ============================================================
// Stdin + time test kernel (gpu-std.4)
// ============================================================

/// Test kernel: read GPU Instant, get wall-clock time, and stdin read.
///
/// `buf` = hostcall buffer
/// `result` = output array of u64[4]:
///   [0] = GPU instant (nanoseconds from %globaltimer)
///   [1] = host wall-clock seconds since epoch
///   [2] = host wall-clock nanoseconds
///   [3] = 1 if stdin read succeeded, 0 if skipped/failed
///         (stdin test is optional — host may not provide input)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_stdin_time_test(
    buf: *mut u8,
    result: *mut u64,
    skip_stdin: u32,
) {
    let thread_x = core::arch::nvptx::_thread_idx_x() as u32;
    let block_x = core::arch::nvptx::_block_idx_x() as u32;
    let block_dim_x = core::arch::nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Test 1: GPU Instant (%globaltimer)
    let t0 = gpu_instant_nanos();
    // Do a small amount of work to see non-zero delta
    // Use volatile reads to prevent the compiler from optimizing this loop away
    let mut dummy: u32 = 0;
    let mut i: u32 = 0;
    while core::ptr::read_volatile(&i) < 1000 {
        dummy = core::ptr::read_volatile(&dummy).wrapping_add(core::ptr::read_volatile(&i));
        i += 1;
    }
    // Use volatile write to ensure dummy is not dead-code eliminated
    core::ptr::write_volatile(result.add(3), dummy as u64);
    let t1 = gpu_instant_nanos();
    *result.add(0) = t1 - t0; // nanosecond delta

    // Test 2: Wall-clock time via hostcall
    let (secs, nanos) = gpu_hostcall_time(buf);
    *result.add(1) = secs;
    *result.add(2) = nanos;

    // Test 3: stdin read (optional, skip if skip_stdin != 0)
    if skip_stdin == 0 {
        let mut stdin_buf: [u8; 56] = [0u8; 56];
        let (bytes, err) = gpu_hostcall_stdin_read(buf, stdin_buf.as_mut_ptr(), 56);
        if err == 0 && bytes > 0 {
            *result.add(3) = 1;
        } else {
            *result.add(3) = 0;
        }
    } else {
        *result.add(3) = 0;
    }
}

// ============================================================
// Error propagation test kernel (error-handling.2)
// ============================================================

/// Test kernel: verify that hostcall error codes propagate to GPU.
///
/// Tests:
/// 1. Open a nonexistent file → should get ERR_NOT_FOUND
/// 2. Close an invalid fd → should get an error
/// 3. Read from an invalid fd → should get an error
///
/// `buf` = hostcall buffer, `result` = output array of u32[6]:
///   [0] = overall success (1 if all tests pass)
///   [1] = test1 error category (expected: ERR_NOT_FOUND = 1)
///   [2] = test2 error category (expected: nonzero)
///   [3] = test3 error category (expected: nonzero)
///   [4] = test1 fd (should be 0 since open failed)
///   [5] = number of tests passed
#[no_mangle]
pub unsafe extern "ptx-kernel" fn error_propagation_test(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    if thread_x != 0 {
        return;
    }

    // Initialize results
    let mut i = 0;
    while i < 6 {
        core::ptr::write_volatile(result.add(i), 0);
        i += 1;
    }

    let mut passed: u32 = 0;

    // Test 1: Open a nonexistent file → should return ERR_NOT_FOUND
    let path = b"__nonexistent_file_12345__";
    let (fd, err_cat) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, FILE_OPEN_READ);
    core::ptr::write_volatile(result.add(1), err_cat as u32);
    core::ptr::write_volatile(result.add(4), fd as u32);
    if err_cat == ERR_NOT_FOUND {
        passed += 1;
    }

    // Test 2: Close an invalid fd (99999) → should return an error
    let (_, err_cat) = gpu_hostcall_close(buf, 99999);
    core::ptr::write_volatile(result.add(2), err_cat as u32);
    if err_cat != 0 {
        passed += 1;
    }

    // Test 3: Read from invalid fd → should return an error
    let mut dummy_buf: [u8; 8] = [0u8; 8];
    let (_, err_cat) = gpu_hostcall_read(buf, 99999, dummy_buf.as_mut_ptr(), 8);
    core::ptr::write_volatile(result.add(3), err_cat as u32);
    if err_cat != 0 {
        passed += 1;
    }

    core::ptr::write_volatile(result.add(5), passed);
    if passed == 3 {
        core::ptr::write_volatile(result.add(0), 1); // All tests passed
    }
}

// ============================================================
// Hostcall latency benchmark kernel (benchmark.2)
// ============================================================

/// Instrumented hc_pop_free: returns (packet_index, cas_retry_count).
#[inline(always)]
unsafe fn hc_pop_free_counted(buf: *mut u8) -> (u16, u32) {
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

/// Benchmark kernel: measure hostcall NOP round-trip latency.
///
/// Each thread performs `num_iters` NOP hostcalls sequentially.
/// Records per-thread: total elapsed ns, total CAS retries, iterations completed.
///
/// Layout of `results` (u64 array, 3 entries per thread):
///   results[tid*3 + 0] = total elapsed nanoseconds for all iterations
///   results[tid*3 + 1] = total CAS retries across all iterations
///   results[tid*3 + 2] = number of completed iterations
///
/// `buf` = hostcall buffer
/// `results` = output array, must have space for num_threads * 3 u64 entries
/// `num_iters` = number of NOP hostcalls per thread
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_latency_bench(
    buf: *mut u8,
    results: *mut u64,
    num_iters: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    let t_start = gpu_instant_nanos();
    let mut total_retries: u64 = 0;
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        // Pop free packet (instrumented)
        let (pkt_idx, retries) = hc_pop_free_counted(buf);
        if pkt_idx == NULL_INDEX {
            // Pool exhaustion — stop this thread
            break;
        }
        total_retries += retries as u64;

        let pkt = buf.add(packet_offset(pkt_idx));

        // Fill NOP packet
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_NOP);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Mark as filled
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Push to ready stack
        let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, pkt_idx);

        // Ring doorbell
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                break;
            }
        }

        // Return packet to free stack
        let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
        gpu_runtime::hostcall::hc_push(free_ptr, buf, pkt_idx);

        completed += 1;
        iter += 1;
    }

    let t_end = gpu_instant_nanos();

    // Write results
    let base = (tid as usize) * 3;
    core::ptr::write_volatile(results.add(base), t_end - t_start);
    core::ptr::write_volatile(results.add(base + 1), total_retries);
    core::ptr::write_volatile(results.add(base + 2), completed);
}

// ============================================================
// Sharding benchmark kernel (per-block-sharding.3)
// ============================================================

/// Instrumented shard-aware hc_pop_free: returns (packet_index, cas_retry_count).
/// Uses shard-local free stack when num_shards > 0, global free stack otherwise.
#[inline(always)]
unsafe fn hc_pop_free_counted_v2(
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

/// Shard-aware benchmark kernel: measure hostcall NOP round-trip latency.
///
/// Identical to `hostcall_latency_bench` but uses shard-aware stacks.
/// Works with both sharded and unsharded buffers (auto-detects via num_shards).
///
/// Layout of `results` (u64 array, 3 entries per thread):
///   results[tid*3 + 0] = total elapsed nanoseconds for all iterations
///   results[tid*3 + 1] = total CAS retries across all iterations
///   results[tid*3 + 2] = number of completed iterations
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_latency_bench_v2(
    buf: *mut u8,
    results: *mut u64,
    num_iters: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    // Read shard info once
    let (num_shards, shard_array_off, _) =
        gpu_runtime::hostcall::read_shard_info(buf as *const u8);
    let free_ptr = gpu_runtime::hostcall::get_free_stack_ptr(buf, num_shards, shard_array_off);
    let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_array_off);

    let t_start = gpu_instant_nanos();
    let mut total_retries: u64 = 0;
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        // Pop free packet (instrumented, shard-aware)
        let (pkt_idx, retries) =
            hc_pop_free_counted_v2(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            break;
        }
        total_retries += retries as u64;

        let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, pkt_idx);
        let pkt = buf.add(pkt_off);

        // Fill NOP packet
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_NOP);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Mark as filled
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Push to ready stack (shard-aware via hc_push which reads shard info)
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, pkt_idx);

        // Ring doorbell (always global)
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                break;
            }
        }

        // Return packet to free stack (shard-aware)
        gpu_runtime::hostcall::hc_push(free_ptr, buf, pkt_idx);

        completed += 1;
        iter += 1;
    }

    let t_end = gpu_instant_nanos();

    // Write results
    let base = (tid as usize) * 3;
    core::ptr::write_volatile(results.add(base), t_end - t_start);
    core::ptr::write_volatile(results.add(base + 1), total_retries);
    core::ptr::write_volatile(results.add(base + 2), completed);
}

// ============================================================
// GPU panic test kernel (gpu-panic.2)
// ============================================================

/// Test kernel: deliberately panic to verify panic message delivery via hostcall.
///
/// Thread 0 initializes the panic handler and then panics with a test message.
/// Other threads simply return.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 before panic — if host sees this AND the
///            panic message, the test passes)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn panic_test_kernel(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;

    // Initialize panic handler with hostcall buffer
    gpu_runtime::panic::gpu_panic_init(buf);

    if global_idx != 0 {
        return;
    }

    // Set marker to indicate we reached this point
    core::ptr::write_volatile(result, 1);

    // Deliberately panic — this should send a message via hostcall then trap
    panic!("test panic from GPU thread 0");
}

// ============================================================
// Bulk data transfer test kernel (large-payload.3)
// ============================================================

/// Test kernel: write a 4KB message to a file via sideband bulk transfer,
/// then read it back and verify the content matches.
///
/// `buf` = hostcall buffer
/// `sideband` = sideband data buffer
/// `result` = output array of u32[4]:
///   [0] = overall success (1 if write+read+verify all pass)
///   [1] = bytes written
///   [2] = bytes read back
///   [3] = content match (1 if all bytes match)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bulk_io_test(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
) {
    use gpu_runtime::sideband::{gpu_bulk_read, gpu_bulk_write, sideband_reset};

    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Initialize panic handler + reset sideband allocator
    gpu_runtime::panic::gpu_panic_init(buf);
    sideband_reset(sideband);

    // Initialize results to failure
    core::ptr::write_volatile(result.add(0), 0);
    core::ptr::write_volatile(result.add(1), 0);
    core::ptr::write_volatile(result.add(2), 0);
    core::ptr::write_volatile(result.add(3), 0);

    // Generate a 4KB test pattern
    const DATA_SIZE: usize = 4096;
    let mut test_data: [u8; DATA_SIZE] = [0u8; DATA_SIZE];
    let mut i: usize = 0;
    while i < DATA_SIZE {
        test_data[i] = (i & 0xFF) as u8;
        i += 1;
    }

    // Step 1: Open file for writing
    let path = b"gpu_bulk_test.bin";
    let (fd, err) = gpu_hostcall_open(
        buf,
        path.as_ptr(),
        path.len() as u32,
        FILE_OPEN_WRITE_CREATE,
    );
    if err != 0 {
        return;
    }

    // Step 2: Bulk write 4KB
    let written = gpu_bulk_write(buf, sideband, fd, test_data.as_ptr(), DATA_SIZE);
    core::ptr::write_volatile(result.add(1), written as u32);

    // Close write file
    gpu_hostcall_close(buf, fd);

    if written != DATA_SIZE {
        return;
    }

    // Reset sideband allocator for read phase
    sideband_reset(sideband);

    // Step 3: Reopen for reading
    let (fd2, err) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, FILE_OPEN_READ);
    if err != 0 {
        return;
    }

    // Step 4: Bulk read 4KB
    let mut read_buf: [u8; DATA_SIZE] = [0u8; DATA_SIZE];
    let bytes_read = gpu_bulk_read(buf, sideband, fd2, read_buf.as_mut_ptr(), DATA_SIZE);
    core::ptr::write_volatile(result.add(2), bytes_read as u32);

    // Close read file
    gpu_hostcall_close(buf, fd2);

    if bytes_read != DATA_SIZE {
        return;
    }

    // Step 5: Verify content matches
    let mut match_ok: bool = true;
    let mut j: usize = 0;
    while j < DATA_SIZE {
        if read_buf[j] != test_data[j] {
            match_ok = false;
        }
        j += 1;
    }

    if match_ok {
        core::ptr::write_volatile(result.add(3), 1);
        core::ptr::write_volatile(result.add(0), 1); // Overall success
    }
}

// ============================================================
// Warp intrinsic tests (warp-future.3)
// ============================================================

/// Test: bar.warp.sync + shfl.sync.idx.b32 warp intrinsics.
///
/// Launches with 32 threads (1 warp). Lane 0 writes a magic value,
/// broadcasts it to all lanes via shfl.sync.idx, then all lanes
/// write the received value to output[lane_id]. If all outputs
/// equal the magic value, both shfl.sync and bar.warp.sync work.
///
/// `output` must have space for 32 u32 entries.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_warp_intrinsics(output: *mut u32) {
    let lid = gpu_atomics::lane_id();
    let mask = gpu_atomics::activemask();

    // Lane 0 provides the magic value; other lanes provide 0
    let my_val = if lid == 0 { 0xCAFE_BABE_u32 } else { 0u32 };

    // Synchronize all lanes before shuffle
    gpu_atomics::syncwarp(mask);

    // Broadcast lane 0's value to all lanes
    let received = gpu_atomics::shfl_sync_idx_u32(mask, my_val, 0);

    // Each lane writes the received value
    *output.add(lid as usize) = received;
}

// ============================================================
// WarpFuture PoC: hand-written warp-level PRINT hostcall (warp-future.4)
// ============================================================

/// State discriminant values for WarpPrintFuture.
const WPF_INIT: u32 = 0;
const WPF_WAIT: u32 = 1;
const WPF_DONE: u32 = 2;

/// Hand-written WarpFuture: all 32 lanes cooperatively send a PRINT hostcall.
///
/// Each lane contributes its lane_id as a byte to the message.
/// Lane 0 handles packet allocation, submission, and release.
/// All lanes stay convergent throughout the state machine.
struct WarpPrintFuture {
    buf: *mut u8,
    state: u32,       // discriminant (lane 0 authoritative)
    pkt_idx: u16,     // packet index (uniform after broadcast)
}

impl WarpPrintFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: WPF_INIT,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for WarpPrintFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        // Broadcast state from lane 0 to all lanes
        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            WPF_INIT => unsafe {
                // Lane 0: pop a free packet
                let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
                if wcx.is_leader() {
                    idx_raw = gpu_runtime::hostcall::hc_pop_free(self.buf) as u32;
                }

                // Broadcast packet index to all lanes
                let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;

                if idx == gpu_protocol::NULL_INDEX {
                    return WarpPoll::Pending; // backpressure — no free packets
                }
                self.pkt_idx = idx;

                // Compute packet pointer
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

                // Build message: "WarpFuture: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
                // Header "WarpFuture: " in slot 0 (msg_len) + slots 1..
                // Each lane writes its character at the right position
                let prefix = b"WarpFuture: ";
                let msg_len = prefix.len() as u32 + 32; // 12 + 32 = 44 bytes

                // Lane 0 writes the length into slot 0
                if wcx.is_leader() {
                    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
                }

                // All lanes write their byte into the message body
                // Message starts at payload + 8 (slot 1)
                let msg_base = payload.add(8);
                let lid = wcx.lane_id;

                // First 12 bytes are the prefix — only lanes 0..11 write those
                if lid < prefix.len() as u32 {
                    core::ptr::write_volatile(
                        msg_base.add(lid as usize),
                        prefix[lid as usize],
                    );
                }

                // Bytes 12..43 are 'A' + lane_id (all 32 lanes write)
                let char_offset = prefix.len() as u32 + lid;
                if char_offset < msg_len {
                    core::ptr::write_volatile(
                        msg_base.add(char_offset as usize),
                        b'A'.wrapping_add(lid as u8),
                    );
                }

                // Lane 0: write thread/block metadata at payload+64
                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        payload.add(64) as *mut u32,
                        nvptx::_block_idx_x() as u32,
                    );
                    core::ptr::write_volatile(
                        payload.add(68) as *mut u32,
                        nvptx::_thread_idx_x() as u32,
                    );
                }

                // Sync: ensure all payload writes are visible
                gpu_atomics::syncwarp(wcx.active_mask);

                // Lane 0: fill header, mark FILLED, push to ready, ring doorbell
                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
                        wcx.active_mask,
                    );
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
                        gpu_protocol::SERVICE_PRINT,
                    );
                    sys_store_release_u32(
                        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
                        0,
                    );
                    sys_store_release_u32(
                        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
                        gpu_protocol::CONTROL_FILLED,
                    );

                    // Push to ready stack + ring doorbell
                    let (num_shards, shard_off, _) =
                        gpu_runtime::hostcall::read_shard_info(self.buf as *const u8);
                    let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(
                        self.buf, num_shards, shard_off,
                    );
                    gpu_runtime::hostcall::hc_push(ready_ptr, self.buf, idx);
                    sys_fetch_add_u64(
                        self.buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64,
                        1,
                    );

                    self.state = WPF_WAIT;
                }

                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            WPF_WAIT => unsafe {
                // All lanes read the same control word — perfectly convergent spin
                let idx = broadcast_u32(wcx.active_mask, self.pkt_idx as u32) as u16;
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let ctrl = sys_spin_load_acquire_u32(
                    pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32,
                );

                if ctrl & gpu_protocol::CONTROL_READY != 0 {
                    // Host responded — release packet
                    if wcx.is_leader() {
                        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
                        self.state = WPF_DONE;
                    }
                    gpu_atomics::syncwarp(wcx.active_mask);
                    return WarpPoll::Ready(true);
                }

                WarpPoll::Pending
            },

            WPF_DONE => {
                WarpPoll::Ready(true)
            },

            _ => WarpPoll::Pending, // unreachable
        }
    }
}

/// WarpFuture PoC kernel: all 32 lanes cooperatively send a PRINT hostcall.
///
/// Uses the WarpFuture trait + WarpExecutor. Lane 0 handles packet management,
/// all lanes write message data in parallel, all lanes spin-wait convergently.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 if WarpFuture completed successfully)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_future_print_test(
    buf: *mut u8,
    result: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = WarpPrintFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    // Lane 0 writes the result
    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// WarpFuture multi-hostcall PoC: 3 sequential PRINT calls (warp-future.6)
// ============================================================

/// State discriminant values for WarpMultiPrintFuture.
/// 7-state machine: INIT1→WAIT1→INIT2→WAIT2→INIT3→WAIT3→DONE
const WMP_INIT1: u32 = 0;
const WMP_WAIT1: u32 = 1;
const WMP_INIT2: u32 = 2;
const WMP_WAIT2: u32 = 3;
const WMP_INIT3: u32 = 4;
const WMP_WAIT3: u32 = 5;
const WMP_DONE:  u32 = 6;

/// Hand-written WarpFuture: 3 sequential PRINT hostcalls.
///
/// Validates that a WarpFuture state machine can compose multiple hostcalls
/// while maintaining warp convergence across all state transitions.
/// Lane 0 manages packets; all 32 lanes write payload cooperatively.
///
/// Messages sent:
///   1: "WarpMulti[1/3]: HELLO_FROM_32_LANES!!"
///   2: "WarpMulti[2/3]: SECOND_CALL_WORKING!"
///   3: "WarpMulti[3/3]: PIPELINE_COMPLETE!!"
struct WarpMultiPrintFuture {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    calls_completed: u32,
}

impl WarpMultiPrintFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: WMP_INIT1,
            pkt_idx: gpu_protocol::NULL_INDEX,
            calls_completed: 0,
        }
    }
}

/// Shared init logic for each of the 3 PRINT hostcalls.
/// Returns the packet pointer (all lanes can use it) and WarpPoll::Pending.
///
/// # Safety
/// Must be called by all active lanes of a warp simultaneously.
#[inline(always)]
unsafe fn warp_multi_init_hostcall(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: &mut u16,
    next_state: u32,
    state: &mut u32,
    call_num: u32,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

    // Lane 0: pop a free packet
    let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = gpu_runtime::hostcall::hc_pop_free(buf) as u32;
    }

    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == gpu_protocol::NULL_INDEX {
        return WarpPoll::Pending; // backpressure
    }
    *pkt_idx = idx;

    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

    // Select message based on call number
    let (prefix, suffix): (&[u8], &[u8]) = match call_num {
        0 => (b"WarpMulti[1/3]: ", b"HELLO_FROM_32_LANES!!"),
        1 => (b"WarpMulti[2/3]: ", b"SECOND_CALL_WORKING!"),
        _ => (b"WarpMulti[3/3]: ", b"PIPELINE_COMPLETE!!"),
    };
    let msg_len = prefix.len() as u32 + suffix.len() as u32;

    // Lane 0: write message length
    if wcx.is_leader() {
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
    }

    // All lanes cooperatively write the message bytes
    let msg_base = payload.add(8);
    let lid = wcx.lane_id;

    // Write prefix bytes (lanes with lid < prefix.len)
    if lid < prefix.len() as u32 {
        core::ptr::write_volatile(
            msg_base.add(lid as usize),
            prefix[lid as usize],
        );
    }

    // Write suffix bytes (lanes with lid < suffix.len)
    if lid < suffix.len() as u32 {
        core::ptr::write_volatile(
            msg_base.add(prefix.len() + lid as usize),
            suffix[lid as usize],
        );
    }

    // Lane 0: write thread/block metadata at payload+64
    if wcx.is_leader() {
        core::ptr::write_volatile(
            payload.add(64) as *mut u32,
            nvptx::_block_idx_x() as u32,
        );
        core::ptr::write_volatile(
            payload.add(68) as *mut u32,
            nvptx::_thread_idx_x() as u32,
        );
    }

    // Ensure all payload writes are visible
    gpu_atomics::syncwarp(wcx.active_mask);

    // Lane 0: fill header, mark FILLED, push to ready, ring doorbell
    if wcx.is_leader() {
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
            wcx.active_mask,
        );
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
            gpu_protocol::SERVICE_PRINT,
        );
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            0,
        );
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            gpu_protocol::CONTROL_FILLED,
        );

        let (num_shards, shard_off, _) =
            gpu_runtime::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(
            buf, num_shards, shard_off,
        );
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(
            buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64,
            1,
        );

        *state = next_state;
    }

    gpu_atomics::syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// Shared wait logic: spin-wait for host response, release packet.
///
/// # Safety
/// Must be called by all active lanes of a warp simultaneously.
#[inline(always)]
unsafe fn warp_multi_wait_hostcall(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state: &mut u32,
    calls_completed: &mut u32,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(
        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32,
    );

    if ctrl & gpu_protocol::CONTROL_READY != 0 {
        if wcx.is_leader() {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            *calls_completed += 1;
            *state = next_state;
        }
        gpu_atomics::syncwarp(wcx.active_mask);

        if next_state == WMP_DONE {
            return WarpPoll::Ready(true);
        }
        return WarpPoll::Pending; // Transition to next INIT state
    }

    WarpPoll::Pending
}

unsafe impl gpu_runtime::warp_future::WarpFuture for WarpMultiPrintFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        // Broadcast state from lane 0 to all lanes
        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            WMP_INIT1 => unsafe {
                warp_multi_init_hostcall(
                    self.buf, wcx, &mut self.pkt_idx, WMP_WAIT1, &mut self.state, 0,
                )
            },
            WMP_WAIT1 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf, wcx, self.pkt_idx, WMP_INIT2, &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_INIT2 => unsafe {
                warp_multi_init_hostcall(
                    self.buf, wcx, &mut self.pkt_idx, WMP_WAIT2, &mut self.state, 1,
                )
            },
            WMP_WAIT2 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf, wcx, self.pkt_idx, WMP_INIT3, &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_INIT3 => unsafe {
                warp_multi_init_hostcall(
                    self.buf, wcx, &mut self.pkt_idx, WMP_WAIT3, &mut self.state, 2,
                )
            },
            WMP_WAIT3 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf, wcx, self.pkt_idx, WMP_DONE, &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// WarpFuture multi-hostcall kernel: 3 sequential PRINT hostcalls in one WarpFuture.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 if all 3 calls succeeded)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_future_multi_print_test(
    buf: *mut u8,
    result: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = WarpMultiPrintFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    // Lane 0 writes the result
    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// WarpFuture proc macro test (warp-future.5)
// ============================================================

// The #[warp_async] proc macro generates:
// - `WarpMacroPrintTest` struct with buf, state, pkt_idx
// - WarpFuture impl with 2 PRINT hostcalls (4 states + DONE)
// - `warp_macro_print_test` kernel entry point
#[warp_macro::warp_async]
unsafe fn warp_macro_print_test(buf: *mut u8) -> bool {
    warp_print!(buf, b"Macro[1/2]: GENERATED_CODE!!");
    warp_print!(buf, b"Macro[2/2]: PROC_MACRO_WORKS!");
}

// ============================================================
// Sharding-aware print test — uses gpu-runtime's hostcall path
// ============================================================

/// Print a message via gpu-runtime's `gpu_hostcall_print` which auto-detects
/// sharded vs legacy buffers. Thread 0 of each block prints "Shard N".
/// Increments `success_count` atomically on success.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sharded_print_test(
    buf: *mut u8,
    success_count: *mut u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;

    if thread_x != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(buf);

    // Format: "Shard N" with block index
    let mut msg_buf: [u8; 16] = [0u8; 16];
    msg_buf[0] = b'S';
    msg_buf[1] = b'h';
    msg_buf[2] = b'a';
    msg_buf[3] = b'r';
    msg_buf[4] = b'd';
    msg_buf[5] = b' ';
    let mut n = block_x;
    let mut pos = 6;
    if n >= 100 {
        msg_buf[pos] = b'0' + (n / 100) as u8;
        pos += 1;
        n %= 100;
    }
    if block_x >= 10 {
        msg_buf[pos] = b'0' + (n / 10) as u8;
        pos += 1;
        n %= 10;
    }
    msg_buf[pos] = b'0' + n as u8;
    pos += 1;

    let ok = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg_buf.as_ptr(), pos as u32);
    if ok {
        gpu_atomics::sys_fetch_add_u32(success_count, 1);
    }
}

// ============================================================
// Parallel file grep kernel (product.8)
// ============================================================

/// Search a byte buffer for lines containing a pattern.
#[inline(always)]
unsafe fn grep_buffer(
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
                    msg[pos] = b'T'; pos += 1;
                    if thread_id >= 10 {
                        msg[pos] = b'0' + (thread_id / 10) as u8; pos += 1;
                    }
                    msg[pos] = b'0' + (thread_id % 10) as u8; pos += 1;
                    msg[pos] = b':'; pos += 1;
                    msg[pos] = b' '; pos += 1;
                    let copy_len = line_len.min(56 - pos);
                    let mut c: usize = 0;
                    while c < copy_len {
                        msg[pos] = *data.add(line_start + c);
                        pos += 1;
                        c += 1;
                    }
                    gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), pos as u32);
                    matches += 1;
                }
            }
            line_start = i + 1;
        }
        i += 1;
    }
    matches
}

/// Parallel file grep kernel: each thread opens, reads, and searches a file.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn parallel_grep_kernel(
    buf: *mut u8,
    sideband: *mut u8,
    results: *mut u64,
    pattern_ptr: *const u8,
    pattern_len: u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    let mut pattern_buf = [0u8; 32];
    let plen = (pattern_len as usize).min(32);
    let mut pi: usize = 0;
    while pi < plen {
        pattern_buf[pi] = core::ptr::read_volatile(pattern_ptr.add(pi));
        pi += 1;
    }

    let path = b"gpu_grep_test.txt";
    let (fd, err) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, 0);
    if err != 0 || fd == 0 {
        core::ptr::write_volatile(results.add(tid as usize), 0u64);
        return;
    }

    let mut file_buf = [0u8; 4096];
    let bytes_read = gpu_runtime::sideband::gpu_bulk_read(
        buf, sideband, fd, file_buf.as_mut_ptr(), 4096,
    );

    gpu_hostcall_close(buf, fd);

    let match_count = grep_buffer(
        buf, file_buf.as_ptr(), bytes_read,
        &pattern_buf[..plen], tid,
    );

    core::ptr::write_volatile(results.add(tid as usize), match_count as u64);
}

// ============================================================
// hybrid-executor.1: WarpFuture + per-thread compute block PoC
// ============================================================
//
// Demonstrates mixing warp-cooperative I/O (WarpFuture) with
// per-thread divergent computation in the same state machine.
//
// State machine:
//   0: INIT_PRINT  - warp-cooperative PRINT "hybrid: start"
//   1: WAIT_PRINT  - wait for host response
//   2: COMPUTE     - per-thread block: each lane computes results[lane_id] = lane_id^2 + 1
//   3: INIT_PRINT2 - warp-cooperative PRINT "hybrid: done"
//   4: WAIT_PRINT2 - wait for host response
//   5: DONE        - return true

const HYB_INIT_PRINT: u32 = 0;
const HYB_WAIT_PRINT: u32 = 1;
const HYB_COMPUTE: u32 = 2;
const HYB_INIT_PRINT2: u32 = 3;
const HYB_WAIT_PRINT2: u32 = 4;
const HYB_DONE: u32 = 5;

struct HybridFuture {
    buf: *mut u8,
    results: *mut u32,
    state: u32,
    pkt_idx: u16,
}

impl HybridFuture {
    #[inline(always)]
    fn new(buf: *mut u8, results: *mut u32) -> Self {
        Self {
            buf,
            results,
            state: HYB_INIT_PRINT,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

/// Helper: warp-cooperative PRINT init — pops packet, writes message, submits.
/// Returns (WarpPoll::Pending, pkt_idx) on success, or Pending with NULL_INDEX on backpressure.
#[inline(always)]
unsafe fn hybrid_warp_print_init(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    msg: &[u8],
    next_state: u32,
    state_cell: &mut u32,
    pkt_idx_cell: &mut u16,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

    let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = gpu_runtime::hostcall::hc_pop_free(buf) as u32;
    }
    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == gpu_protocol::NULL_INDEX {
        return WarpPoll::Pending;
    }
    *pkt_idx_cell = idx;

    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);
    let msg_len = msg.len() as u32;

    if wcx.is_leader() {
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
    }

    // Cooperative write: all lanes write first 32 bytes
    let msg_base = payload.add(8);
    let lid = wcx.lane_id;
    if lid < msg_len && lid < 32 {
        core::ptr::write_volatile(msg_base.add(lid as usize), msg[lid as usize]);
    }
    // Lane 0 writes remaining bytes
    if wcx.is_leader() && msg_len > 32 {
        let mut j: u32 = 32;
        while j < msg_len {
            core::ptr::write_volatile(msg_base.add(j as usize), msg[j as usize]);
            j += 1;
        }
    }

    // Metadata
    if wcx.is_leader() {
        core::ptr::write_volatile(payload.add(64) as *mut u32, nvptx::_block_idx_x() as u32);
        core::ptr::write_volatile(payload.add(68) as *mut u32, nvptx::_thread_idx_x() as u32);
    }

    gpu_atomics::syncwarp(wcx.active_mask);

    if wcx.is_leader() {
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
            wcx.active_mask,
        );
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
            gpu_protocol::SERVICE_PRINT,
        );
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            gpu_protocol::CONTROL_FILLED,
        );
        let (num_shards, shard_off, _) =
            gpu_runtime::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64, 1);
        *state_cell = next_state;
    }

    gpu_atomics::syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// Helper: warp-cooperative WAIT — spin on control word, release packet on READY.
#[inline(always)]
unsafe fn hybrid_warp_wait(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state_cell: &mut u32,
) -> Option<()> {
    use gpu_runtime::warp_future::broadcast_u32;

    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32);

    if ctrl & gpu_protocol::CONTROL_READY != 0 {
        if wcx.is_leader() {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            *state_cell = next_state;
        }
        gpu_atomics::syncwarp(wcx.active_mask);
        Some(())
    } else {
        None
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for HybridFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === WarpFuture I/O: cooperative PRINT "hybrid: start" ===
            HYB_INIT_PRINT => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"hybrid: start",
                    HYB_WAIT_PRINT, &mut self.state, &mut self.pkt_idx,
                )
            },

            HYB_WAIT_PRINT => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    HYB_COMPUTE, &mut self.state,
                ).is_some() {
                    WarpPoll::Pending // continue to next state on next poll
                } else {
                    WarpPoll::Pending
                }
            },

            // === Per-thread compute block ===
            // All lanes enter together (state is broadcast), but each lane
            // computes independently. syncwarp() at exit ensures reconvergence
            // before transitioning back to WarpFuture I/O.
            HYB_COMPUTE => unsafe {
                let lid = wcx.lane_id;

                // --- Per-thread divergent computation ---
                // Each lane computes a different value: lane_id^2 + 1
                // In a real workload, this could be any per-lane logic with
                // different iteration counts, branches, etc.
                let value = lid * lid + 1;

                // Each lane writes its result independently
                core::ptr::write_volatile(
                    self.results.add(lid as usize),
                    value,
                );

                // --- End per-thread block ---
                // syncwarp: reconverge all lanes before returning to WarpFuture mode
                gpu_atomics::syncwarp(wcx.active_mask);

                // Lane 0 transitions state
                if wcx.is_leader() {
                    self.state = HYB_INIT_PRINT2;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === WarpFuture I/O: cooperative PRINT "hybrid: done" ===
            HYB_INIT_PRINT2 => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"hybrid: done",
                    HYB_WAIT_PRINT2, &mut self.state, &mut self.pkt_idx,
                )
            },

            HYB_WAIT_PRINT2 => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    HYB_DONE, &mut self.state,
                ).is_some() {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            HYB_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// hybrid-executor.1 kernel: WarpFuture PRINT → per-thread compute → WarpFuture PRINT
///
/// `buf` = hostcall buffer
/// `results` = output u32[32] array (one per lane)
/// `status` = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hybrid_executor_test(
    buf: *mut u8,
    results: *mut u32,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = HybridFuture::new(buf, results);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// hybrid-executor.2: Variable-duration + multi-switch stress test
// ============================================================
//
// 3 I/O phases + 2 compute blocks, testing:
// - Variable-duration per-thread work (lane_id-dependent iteration count)
// - Multiple switching points in one state machine
// - 11-state machine: INIT1→WAIT1→COMPUTE1→INIT2→WAIT2→COMPUTE2→INIT3→WAIT3→DONE
//
// COMPUTE1: sum 1..=(lane_id*100+1), ~100x duration variance across lanes
// COMPUTE2: XOR-fold lane_id-dependent seed, different duration per lane

const HYB2_INIT1: u32 = 0;
const HYB2_WAIT1: u32 = 1;
const HYB2_COMPUTE1: u32 = 2;
const HYB2_INIT2: u32 = 3;
const HYB2_WAIT2: u32 = 4;
const HYB2_COMPUTE2: u32 = 5;
const HYB2_INIT3: u32 = 6;
const HYB2_WAIT3: u32 = 7;
const HYB2_DONE: u32 = 8;

struct HybridStressFuture {
    buf: *mut u8,
    results: *mut u32,
    state: u32,
    pkt_idx: u16,
}

impl HybridStressFuture {
    #[inline(always)]
    fn new(buf: *mut u8, results: *mut u32) -> Self {
        Self {
            buf,
            results,
            state: HYB2_INIT1,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for HybridStressFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === Phase 1: WarpFuture PRINT "stress: phase1" ===
            HYB2_INIT1 => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"stress: phase1",
                    HYB2_WAIT1, &mut self.state, &mut self.pkt_idx,
                )
            },
            HYB2_WAIT1 => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    HYB2_COMPUTE1, &mut self.state,
                ).is_some() {
                    WarpPoll::Pending
                } else {
                    WarpPoll::Pending
                }
            },

            // === COMPUTE1: Variable-duration sum ===
            // Each lane sums 1..=(lane_id*100+1)
            // Lane 0: 1 iteration, Lane 31: 3101 iterations (~3100x variance)
            HYB2_COMPUTE1 => unsafe {
                let lid = wcx.lane_id;
                let iters = lid * 100 + 1;
                let mut sum: u32 = 0;
                let mut i: u32 = 1;
                while i <= iters {
                    sum = sum.wrapping_add(i);
                    i += 1;
                }
                // Write result: results[lane_id]
                core::ptr::write_volatile(self.results.add(lid as usize), sum);

                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = HYB2_INIT2;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Phase 2: WarpFuture PRINT "stress: phase2" ===
            HYB2_INIT2 => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"stress: phase2",
                    HYB2_WAIT2, &mut self.state, &mut self.pkt_idx,
                )
            },
            HYB2_WAIT2 => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    HYB2_COMPUTE2, &mut self.state,
                ).is_some() {
                    WarpPoll::Pending
                } else {
                    WarpPoll::Pending
                }
            },

            // === COMPUTE2: XOR-fold with lane-dependent iteration count ===
            // Each lane XOR-folds a different seed for (lane_id+1)*50 iterations
            HYB2_COMPUTE2 => unsafe {
                let lid = wcx.lane_id;
                let iters = (lid + 1) * 50;
                let mut val: u32 = 0xDEAD_0000 | lid;
                let mut i: u32 = 0;
                while i < iters {
                    val ^= val << 13;
                    val ^= val >> 17;
                    val ^= val << 5;
                    i += 1;
                }
                // Write result: results[32 + lane_id]
                core::ptr::write_volatile(self.results.add(32 + lid as usize), val);

                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = HYB2_INIT3;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Phase 3: WarpFuture PRINT "stress: phase3" ===
            HYB2_INIT3 => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"stress: phase3",
                    HYB2_WAIT3, &mut self.state, &mut self.pkt_idx,
                )
            },
            HYB2_WAIT3 => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    HYB2_DONE, &mut self.state,
                ).is_some() {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            HYB2_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// hybrid-executor.2 kernel: stress test with variable-duration per-thread compute + multi-switch
///
/// `buf` = hostcall buffer
/// `results` = output u32[64] array (32 per compute phase)
/// `status` = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hybrid_stress_test(
    buf: *mut u8,
    results: *mut u32,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = HybridStressFuture::new(buf, results);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// async-pipeline: Warp-cooperative hostcall helpers
// ============================================================

/// General-purpose warp-cooperative hostcall submit.
/// Lane 0 pops a packet, fills payload via closure (lane 0 only), submits to ready stack.
/// All lanes participate in broadcast of packet index.
#[inline(always)]
unsafe fn warp_hostcall_submit(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    service: u32,
    fill_payload: impl FnOnce(*mut u8),
    next_state: u32,
    state_cell: &mut u32,
    pkt_idx_cell: &mut u16,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

    let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = gpu_runtime::hostcall::hc_pop_free(buf) as u32;
    }
    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == gpu_protocol::NULL_INDEX {
        return WarpPoll::Pending;
    }
    *pkt_idx_cell = idx;

    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

    // Only lane 0 fills the payload
    if wcx.is_leader() {
        fill_payload(payload);
    }

    gpu_atomics::syncwarp(wcx.active_mask);

    if wcx.is_leader() {
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
            wcx.active_mask,
        );
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
            service,
        );
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            gpu_protocol::CONTROL_FILLED,
        );
        let (num_shards, shard_off, _) =
            gpu_runtime::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64, 1);
        *state_cell = next_state;
    }

    gpu_atomics::syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// General-purpose warp-cooperative wait. Returns Some(u64) from payload slot 0 when ready.
/// Releases the packet and transitions state on completion.
#[inline(always)]
unsafe fn warp_hostcall_wait_u64(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state_cell: &mut u32,
) -> Option<u64> {
    use gpu_runtime::warp_future::broadcast_u32;

    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(
        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32,
    );

    if ctrl & gpu_protocol::CONTROL_READY != 0 {
        let mut val: u64 = 0;
        if wcx.is_leader() {
            val = core::ptr::read_volatile(
                pkt.add(gpu_protocol::PKT_OFF_PAYLOAD) as *const u64,
            );
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            *state_cell = next_state;
        }
        // Broadcast u64 as two u32 halves
        let lo = broadcast_u32(wcx.active_mask, val as u32) as u64;
        let hi = broadcast_u32(wcx.active_mask, (val >> 32) as u32) as u64;
        gpu_atomics::syncwarp(wcx.active_mask);
        Some(lo | (hi << 32))
    } else {
        None
    }
}

// ============================================================
// async-pipeline: File transform demo — 16-state WarpFuture
// ============================================================
//
// GPU-autonomous pipeline: open → read → transform → open → write → close → close → print
// All I/O is warp-cooperative. Compute is per-thread divergent.
// One kernel launch, zero CPU intervention between steps.

const FTP_OPEN_IN: u32 = 0;
const FTP_WAIT_OPEN_IN: u32 = 1;
const FTP_BULK_READ: u32 = 2;
const FTP_WAIT_READ: u32 = 3;
const FTP_COMPUTE: u32 = 4;
const FTP_OPEN_OUT: u32 = 5;
const FTP_WAIT_OPEN_OUT: u32 = 6;
const FTP_BULK_WRITE: u32 = 7;
const FTP_WAIT_WRITE: u32 = 8;
const FTP_CLOSE_IN: u32 = 9;
const FTP_WAIT_CLOSE_IN: u32 = 10;
const FTP_CLOSE_OUT: u32 = 11;
const FTP_WAIT_CLOSE_OUT: u32 = 12;
const FTP_PRINT: u32 = 13;
const FTP_WAIT_PRINT: u32 = 14;
const FTP_DONE: u32 = 15;

/// Data size: 32 lanes × 32 bytes = 1024 bytes.
const FTP_DATA_SIZE: u64 = 1024;

struct FileTransformFuture {
    buf: *mut u8,
    sideband: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd_in: u64,
    fd_out: u64,
    sideband_offset: u64,
    bytes_read: u64,
}

impl FileTransformFuture {
    unsafe fn new(buf: *mut u8, sideband: *mut u8) -> Self {
        Self {
            buf,
            sideband,
            state: FTP_OPEN_IN,
            pkt_idx: gpu_protocol::NULL_INDEX,
            fd_in: 0,
            fd_out: 0,
            sideband_offset: 0,
            bytes_read: 0,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for FileTransformFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === Step 1: Open input file ===
            FTP_OPEN_IN => unsafe {
                let path = b"gpu_input.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    FTP_WAIT_OPEN_IN, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_OPEN_IN => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_BULK_READ, &mut self.state,
                ) {
                    if wcx.is_leader() { self.fd_in = fd; }
                }
                WarpPoll::Pending
            },

            // === Step 2: Read data via sideband bulk transfer ===
            FTP_BULK_READ => unsafe {
                if wcx.is_leader() {
                    gpu_runtime::sideband::sideband_reset(self.sideband);
                    self.sideband_offset = gpu_runtime::sideband::sideband_alloc(
                        self.sideband, FTP_DATA_SIZE,
                    );
                }
                gpu_atomics::syncwarp(wcx.active_mask);

                let fd = self.fd_in;
                let sb_off = self.sideband_offset;
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, FTP_DATA_SIZE);
                    },
                    FTP_WAIT_READ, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_READ => unsafe {
                if let Some(n) = warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_COMPUTE, &mut self.state,
                ) {
                    if wcx.is_leader() { self.bytes_read = n; }
                }
                WarpPoll::Pending
            },

            // === Step 3: Per-thread compute — toggle ASCII case ===
            // Each lane processes its 32-byte slice of the sideband data in-place.
            // Divergent: each lane may process different byte counts.
            FTP_COMPUTE => unsafe {
                let lid = wcx.lane_id;
                let offset = broadcast_u32(wcx.active_mask, self.sideband_offset as u32) as usize;
                let data_base = self.sideband.add(
                    gpu_protocol::SIDEBAND_DATA_OFFSET + offset,
                );
                let lane_base = data_base.add(lid as usize * 32);
                let bytes_read = broadcast_u32(wcx.active_mask, self.bytes_read as u32);
                let lane_start = lid * 32;

                let mut i: u32 = 0;
                while i < 32 && lane_start + i < bytes_read {
                    let b = core::ptr::read_volatile(lane_base.add(i as usize));
                    let toggled = if (b >= b'A' && b <= b'Z') || (b >= b'a' && b <= b'z') {
                        b ^ 0x20
                    } else {
                        b
                    };
                    core::ptr::write_volatile(lane_base.add(i as usize), toggled);
                    i += 1;
                }

                // Flush all lanes' sideband writes to system visibility
                membar_sys();
                gpu_atomics::syncwarp(wcx.active_mask);

                if wcx.is_leader() { self.state = FTP_OPEN_OUT; }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Step 4: Open output file ===
            FTP_OPEN_OUT => unsafe {
                let path = b"gpu_output.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    FTP_WAIT_OPEN_OUT, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_OPEN_OUT => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_BULK_WRITE, &mut self.state,
                ) {
                    if wcx.is_leader() { self.fd_out = fd; }
                }
                WarpPoll::Pending
            },

            // === Step 5: Write transformed data via sideband ===
            FTP_BULK_WRITE => unsafe {
                let fd = self.fd_out;
                let sb_off = self.sideband_offset;
                let len = self.bytes_read;
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_BULK_WRITE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, len);
                    },
                    FTP_WAIT_WRITE, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_WRITE => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_CLOSE_IN, &mut self.state,
                ).is_some() {}
                WarpPoll::Pending
            },

            // === Step 6: Close input file ===
            FTP_CLOSE_IN => unsafe {
                let fd = self.fd_in;
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    FTP_WAIT_CLOSE_IN, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_CLOSE_IN => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_CLOSE_OUT, &mut self.state,
                ).is_some() {}
                WarpPoll::Pending
            },

            // === Step 7: Close output file ===
            FTP_CLOSE_OUT => unsafe {
                let fd = self.fd_out;
                warp_hostcall_submit(
                    self.buf, wcx, SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    FTP_WAIT_CLOSE_OUT, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_CLOSE_OUT => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    FTP_PRINT, &mut self.state,
                ).is_some() {}
                WarpPoll::Pending
            },

            // === Step 8: Print completion message ===
            FTP_PRINT => unsafe {
                hybrid_warp_print_init(
                    self.buf, wcx, b"pipeline: done",
                    FTP_WAIT_PRINT, &mut self.state, &mut self.pkt_idx,
                )
            },

            FTP_WAIT_PRINT => unsafe {
                if hybrid_warp_wait(
                    self.buf, wcx, self.pkt_idx,
                    FTP_DONE, &mut self.state,
                ).is_some() {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            FTP_DONE => WarpPoll::Ready(true),

            _ => WarpPoll::Ready(false),
        }
    }
}

/// async-pipeline demo kernel: GPU-autonomous file transform pipeline.
///
/// The GPU self-coordinates 8 I/O steps + 1 compute step in a single kernel launch:
///   open(in) → read(in) → transform → open(out) → write(out) → close(in) → close(out) → print
///
/// No CPU intervention between steps — the GPU drives the entire pipeline via WarpFuture.
///
/// `buf`      = hostcall buffer (CUDA mapped memory)
/// `sideband` = sideband buffer for bulk data transfer (CUDA mapped memory)
/// `status`   = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn file_transform_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = FileTransformFuture::new(buf, sideband);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}
