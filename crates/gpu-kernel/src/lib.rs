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
// Hostcall protocol (GPU side) — hostcall.4
// ============================================================

/// Pop a packet from the free stack. Returns packet index or NULL_INDEX.
#[inline(always)]
unsafe fn hc_pop_free(buf: *mut u8) -> u16 {
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return NULL_INDEX;
        }
        let pkt = buf.add(packet_offset(idx));
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if sys_cas_u64(free_ptr, old_head, next) == old_head {
            return idx;
        }
    }
}

/// Push a packet onto a tagged-pointer stack (free or ready).
#[inline(always)]
unsafe fn hc_push(stack_ptr: *mut u64, buf: *mut u8, pkt_idx: u16) {
    let pkt = buf.add(packet_offset(pkt_idx));
    loop {
        let old_head = sys_load_acquire_u64(stack_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if sys_cas_u64(stack_ptr, old_head, new_tagged) == old_head {
            break;
        }
    }
}

/// GPU-side hostcall: send a PRINT request with a short message.
///
/// Only lane 0 (thread 0) should call this. The message is copied into
/// the packet payload (mapped memory). Max 56 bytes.
///
/// Returns true on success, false on pool exhaustion or timeout.
#[inline(always)]
unsafe fn gpu_hostcall_print(buf: *mut u8, msg: *const u8, msg_len: u32) -> bool {
    // Step 1: Pop free packet
    let pkt_idx = hc_pop_free(buf);
    if pkt_idx == NULL_INDEX {
        return false;
    }

    let pkt = buf.add(packet_offset(pkt_idx));

    // Step 2: Fill packet header
    let mask = activemask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
    // Clear READY/ERROR with a release store (ensures prior state is clean)
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Step 3: Fill payload (lane 0 only)
    // Slot 0 = message length, Slots 1-7 = message bytes (up to 56 bytes)
    let payload = pkt.add(PKT_OFF_PAYLOAD);
    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

    let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
        PRINT_MAX_MSG_LEN as u32
    } else {
        msg_len
    };
    let dst = payload.add(8); // skip slot 0
    let mut i: u32 = 0;
    while i < copy_len {
        core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
        i += 1;
    }

    // Step 4: Mark packet as filled (release store ensures all prior writes visible)
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

    // Step 5: Push to ready stack
    let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
    hc_push(ready_ptr, buf, pkt_idx);

    // Step 6: Ring doorbell
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    // Step 7: Spin-wait for host response
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
    let mut spins: u32 = 0;
    let success;
    loop {
        let ctrl = sys_spin_load_acquire_u32(control_ptr);
        if ctrl & CONTROL_READY != 0 {
            success = true;
            break;
        }
        spins += 1;
        if spins >= GPU_MAX_SPIN {
            success = false;
            break;
        }
    }

    // Step 8: Return packet to free stack
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    hc_push(free_ptr, buf, pkt_idx);

    success
}

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
    let ok = gpu_hostcall_print(buf, msg.as_ptr(), 15);
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

    let ok = gpu_hostcall_print(buf, msg_buf.as_ptr(), pos as u32);
    if ok {
        gpu_atomics::sys_fetch_add_u32(success_count, 1);
    }
}

// ============================================================
// File I/O hostcall helpers (gpu-std.3)
// ============================================================

/// Generic hostcall: allocate packet, set service, fill payload via callback,
/// push to ready stack, ring doorbell, spin-wait for response.
/// Returns true on success. On success, the payload contains the host's response.
#[inline(always)]
unsafe fn gpu_hostcall_request(
    buf: *mut u8,
    service: u32,
    fill_payload: impl FnOnce(*mut u8),
) -> (*mut u8, bool) {
    // Step 1: Pop free packet
    let pkt_idx = hc_pop_free(buf);
    if pkt_idx == NULL_INDEX {
        return (core::ptr::null_mut(), false);
    }

    let pkt = buf.add(packet_offset(pkt_idx));

    // Step 2: Fill packet header
    let mask = activemask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Step 3: Fill payload
    fill_payload(pkt.add(PKT_OFF_PAYLOAD));

    // Step 4: Mark packet as filled (release store ensures all prior writes visible)
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

    // Step 5: Push to ready stack
    let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
    hc_push(ready_ptr, buf, pkt_idx);

    // Step 6: Ring doorbell
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    // Step 7: Spin-wait for host response
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
    let mut spins: u32 = 0;
    let success;
    loop {
        let ctrl = sys_spin_load_acquire_u32(control_ptr);
        if ctrl & CONTROL_READY != 0 {
            success = (ctrl & CONTROL_ERROR) == 0;
            break;
        }
        spins += 1;
        if spins >= GPU_MAX_SPIN {
            // Timeout — return packet to free stack
            let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
            hc_push(free_ptr, buf, pkt_idx);
            return (core::ptr::null_mut(), false);
        }
    }

    // Do NOT return packet yet — caller needs to read response payload
    (pkt, success)
}

/// Return a packet to the free stack after reading response.
#[inline(always)]
unsafe fn gpu_hostcall_release(buf: *mut u8, pkt: *mut u8) {
    // Calculate packet index from pointer offset
    let offset = (pkt as usize) - (buf as usize) - BUFFER_HEADER_SIZE;
    let pkt_idx = (offset / PACKET_SIZE) as u16;
    let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    hc_push(free_ptr, buf, pkt_idx);
}

/// GPU-side hostcall: open a file.
/// Returns `(fd, 0)` on success, `(0, error_category)` on failure.
#[inline(always)]
unsafe fn gpu_hostcall_open(buf: *mut u8, path: *const u8, path_len: u32, flags: u32) -> (u64, u16) {
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
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
    gpu_hostcall_release(buf, pkt);

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
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
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
    gpu_hostcall_release(buf, pkt);

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
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

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
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_READ, |payload| {
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
        gpu_hostcall_release(buf, pkt);
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
    gpu_hostcall_release(buf, pkt);
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
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_STDIN, |payload| {
        // Slot 0: max bytes to read
        core::ptr::write_volatile(payload as *mut u64, max_len as u64);
    });

    if pkt.is_null() {
        return (0, ERR_HOST_TIMEOUT);
    }

    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);

    if !success {
        gpu_hostcall_release(buf, pkt);
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
    gpu_hostcall_release(buf, pkt);
    (slot0, 0)
}

/// GPU-side hostcall: get wall-clock time from host.
/// Returns (seconds_since_epoch, nanoseconds) on success, (0, 0) on failure.
#[inline(always)]
unsafe fn gpu_hostcall_time(buf: *mut u8) -> (u64, u64) {
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_TIME, |_payload| {
        // No request payload needed
    });

    if pkt.is_null() || !success {
        if !pkt.is_null() {
            gpu_hostcall_release(buf, pkt);
        }
        return (0, 0);
    }

    let secs = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    let nanos = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD).add(8) as *const u64);
    gpu_hostcall_release(buf, pkt);
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
        hc_push(ready_ptr, buf, pkt_idx);

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
        hc_push(free_ptr, buf, pkt_idx);

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
