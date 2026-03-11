#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]
use core::arch::nvptx;
use core::panic::PanicInfo;
use gpu_atomics::{
    membar_sys, sys_store_release_u32, sys_load_acquire_u32, sys_cas_u32, st_global_u32,
    sys_cas_u64, sys_fetch_add_u64, sys_exchange_u64,
    sys_load_acquire_u64, sys_spin_load_acquire_u32, activemask, lane_id,
};
use gpu_protocol::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

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

    // Step 4: membar.sys to ensure all packet writes are visible at system scope
    membar_sys();

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
