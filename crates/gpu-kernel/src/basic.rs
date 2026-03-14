// Basic asm/atomic test kernels — all are `pub unsafe extern "ptx-kernel"` entry points.

use core::arch::nvptx;
use gpu_atomics::{
    activemask, lane_id, membar_sys, st_global_u32, sys_cas_u32, sys_cas_u64, sys_exchange_u64,
    sys_fetch_add_u64, sys_load_acquire_u32, sys_spin_load_acquire_u32, sys_store_release_u32,
};

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
pub unsafe extern "ptx-kernel" fn vector_add(a: *const f32, b: *const f32, c: *mut f32, len: u32) {
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
// Multi-thread malloc test (std-hardening.3)
// ============================================================

/// Test: concurrent malloc from 32 threads.
///
/// Each thread calls gpu_libc::malloc(64) and writes the returned pointer
/// to output[tid]. The host verifies that all 32 pointers are non-null
/// and non-overlapping (each allocation is 64 bytes apart at minimum).
///
/// Launch with: block_dim=(32,1,1), grid_dim=(1,1,1)
/// Args: heap (ptr to heap region), heap_size (u64), output (*mut u64, len>=32)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_multithread_malloc(
    heap: *mut u8,
    heap_size: u64,
    output: *mut u64,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    // Thread 0 initializes the heap (all threads must see this before malloc)
    if tid == 0 {
        gpu_libc::gpu_heap_init(heap, heap_size as usize);
    }
    // Barrier: ensure heap init is visible to all threads
    core::arch::asm!("bar.sync 0;");

    // Each thread allocates 64 bytes
    let ptr = gpu_libc::malloc(64);
    *output.add(tid as usize) = ptr as u64;

    // Write a unique pattern to verify no overlap
    if !ptr.is_null() {
        let p = ptr as *mut u32;
        *p = tid; // first 4 bytes = thread ID
    }
}
