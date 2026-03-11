#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]
#![feature(link_llvm_intrinsics)]

use core::arch::nvptx;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// ============================================================
// Step 1: Inline PTX asm test — CRITICAL EXPERIMENT
// ============================================================

/// Test: membar.sys via inline PTX asm
/// If this compiles, inline PTX asm works on nvptx64.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_membar_sys(output: *mut u32, len: u32) {
    core::arch::asm!("membar.sys;", options(nostack));
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        *output.add(idx as usize) = 0xDEAD_BEEFu32;
    }
}

/// Test: st.release.sys.global.u32 via inline PTX asm
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_st_release_sys(ptr: *mut u32, val: u32) {
    core::arch::asm!(
        "st.release.sys.global.u32 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
}

/// Test: ld.acquire.sys.global.u32 via inline PTX asm
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_ld_acquire_sys(ptr: *const u32, output: *mut u32) {
    let result: u32;
    core::arch::asm!(
        "ld.acquire.sys.global.u32 {result}, [{ptr}];",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        options(nostack, readonly),
    );
    core::arch::asm!(
        "st.global.b32 [{output}], {result};",
        output = in(reg64) output,
        result = in(reg32) result,
        options(nostack),
    );
}

/// Test: atom.cas.sys.global.b32 via inline PTX asm
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_asm_cas_sys(
    ptr: *mut u32,
    expected: u32,
    desired: u32,
    output: *mut u32,
) {
    let result: u32;
    core::arch::asm!(
        "atom.cas.sys.global.b32 {result}, [{ptr}], {expected}, {desired};",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        expected = in(reg32) expected,
        desired = in(reg32) desired,
        options(nostack),
    );
    *output = result;
}

// ============================================================
// Step 2 fallback: NVVM intrinsics via extern "C"
// (tested if inline asm fails)
// ============================================================

extern "C" {
    #[link_name = "llvm.nvvm.membar.sys"]
    fn nvvm_membar_sys();

    #[link_name = "llvm.nvvm.atomic.add.gen.i.sys.i32.p0i32"]
    fn nvvm_atomic_add_sys_i32(ptr: *mut i32, val: i32) -> i32;
}

/// Test: membar.sys via llvm.nvvm.membar.sys intrinsic
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_nvvm_membar_sys(flag: *mut u32) {
    nvvm_membar_sys();
    *flag = 1u32;
}

/// Test: scoped atomic add via llvm.nvvm.atomic.add.gen.i.sys
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_nvvm_atomic_add_sys(ptr: *mut i32, val: i32, output: *mut i32) {
    let result = nvvm_atomic_add_sys_i32(ptr, val);
    *output = result;
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
// Step 5: Integration kernel using sys atomics
// ============================================================

/// Integration test kernel:
/// 1. Write a value to data_ptr using st.release.sys.global.u32
/// 2. Set flag_ptr = 1 using st.release.sys.global.u32
/// Host reads flag and verifies data.
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
        core::arch::asm!(
            "st.release.sys.global.u32 [{ptr}], {val};",
            ptr = in(reg64) data_ptr,
            val = in(reg32) value,
            options(nostack),
        );
        // Fence for extra safety
        core::arch::asm!("membar.sys;", options(nostack));
        // Set flag with system-scope release
        core::arch::asm!(
            "st.release.sys.global.u32 [{ptr}], {val};",
            ptr = in(reg64) flag_ptr,
            val = in(reg32) 1u32,
            options(nostack),
        );
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
