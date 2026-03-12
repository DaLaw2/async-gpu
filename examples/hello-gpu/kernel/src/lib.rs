//! Hello GPU — minimal kernel using gpu-runtime.
//!
//! Demonstrates a single PRINT hostcall from GPU to host.

#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::panic::PanicInfo;
use gpu_runtime::prelude::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// A simple kernel that sends a message to the host via the hostcall protocol.
///
/// Only thread 0 performs the hostcall. All other threads return immediately.
///
/// # Arguments
/// * `buf` — Pointer to the hostcall buffer (device-mapped pinned memory)
/// * `result` — Pointer to a u32 where thread 0 writes 1 (success) or 0 (failure)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hello_gpu(buf: *mut u8, result: *mut u32) {
    let thread_x = core::arch::nvptx::_thread_idx_x() as u32;
    let block_x = core::arch::nvptx::_block_idx_x() as u32;
    let block_dim_x = core::arch::nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;

    if global_idx != 0 {
        return;
    }

    let msg = b"Hello from GPU via gpu-runtime!";
    let ok = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

/// Vector addition kernel — no hostcall needed, just pure compute.
///
/// Each thread computes `c[i] = a[i] + b[i]`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vector_add(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: u32,
) {
    let thread_x = core::arch::nvptx::_thread_idx_x() as u32;
    let block_x = core::arch::nvptx::_block_idx_x() as u32;
    let block_dim_x = core::arch::nvptx::_block_dim_x() as u32;

    let idx = block_x * block_dim_x + thread_x;
    if idx < len {
        let val = *a.add(idx as usize) + *b.add(idx as usize);
        *c.add(idx as usize) = val;
    }
}
