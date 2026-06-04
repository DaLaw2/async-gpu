// Round 3: Test gpu-libc + alloc integration on nvptx64.
//
// This tests:
// 1. gpu-libc functions are accessible via cross-crate LTO
// 2. alloc works with gpu-libc's bump allocator
// 3. format! macro works (core::fmt is no_std compatible)

#![no_std]
#![feature(abi_gpu_kernel)]

extern crate alloc;
extern crate gpu_libc;

use alloc::string::String;
use alloc::vec;
use alloc::format;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { gpu_libc::abort() }
}

// Use gpu-libc's malloc/free as the global allocator
use core::alloc::{GlobalAlloc, Layout};

struct GpuAllocator;

unsafe impl GlobalAlloc for GpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= gpu_libc::MALLOC_ALIGN {
            gpu_libc::malloc(layout.size()) as *mut u8
        } else {
            let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
            let ret = gpu_libc::posix_memalign(&mut ptr, layout.align(), layout.size());
            if ret == 0 {
                ptr as *mut u8
            } else {
                core::ptr::null_mut()
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        gpu_libc::free(ptr as *mut core::ffi::c_void);
    }

    unsafe fn realloc(&self, ptr: *mut u8, _layout: Layout, new_size: usize) -> *mut u8 {
        gpu_libc::realloc(ptr as *mut core::ffi::c_void, new_size) as *mut u8
    }
}

#[global_allocator]
static ALLOCATOR: GpuAllocator = GpuAllocator;

/// Initialize the GPU heap, then test Vec allocation.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_alloc_vec(heap: *mut u8, heap_size: u64, output: *mut u32) {
    // Initialize bump allocator
    gpu_libc::gpu_heap_init(heap, heap_size as usize);

    // Test: create a Vec and push some values
    let v = vec![1u32, 2, 3, 4, 5];
    *output = v.iter().sum::<u32>();
}

/// Test format! macro with GPU heap.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_format(heap: *mut u8, heap_size: u64, output: *mut u32) {
    gpu_libc::gpu_heap_init(heap, heap_size as usize);

    let s = format!("value = {}", 42u32);
    *output = s.len() as u32;
}

/// Test String operations.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_string(heap: *mut u8, heap_size: u64, output: *mut u32) {
    gpu_libc::gpu_heap_init(heap, heap_size as usize);

    let mut s = String::from("Hello");
    s.push_str(", GPU!");
    *output = s.len() as u32;
}

/// Test calling gpu-libc's write stub (should return -1/ENOSYS).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_write_stub(output: *mut i64) {
    let msg = b"test\0";
    let result = gpu_libc::write(1, msg.as_ptr() as *const core::ffi::c_void, 4);
    *output = result as i64;
}
