// Test: can we compile with -Zbuild-std=std for nvptx64?
// Requires patched std source with cuda target support.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_ptx)]

use std::io::Write;

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_hello_kernel(result: *mut u32) {
    // Test that std types are available
    let v = vec![1u32, 2, 3, 4, 5];
    let sum: u32 = v.iter().sum();

    // Test String from std (uses alloc)
    let s = String::from("Hello from GPU std!");
    let len = s.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, sum + len);
    }
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_format_kernel(result: *mut u32) {
    // Test format! macro (uses alloc + fmt)
    let formatted = format!("value = {}", 42u32);
    let len = formatted.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, len);
    }
}
