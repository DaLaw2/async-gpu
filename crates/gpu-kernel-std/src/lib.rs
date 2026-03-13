// GPU kernels using real Rust std (println!, Vec, String, format!).
//
// Unlike std-build-test which duplicates 430+ lines of hostcall inline PTX,
// this crate depends on gpu-runtime for the hostcall protocol implementation.
// This eliminates code duplication and makes std kernels first-class citizens.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_ptx)]

use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Global hostcall buffer pointer for stdio. Set by kernel at entry.
static STDIO_HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);

/// External function called by std's CUDA PAL Stdout::write().
/// Routes through gpu-runtime's hostcall PRINT implementation.
#[unsafe(no_mangle)]
pub fn gpu_stdout_write(buf: *const u8, len: usize) -> usize {
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || buf.is_null() || len == 0 {
        return len; // silently discard if no hostcall buffer set
    }
    // Send via gpu-runtime's hostcall PRINT service (56-byte chunks)
    const MAX_CHUNK: usize = 56;
    let mut offset = 0usize;
    while offset < len {
        let chunk_len = core::cmp::min(len - offset, MAX_CHUNK);
        let result = unsafe {
            gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, buf.add(offset), chunk_len as u32)
        };
        if result.is_err() {
            return offset; // partial write on failure
        }
        offset += chunk_len;
    }
    len
}

/// External function called by std's CUDA PAL Stdin::read().
/// Currently returns 0 (EOF) — stdin support to be added when gpu-runtime
/// exposes a stdin hostcall function.
#[unsafe(no_mangle)]
pub fn gpu_stdin_read(_out_buf: *mut u8, _max_len: usize) -> usize {
    // TODO: implement via gpu-runtime hostcall when SERVICE_STDIN is exposed
    0
}

/// Set the hostcall buffer pointer for stdio. Must be called at kernel entry.
fn stdio_init(buf: *mut u8) {
    STDIO_HOSTCALL_BUF.store(buf as u64, AtomicOrdering::Relaxed);
}

// ============================================================
// Test kernels — demonstrate real std on GPU
// ============================================================

/// Test kernel: println! via patched std (no custom macros).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_println_test(buf: *mut u8) {
    stdio_init(buf);

    println!("Hello from gpu-kernel-std!");
    println!("This uses real Rust std println!, not a custom macro.");
}

/// Test kernel: Vec + String + format! via std allocator.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_vec_format_test(buf: *mut u8) {
    stdio_init(buf);

    // Vec allocation and manipulation
    let mut v: Vec<i32> = Vec::new();
    for i in 0..10 {
        v.push(i * i);
    }
    println!("Vec contents: {:?}", v);
    println!("Vec len={}, capacity={}", v.len(), v.capacity());

    // String formatting
    let name = String::from("GPU");
    let msg = format!("Hello from {}! Vec sum = {}", name, v.iter().sum::<i32>());
    println!("{}", msg);

    // Vec with_capacity
    let mut v2: Vec<u8> = Vec::with_capacity(64);
    v2.extend_from_slice(b"gpu-kernel-std works!");
    println!("String from bytes: {}", core::str::from_utf8(&v2).unwrap());
}

/// Test kernel: multiple allocations and drops to verify allocator.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_alloc_stress_test(buf: *mut u8) {
    stdio_init(buf);

    // Allocate and drop multiple Vecs to test allocator reuse
    for i in 0..5u32 {
        let v: Vec<u32> = (0..20).map(|x| x + i * 100).collect();
        let sum: u32 = v.iter().sum();
        println!("Round {}: Vec[20] sum = {}", i, sum);
        // v is dropped here — allocator should reclaim
    }
    println!("Alloc stress test complete — 5 rounds of alloc/drop");
}
