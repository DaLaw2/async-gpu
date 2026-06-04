//! Thread Demo — GPU threading that looks like CPU Rust.
//!
//! This example demonstrates async-gpu's threading model:
//! - Each GPU warp (32 lanes) acts as a single "thread"
//! - `thread::spawn()` wakes a sleeping warp and assigns work
//! - `JoinHandle::join()` blocks until the spawned thread completes
//! - `thread::available_parallelism()` returns the number of free warps
//!
//! The GPU kernel code (in crates/kernel/gpu-kernel/src/thread_test.rs):
//!
//! ```rust
//! use gpu_runtime::thread;
//!
//! pub unsafe extern "ptx-kernel" fn thread_spawn_test(result: *mut u32) {
//!     thread::gpu_main(|| {
//!         let h1 = thread::spawn(|| 42u32);
//!         let h2 = thread::spawn(|| 99u32);
//!         let r1 = h1.join();
//!         let r2 = h2.join();
//!         // Write results...
//!     });
//! }
//! ```
//!
//! The host side: ONE line to launch and get results.

use gpu_host::gpu;

fn main() {
    println!("=== Thread Demo: GPU threading like CPU Rust ===\n");

    // One line: launch the kernel, get results back.
    // 4 warps (128 threads): warp 0 = main, warps 1-3 = workers.
    let result: Vec<u32> = gpu::launch("thread_spawn_test", 4, 128)
        .expect("GPU launch failed");

    println!("Thread 1 computed: {}", result[0]); // 42
    println!("Thread 2 computed: {}", result[1]); // 99
    println!("Available parallelism: {} warps", result[2]); // 3
    println!("Main thread ID: warp {}", result[3]); // 0
    println!("\nAll threads completed. Results joined successfully.");

    // With reuse: spawn 4 tasks on 3 worker warps
    let result2: Vec<u32> = gpu::launch("thread_reuse_test", 5, 128)
        .expect("GPU launch failed");

    println!("\n--- Warp Reuse Demo ---");
    println!("Spawned 4 tasks on 3 warps (one warp reused):");
    for i in 0..4 {
        println!("  Task {}: computed {}", i, result2[i]);
    }
    println!("Total: {}", result2[4]);
}
