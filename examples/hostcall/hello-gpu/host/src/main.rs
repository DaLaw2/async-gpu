//! Hello GPU — GPU capabilities in one-liner API calls.
//!
//! Each demo is a single function call. No manual PTX loading, no
//! buffer management, no launch configs. Just call and get results.
//!
//! Three demos:
//! 1. GPU println — kernel sends "Hello from GPU!" to host stdout
//! 2. GPU file I/O — kernel creates, writes, reads, and verifies a file
//! 3. GPU threading — thread::spawn on separate warps, join results

use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    println!("=== Hello GPU Example ===\n");

    // ---- Demo 1: GPU println ----
    // The kernel calls gpu_hostcall_print("Hello from GPU!").
    // One line: launch kernel, get success flag back.
    println!("--- Demo 1: GPU println ---");
    let result: Vec<u32> = gpu::run_with_output("hostcall_print_hello", 1)?;
    println!(
        "[host] hostcall_print_hello: {}\n",
        if result[0] == 1 { "PASSED" } else { "FAILED" }
    );

    // ---- Demo 2: GPU file I/O ----
    // The kernel opens a file, writes "Hello from GPU file I/O!",
    // reads it back, and verifies the content matches.
    // Four result slots: [success, fd, bytes_written, bytes_read].
    println!("--- Demo 2: GPU file I/O ---");
    let result: Vec<u32> = gpu::run_with_output("hostcall_file_test", 4)?;
    let success = result[0] == 1;
    println!("[host] File created and written from GPU");
    println!("[host] Bytes written: {}, bytes read back: {}", result[2], result[3]);
    println!(
        "[host] hostcall_file_test: {}\n",
        if success { "PASSED" } else { "FAILED" }
    );
    // Clean up the test file
    let _ = std::fs::remove_file("gpu_test_output.txt");

    // ---- Demo 3: GPU threading ----
    // The kernel spawns two threads on separate GPU warps:
    //   Thread 1 returns 42, Thread 2 returns 99.
    // Host gets results via a shared output buffer.
    // Result: [thread1_value, thread2_value, available_parallelism, main_warp_id]
    println!("--- Demo 3: GPU threading ---");
    let result: Vec<u32> = gpu::launch("thread_spawn_test", 4, 128)?;
    println!("[host] Thread 1 computed: {} (expected 42)", result[0]);
    println!("[host] Thread 2 computed: {} (expected 99)", result[1]);
    println!("[host] Available parallelism: {} warps", result[2]);
    println!("[host] Main thread: warp {}", result[3]);
    let ok = result[0] == 42 && result[1] == 99;
    println!(
        "[host] thread_spawn_test: {}\n",
        if ok { "PASSED" } else { "FAILED" }
    );

    println!("=== All demos complete! ===");
    Ok(())
}
