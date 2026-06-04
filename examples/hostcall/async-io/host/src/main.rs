//! Async I/O — GPU file operations in one-liner calls.
//!
//! Demonstrates GPU kernels performing file I/O via the hostcall system:
//! 1. File I/O — create, write, read, and verify a file from GPU
//! 2. Pipelined I/O — overlap computation with pending file operations
//!
//! Each demo is a single function call. The hostcall system transparently
//! bridges GPU file operations to the host filesystem.

use gpu_host::gpu;

fn main() -> gpu_host::Result<()> {
    println!("=== Async I/O Example ===\n");

    // ---- Demo 1: File I/O from GPU ----
    // The kernel: open → write "Hello from GPU file I/O!" → close → reopen → read → verify
    // Result: [success, fd, bytes_written, bytes_read]
    println!("--- Demo 1: File I/O (create + write + read + verify) ---");
    let result: Vec<u32> = gpu::run_with_output("hostcall_file_test", 4)?;
    let success = result[0] == 1;
    println!("[host] Written {} bytes, read back {} bytes", result[2], result[3]);
    println!("[host] Content verified: {}", if success { "match" } else { "MISMATCH" });
    if let Ok(content) = std::fs::read_to_string("gpu_test_output.txt") {
        println!("[host] File content: {:?}", content.trim());
    }
    println!(
        "[host] hostcall_file_test: {}\n",
        if success { "PASSED" } else { "FAILED" }
    );
    let _ = std::fs::remove_file("gpu_test_output.txt");

    // ---- Demo 2: Pipelined compute + I/O ----
    // The kernel overlaps GPU computation with pending file I/O:
    //   submit file open → compute while waiting → write results
    // This pattern is impossible with CUDA graphs (fixed execution trace).
    println!("--- Demo 2: Pipelined Compute + I/O ---");
    let result: Vec<u32> = gpu::run_with_output("pipelined_compute", 1)?;
    println!(
        "[host] pipelined_compute: {}\n",
        if result[0] == 1 { "PASSED" } else { "FAILED" }
    );
    let _ = std::fs::remove_file("pipelined_output.txt");

    println!("=== Async I/O Example Complete ===");
    Ok(())
}
