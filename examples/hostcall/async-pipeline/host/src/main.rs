//! Async Pipeline — warp-cooperative GPU pipelines in one-liner calls.
//!
//! Demonstrates GPU kernels that use async/await-style state machines
//! (WarpFuture) to overlap I/O with computation. Each demo is a single
//! function call — no manual PTX loading or buffer management.
//!
//! Two demos:
//! 1. Branching pipeline — conditional file creation based on runtime data
//! 2. Pipelined compute — overlaps GPU computation with pending file I/O

use gpu_host::gpu;

fn main() -> gpu_host::Result<()> {
    println!("=== Async Pipeline Example ===\n");

    // ---- Demo 1: Branching Pipeline ----
    // The kernel uses a WarpFuture state machine to:
    //   1. Open a file for writing
    //   2. Write data based on runtime conditions (branching)
    //   3. Close the file
    // All I/O is async — the warp yields while waiting for host responses.
    println!("--- Demo 1: Branching Pipeline ---");
    println!("[host] Launching branching_pipeline kernel...");
    let result: Vec<u32> = gpu::run_with_output("branching_pipeline", 1)?;
    let success = result[0] == 1;
    println!(
        "[host] branching_pipeline: {}\n",
        if success { "PASSED" } else { "FAILED" }
    );

    // ---- Demo 2: Pipelined Compute ----
    // The kernel overlaps computation with I/O:
    //   1. Submit file open (async)
    //   2. While waiting, do GPU-local computation
    //   3. Wait for file open to complete
    //   4. Write results
    // This is impossible with CUDA graphs — the compute happens
    // *between* I/O submit and I/O complete.
    println!("--- Demo 2: Pipelined Compute ---");
    println!("[host] Launching pipelined_compute kernel...");
    let result: Vec<u32> = gpu::run_with_output("pipelined_compute", 1)?;
    let success = result[0] == 1;
    println!(
        "[host] pipelined_compute: {}\n",
        if success { "PASSED" } else { "FAILED" }
    );

    // Clean up files created by the kernels
    let _ = std::fs::remove_file("branching_output.txt");
    let _ = std::fs::remove_file("pipelined_output.txt");

    println!("=== Async Pipeline Example Complete ===");
    Ok(())
}
