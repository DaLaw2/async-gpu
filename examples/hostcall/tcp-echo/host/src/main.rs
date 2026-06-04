//! TCP Echo -- host binary demonstrating GPU-initiated TCP networking.
//!
//! 1. Starts a local TCP echo server on a random port
//! 2. Launches a GPU kernel that connects, sends a message, reads the echo
//! 3. Verifies the kernel received the correct response
//!
//! Uses the `gpu::custom()` builder API with hostcall for GPU-host RPC
//! and a mapped buffer for the kernel's output.

use gpu_host::gpu;
use std::io::{Read, Write};
use std::net::TcpListener;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

/// Expected message the GPU kernel sends.
const EXPECTED_MSG: &str = "Hello from GPU!";

fn main() -> gpu_host::Result<()> {
    println!("=== TCP Echo Example ===\n");

    // Step 1: Start a TCP echo server on a random port
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");
    let port = listener.local_addr().unwrap().port();
    println!("[host] TCP echo server listening on 127.0.0.1:{port}");

    // Spawn the echo server in a background thread.
    let echo_handle = std::thread::spawn(move || {
        let (mut stream, peer) = listener.accept().expect("Failed to accept connection");
        println!("[echo] Accepted connection from {peer}");

        let mut buf = [0u8; 256];
        let n = stream.read(&mut buf).expect("Failed to read from client");
        println!(
            "[echo] Received {} bytes: {:?}",
            n,
            std::str::from_utf8(&buf[..n]).unwrap_or("<invalid utf8>")
        );

        stream.write_all(&buf[..n]).expect("Failed to echo data");
        println!("[echo] Echoed {n} bytes back");
    });

    // Step 2: Prepare GPU context with hostcall
    let ctx = gpu::custom("tcp_echo_kernel")
        .ptx(KERNEL_PTX)
        .threads(32)
        .hostcall_packets(8)
        .prepare()?;

    let mut output_buf = ctx.mapped_buffer::<u32>(1)?;
    unsafe { output_buf.write(0, 0) };

    // Extract pointers before launch (they're Copy u64 values)
    let hc_ptr = ctx.hostcall_ptr();
    let sb_ptr = ctx.sideband_ptr();
    let out_ptr = output_buf.dev_ptr() as u64;

    // Step 3: Launch kernel
    println!("--- TCP Echo: GPU connects, sends, reads echo ---");
    let gpu_result = unsafe { ctx.launch((hc_ptr, sb_ptr, port as u32, out_ptr))? };

    std::thread::sleep(std::time::Duration::from_millis(50));
    let result_val = unsafe { output_buf.read(0) };

    if result_val == 0xDEAD {
        println!("[host] tcp_echo_kernel: FAILED (error sentinel 0xDEAD)");
    } else {
        let expected_len = EXPECTED_MSG.len() as u32;
        let passed = result_val == expected_len;
        println!(
            "[host] tcp_echo_kernel: {} (response length: {}, expected: {})",
            if passed { "PASSED" } else { "FAILED" },
            result_val,
            expected_len,
        );
    }
    println!();

    // Drop mapped buffer before GpuResult to ensure CUDA context is still alive
    drop(output_buf);
    gpu_result.finish();

    // Wait for the echo server thread to finish
    echo_handle.join().expect("Echo server thread panicked");

    println!("=== TCP Echo example complete! ===");
    Ok(())
}
