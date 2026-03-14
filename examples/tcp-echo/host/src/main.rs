//! TCP Echo -- host binary demonstrating GPU-initiated TCP networking.
//!
//! 1. Starts a local TCP echo server on a random port
//! 2. Launches a GPU kernel that connects, sends a message, reads the echo
//! 3. Verifies the kernel received the correct response
//!
//! Uses the three core SDK types:
//! - [`GpuRuntime`] -- device init, PTX loading, kernel launch
//! - [`HostcallBuffer`] -- GPU-host RPC communication
//! - [`MappedBuffer`] -- RAII pinned device-mapped memory

use cudarc::driver::LaunchAsync;
use gpu_host::{GpuHostError, GpuRuntime, HostcallBuffer, MappedBuffer};
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
    // It accepts one connection, reads data, echoes it back, then exits.
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

    // Step 2: Initialize GPU runtime
    let rt = GpuRuntime::new(0)?;
    println!("[host] CUDA device initialized.");

    rt.load_ptx(KERNEL_PTX, "tcpecho", &["tcp_echo_kernel"])?;
    println!("[host] PTX module loaded.\n");

    let hcbuf = HostcallBuffer::new(8)?;
    let mut output_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    let cfg = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);

    // Step 3: Launch kernel with hostcall listener
    std::thread::scope(|scope| -> gpu_host::Result<()> {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {}", s);
            });
        });

        println!("--- TCP Echo: GPU connects, sends, reads echo ---");
        unsafe { output_buf.write(0, 0) };
        {
            let f = rt
                .get_func("tcpecho", "tcp_echo_kernel")
                .ok_or(GpuHostError::KernelNotFound("tcp_echo_kernel"))?;
            unsafe {
                f.launch(
                    cfg,
                    (
                        hcbuf.dev_ptr as u64,
                        hcbuf.sideband_dev_ptr as u64,
                        port as u32,
                        output_buf.dev_ptr() as u64,
                    ),
                )?;
            }
            rt.synchronize()?;
            std::thread::sleep(std::time::Duration::from_millis(50));
            let result = unsafe { output_buf.read(0) };

            if result == 0xDEAD {
                println!("[host] tcp_echo_kernel: FAILED (error sentinel 0xDEAD)");
            } else {
                let expected_len = EXPECTED_MSG.len() as u32;
                let passed = result == expected_len;
                println!(
                    "[host] tcp_echo_kernel: {} (response length: {}, expected: {})",
                    if passed { "PASSED" } else { "FAILED" },
                    result,
                    expected_len,
                );
            }
            println!();
        }

        hcbuf.signal_shutdown();
        let _ = listener;
        Ok(())
    })?;

    // Wait for the echo server thread to finish
    echo_handle.join().expect("Echo server thread panicked");

    println!("=== TCP Echo example complete! ===");
    Ok(())
}
