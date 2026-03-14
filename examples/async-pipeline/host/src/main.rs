//! Async Pipeline — host driver for warp-cooperative data pipeline demo.
//!
//! 1. Creates input file "pipeline_input.txt"
//! 2. Launches GPU kernel that reads → transforms → writes asynchronously
//! 3. Verifies output file "pipeline_output.txt" contains transformed data
//!
//! The GPU kernel uses `#[warp_cooperative] async fn` with real hostcall
//! Futures — each `.await` yields the warp so other warps can execute.

use cudarc::driver::LaunchAsync;
use gpu_host::{GpuHostError, GpuRuntime, HostcallBuffer, MappedBuffer};

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Async Pipeline Demo ===");
    println!("  #[warp_cooperative] async fn with real hostcall I/O\n");

    // Step 1: Create input file
    let input_data = b"Hello from GPU async pipeline!";
    std::fs::write("pipeline_input.txt", input_data)?;
    println!("[host] Created pipeline_input.txt ({} bytes)", input_data.len());

    // Step 2: Initialize GPU runtime
    let rt = GpuRuntime::new(0)?;
    rt.load_ptx(KERNEL_PTX, "pipeline", &["async_data_pipeline"])?;
    println!("[host] PTX loaded.\n");

    // Step 3: Set up hostcall buffer + result buffer
    let hcbuf = HostcallBuffer::new(8)?;
    let result_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    // Single thread — warp-cooperative is proven by PTX instructions (bar.warp.sync, shfl.sync).
    // Multi-lane I/O requires warp-batched hostcall (future work).
    let cfg = GpuRuntime::launch_config((1, 1, 1), (1, 1, 1), 0);

    // Step 4: Launch kernel with hostcall listener
    std::thread::scope(|scope| -> gpu_host::Result<()> {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {s}");
            });
        });

        println!("[host] Launching async_data_pipeline kernel...");
        let f = rt
            .get_func("pipeline", "async_data_pipeline")
            .ok_or(GpuHostError::KernelNotFound("async_data_pipeline"))?;
        unsafe {
            f.launch(cfg, (hcbuf.dev_ptr as u64, result_buf.dev_ptr() as u64))?;
        }
        rt.synchronize()?;
        // Give listener time to process final messages
        std::thread::sleep(std::time::Duration::from_millis(50));

        hcbuf.signal_shutdown();
        let _ = listener;
        Ok(())
    })?;

    // Step 5: Check results
    let result = unsafe { result_buf.read(0) };
    println!("\n[host] Kernel result: 0x{result:X} ({result})");

    if result >= 0xE000 {
        println!("[host] FAILED — error code 0x{result:X}");
        cleanup();
        return Ok(());
    }

    // Step 6: Verify output file
    match std::fs::read("pipeline_output.txt") {
        Ok(output_data) => {
            println!(
                "[host] pipeline_output.txt: {} bytes",
                output_data.len()
            );

            // Verify: each byte should be input + 1
            let expected: Vec<u8> = input_data.iter().map(|b| b.wrapping_add(1)).collect();
            let matches = output_data.len() == expected.len()
                && output_data.iter().zip(expected.iter()).all(|(a, b)| a == b);

            if matches {
                println!("[host] Verification: PASSED");
                println!(
                    "[host]   Input:  {:?}",
                    std::str::from_utf8(input_data).unwrap()
                );
                println!(
                    "[host]   Output: {:?}",
                    std::str::from_utf8(&output_data).unwrap_or("<non-utf8>")
                );
            } else {
                println!("[host] Verification: FAILED");
                println!("[host]   Expected {} bytes, got {}", expected.len(), output_data.len());
                for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
                    if got != exp {
                        println!("[host]   Mismatch at byte {i}: got 0x{got:02X}, expected 0x{exp:02X}");
                    }
                }
            }
        }
        Err(e) => {
            println!("[host] pipeline_output.txt not found: {e}");
            println!("[host] FAILED — kernel did not produce output file");
        }
    }

    cleanup();
    println!("\n=== Async Pipeline Demo Complete ===");
    Ok(())
}

fn cleanup() {
    let _ = std::fs::remove_file("pipeline_input.txt");
    let _ = std::fs::remove_file("pipeline_output.txt");
}
