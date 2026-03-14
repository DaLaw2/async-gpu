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
    let rt = GpuRuntime::new(0)?;
    rt.load_ptx(
        KERNEL_PTX,
        "pipeline",
        &["async_data_pipeline", "async_bulk_pipeline"],
    )?;

    run_small_io_demo(&rt)?;
    run_bulk_io_demo(&rt)?;

    println!("\n=== All Async Pipeline Demos Complete ===");
    Ok(())
}

// ---------------------------------------------------------------------------
// Demo 1: Small I/O (packet-payload, ≤48 bytes)
// ---------------------------------------------------------------------------

fn run_small_io_demo(rt: &GpuRuntime) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 1: Small I/O Pipeline ===");
    println!("  #[warp_cooperative] async fn with packet-payload I/O\n");

    let input_data = b"Hello from GPU async pipeline!";
    std::fs::write("pipeline_input.txt", input_data)?;
    println!("[host] Created pipeline_input.txt ({} bytes)", input_data.len());

    let hcbuf = HostcallBuffer::new(8)?;
    let result_buf = MappedBuffer::<u32>::new_zeroed(1)?;
    let cfg = GpuRuntime::launch_config((1, 1, 1), (1, 1, 1), 0);

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
        std::thread::sleep(std::time::Duration::from_millis(50));
        hcbuf.signal_shutdown();
        let _ = listener;
        Ok(())
    })?;

    let result = unsafe { result_buf.read(0) };
    println!("\n[host] Kernel result: 0x{result:X} ({result})");

    if result >= 0xE000 {
        println!("[host] FAILED — error code 0x{result:X}");
    } else {
        verify_file(
            "pipeline_output.txt",
            input_data,
            |b| if b >= b'a' && b <= b'z' { b - 32 } else { b },
        );
    }

    let _ = std::fs::remove_file("pipeline_input.txt");
    let _ = std::fs::remove_file("pipeline_output.txt");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Demo 2: Bulk I/O (sideband, up to 1MB)
// ---------------------------------------------------------------------------

fn run_bulk_io_demo(rt: &GpuRuntime) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 2: Bulk I/O Pipeline ===");
    println!("  #[warp_cooperative] async fn with sideband bulk I/O\n");

    // Write a longer input to exercise sideband (>48 bytes to prove bulk path)
    let input_data = b"The quick brown fox jumps over the lazy dog. Bulk sideband I/O test from GPU!";
    std::fs::write("bulk_input.txt", input_data)?;
    println!("[host] Created bulk_input.txt ({} bytes)", input_data.len());

    let hcbuf = HostcallBuffer::new(8)?;
    let result_buf = MappedBuffer::<u32>::new_zeroed(1)?;
    let cfg = GpuRuntime::launch_config((1, 1, 1), (1, 1, 1), 0);

    std::thread::scope(|scope| -> gpu_host::Result<()> {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {s}");
            });
        });

        println!("[host] Launching async_bulk_pipeline kernel...");
        let f = rt
            .get_func("pipeline", "async_bulk_pipeline")
            .ok_or(GpuHostError::KernelNotFound("async_bulk_pipeline"))?;
        unsafe {
            f.launch(
                cfg,
                (
                    hcbuf.dev_ptr as u64,
                    hcbuf.sideband_dev_ptr as u64,
                    result_buf.dev_ptr() as u64,
                ),
            )?;
        }
        rt.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        hcbuf.signal_shutdown();
        let _ = listener;
        Ok(())
    })?;

    let result = unsafe { result_buf.read(0) };
    println!("\n[host] Kernel result: 0x{result:X} ({result})");

    if result >= 0xE000 {
        println!("[host] FAILED — error code 0x{result:X}");
    } else {
        // The kernel swaps case: lowercase→uppercase, uppercase→lowercase
        verify_file("bulk_output.txt", input_data, |b| {
            if b >= b'a' && b <= b'z' {
                b - 32
            } else if b >= b'A' && b <= b'Z' {
                b + 32
            } else {
                b
            }
        });
    }

    let _ = std::fs::remove_file("bulk_input.txt");
    let _ = std::fs::remove_file("bulk_output.txt");
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared verification helper
// ---------------------------------------------------------------------------

fn verify_file(path: &str, input: &[u8], transform: impl Fn(u8) -> u8) {
    match std::fs::read(path) {
        Ok(output_data) => {
            println!("[host] {path}: {} bytes", output_data.len());
            let expected: Vec<u8> = input.iter().map(|&b| transform(b)).collect();
            let matches = output_data.len() == expected.len()
                && output_data.iter().zip(expected.iter()).all(|(a, b)| a == b);

            if matches {
                println!("[host] Verification: PASSED");
                println!(
                    "[host]   Input:  {:?}",
                    std::str::from_utf8(input).unwrap_or("<non-utf8>")
                );
                println!(
                    "[host]   Output: {:?}",
                    std::str::from_utf8(&output_data).unwrap_or("<non-utf8>")
                );
            } else {
                println!("[host] Verification: FAILED");
                println!(
                    "[host]   Expected {} bytes, got {}",
                    expected.len(),
                    output_data.len()
                );
                for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
                    if got != exp {
                        println!(
                            "[host]   Mismatch at byte {i}: got 0x{got:02X}, expected 0x{exp:02X}"
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("[host] {path} not found: {e}");
            println!("[host] FAILED — kernel did not produce output file");
        }
    }
}
