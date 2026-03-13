//! Async I/O — host binary demonstrating multi-step file I/O from GPU.
//!
//! Runs two GPU kernels via the gpu-host SDK:
//! 1. write_pipeline — write 3 files from GPU in sequence
//! 2. transform_pipeline — read a file, uppercase on GPU, write result
//!
//! Uses the three core SDK types:
//! - [`GpuRuntime`] — device init, PTX loading, kernel launch
//! - [`HostcallBuffer`] — GPU-host RPC communication
//! - [`MappedBuffer`] — RAII pinned device-mapped memory

use cudarc::driver::LaunchAsync;
use gpu_host::{GpuHostError, GpuRuntime, HostcallBuffer, MappedBuffer};

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    println!("=== Async I/O Example ===\n");

    let rt = GpuRuntime::new(0)?;
    println!("[host] CUDA device initialized.");

    rt.load_ptx(
        KERNEL_PTX,
        "asyncio",
        &["write_pipeline", "transform_pipeline"],
    )?;
    println!("[host] PTX module loaded.\n");

    let hcbuf = HostcallBuffer::new(8)?;
    let mut result_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    let cfg = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);

    std::thread::scope(|scope| -> gpu_host::Result<()> {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {}", s);
            });
        });

        // ---- Demo 1: write_pipeline ----
        println!("--- Demo 1: write_pipeline (3 files from GPU) ---");
        unsafe { result_buf.write(0, 0) };
        {
            let f = rt
                .get_func("asyncio", "write_pipeline")
                .ok_or(GpuHostError::KernelNotFound("write_pipeline"))?;
            unsafe {
                f.launch(cfg, (hcbuf.dev_ptr as u64, result_buf.dev_ptr() as u64))?;
            }
            rt.synchronize()?;
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { result_buf.read(0) };
            println!("[host] write_pipeline: {}/3 files written", r);

            for i in 0..3 {
                let name = format!("gpu_file_{i}.txt");
                if let Ok(content) = std::fs::read_to_string(&name) {
                    println!("[host]   {name}: {:?}", content.trim());
                }
            }
            println!();
        }

        // ---- Demo 2: transform_pipeline ----
        println!("--- Demo 2: transform_pipeline (read -> uppercase -> write) ---");
        unsafe { result_buf.write(0, 0) };
        {
            let f = rt
                .get_func("asyncio", "transform_pipeline")
                .ok_or(GpuHostError::KernelNotFound("transform_pipeline"))?;
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
            let r = unsafe { result_buf.read(0) };
            println!(
                "[host] transform_pipeline: {}",
                if r == 1 { "PASSED" } else { "FAILED" }
            );
            if let Ok(content) = std::fs::read_to_string("gpu_upper.txt") {
                println!("[host]   gpu_upper.txt: {:?}", content.trim());
            }
            println!();
        }

        hcbuf.signal_shutdown();
        let _ = listener;
        Ok(())
    })?;

    // Cleanup
    for name in &[
        "gpu_file_0.txt",
        "gpu_file_1.txt",
        "gpu_file_2.txt",
        "gpu_upper.txt",
    ] {
        let _ = std::fs::remove_file(name);
    }

    println!("=== Async I/O example complete! ===");
    Ok(())
}
