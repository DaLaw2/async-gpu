//! Hello GPU — standalone example using the gpu-host SDK.
//!
//! Demonstrates four GPU kernels:
//! 1. vector_add — pure compute, no hostcall
//! 2. hello_gpu — PRINT hostcall (GPU prints to host terminal)
//! 3. file_io_demo — file OPEN + WRITE + CLOSE from GPU
//! 4. bulk_read_demo — bulk READ via sideband buffer
//!
//! Uses the core SDK types:
//! - [`GpuRuntime`] — device init, PTX loading, kernel launch
//! - [`HostcallSession`] — managed GPU-host RPC communication
//! - [`MappedBuffer`] — RAII pinned device-mapped memory

use cudarc::driver::LaunchAsync;
use gpu_host::{GpuRuntime, HostcallSession, MappedBuffer};

// Embed the PTX compiled by build.rs
const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    println!("=== Hello GPU Example ===\n");

    // Initialize CUDA device via SDK
    let rt = GpuRuntime::new(0)?;
    println!("[host] CUDA device initialized.");

    // Load PTX module
    rt.load_ptx(
        KERNEL_PTX,
        "hello",
        &["hello_gpu", "vector_add", "file_io_demo", "bulk_read_demo"],
    )?;
    println!("[host] PTX module loaded.\n");

    // ---- Demo 1: vector_add (pure compute, no hostcall) ----
    println!("--- Demo 1: vector_add ---");
    {
        const N: usize = 64;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

        let a_dev = rt.htod_sync_copy(&a)?;
        let b_dev = rt.htod_sync_copy(&b)?;
        let mut c_dev = rt.alloc_zeros::<f32>(N)?;

        let f = rt.require_func("hello", "vector_add")?;
        let cfg = GpuRuntime::launch_config((1, 1, 1), (N as u32, 1, 1), 0);
        unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, N as u32))? };
        let result = rt.dtoh_sync_copy(&c_dev)?;

        let ok = result.iter().all(|&v| (v - N as f32).abs() < 0.001);
        println!(
            "[host] vector_add: {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
    }

    // ---- Demos 2-4: hostcall-based kernels ----
    // Start hostcall session (spawns listener thread automatically)
    let session = HostcallSession::start(8)?;

    // MappedBuffer<u32> — RAII pinned memory, auto-freed on drop
    let mut result_buf = MappedBuffer::<u32>::new_zeroed(1)?;
    let mut bytes_read_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    let cfg1 = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);

    // ---- Demo 2: hello_gpu (PRINT hostcall) ----
    println!("--- Demo 2: hello_gpu (PRINT hostcall) ---");
    unsafe { result_buf.write(0, 0) };
    {
        let f = rt.require_func("hello", "hello_gpu")?;
        unsafe {
            f.launch(cfg1, (session.dev_ptr(), result_buf.dev_ptr() as u64))?;
        }
        rt.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        let r = unsafe { result_buf.read(0) };
        println!(
            "[host] hello_gpu: {}\n",
            if r == 1 { "PASSED" } else { "FAILED" }
        );
    }

    // ---- Demo 3: file_io_demo (OPEN + WRITE + CLOSE) ----
    println!("--- Demo 3: file_io_demo (file I/O from GPU) ---");
    unsafe { result_buf.write(0, 0) };
    {
        let f = rt.require_func("hello", "file_io_demo")?;
        unsafe {
            f.launch(cfg1, (session.dev_ptr(), result_buf.dev_ptr() as u64))?;
        }
        rt.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        let r = unsafe { result_buf.read(0) };
        println!(
            "[host] file_io_demo: {}",
            if r == 1 { "PASSED" } else { "FAILED" }
        );
        if let Ok(content) = std::fs::read_to_string("gpu_output.txt") {
            println!("[host] Verified file content: {:?}\n", content.trim());
        }
    }

    // ---- Demo 4: bulk_read_demo (OPEN + BULK_READ + CLOSE via sideband) ----
    println!("--- Demo 4: bulk_read_demo (sideband bulk read) ---");
    unsafe {
        result_buf.write(0, 0);
        bytes_read_buf.write(0, 0);
    }
    {
        let f = rt.require_func("hello", "bulk_read_demo")?;
        unsafe {
            f.launch(
                cfg1,
                (
                    session.dev_ptr(),
                    session.sideband_dev_ptr(),
                    result_buf.dev_ptr() as u64,
                    bytes_read_buf.dev_ptr() as u64,
                ),
            )?;
        }
        rt.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        let r = unsafe { result_buf.read(0) };
        let n = unsafe { bytes_read_buf.read(0) };
        println!(
            "[host] bulk_read_demo: {} ({} bytes read)\n",
            if r == 1 { "PASSED" } else { "FAILED" },
            n
        );
    }

    // Shutdown session (stops listener, closes files)
    session.shutdown();

    // Cleanup (MappedBuffer auto-frees on drop)
    let _ = std::fs::remove_file("gpu_output.txt");

    println!("=== All demos complete! ===");
    Ok(())
}
