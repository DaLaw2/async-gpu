//! Tokio GPU Offload — async kernel launch + event streaming.
//!
//! Demonstrates using the `GpuTask` API to launch GPU kernels from a tokio
//! runtime without blocking the async executor. Events from the GPU (print
//! messages) are received asynchronously via `next_event().await`.
//!
//! This example uses the embedded kernel PTX from `gpu_host::ptx::KERNEL_IO`,
//! which includes `hostcall_print_hello` — a kernel that prints via hostcall.

use async_gpu::GpuRuntime;
use async_gpu::MappedBuffer;
use async_gpu::{AsyncGpuRuntime, GpuTask, HostcallEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tokio GPU Offload Example ===\n");

    // Initialize async GPU runtime
    let rt = AsyncGpuRuntime::new(0)?;
    println!("[host] CUDA device initialized (async).");

    // Load PTX module (synchronous — fast, no GPU work queued)
    rt.load_ptx(
        async_gpu::ptx::KERNEL_IO,
        "kernel",
        &["hostcall_print_hello"],
    )?;
    println!("[host] PTX module loaded.\n");

    // Create a GpuTask — manages hostcall session + event stream
    let mut task = GpuTask::new(&rt, 4)?;

    // Allocate mapped result buffer (GPU writes, host reads)
    let result_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    // Get kernel function handle
    let func = rt.inner().require_func("kernel", "hostcall_print_hello")?;
    let cfg = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);

    // --- Demo: launch kernel asynchronously ---
    println!("[host] Launching kernel via GpuTask::launch().await...");
    let start = std::time::Instant::now();

    // This does NOT block the tokio runtime — kernel launch + synchronize
    // run on tokio's blocking thread pool.
    task.launch(func, cfg, (task.session_dev_ptr(), result_buf.dev_ptr()))
        .await?;

    let elapsed = start.elapsed();
    println!("[host] Kernel completed in {elapsed:?}.\n");

    // Give listener time to forward remaining events
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // --- Demo: receive GPU events asynchronously ---
    println!("[host] Draining GPU events...");
    let mut event_count = 0;
    loop {
        match tokio::time::timeout(std::time::Duration::from_millis(50), task.next_event()).await {
            Ok(Some(HostcallEvent::Print(msg))) => {
                let s = String::from_utf8_lossy(&msg);
                println!("  [GPU → host] {s}");
                event_count += 1;
            }
            Ok(Some(HostcallEvent::Shutdown)) => break,
            Ok(None) => break,
            Err(_) => break, // timeout = no more events
        }
    }

    // Check kernel result
    let result = unsafe { result_buf.read(0) };
    println!("\n[host] Kernel result: {result} (expected 1)");
    println!("[host] Events received: {event_count}");

    // Shutdown
    task.shutdown().await;

    if result == 1 {
        println!("\n=== Tokio GPU Offload — PASSED ===");
    } else {
        println!("\n=== Tokio GPU Offload — FAILED ===");
        std::process::exit(1);
    }

    Ok(())
}
