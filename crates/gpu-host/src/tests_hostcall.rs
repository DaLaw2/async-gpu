//! Hostcall + Embassy + Async hostcall tests.

use std::sync::Arc;

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::hostcall;
use crate::mapped_mem::{alloc_mapped_result_array, alloc_mapped_u32, free_mapped_mem};

/// Step 8: Hostcall print test (hostcall.4).
pub(crate) fn run_hostcall_print_hello(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 8: Hostcall print (single message) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    println!(
        "  Hostcall buffer allocated: {} bytes, {} packets",
        hc_buf.size, hc_buf.num_packets
    );
    println!(
        "  Host ptr: {:p}, Device ptr: 0x{:016X}",
        hc_buf.host_ptr, dev_ptr
    );

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["hostcall_print_hello"]);
    let f = dev
        .get_func("kernel", "hostcall_print_hello")
        .ok_or(GpuHostError::KernelNotFound("hostcall_print_hello"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching hostcall_print_hello kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_hello",
            detail: format!("kernel reported failure (result={result_val})"),
        });
    }
    if received.len() != 1 || !received[0].contains("Hello from GPU!") {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_hello",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  hostcall_print_hello: PASSED!");
    println!("    GPU sent \"{}\" via hostcall protocol", received[0]);
    println!("    Host listener received and printed it correctly");
    Ok(())
}

/// Step 9: Multi-warp hostcall print test.
pub(crate) fn run_hostcall_print_multi(dev: Arc<CudaDevice>, num_blocks: u32) -> Result<()> {
    println!("\n--- Step 9: Hostcall print (multi-block, {num_blocks} blocks) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_packets = (num_blocks as u16).max(8);
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr;

    println!(
        "  Hostcall buffer: {} packets, {} bytes",
        num_packets, hc_buf.size
    );

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["hostcall_print_multi"]);
    let f = dev
        .get_func("kernel", "hostcall_print_multi")
        .ok_or(GpuHostError::KernelNotFound("hostcall_print_multi"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching hostcall_print_multi ({num_blocks} blocks × 32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    let received = messages.lock().unwrap();
    println!(
        "  Results: {} blocks succeeded, {} messages received",
        success_count,
        received.len()
    );

    if success_count != num_blocks {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_multi",
            detail: format!("expected {num_blocks} successes, got {success_count}"),
        });
    }
    if received.len() != num_blocks as usize {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_multi",
            detail: format!("expected {} messages, got {}", num_blocks, received.len()),
        });
    }

    println!("  hostcall_print_multi: PASSED!");
    println!("    {num_blocks} concurrent warps printed via hostcall successfully");
    Ok(())
}

/// Test: ImmediateFuture — single spawn + poll, writes 1 on success.
pub(crate) fn run_embassy_immediate(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 1: ImmediateFuture (spawn + single poll) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    dev.load_ptx(ptx, "embassy", &["embassy_test_kernel"])?;

    let f = dev
        .get_func("embassy", "embassy_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_test_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    if host_result[0] != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_test_kernel",
            detail: format!("expected result=1, got {}", host_result[0]),
        });
    }

    println!("  embassy_test_kernel: PASSED (ImmediateFuture polled to Ready, result=1)");
    Ok(())
}

/// Test: CountdownFuture — multi-poll async task.
pub(crate) fn run_embassy_countdown(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 2: CountdownFuture (multi-poll) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    let _ = dev.load_ptx(ptx, "embassy", &["embassy_countdown_kernel"]);

    let f = dev
        .get_func("embassy", "embassy_countdown_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_countdown_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let poll_rounds = host_result[0];
    let success = host_result[1];

    println!("  Poll rounds executed: {poll_rounds}");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_countdown_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    println!(
        "  embassy_countdown_kernel: PASSED (CountdownFuture completed after {poll_rounds} poll rounds)"
    );
    Ok(())
}

/// Test: Two concurrent tasks on the same executor.
pub(crate) fn run_embassy_two_task(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 3: Two concurrent tasks ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    let _ = dev.load_ptx(ptx, "embassy", &["embassy_two_task_kernel"]);

    let f = dev
        .get_func("embassy", "embassy_two_task_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_two_task_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let poll_rounds = host_result[0];
    let success = host_result[1];

    println!("  Poll rounds executed: {poll_rounds}");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_two_task_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    println!(
        "  embassy_two_task_kernel: PASSED (2 concurrent tasks completed after {poll_rounds} poll rounds)"
    );
    Ok(())
}

/// Test: Synchronous countdown baseline for register pressure comparison.
pub(crate) fn run_sync_countdown(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 4: Sync countdown baseline ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    let _ = dev.load_ptx(ptx, "embassy", &["sync_countdown_kernel"]);

    let f = dev
        .get_func("embassy", "sync_countdown_kernel")
        .ok_or(GpuHostError::KernelNotFound("sync_countdown_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let value = host_result[0];
    let success = host_result[1];

    if value != 42 || success != 1 {
        return Err(GpuHostError::Verification {
            test: "sync_countdown_kernel",
            detail: format!("expected (42, 1), got ({value}, {success})"),
        });
    }

    println!("  sync_countdown_kernel: PASSED (result=42, sync loop completed)");
    Ok(())
}

/// File I/O round-trip test: open → write → close → open → read → close → verify.
pub(crate) fn run_hostcall_file_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hostcall file I/O test (gpu-std.3) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    println!(
        "  Hostcall buffer allocated: {} bytes, {} packets",
        hc_buf.size, hc_buf.num_packets
    );

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 4)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Print from GPU: \"{s}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["hostcall_file_test"]);
    let f = dev
        .get_func("kernel", "hostcall_file_test")
        .ok_or(GpuHostError::KernelNotFound("hostcall_file_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();

    println!("  Launching hostcall_file_test kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let overall = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let bytes_written = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    let bytes_read = unsafe { std::ptr::read_volatile(result_host_ptr.add(3)) };

    unsafe { free_mapped_mem(result_host_ptr)? };

    let file_content = std::fs::read_to_string("gpu_test_output.txt");
    if let Ok(content) = &file_content {
        println!("  File on disk: \"{}\"", content.trim_end());
    }
    let _ = std::fs::remove_file("gpu_test_output.txt");

    if overall != 1 {
        return Err(GpuHostError::Verification {
            test: "hostcall_file_test",
            detail: format!("overall={overall}, written={bytes_written}, read={bytes_read}"),
        });
    }

    println!(
        "  hostcall_file_test: PASSED! (wrote {bytes_written} bytes, read {bytes_read} bytes, {elapsed:?})"
    );
    Ok(())
}

/// Test: single async hostcall print via HostcallFuture + Embassy executor.
pub(crate) fn run_async_hostcall_single(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 1: Async hostcall (single HostcallFuture) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "async_hostcall", &["async_hostcall_single_kernel"])?;
    let f = dev
        .get_func("async_hostcall", "async_hostcall_single_kernel")
        .ok_or(GpuHostError::KernelNotFound("async_hostcall_single_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching async_hostcall_single_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_single_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.is_empty() || !received[0].contains("Async hello") {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_single_kernel",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  async_hostcall_single_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Message: \"{}\"", received[0]);
    println!("    HostcallFuture correctly yielded and resumed across poll rounds");
    Ok(())
}

/// Test: two concurrent async hostcall prints via HostcallFuture + Embassy executor.
pub(crate) fn run_async_hostcall_two(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 2: Async hostcall (two concurrent HostcallFutures) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    let _ = dev.load_ptx(ptx, "async_hostcall", &["async_hostcall_two_kernel"]);
    let f = dev
        .get_func("async_hostcall", "async_hostcall_two_kernel")
        .ok_or(GpuHostError::KernelNotFound("async_hostcall_two_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching async_hostcall_two_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_two_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_two_kernel",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  async_hostcall_two_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Messages: {:?}", *received);
    println!("    Two HostcallFutures ran concurrently on one Embassy executor!");
    Ok(())
}

/// Test: futures_util::future::join — third-party async combinator on GPU.
pub(crate) fn run_futures_join(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 3: futures_util::future::join on GPU ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    let _ = dev.load_ptx(ptx, "async_hostcall", &["futures_join_kernel"]);
    let f = dev
        .get_func("async_hostcall", "futures_join_kernel")
        .ok_or(GpuHostError::KernelNotFound("futures_join_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching futures_join_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "futures_join_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "futures_join_kernel",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  futures_join_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Messages: {:?}", *received);
    println!("    futures_util::future::join works on GPU! Third-party async crate confirmed.");
    Ok(())
}

/// Test: GPU %globaltimer + wall-clock time via hostcall.
pub(crate) fn run_hostcall_time_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- GPU-std Test: Instant (%globaltimer) + SystemTime (hostcall) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let (result_host_ptr, result_dev_ptr) = unsafe {
        let cu = cuda_lib();
        let size = 4 * std::mem::size_of::<u64>();

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let result = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuHostError::CudaAlloc(result));
        }

        let mut d_ptr: sys::CUdeviceptr = 0;
        let result = cu.cuMemHostGetDevicePointer_v2(&mut d_ptr, host_ptr, 0);
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuMemFreeHost(host_ptr);
            return Err(GpuHostError::CudaGetDevPtr(result));
        }

        std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        (host_ptr as *mut u64, d_ptr)
    };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Print from GPU: \"{s}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["hostcall_stdin_time_test"]);
    let f = dev
        .get_func("kernel", "hostcall_stdin_time_test")
        .ok_or(GpuHostError::KernelNotFound("hostcall_stdin_time_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Skip stdin (skip_stdin=1) to avoid blocking
    println!("  Launching hostcall_stdin_time_test (skip_stdin=1)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr, 1u32))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let gpu_instant_delta = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let host_secs = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let host_nanos = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };

    unsafe {
        let cu = cuda_lib();
        cu.cuMemFreeHost(result_host_ptr as *mut std::ffi::c_void);
    }

    println!("  GPU Instant delta: {gpu_instant_delta} ns (time for 1000 add iterations)");
    println!("  Host wall-clock: epoch_secs={host_secs}, nanos={host_nanos}");

    if gpu_instant_delta == 0 {
        return Err(GpuHostError::Verification {
            test: "hostcall_stdin_time_test",
            detail: "GPU instant delta is 0 — %globaltimer may not be working".to_string(),
        });
    }
    if host_secs < 1700000000 {
        return Err(GpuHostError::Verification {
            test: "hostcall_stdin_time_test",
            detail: format!("host secs={host_secs} seems too low for epoch time"),
        });
    }

    println!("  hostcall_stdin_time_test: PASSED!");
    println!("    GPU %globaltimer works (non-zero delta)");
    println!("    Host SystemTime works (reasonable epoch seconds)");
    Ok(())
}

/// Test: 32 threads each emit a gpu_trace!() event, verify all 32 complete.
///
/// Trace events are printed to stderr by the host-side handler. We verify
/// that all 32 threads ran to completion via an atomic success counter.
pub(crate) fn run_trace_multithread_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Trace multi-thread test (32 threads) ---");

    let num_packets = 64u16; // Enough for 32 threads
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr;

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {
            // Trace events go to stderr via handle_trace, not through on_print
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["trace_multithread_test"]);
    let f = dev
        .get_func("kernel", "trace_multithread_test")
        .ok_or(GpuHostError::KernelNotFound("trace_multithread_test"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching trace_multithread_test (1 block × 32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    println!("  Results: {success_count}/32 threads completed trace events");

    if success_count != 32 {
        return Err(GpuHostError::Verification {
            test: "trace_multithread_test",
            detail: format!("expected 32 successes, got {success_count}"),
        });
    }

    println!("  trace_multithread_test: PASSED!");
    Ok(())
}

/// Test: threads trace + assert (true condition). Verify assert does not trap.
pub(crate) fn run_trace_assert_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Trace + assert test (32 threads, assert true) ---");

    let num_packets = 64u16;
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr;

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {});
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "kernel", &["trace_assert_test"]);
    let f = dev
        .get_func("kernel", "trace_assert_test")
        .ok_or(GpuHostError::KernelNotFound("trace_assert_test"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching trace_assert_test (1 block × 32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed (no trap — assert passed).");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    println!("  Results: {success_count}/32 threads completed");

    if success_count != 32 {
        return Err(GpuHostError::Verification {
            test: "trace_assert_test",
            detail: format!("expected 32 successes, got {success_count}"),
        });
    }

    println!("  trace_assert_test: PASSED!");
    Ok(())
}
