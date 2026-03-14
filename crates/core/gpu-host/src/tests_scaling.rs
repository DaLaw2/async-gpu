//! Multi-warp/block tests + misc (error propagation, println direct, slab allocator).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::hostcall;
use crate::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// Test: 32-thread multi-warp synchronous hostcall scaling (product.3).
pub(crate) fn run_multi_warp_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 3: Multi-warp sync hostcall (32 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(64)?;
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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_warp", &["multi_warp_sync_kernel"])?;
    let f = dev
        .get_func("multi_warp", "multi_warp_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_warp_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching multi_warp_sync_kernel (32 threads, full warp)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(500));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != 32 {
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!("expected thread_count=32, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    println!("  multi_warp_sync_kernel: {msg_count} messages received from 32 threads");

    let mut thread_seen = [false; 32];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 16 {
            let tens = msg.as_bytes()[7];
            let ones = msg.as_bytes()[8];
            if (b'0'..=b'3').contains(&tens) && ones.is_ascii_digit() {
                let tid = ((tens - b'0') * 10 + (ones - b'0')) as usize;
                if tid < 32 {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < 32 {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!(
                "expected messages from all 32 threads, got {unique_threads} unique. Missing: {missing:?}"
            ),
        });
    }

    println!("  multi_warp_sync_kernel: PASSED!");
    println!("    All 32 threads sent unique messages concurrently");
    println!("    Thread count: {thread_count}");
    println!("    Messages received: {msg_count}");
    println!("    Unique threads verified: {unique_threads}/32");
    println!("    Multi-warp hostcall scaling demonstrated end-to-end");
    Ok(())
}

/// Test: multi-block synchronous hostcall (multiblock.1).
pub(crate) fn run_multi_block_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 1: 4-block sync hostcall (128 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_blocks: u32 = 4;
    let threads_per_block: u32 = 32;
    let total_threads = (num_blocks * threads_per_block) as usize;
    let num_packets: u16 = 256;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_block", &["multi_block_sync_kernel"])?;
    let f = dev
        .get_func("multi_block", "multi_block_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "  Launching multi_block_sync_kernel ({num_blocks} blocks × {threads_per_block} threads = {total_threads} total)..."
    );
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(500));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != total_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!("expected thread_count={total_threads}, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    println!(
        "  multi_block_sync_kernel: {msg_count} messages received from {total_threads} threads"
    );

    let mut thread_seen = vec![false; total_threads];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 17 {
            let hundreds = msg.as_bytes()[7];
            let tens = msg.as_bytes()[8];
            let ones = msg.as_bytes()[9];
            if hundreds.is_ascii_digit() && tens.is_ascii_digit() && ones.is_ascii_digit() {
                let tid = ((hundreds - b'0') as usize) * 100
                    + ((tens - b'0') as usize) * 10
                    + (ones - b'0') as usize;
                if tid < total_threads {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < total_threads {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!(
                "expected messages from all {total_threads} threads, got {unique_threads} unique. Missing: {missing:?}"
            ),
        });
    }

    println!("  multi_block_sync_kernel: PASSED! ({elapsed:?})");
    println!("    All {total_threads} threads across {num_blocks} blocks sent unique messages");
    println!("    Messages received: {msg_count}");
    println!("    Unique threads verified: {unique_threads}/{total_threads}");
    Ok(())
}

/// Test: multi-block scaling to 512 threads (multiblock.2).
pub(crate) fn run_multi_block_512_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 2: 8-block × 64-thread sync hostcall (512 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_blocks: u32 = 8;
    let threads_per_block: u32 = 64;
    let total_threads = (num_blocks * threads_per_block) as usize;
    let num_packets: u16 = 1024;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_block_512", &["multi_block_sync_kernel"])?;
    let f = dev
        .get_func("multi_block_512", "multi_block_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "  Launching multi_block_sync_kernel ({num_blocks} blocks × {threads_per_block} threads = {total_threads} total)..."
    );
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(1000));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != total_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!("expected thread_count={total_threads}, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    let mut thread_seen = vec![false; total_threads];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 17 {
            let hundreds = msg.as_bytes()[7];
            let tens = msg.as_bytes()[8];
            let ones = msg.as_bytes()[9];
            if hundreds.is_ascii_digit() && tens.is_ascii_digit() && ones.is_ascii_digit() {
                let tid = ((hundreds - b'0') as usize) * 100
                    + ((tens - b'0') as usize) * 10
                    + (ones - b'0') as usize;
                if tid < total_threads {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < total_threads {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        let shown: Vec<usize> = missing.iter().take(20).copied().collect();
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!(
                "expected messages from all {total_threads} threads, got {unique_threads} unique. Missing (first 20): {shown:?}"
            ),
        });
    }

    let duplicates = msg_count - unique_threads;

    println!("  multi_block_512: PASSED! ({elapsed:?})");
    println!("    All {total_threads} threads across {num_blocks} blocks sent unique messages");
    println!(
        "    Messages received: {msg_count} ({unique_threads} unique, {duplicates} duplicates)"
    );
    println!("    Unique threads verified: {unique_threads}/{total_threads}");
    Ok(())
}

/// Test: 4-step sequential async pipeline (product.2).
pub(crate) fn run_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 2: 4-step async pipeline ---");

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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_PIPELINE_PTX);
    dev.load_ptx(ptx, "async_pipeline", &["pipeline_kernel"])?;
    let f = dev
        .get_func("async_pipeline", "pipeline_kernel")
        .ok_or(GpuHostError::KernelNotFound("pipeline_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching pipeline_kernel (4-step sequential async)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "pipeline_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }

    let expected_steps = [
        "step 1: READ",
        "step 2: PROCESS",
        "step 3: WRITE",
        "step 4: PRINT",
    ];
    let mut found_steps = 0;
    for expected in &expected_steps {
        if received.iter().any(|m| m.contains(expected)) {
            found_steps += 1;
        }
    }

    if found_steps != 4 {
        return Err(GpuHostError::Verification {
            test: "pipeline_kernel",
            detail: format!(
                "expected 4 pipeline steps, found {}. Messages: {:?}",
                found_steps, *received
            ),
        });
    }

    println!("  pipeline_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Steps completed: 4/4");
    for (i, msg) in received.iter().enumerate() {
        println!("    Step {}: \"{}\"", i + 1, msg);
    }
    println!("    4-step sequential async pipeline demonstrated end-to-end");
    Ok(())
}

/// Showcase demo: all features combined (product.4).
pub(crate) fn run_showcase_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 4: Showcase Demo (Vec + format! + stdin + stdout on GPU) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let input_data: Vec<u32> = vec![10, 25, 30, 7, 42, 15, 88, 3];
    let input_dev: CudaSlice<u32> = dev.htod_copy(input_data.clone())?;
    let input_len = input_data.len() as u32;

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);

    let stdin_data = b"Rustacean\n".to_vec();
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen_with_stdin(
            |msg| {
                let s = String::from_utf8_lossy(msg).to_string();
                println!("  [GPU] {s}");
                messages_clone.lock().unwrap().push(s);
            },
            stdin_data,
        );
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    let _ = dev.load_ptx(ptx, "std_test", &["showcase_kernel"]);
    let f = dev
        .get_func("std_test", "showcase_kernel")
        .ok_or(GpuHostError::KernelNotFound("showcase_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching showcase_kernel (Vec + format! + stdin + stdout)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, &input_dev, input_len, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let msg_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!("kernel reported failure (success={success})"),
        });
    }

    if msg_count < 4 {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!("expected 4 stdout messages, got {msg_count}"),
        });
    }

    let has_greeting = received
        .iter()
        .any(|m| m.contains("Hello") && m.contains("Rustacean"));
    let has_stats = received.iter().any(|m| m.contains("sum=220"));
    let has_goodbye = received.iter().any(|m| m.contains("Goodbye"));

    if !has_greeting || !has_stats || !has_goodbye {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!(
                "missing expected content (greeting={}, stats={}, goodbye={}). Messages: {:?}",
                has_greeting, has_stats, has_goodbye, *received
            ),
        });
    }

    println!("  showcase_kernel: PASSED!");
    println!("    Features demonstrated:");
    println!("      - stdin read (got name \"Rustacean\" from host)");
    println!(
        "      - Vec<u32> built from {} runtime kernel arguments",
        input_data.len()
    );
    println!("      - Iterator methods: sum, min, max, filter, collect");
    println!("      - format!() with heap-allocated String");
    println!("      - writeln!(stdout()) through PAL hostcall ({msg_count}x)");
    println!("    Total time: {elapsed:?}");
    println!("    This is VectorWare-level Rust std on GPU!");
    Ok(())
}

/// Test: multi-block async with Embassy executors (multiblock.3).
pub(crate) fn run_multi_block_async_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 3: Multi-block async (4 blocks × 1 thread) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(8)?;
    let dev_ptr = hc_buf.dev_ptr;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let num_threads: usize = 4;
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_threads + 1)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Async: {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    let _ = dev.load_ptx(ptx, "multi_block_async", &["multi_block_async_kernel"]);
    let f = dev
        .get_func("multi_block_async", "multi_block_async_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_async_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_threads as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching multi_block_async_kernel ({num_threads} blocks × 1 thread)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let completed = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let mut poll_rounds = Vec::new();
    for i in 0..num_threads {
        let rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(1 + i)) };
        poll_rounds.push(rounds);
    }
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    println!("  Completed: {completed}/{num_threads} threads");
    println!("  Poll rounds per thread: {poll_rounds:?}");
    println!("  Messages received: {}", received.len());
    for msg in received.iter() {
        println!("    \"{msg}\"");
    }

    if completed != num_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_async_kernel",
            detail: format!("only {completed}/{num_threads} threads completed"),
        });
    }

    let mut seen = [false; 4];
    for msg in received.iter() {
        for i in 0..4u8 {
            if msg.contains(&format!("block {i}")) {
                seen[i as usize] = true;
            }
        }
    }
    let all_seen = seen.iter().all(|&s| s);
    if !all_seen {
        return Err(GpuHostError::Verification {
            test: "multi_block_async_kernel",
            detail: format!("missing messages from some blocks. Seen: {seen:?}"),
        });
    }

    println!("  multi_block_async_kernel: PASSED! ({elapsed:?})");
    println!("    {num_threads} blocks × 1 thread, each with its own Embassy executor");
    println!("    All threads completed async hostcall independently!");
    Ok(())
}

/// Test: error propagation through hostcall (error-handling.2).
pub(crate) fn run_error_propagation_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Error Handling Test: error propagation (hostcall \u{2192} GPU) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr;

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 6)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {});
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "error_test", &["error_propagation_test"]);
    let f = dev
        .get_func("error_test", "error_propagation_test")
        .ok_or(GpuHostError::KernelNotFound("error_propagation_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching error_propagation_test kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let err1_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let err2_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    let err3_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(3)) };
    let err1_fd = unsafe { std::ptr::read_volatile(result_host_ptr.add(4)) };
    let passed = unsafe { std::ptr::read_volatile(result_host_ptr.add(5)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let err_not_found: u32 = 1;

    println!(
        "  Test 1 (open nonexistent): category={err1_cat} (expected {err_not_found}), fd={err1_fd}"
    );
    println!("  Test 2 (close invalid fd): category={err2_cat} (expected nonzero)");
    println!("  Test 3 (read invalid fd):  category={err3_cat} (expected nonzero)");
    println!("  Tests passed: {passed}/3");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "error_propagation_test",
            detail: format!(
                "not all tests passed ({passed}/3). categories: open={err1_cat}, close={err2_cat}, read={err3_cat}"
            ),
        });
    }

    println!("  error_propagation_test: PASSED! ({elapsed:?})");
    println!("    Structured error codes propagate from host to GPU correctly!");
    println!("    File-not-found → ERR_NOT_FOUND ({err_not_found}), Invalid fd → error category");
    Ok(())
}

pub(crate) fn run_println_direct_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- OnceLock Test: println!() direct (no writeln! workaround) ---");

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
            println!("  [GPU println] {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    let _ = dev.load_ptx(ptx, "println_test", &["println_direct_test_kernel"]);
    let f = dev
        .get_func("println_test", "println_direct_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("println_direct_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let input_val: u32 = 42;
    println!("  Launching println_direct_test_kernel (println! with value={input_val})...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, input_val, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "println_direct_test_kernel",
            detail: format!("kernel reported failure (success={success})"),
        });
    }

    let full_output: String = received.iter().map(|s| s.as_str()).collect();

    let has_hello = full_output.contains("hello from GPU");
    let has_value = full_output.contains("value = 42");
    let has_multi = full_output.contains("x=84") && full_output.contains("y=52");

    if !has_hello || !has_value || !has_multi {
        return Err(GpuHostError::Verification {
            test: "println_direct_test_kernel",
            detail: format!(
                "missing expected content (hello={has_hello}, value={has_value}, multi={has_multi}). Full output: {full_output:?}"
            ),
        });
    }

    println!("  println_direct_test_kernel: PASSED! ({elapsed:?})");
    println!("    println!() works directly on GPU — no writeln! workaround needed!");
    println!(
        "    {} println! calls completed, {} messages received",
        count,
        received.len()
    );
    println!("    Full VectorWare parity achieved for println!");
    Ok(())
}

/// Test: slab allocator deallocation (allocator.2).
pub(crate) fn run_slab_dealloc_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Allocator Test: Slab dealloc (10 Vec + 10 String cycles) ---");

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    let _ = dev.load_ptx(ptx, "slab_test", &["slab_dealloc_test_kernel"]);
    let f = dev
        .get_func("slab_test", "slab_dealloc_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("slab_dealloc_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching slab_dealloc_test_kernel (10 Vec + 10 String alloc/dealloc cycles)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (0u64, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let cycles = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "slab_dealloc_test_kernel",
            detail: format!("expected 20 successful cycles, got {cycles}"),
        });
    }

    println!("  slab_dealloc_test_kernel: PASSED! ({elapsed:?})");
    println!("    Completed {cycles} alloc/dealloc cycles (10 Vec + 10 String)");
    println!("    Memory reuse confirmed — slab allocator deallocates correctly");
    Ok(())
}

/// Test: concurrent slab allocator stress test (allocator.3).
pub(crate) fn run_slab_concurrent_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Allocator Test: 32-thread concurrent alloc/dealloc ---");

    let num_threads: u32 = 32;
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_threads as usize)? };

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    let _ = dev.load_ptx(ptx, "slab_concurrent", &["slab_concurrent_test_kernel"]);
    let f = dev
        .get_func("slab_concurrent", "slab_concurrent_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("slab_concurrent_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching slab_concurrent_test_kernel (32 threads × 5 cycles each)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (0u64, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    let mut total_ok: u32 = 0;
    let mut threads_ok: u32 = 0;
    let mut failed_threads = Vec::new();
    for i in 0..num_threads as usize {
        let cycles = unsafe { std::ptr::read_volatile(result_host_ptr.add(i)) };
        total_ok += cycles;
        if cycles == 5 {
            threads_ok += 1;
        } else {
            failed_threads.push((i, cycles));
        }
    }
    unsafe { free_mapped_mem(result_host_ptr)? };

    if threads_ok < num_threads {
        return Err(GpuHostError::Verification {
            test: "slab_concurrent_test_kernel",
            detail: format!(
                "expected all 32 threads to complete 5 cycles, got {threads_ok}/{num_threads} threads OK. Failed: {failed_threads:?}"
            ),
        });
    }

    println!("  slab_concurrent_test_kernel: PASSED! ({elapsed:?})");
    println!("    All {num_threads} threads completed 5 alloc/dealloc cycles each");
    println!(
        "    Total successful cycles: {}/{}",
        total_ok,
        num_threads * 5
    );
    println!("    Concurrent slab allocator verified under contention");
    Ok(())
}
