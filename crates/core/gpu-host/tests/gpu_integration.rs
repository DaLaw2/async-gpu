#![cfg(feature = "demo")]
//! GPU integration tests — proof of concept for `cargo test` on GPU kernels.
//!
//! Run with: `cargo test --features demo --test gpu_integration -- --test-threads=1`
//!
//! These tests require a CUDA-capable GPU and the `demo` feature (for
//! `mapped_mem` access). They will fail on machines without NVIDIA GPU +
//! CUDA driver installed.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};
use gpu_host::ptx;
use gpu_host::runtime::GpuRuntime;

/// Shared CUDA device — initialized once, reused across all tests.
/// Tests must run with `--test-threads=1` to avoid CUDA context conflicts.
///
/// Also re-binds the CUDA context to the calling thread, since prior tests
/// (e.g., stream creation/destruction) may have altered the thread-local context.
fn shared_device() -> Arc<CudaDevice> {
    static DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();
    let dev =
        Arc::clone(DEVICE.get_or_init(|| CudaDevice::new(0).expect("CUDA device init failed")));
    dev.bind_to_thread().expect("bind CUDA context to thread");
    dev
}

/// Basic test: launch `write_thread_idx` kernel and verify thread 0 wrote its index.
///
/// This tests the fundamental GPU launch path: PTX loading → kernel dispatch →
/// device-mapped memory read-back.
#[test]
fn test_write_thread_idx() {
    let dev = shared_device();

    let ptx_src = cudarc::nvrtc::Ptx::from_src(ptx::KERNEL);
    let _ = dev.load_ptx(ptx_src, "kernel", &["write_thread_idx"]);
    let f = dev
        .get_func("kernel", "write_thread_idx")
        .expect("write_thread_idx not found");

    let (result_host, result_dev) =
        unsafe { alloc_mapped_result_array(&dev, 32).expect("alloc mapped") };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (result_dev, 32u32)).expect("kernel launch");
    }
    dev.synchronize().expect("sync");

    // Verify: each thread wrote its own index
    for i in 0..32 {
        let val = unsafe { std::ptr::read_volatile(result_host.add(i)) };
        assert_eq!(val, i as u32, "thread {i} should write its index");
    }

    unsafe { free_mapped_mem(result_host).expect("free mapped") };
}

/// Hostcall test: launch `hostcall_print_hello` and verify it succeeds.
///
/// This tests the hostcall protocol: buffer allocation → listener thread →
/// GPU print → host receives message → cleanup.
#[test]
fn test_hostcall_print_hello() {
    use gpu_host::hostcall;
    use std::sync::Arc as StdArc;

    let dev = shared_device();

    let hc_buf = hostcall::HostcallBuffer::new(4).expect("hostcall buffer");
    let dev_ptr = hc_buf.dev_ptr;
    let (result_host, result_dev) =
        unsafe { alloc_mapped_result_array(&dev, 1).expect("alloc mapped") };

    let received = StdArc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let received_clone = StdArc::clone(&received);

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            received_clone.lock().unwrap().push(s);
        });
    });

    let ptx_src = cudarc::nvrtc::Ptx::from_src(ptx::KERNEL);
    let _ = dev.load_ptx(ptx_src, "kernel", &["hostcall_print_hello"]);
    let f = dev
        .get_func("kernel", "hostcall_print_hello")
        .expect("hostcall_print_hello not found");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (dev_ptr, result_dev)).expect("kernel launch");
    }
    dev.synchronize().expect("sync");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener.join().unwrap();

    let result = unsafe { std::ptr::read_volatile(result_host) };
    unsafe { free_mapped_mem(result_host).expect("free mapped") };

    assert_eq!(result, 1, "hostcall print should succeed");

    let msgs = received.lock().unwrap();
    assert!(
        msgs.iter().any(|m| m.contains("Hello from GPU")),
        "should receive 'Hello from GPU' message, got: {msgs:?}"
    );
}

/// Buffered print test: verify SERVICE_BULK_PRINT works end-to-end.
///
/// This tests the printf-batch infrastructure: print_buffer::init/print/flush
/// on GPU → SERVICE_BULK_PRINT handler on host → messages received.
#[test]
fn test_buffered_print() {
    use gpu_host::hostcall;
    use std::sync::Arc as StdArc;

    let dev = shared_device();

    let hc_buf = hostcall::HostcallBuffer::new(4).expect("hostcall buffer");
    let dev_ptr = hc_buf.dev_ptr;
    let sb_dev_ptr = hc_buf.sideband_dev_ptr;

    let (result_host, result_dev) =
        unsafe { alloc_mapped_result_array(&dev, 1).expect("alloc mapped") };

    let messages = StdArc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let messages_clone = StdArc::clone(&messages);

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx_src = cudarc::nvrtc::Ptx::from_src(ptx::KERNEL);
    let _ = dev.load_ptx(ptx_src, "kernel", &["buffered_print_test"]);
    let f = dev
        .get_func("kernel", "buffered_print_test")
        .expect("buffered_print_test not found");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (dev_ptr, sb_dev_ptr, result_dev))
            .expect("kernel launch");
    }
    dev.synchronize().expect("sync");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener.join().unwrap();

    let result = unsafe { std::ptr::read_volatile(result_host) };
    unsafe { free_mapped_mem(result_host).expect("free mapped") };

    assert_eq!(result, 1, "buffered print kernel should succeed");

    let msgs = messages.lock().unwrap();
    assert!(
        msgs.len() >= 12,
        "expected 12+ messages, got {}",
        msgs.len()
    );
}

/// Multi-GPU enumeration test: verify device_count, device_name, device_ordinal.
///
/// If multiple GPUs are available, creates a second GpuRuntime on device 1.
#[test]
fn test_multi_gpu_enumeration() {
    let count = GpuRuntime::device_count().expect("device_count");
    assert!(count >= 1, "expected at least 1 CUDA device, got {count}");
    println!("CUDA device count: {count}");

    let rt0 = GpuRuntime::new(0).expect("GpuRuntime device 0");
    assert_eq!(rt0.device_ordinal(), 0);

    let name0 = rt0.device_name().expect("device_name(0)");
    assert!(!name0.is_empty(), "device name should not be empty");
    println!("Device 0: {name0}");

    if count >= 2 {
        let rt1 = GpuRuntime::new(1).expect("GpuRuntime device 1");
        assert_eq!(rt1.device_ordinal(), 1);

        let name1 = rt1.device_name().expect("device_name(1)");
        assert!(!name1.is_empty(), "device 1 name should not be empty");
        println!("Device 1: {name1}");
    }
}

/// Stream test: launch a kernel on a non-default stream via GpuStream wrapper.
///
/// Tests the full GpuStream API path: GpuRuntime::create_stream → launch →
/// synchronize → join_default → verify output.
#[test]
fn test_cuda_stream_launch() {
    let rt = GpuRuntime::new(0).expect("GpuRuntime init");
    let dev = rt.device();

    // Load the write_thread_idx PTX
    rt.load_ptx(ptx::KERNEL, "kernel_stream", &["write_thread_idx"])
        .expect("load PTX");
    let func = dev
        .get_func("kernel_stream", "write_thread_idx")
        .expect("write_thread_idx not found");

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Allocate mapped memory for the result
    let (result_ptr, result_dev) =
        unsafe { alloc_mapped_result_array(dev, 32).expect("mapped alloc") };

    // Create a GpuStream and launch the kernel on it
    let stream = rt.create_stream().expect("create stream");
    unsafe {
        stream
            .launch(func, config, (result_dev, 32u32))
            .expect("launch on stream");
    }

    // Sync the stream, join back to default, then full device sync
    stream.synchronize().expect("stream sync");
    stream.join_default().expect("join default");
    dev.synchronize().expect("device sync");

    // Verify results — each thread should have written its index
    for i in 0..32 {
        let val = unsafe { std::ptr::read_volatile(result_ptr.add(i)) };
        assert_eq!(val, i as u32, "thread {i} should write its index");
    }
    println!("GpuStream launch: all 32 threads wrote correct indices");

    unsafe { free_mapped_mem(result_ptr).expect("free mapped mem") };
}
