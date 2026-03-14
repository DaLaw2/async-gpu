//! GPU integration tests — proof of concept for `cargo test` on GPU kernels.
//!
//! Run with: `cargo test --test gpu_integration -- --test-threads=1`
//!
//! These tests require a CUDA-capable GPU. They will fail on machines without
//! NVIDIA GPU + CUDA driver installed.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};
use gpu_host::ptx;

/// Shared CUDA device — initialized once, reused across all tests.
/// Tests must run with `--test-threads=1` to avoid CUDA context conflicts.
fn shared_device() -> Arc<CudaDevice> {
    static DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();
    Arc::clone(DEVICE.get_or_init(|| CudaDevice::new(0).expect("CUDA device init failed")))
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
        f.launch(cfg, (result_dev,)).expect("kernel launch");
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
