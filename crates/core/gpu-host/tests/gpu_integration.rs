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

/// Minimal PTX for a map kernel: f(x) = x * 2.0 + 1.0
///
/// Grid-stride loop, same semantics as `par_iter_map_collect_multiblock`.
/// Only ~20 instructions — JIT compiles in milliseconds (vs 10+ min for full kernel PTX).
///
/// Kernel signature: `fn(input: *const f32, output: *mut f32, n: u32)`
/// Minimal PTX for a map kernel: f(x) = x * 2.0 + 1.0
///
/// Grid-stride loop, same semantics as `par_iter_map_collect_multiblock`.
/// Only ~20 instructions — JIT compiles in milliseconds (vs 10+ min for full kernel PTX).
///
/// Kernel signature: `fn(input: *const f32, output: *mut f32, n: u32)`
const MAP_KERNEL_PTX: &str = "\
.version 7.8\n\
.target sm_75\n\
.address_size 64\n\
\n\
.visible .entry gpuvec_map_f32(\n\
    .param .u64 input,\n\
    .param .u64 output,\n\
    .param .u32 n\n\
)\n\
{\n\
    .reg .u32  %r<10>;\n\
    .reg .u64  %rd<6>;\n\
    .reg .f32  %f<3>;\n\
    .reg .pred %p;\n\
\n\
    ld.param.u64    %rd0, [input];\n\
    ld.param.u64    %rd1, [output];\n\
    ld.param.u32    %r0,  [n];\n\
\n\
    mov.u32         %r1,  %tid.x;\n\
    mov.u32         %r2,  %ntid.x;\n\
    mov.u32         %r3,  %ctaid.x;\n\
    mov.u32         %r4,  %nctaid.x;\n\
\n\
    mad.lo.u32      %r5,  %r3, %r2, %r1;\n\
    mul.lo.u32      %r6,  %r4, %r2;\n\
\n\
LOOP:\n\
    setp.ge.u32     %p, %r5, %r0;\n\
    @%p bra         DONE;\n\
\n\
    mul.wide.u32    %rd2, %r5, 4;\n\
    add.u64         %rd3, %rd0, %rd2;\n\
    ld.global.f32   %f0,  [%rd3];\n\
\n\
    fma.rn.f32      %f1,  %f0, 0f40000000, 0f3F800000;\n\
\n\
    add.u64         %rd4, %rd1, %rd2;\n\
    st.global.f32   [%rd4], %f1;\n\
\n\
    add.u32         %r5, %r5, %r6;\n\
    bra             LOOP;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

/// GpuVec zero-copy test: `launch_with_gpuvec` passes GpuVec device pointers directly.
///
/// Tests the "no cudaMemcpy" pattern: GpuVec uses pinned device-mapped memory,
/// so the GPU reads/writes directly without explicit host-to-device or
/// device-to-host transfers.
#[test]
fn test_gpuvec_launch_zero_copy() {
    use gpu_host::gpu;
    use gpu_host::memory::GpuVec;

    // Ensure CUDA context is alive for the duration of the test
    let _dev = shared_device();

    let n = 1024;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let input = GpuVec::from_vec(input_data.clone()).unwrap();
    let mut output = GpuVec::<f32>::zeroed(n).unwrap();

    // Launch the map kernel: f(x) = x * 2.0 + 1.0
    // Uses inline PTX (JIT compiles in milliseconds)
    gpu::launch_with_gpuvec(MAP_KERNEL_PTX, "gpuvec_map_f32", &input, &mut output, 256)
        .expect("launch_with_gpuvec should succeed");

    // Results are immediately readable — zero-copy
    let results = output.as_slice();
    for i in 0..n {
        let expected = input_data[i] * 2.0 + 1.0;
        assert!(
            (results[i] - expected).abs() < 1e-5,
            "mismatch at index {i}: got {}, expected {expected}",
            results[i]
        );
    }
}

/// GpuVec::map_gpu convenience test: one-liner transform returning a new GpuVec.
///
/// Tests that `map_gpu` creates an output buffer, launches the kernel, and returns
/// results in a new GpuVec — all with zero explicit memory transfers.
#[test]
fn test_gpuvec_map_gpu() {
    use gpu_host::memory::GpuVec;

    let _dev = shared_device();

    let n = 2048;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let input = GpuVec::from_vec(input_data.clone()).unwrap();

    // One-liner: input → kernel → output GpuVec
    let output = input
        .map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256)
        .expect("map_gpu should succeed");

    assert_eq!(output.len(), n);

    // Verify correctness
    let results = output.as_slice();
    for i in 0..n {
        let expected = input_data[i] * 2.0 + 1.0;
        assert!(
            (results[i] - expected).abs() < 1e-5,
            "mismatch at index {i}: got {}, expected {expected}",
            results[i]
        );
    }

    // Verify into_vec also works
    let result_vec = output.into_vec();
    assert_eq!(result_vec.len(), n);
    assert!((result_vec[0] - (0.0 * 2.0 + 1.0)).abs() < 1e-5);
    assert!((result_vec[n - 1] - ((n - 1) as f32 * 0.1 * 2.0 + 1.0)).abs() < 1e-5);
}

/// GpuVec zero-copy with large data: verify multi-block dispatch works at scale.
///
/// Uses 1M elements to stress-test the grid-stride loop and zero-copy path.
#[test]
fn test_gpuvec_large_data() {
    use gpu_host::memory::GpuVec;

    let _dev = shared_device();

    let n = 1_048_576; // 1M elements
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let input = GpuVec::from_vec(input_data.clone()).unwrap();

    let output = input
        .map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256)
        .expect("map_gpu with 1M elements should succeed");

    // Spot-check values (full check would be slow)
    let results = output.as_slice();
    for &i in &[0, 1, 100, 1000, 10_000, 100_000, n - 1] {
        let expected = input_data[i] * 2.0 + 1.0;
        assert!(
            (results[i] - expected).abs() < 1e-3,
            "mismatch at index {i}: got {}, expected {expected}",
            results[i]
        );
    }
}
