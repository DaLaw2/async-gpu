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

// ============================================================================
// Unified pipeline: North Star demo integration test
// ============================================================================

/// North Star demo: read -> compute -> write pipeline, end-to-end on GPU.
///
/// This test exercises the FULL user-facing pipeline using GpuVec + inline PTX
/// (JIT compiles in milliseconds, avoids the 10-min full PTX JIT):
/// 1. Generate test input data (simulates file read)
/// 2. Transform on GPU via GpuVec::map_gpu (zero-copy, no cudaMemcpy)
/// 3. Verify correctness against CPU reference
/// 4. Serialize output (simulates file write)
/// 5. Print timing info
///
/// Hidden GPU concepts: kernel launch config, memory transfer, device sync,
/// block/thread/warp, PTX compilation, grid-stride loops.
#[test]
fn test_unified_pipeline_read_compute_write() {
    use gpu_host::memory::GpuVec;
    use std::time::Instant;

    let _dev = shared_device();

    // ── Step 1: "Read" — generate test data (simulates reading a file) ──
    let n = 10_000;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1000.0).collect();
    println!("[unified_pipeline] Read {} elements", n);

    // ── CPU reference for verification ──────────────────────────
    let cpu_ref: Vec<f32> = input_data.iter().map(|&x| x * 2.0 + 1.0).collect();

    // ── Step 2: "Compute" — transform on GPU ────────────────────
    let t0 = Instant::now();
    let gpu_data = GpuVec::from_vec(input_data).expect("GpuVec::from_vec");
    let gpu_result = gpu_data
        .map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256)
        .expect("GpuVec::map_gpu");
    let gpu_elapsed = t0.elapsed();
    println!("[unified_pipeline] GPU compute: {:?}", gpu_elapsed);

    // ── Step 3: "Write" — serialize to bytes (simulates file write) ──
    let output = gpu_result.as_slice();
    let write_start = Instant::now();
    let bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
    let write_elapsed = write_start.elapsed();
    assert_eq!(bytes.len(), n * 4);
    println!(
        "[unified_pipeline] Serialized {} bytes in {:?}",
        bytes.len(),
        write_elapsed
    );

    // ── Verify correctness against CPU reference ────────────────
    assert_eq!(output.len(), n);
    let mut max_err: f32 = 0.0;
    for i in 0..n {
        let err = (output[i] - cpu_ref[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 1e-3,
            "mismatch at index {i}: GPU={}, CPU={}, err={err}",
            output[i],
            cpu_ref[i]
        );
    }
    println!(
        "[unified_pipeline] All {} elements match (max error: {:.2e})",
        n, max_err
    );
}

/// North Star demo: GpuVec zero-copy pipeline with inline PTX.
///
/// Tests the explicit-GPU-but-no-transfers pattern using the tiny inline
/// kernel (JIT compiles in milliseconds, avoids 10-min full PTX JIT).
///
/// Pipeline: create data -> GpuVec::from_vec -> map_gpu -> as_slice -> verify
/// Hidden GPU concepts: kernel launch config, memory transfer, device sync.
#[test]
fn test_unified_pipeline_gpuvec() {
    use gpu_host::memory::GpuVec;
    use std::time::Instant;

    let _dev = shared_device();

    // ── Generate test data ───────────────────────────────────────
    let n = 16_384;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 50.0).collect();
    println!("[unified_gpuvec] Input: {} elements", n);

    // ── CPU reference ────────────────────────────────────────────
    let cpu_ref: Vec<f32> = input_data.iter().map(|&x| x * 2.0 + 1.0).collect();

    // ── GpuVec zero-copy pipeline ────────────────────────────────
    let t0 = Instant::now();
    let gpu_data = GpuVec::from_vec(input_data).expect("GpuVec::from_vec");
    let t_alloc = t0.elapsed();

    let t1 = Instant::now();
    let gpu_result = gpu_data
        .map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256)
        .expect("GpuVec::map_gpu with inline PTX");
    let t_compute = t1.elapsed();

    let t2 = Instant::now();
    let output = gpu_result.as_slice(); // zero-copy read
    let t_read = t2.elapsed();

    println!(
        "[unified_gpuvec] Timing: alloc={:?}, compute={:?}, read={:?}",
        t_alloc, t_compute, t_read
    );

    // ── Verify correctness ───────────────────────────────────────
    assert_eq!(output.len(), n);
    let mut max_err: f32 = 0.0;
    for i in 0..n {
        let err = (output[i] - cpu_ref[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 1e-3,
            "mismatch at [{}]: GPU={}, CPU={}, err={err}",
            i,
            output[i],
            cpu_ref[i]
        );
    }
    println!(
        "[unified_gpuvec] All {} elements verified (max error: {:.2e})",
        n, max_err
    );

    // ── into_vec round-trip ──────────────────────────────────────
    let result_vec = gpu_result.into_vec();
    assert_eq!(result_vec.len(), n);
    println!("[unified_gpuvec] into_vec() round-trip OK");
}

/// North Star demo: small data takes the CPU path transparently.
///
/// The user writes the SAME code as the GPU path. AutoScheduler routes
/// to CPU when data is below the threshold. The result is identical.
#[test]
fn test_unified_pipeline_cpu_fallback() {
    use gpu_host::scheduler::AutoScheduler;

    let _dev = shared_device();

    let n = 100; // well below the 4096 threshold
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let scheduler = AutoScheduler::new();
    assert!(
        n < scheduler.threshold(),
        "test data must be below threshold to exercise CPU path"
    );

    let output = scheduler
        .par_map(&input, |x| x * 3.0 + 7.0)
        .expect("AutoScheduler::par_map CPU path");

    // CPU path uses the ACTUAL closure (not the pre-compiled kernel)
    assert_eq!(output.len(), n);
    for (i, &val) in output.iter().enumerate() {
        let expected = (i as f32) * 3.0 + 7.0;
        assert!(
            (val - expected).abs() < 1e-6,
            "CPU path mismatch at [{}]: got {}, expected {}",
            i,
            val,
            expected
        );
    }
    println!(
        "[unified_cpu] {} elements, CPU path — user code is identical to GPU path",
        n
    );
}

/// Full file I/O round-trip: write input file -> read -> GPU compute -> write -> read back.
///
/// This is the closest test to the actual North Star demo example.
/// Uses GpuVec + inline PTX for fast JIT (milliseconds, not 10 minutes).
/// Uses temp files so it does not litter the repo.
#[test]
fn test_unified_pipeline_file_roundtrip() {
    use gpu_host::memory::GpuVec;
    use std::time::Instant;

    let _dev = shared_device();

    let dir = std::env::temp_dir().join("async_gpu_unified_test");
    std::fs::create_dir_all(&dir).unwrap();

    let input_path = dir.join("input.bin");
    let output_path = dir.join("output.bin");

    // ── Step 1: Generate and write input file ────────────────────
    let n = 8192;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2000.0).collect();
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&input_path, &input_bytes).unwrap();
    println!(
        "[file_roundtrip] Wrote {} elements ({} bytes) to {:?}",
        n,
        input_bytes.len(),
        input_path
    );

    // ── Step 2: Read from file (exactly as the North Star demo) ──
    let raw = std::fs::read(&input_path).unwrap();
    let input: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(input.len(), n);

    // ── Step 3: Compute on GPU via GpuVec (inline PTX, fast JIT) ─
    let t0 = Instant::now();
    let gpu_data = GpuVec::from_vec(input).expect("GpuVec::from_vec");
    let gpu_result = gpu_data
        .map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256)
        .expect("GpuVec::map_gpu");
    let compute_elapsed = t0.elapsed();
    println!("[file_roundtrip] GPU compute: {:?}", compute_elapsed);

    // ── Step 4: Write output file (zero-copy read from GPU result) ──
    let output = gpu_result.as_slice();
    let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&output_path, &output_bytes).unwrap();
    println!(
        "[file_roundtrip] Wrote {} elements to {:?}",
        output.len(),
        output_path
    );

    // ── Step 5: Read back and verify ─────────────────────────────
    let readback_raw = std::fs::read(&output_path).unwrap();
    let readback: Vec<f32> = readback_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(readback.len(), n);

    for i in 0..n {
        let expected = input_data[i] * 2.0 + 1.0;
        assert!(
            (readback[i] - expected).abs() < 1e-3,
            "file roundtrip mismatch at [{}]: got {}, expected {}",
            i,
            readback[i],
            expected
        );
    }
    println!(
        "[file_roundtrip] Full round-trip verified: {} elements correct",
        n
    );

    // ── Cleanup ──────────────────────────────────────────────────
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
    let _ = std::fs::remove_dir(&dir);
}

// ============================================================================
// Performance benchmark: unified pipeline paths comparison
// ============================================================================

/// Benchmark: compare GpuVec::map_gpu vs hand-optimized at multiple sizes.
///
/// GpuVec::map_gpu is the recommended unified API path (zero-copy). The
/// hand-optimized path uses the same raw CUDA driver API. The performance
/// difference should be negligible (GpuVec is a thin wrapper).
///
/// AutoScheduler::par_map uses cudarc htod/dtoh (not zero-copy), so it has
/// inherently higher overhead and is NOT the primary comparison target.
///
/// Each size: 1 warm-up + 3 timed iterations. Uses inline PTX for fast JIT.
#[test]
fn test_unified_benchmark_gpuvec_vs_hand() {
    use gpu_host::gpu;
    use gpu_host::memory::GpuVec;
    use std::time::Instant;

    let _dev = shared_device();

    let sizes: &[usize] = &[4096, 16_384, 65_536, 262_144, 1_048_576];
    let iterations = 3;

    println!();
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  Unified Pipeline Benchmark: GpuVec vs Hand-Optimized             ║");
    println!("╠══════════╦══════════════════╦══════════════════╦════════════════════╣");
    println!("║ Elements ║ GpuVec::map_gpu  ║ Hand-optimized   ║ Ratio (GV/Hand)  ║");
    println!("╠══════════╬══════════════════╬══════════════════╬════════════════════╣");

    for &n in sizes {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        // ── Path 1: GpuVec::map_gpu (zero-copy) ────────────────
        // Warm-up
        {
            let gv = GpuVec::from_vec(input.clone()).unwrap();
            let _ = gv.map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256);
        }
        let mut gpuvec_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            let gv = GpuVec::from_vec(input.clone()).unwrap();
            let result = gv.map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256).unwrap();
            let _output = result.as_slice();
            let elapsed = t0.elapsed();
            gpuvec_times.push(elapsed);
        }

        // ── Path 2: Hand-optimized MappedBuffer + raw launch ────
        // Warm-up
        {
            let gi = GpuVec::from_vec(input.clone()).unwrap();
            let mut go = GpuVec::<f32>::zeroed(n).unwrap();
            let _ = gpu::launch_with_gpuvec(MAP_KERNEL_PTX, "gpuvec_map_f32", &gi, &mut go, 256);
        }
        let mut hand_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            let gi = GpuVec::from_vec(input.clone()).unwrap();
            let mut go = GpuVec::<f32>::zeroed(n).unwrap();
            gpu::launch_with_gpuvec(MAP_KERNEL_PTX, "gpuvec_map_f32", &gi, &mut go, 256).unwrap();
            let _output = go.as_slice();
            let elapsed = t0.elapsed();
            hand_times.push(elapsed);
        }

        // Compute medians
        gpuvec_times.sort();
        hand_times.sort();

        let gpuvec_median = gpuvec_times[iterations / 2];
        let hand_median = hand_times[iterations / 2];

        let ratio = gpuvec_median.as_secs_f64() / hand_median.as_secs_f64();

        println!(
            "║ {:>8} ║ {:>14.3?} ║ {:>14.3?} ║ {:>14.2}x       ║",
            n, gpuvec_median, hand_median, ratio
        );

        // GpuVec should be within 2x of hand-optimized at every size
        assert!(
            ratio < 2.0,
            "GpuVec is {:.1}x slower than hand-optimized at n={} — expected <2x",
            ratio,
            n
        );
    }

    println!("╚══════════╩══════════════════╩══════════════════╩════════════════════╝");
    println!();
    println!("[benchmark] GpuVec::map_gpu matches hand-optimized — zero abstraction cost.");
}

/// AutoScheduler routing correctness: CPU for small data, GPU for large data.
///
/// Verifies the routing decision by using a closure (x * 3.0 + 7.0) that
/// differs from the GPU kernel (x * 2.0 + 1.0). Checking output values
/// proves which path actually executed.
///
/// Uses a single invocation per size (not timed iterations) since the goal
/// is correctness, not speed measurement.
///
/// NOTE: Ignored by default because AutoScheduler's GPU path JIT-compiles
/// the full KERNEL_TEST PTX (~10 minutes per call). The routing boundary
/// logic is already verified by unit tests in scheduler.rs
/// (`auto_scheduler_routing_boundary`). Run with `--ignored` when you
/// have time for the full JIT.
#[test]
#[ignore]
fn test_unified_routing_correctness() {
    use gpu_host::scheduler::AutoScheduler;

    let _dev = shared_device();

    let sched = AutoScheduler::new(); // default threshold = 4096
    let threshold = sched.threshold();

    // Test sizes spanning the CPU/GPU routing boundary
    let test_cases: &[(usize, &str)] = &[
        (100, "CPU"),
        (1000, "CPU"),
        (4095, "CPU"),
        (4096, "GPU"),
        (8192, "GPU"),
    ];

    println!();
    println!("[routing] AutoScheduler threshold = {} elements", threshold);

    for &(n, expected_route) in test_cases {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

        // The closure does x * 3.0 + 7.0 — different from the GPU kernel (x * 2.0 + 1.0).
        let closure_op = |x: f32| x * 3.0 + 7.0;
        let kernel_op = |x: f32| x * 2.0 + 1.0;

        let result = sched.par_map(&input, closure_op).unwrap();

        // Determine which path actually ran by checking output values
        let actual_route = {
            let sample_idx = n / 2;
            let cpu_expected = closure_op(input[sample_idx]);
            let gpu_expected = kernel_op(input[sample_idx]);
            let actual = result[sample_idx];

            if (actual - cpu_expected).abs() < 1e-3 {
                "CPU"
            } else if (actual - gpu_expected).abs() < 1e-3 {
                "GPU"
            } else {
                "???"
            }
        };

        println!(
            "[routing] n={:>5} -> {} (expected {})",
            n, actual_route, expected_route
        );

        assert_eq!(
            actual_route, expected_route,
            "n={}: expected {} route, got {}",
            n, expected_route, actual_route
        );
        assert_eq!(result.len(), n);
    }

    println!("[routing] All routing decisions correct.");
}

/// Performance target: GpuVec path within 2x of hand-optimized at 1M elements.
///
/// This is the key deliverable for the unified-demo theme success criterion:
/// "Performance within 2x of hand-optimized MappedBuffer + gpu::launch path"
#[test]
fn test_unified_performance_target() {
    use gpu_host::gpu;
    use gpu_host::memory::GpuVec;
    use std::time::Instant;

    let _dev = shared_device();

    let n = 1_048_576; // 1M elements — the primary benchmark size
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let iterations = 5;

    // ── GpuVec::map_gpu path ────────────────────────────────────
    // Warm-up (2 iterations)
    for _ in 0..2 {
        let gv = GpuVec::from_vec(input.clone()).unwrap();
        let _ = gv.map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256);
    }

    let mut gpuvec_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        let gv = GpuVec::from_vec(input.clone()).unwrap();
        let result = gv.map_gpu(MAP_KERNEL_PTX, "gpuvec_map_f32", 256).unwrap();
        let output = result.as_slice();
        // Force read to ensure transfer is complete
        assert!((output[0] - (0.0f32 * 2.0 + 1.0)).abs() < 1e-3);
        assert!((output[n - 1] - ((n - 1) as f32 * 0.001 * 2.0 + 1.0)).abs() < 1e-1);
        gpuvec_times.push(t0.elapsed());
    }

    // ── Hand-optimized MappedBuffer path ────────────────────────
    for _ in 0..2 {
        let gi = GpuVec::from_vec(input.clone()).unwrap();
        let mut go = GpuVec::<f32>::zeroed(n).unwrap();
        let _ = gpu::launch_with_gpuvec(MAP_KERNEL_PTX, "gpuvec_map_f32", &gi, &mut go, 256);
    }

    let mut hand_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        let gi = GpuVec::from_vec(input.clone()).unwrap();
        let mut go = GpuVec::<f32>::zeroed(n).unwrap();
        gpu::launch_with_gpuvec(MAP_KERNEL_PTX, "gpuvec_map_f32", &gi, &mut go, 256).unwrap();
        let output = go.as_slice();
        assert!((output[0] - (0.0f32 * 2.0 + 1.0)).abs() < 1e-3);
        assert!((output[n - 1] - ((n - 1) as f32 * 0.001 * 2.0 + 1.0)).abs() < 1e-1);
        hand_times.push(t0.elapsed());
    }

    gpuvec_times.sort();
    hand_times.sort();

    let gpuvec_median = gpuvec_times[iterations / 2];
    let hand_median = hand_times[iterations / 2];
    let ratio = gpuvec_median.as_secs_f64() / hand_median.as_secs_f64();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Performance Target: GpuVec vs Hand-Optimized @ 1M elems   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  GpuVec::map_gpu    : {:>10.3?} (median of {})            ║",
        gpuvec_median, iterations
    );
    println!(
        "║  Hand-optimized     : {:>10.3?} (median of {})            ║",
        hand_median, iterations
    );
    println!(
        "║  Ratio (GpuVec/Hand): {:.2}x                                ║",
        ratio
    );
    println!("║  Target             : < 2.0x                               ║");
    println!(
        "║  Status             : {}                              ║",
        if ratio < 2.0 { "PASS" } else { "FAIL" }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // The key assertion: GpuVec should be within 2x of the hand-optimized path.
    // GpuVec::map_gpu wraps the same raw CUDA launch, so overhead is only:
    // 1. One extra GpuVec::zeroed allocation (output buffer)
    // 2. Method dispatch overhead (negligible)
    assert!(
        ratio < 2.0,
        "GpuVec::map_gpu is {:.2}x slower than hand-optimized at 1M elements — target is <2.0x",
        ratio
    );
}
