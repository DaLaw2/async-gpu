//! Hello GPU — host binary demonstrating the full async_gpu stack.
//!
//! Runs four GPU kernels in sequence:
//! 1. vector_add — pure compute, no hostcall
//! 2. hello_gpu — PRINT hostcall
//! 3. file_io_demo — file OPEN + WRITE + CLOSE from GPU
//! 4. bulk_read_demo — bulk READ via sideband buffer
//!
//! Uses gpu-host's HostcallBuffer for the listener, which handles all
//! service types with I/O thread separation (ADR-6).

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::hostcall::HostcallBuffer;

// Embed the PTX compiled by build.rs
const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

/// Allocate pinned, device-mapped memory for a single u32.
unsafe fn alloc_mapped_u32() -> (*mut u32, sys::CUdeviceptr) {
    let cu = cuda_lib();
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, std::mem::size_of::<u32>(), flags);
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc failed");

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    assert_eq!(
        result,
        sys::CUresult::CUDA_SUCCESS,
        "cuMemHostGetDevicePointer failed"
    );

    (host_ptr as *mut u32, dev_ptr)
}

fn main() {
    println!("=== Hello GPU Example ===\n");

    // Initialize CUDA
    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("[host] CUDA device initialized.");

    // Load PTX module (auto-compiled by build.rs)
    let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_PTX);
    dev.load_ptx(
        ptx,
        "hello",
        &["hello_gpu", "vector_add", "file_io_demo", "bulk_read_demo"],
    )
    .expect("Failed to load PTX module");
    println!("[host] PTX module loaded.\n");

    // ---- Demo 1: vector_add (pure compute, no hostcall) ----
    println!("--- Demo 1: vector_add ---");
    {
        const N: usize = 64;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

        let a_dev = dev.htod_sync_copy(&a).unwrap();
        let b_dev = dev.htod_sync_copy(&b).unwrap();
        let mut c_dev = dev.alloc_zeros::<f32>(N).unwrap();

        let f = dev.get_func("hello", "vector_add").unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, N as u32)).unwrap() };
        let result = dev.dtoh_sync_copy(&c_dev).unwrap();

        let ok = result.iter().all(|&v| (v - N as f32).abs() < 0.001);
        println!("[host] vector_add: {}\n", if ok { "PASSED" } else { "FAILED" });
    }

    // ---- Demos 2-4: hostcall-based kernels ----
    // Create HostcallBuffer (handles all services: PRINT, FILE, BULK, PANIC, etc.)
    let hcbuf = HostcallBuffer::new(8).expect("HostcallBuffer allocation failed");
    let (result_ptr, result_dev) = unsafe { alloc_mapped_u32() };
    let (bytes_read_ptr, bytes_read_dev) = unsafe { alloc_mapped_u32() };

    let cfg1 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Use thread::scope so the listener thread borrows &hcbuf safely
    std::thread::scope(|scope| {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {}", s);
            });
        });

        // ---- Demo 2: hello_gpu (PRINT hostcall) ----
        println!("--- Demo 2: hello_gpu (PRINT hostcall) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("hello", "hello_gpu").unwrap();
            unsafe {
                f.launch(cfg1, (hcbuf.dev_ptr as u64, result_dev as u64))
                    .unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            println!(
                "[host] hello_gpu: {}\n",
                if r == 1 { "PASSED" } else { "FAILED" }
            );
        }

        // ---- Demo 3: file_io_demo (OPEN + WRITE + CLOSE) ----
        println!("--- Demo 3: file_io_demo (file I/O from GPU) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("hello", "file_io_demo").unwrap();
            unsafe {
                f.launch(cfg1, (hcbuf.dev_ptr as u64, result_dev as u64))
                    .unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
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
            std::ptr::write_volatile(result_ptr, 0);
            std::ptr::write_volatile(bytes_read_ptr, 0);
        }
        {
            let f = dev.get_func("hello", "bulk_read_demo").unwrap();
            unsafe {
                f.launch(
                    cfg1,
                    (
                        hcbuf.dev_ptr as u64,
                        hcbuf.sideband_dev_ptr as u64,
                        result_dev as u64,
                        bytes_read_dev as u64,
                    ),
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            let n = unsafe { std::ptr::read_volatile(bytes_read_ptr) };
            println!(
                "[host] bulk_read_demo: {} ({} bytes read)\n",
                if r == 1 { "PASSED" } else { "FAILED" },
                n
            );
        }

        // Shutdown listener
        hcbuf.signal_shutdown();
        let _ = listener;
    });

    // Cleanup
    unsafe {
        let cu = cuda_lib();
        cu.cuMemFreeHost(result_ptr as *mut std::ffi::c_void);
        cu.cuMemFreeHost(bytes_read_ptr as *mut std::ffi::c_void);
    }
    let _ = std::fs::remove_file("gpu_output.txt");

    println!("=== All demos complete! ===");
}
