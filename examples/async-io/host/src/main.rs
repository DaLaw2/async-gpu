//! Async I/O — host binary demonstrating multi-step file I/O from GPU.
//!
//! Runs two GPU kernels:
//! 1. write_pipeline — write 3 files from GPU in sequence
//! 2. transform_pipeline — read a file, uppercase on GPU, write result

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::hostcall::HostcallBuffer;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

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
    println!("=== Async I/O Example ===\n");

    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("[host] CUDA device initialized.");

    let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_PTX);
    dev.load_ptx(ptx, "asyncio", &["write_pipeline", "transform_pipeline"])
        .expect("Failed to load PTX module");
    println!("[host] PTX module loaded.\n");

    let hcbuf = HostcallBuffer::new(8).expect("HostcallBuffer allocation failed");
    let (result_ptr, result_dev) = unsafe { alloc_mapped_u32() };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    std::thread::scope(|scope| {
        let listener = scope.spawn(|| {
            hcbuf.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {}", s);
            });
        });

        // ---- Demo 1: write_pipeline ----
        println!("--- Demo 1: write_pipeline (3 files from GPU) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("asyncio", "write_pipeline").unwrap();
            unsafe {
                f.launch(cfg, (hcbuf.dev_ptr as u64, result_dev as u64))
                    .unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            println!("[host] write_pipeline: {}/3 files written", r);

            // Verify files exist
            for i in 0..3 {
                let name = format!("gpu_file_{i}.txt");
                if let Ok(content) = std::fs::read_to_string(&name) {
                    println!("[host]   {name}: {:?}", content.trim());
                }
            }
            println!();
        }

        // ---- Demo 2: transform_pipeline ----
        println!("--- Demo 2: transform_pipeline (read → uppercase → write) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("asyncio", "transform_pipeline").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        hcbuf.dev_ptr as u64,
                        hcbuf.sideband_dev_ptr as u64,
                        result_dev as u64,
                    ),
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
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
    });

    // Cleanup
    unsafe {
        let cu = cuda_lib();
        cu.cuMemFreeHost(result_ptr as *mut std::ffi::c_void);
    }
    for name in &[
        "gpu_file_0.txt",
        "gpu_file_1.txt",
        "gpu_file_2.txt",
        "gpu_upper.txt",
    ] {
        let _ = std::fs::remove_file(name);
    }

    println!("=== Async I/O example complete! ===");
}
