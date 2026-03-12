//! Hello GPU — minimal host binary that loads and runs the kernel.
//!
//! This example demonstrates:
//! 1. PTX auto-compilation via build.rs
//! 2. CUDA device initialization with cudarc
//! 3. Hostcall buffer setup with pinned mapped memory
//! 4. Kernel launch + host listener for PRINT hostcalls
//! 5. Synchronization and result verification

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_protocol::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
// Embed the PTX compiled by build.rs.
// Note: llvm-bitcode-linker may emit `.target sm_30` in the header even when
// compiled with `-C target-cpu=sm_86`. We patch it at load time.
const KERNEL_PTX_RAW: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

/// Allocate pinned, device-mapped memory for a single u32.
unsafe fn alloc_mapped_u32() -> (*mut u32, sys::CUdeviceptr) {
    let cu = cuda_lib();
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, std::mem::size_of::<u32>(), flags);
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc failed");

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer failed");

    (host_ptr as *mut u32, dev_ptr)
}

/// Minimal hostcall listener — handles PRINT and NOP services.
///
/// Runs until shutdown flag is set. In production, use gpu-host's HostcallBuffer.
fn listen_hostcall(buf: *mut u8, num_packets: u16) {
    let doorbell = unsafe { &*(buf.add(BUF_OFF_DOORBELL) as *const AtomicU64) };
    let ready_stack = unsafe { &*(buf.add(BUF_OFF_READY_STACK) as *const AtomicU64) };
    let shutdown = unsafe { &*(buf.add(BUF_OFF_SHUTDOWN) as *const AtomicU32) };

    let mut last_doorbell: u64 = 0;

    loop {
        if shutdown.load(Ordering::Acquire) != 0 {
            break;
        }

        let current = doorbell.load(Ordering::Acquire);
        if current == last_doorbell {
            std::hint::spin_loop();
            continue;
        }
        last_doorbell = current;

        // Grab all ready packets
        let head = ready_stack.swap(null_tagged(), Ordering::AcqRel);
        let mut cur = head;

        while tagged_index(cur) != NULL_INDEX {
            let idx = tagged_index(cur);
            let pkt = unsafe { buf.add(packet_offset(idx)) };

            unsafe {
                let next = std::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
                let control = &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                let ctrl = control.load(Ordering::Acquire);

                if ctrl & CONTROL_FILLED == 0 {
                    cur = next;
                    continue;
                }

                let service = std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32);

                match service {
                    SERVICE_PRINT => {
                        let payload = pkt.add(PKT_OFF_PAYLOAD);
                        let msg_len = std::ptr::read_volatile(payload as *const u64) as usize;
                        let msg_len = msg_len.min(PRINT_MAX_MSG_LEN);
                        let msg_ptr = payload.add(8);
                        let mut msg_buf = [0u8; PRINT_MAX_MSG_LEN];
                        for i in 0..msg_len {
                            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
                        }
                        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid utf8>");
                        println!("[GPU] {}", msg);
                    }
                    SERVICE_NOP => {}
                    other => {
                        eprintln!("[HOST] Unknown service: {}", other);
                    }
                }

                control.store(CONTROL_READY, Ordering::Release);
                cur = next;
            }
        }
    }

    let _ = num_packets; // suppress unused warning
}

/// Allocate and initialize a hostcall buffer.
unsafe fn create_hostcall_buffer(num_packets: u16) -> (*mut u8, sys::CUdeviceptr) {
    let cu = cuda_lib();
    let size = buffer_size(num_packets);

    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);

    let buf = host_ptr as *mut u8;
    std::ptr::write_bytes(buf, 0, size);

    // Initialize header
    std::ptr::write_volatile(buf.add(BUF_OFF_READY_STACK) as *mut u64, null_tagged());
    std::ptr::write_volatile(buf.add(BUF_OFF_DOORBELL) as *mut u64, 0u64);
    std::ptr::write_volatile(buf.add(BUF_OFF_SHUTDOWN) as *mut u32, 0u32);
    std::ptr::write_volatile(buf.add(BUF_OFF_NUM_PACKETS) as *mut u32, num_packets as u32);
    std::ptr::write_volatile(buf.add(BUF_OFF_WARP_SIZE) as *mut u32, WARP_SIZE);

    // Build free stack
    for i in 0..num_packets {
        let pkt = buf.add(packet_offset(i));
        let next = if i + 1 < num_packets {
            make_tagged(0, i + 1)
        } else {
            null_tagged()
        };
        std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next);
        std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
    }
    std::ptr::write_volatile(buf.add(BUF_OFF_FREE_STACK) as *mut u64, make_tagged(0, 0));

    (buf, dev_ptr)
}

fn main() {
    println!("=== Hello GPU Example ===\n");

    // Step 1: Initialize CUDA
    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("CUDA device initialized.");

    // Step 2: Load PTX (auto-compiled by build.rs)
    // Patch the target directive — llvm-bitcode-linker may emit sm_30 instead of sm_86
    let ptx_text = KERNEL_PTX_RAW.replace(".target sm_30", ".target sm_86");
    let ptx = cudarc::nvrtc::Ptx::from_src(&ptx_text);
    dev.load_ptx(ptx, "hello", &["hello_gpu", "vector_add"])
        .expect("Failed to load PTX module");
    println!("PTX module loaded.");

    // Step 3: Run vector_add (no hostcall needed)
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

        let all_correct = result.iter().all(|&v| (v - N as f32).abs() < 0.001);
        println!("vector_add: {} (all elements = {})", if all_correct { "PASSED" } else { "FAILED" }, N);
    }

    // Step 4: Run hello_gpu with hostcall
    {
        let (buf, buf_dev_ptr) = unsafe { create_hostcall_buffer(4) };
        let (result_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32() };
        unsafe { std::ptr::write_volatile(result_ptr, 0) };

        // Start listener thread
        let buf_for_listener = buf as usize; // Send as usize for Send
        let listener = std::thread::spawn(move || {
            listen_hostcall(buf_for_listener as *mut u8, 4);
        });

        // Launch kernel
        let f = dev.get_func("hello", "hello_gpu").unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (buf_dev_ptr as u64, result_dev_ptr as u64)).unwrap();
        }
        dev.synchronize().unwrap();

        // Wait for listener to process, then shut down
        std::thread::sleep(std::time::Duration::from_millis(100));
        let shutdown = unsafe { &*(buf.add(BUF_OFF_SHUTDOWN) as *const AtomicU32) };
        shutdown.store(1, Ordering::Release);
        listener.join().unwrap();

        let result = unsafe { std::ptr::read_volatile(result_ptr) };
        println!("hello_gpu: {} (hostcall print via gpu-runtime)", if result == 1 { "PASSED" } else { "FAILED" });

        // Cleanup
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(buf as *mut std::ffi::c_void);
            cu.cuMemFreeHost(result_ptr as *mut std::ffi::c_void);
        }
    }

    println!("\nDone!");
}
