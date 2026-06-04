//! Test kernels for gpu_runtime::thread — warp-as-thread model.
//! Also demonstrates extern "gpu-kernel" ABI (when compiled with patched rustc).
//!
//! thread_spawn_test: spawn 2 threads, each computes a value, join results.
//! Launched with block_dim=(128, 1, 1) = 4 warps.

use gpu_runtime::thread;

/// Test kernel: spawn 2 threads, join, write results.
///
/// Warp 0 is the main thread. It spawns work on warps 1 and 2.
/// Each spawned closure computes a simple value and returns it.
/// Results are written to the output buffer.
///
/// Launch with: block_dim=(128,1,1), shared_mem_bytes=0
/// Output: result[0] = thread_1_result (42), result[1] = thread_2_result (99)
///         result[2] = available_parallelism (3 with 4 warps)
///         result[3] = main_thread_id (0)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn thread_spawn_test(result: *mut u32) {
    thread::gpu_main(|| {
        // Record main thread info
        let main_id = thread::current_id();
        let parallelism = thread::available_parallelism();

        // Spawn two threads doing independent work
        let h1 = thread::spawn(|| -> u32 { 42u32 });
        let h2 = thread::spawn(|| -> u32 { 99u32 });

        // Join and collect results
        let r1 = h1.join();
        let r2 = h2.join();

        // Write results (only lane 0 of warp 0 writes)
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, r1);
                core::ptr::write_volatile(result.add(1), r2);
                core::ptr::write_volatile(result.add(2), parallelism as u32);
                core::ptr::write_volatile(result.add(3), main_id);
            }
        }
    });
}

/// Test kernel: spawn and reuse — spawn 4 threads sequentially on 3 available warps.
///
/// Verifies that warps return to IDLE after join and can be reused.
///
/// Launch with: block_dim=(128,1,1)
/// Output: result[0..3] = sum of each spawned computation
#[no_mangle]
pub unsafe extern "ptx-kernel" fn thread_reuse_test(result: *mut u32) {
    thread::gpu_main(|| {
        let mut total: u32 = 0;

        // Spawn 4 tasks sequentially (only 3 warps available, so one must be reused)
        for i in 0u32..4 {
            let h = thread::spawn(move || -> u32 { (i + 1) * 10 });
            let r = h.join();
            total += r;
            if gpu_runtime::index::thread_idx_x() == 0 {
                unsafe {
                    core::ptr::write_volatile(result.add(i as usize), r);
                }
            }
        }

        // result[4] = total (10 + 20 + 30 + 40 = 100)
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result.add(4), total);
            }
        }
    });
}

/// Debug: cooperative with zero-capture closure — just writes to a global static.
static COOP_RESULT: [core::sync::atomic::AtomicU32; 4] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 4]
};

#[no_mangle]
pub unsafe extern "ptx-kernel" fn cooperative_debug(result: *mut u32) {
    thread::gpu_main(|| {
        // Zero-capture closure: all data accessed via statics
        unsafe {
            thread::cooperative(&|| {
                let wid = thread::current_id();
                let lid = gpu_runtime::index::thread_idx_x() % 32;
                if lid == 0 {
                    COOP_RESULT[wid as usize].store(wid + 100, core::sync::atomic::Ordering::Relaxed);
                }
            });
        }

        // Copy results to output
        if gpu_runtime::index::thread_idx_x() == 0 {
            for i in 0..4usize {
                core::ptr::write_volatile(
                    result.add(i),
                    COOP_RESULT[i].load(core::sync::atomic::Ordering::Relaxed),
                );
            }
        }
    });
}

/// Test: cooperative compute — all warps process data in parallel.
///
/// Fills an output array: output[i] = i * 2 + 1.
/// Uses cooperative() so all 4 warps share the work.
/// Data pointer passed via global atomic (not closure capture).
///
/// Launch with: block_dim=(128,1,1)
/// Output: result[i] = i * 2 + 1 for i in 0..256
static COOP_OUT_PTR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static COOP_N: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

#[no_mangle]
pub unsafe extern "ptx-kernel" fn cooperative_compute_test(result: *mut u32) {
    thread::gpu_main(|| {
        // Pass data via global atomics (closure captures point to local memory)
        COOP_OUT_PTR.store(result as u64, core::sync::atomic::Ordering::Relaxed);
        COOP_N.store(256, core::sync::atomic::Ordering::Relaxed);

        unsafe {
            thread::cooperative(&|| {
                let out = COOP_OUT_PTR.load(core::sync::atomic::Ordering::Relaxed) as *mut u32;
                let n = COOP_N.load(core::sync::atomic::Ordering::Relaxed);
                let wid = thread::current_id();
                let total_warps = (thread::available_parallelism() + 1) as u32;
                let lid = gpu_runtime::index::thread_idx_x() % 32;

                if lid == 0 {
                    let mut i = wid;
                    while i < n {
                        core::ptr::write_volatile(out.add(i as usize), i * 2 + 1);
                        i += total_warps;
                    }
                }
            });
        }
    });
}

/// Demo: extern "gpu-kernel" ABI — the native Rust GPU entry point.
///
/// This is identical to thread_spawn_test but uses extern "gpu-kernel" instead
/// of extern "ptx-kernel". Requires patched rustc (feature = "gpu_kernel_abi").
///
/// Launch with: block_dim=(128,1,1)
/// Output: result[0] = 42, result[1] = 99
#[cfg(feature = "gpu_kernel_abi")]
#[no_mangle]
pub unsafe extern "gpu-kernel" fn gpu_kernel_demo(result: *mut u32) {
    thread::gpu_main(|| {
        let h1 = thread::spawn(|| -> u32 { 42u32 });
        let h2 = thread::spawn(|| -> u32 { 99u32 });
        let r1 = h1.join();
        let r2 = h2.join();

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, r1);
                core::ptr::write_volatile(result.add(1), r2);
            }
        }
    });
}
