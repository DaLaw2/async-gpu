//! Test kernels for gpu_runtime::thread — warp-as-thread model.
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
pub unsafe extern "gpu-kernel" fn thread_spawn_test(result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn thread_reuse_test(result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn cooperative_debug(result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn cooperative_compute_test(result: *mut u32) {
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

/// Test: cooperative_map — all warps double each element, zero global atomics.
///
/// Input: result[0..256] pre-filled with i (0..256)
/// Output: result[i] = i * 2 for all i
///
/// Unlike cooperative_compute_test, this uses NO global atomics for data passing.
/// All data flows through cooperative_map's explicit (src, dst, len) parameters.
///
/// Launch with: block_dim=(128,1,1)
/// Output: result[i] = i * 2 for i in 0..256
///
/// The kernel allocates a Vec for input (heap → global address space, visible
/// to all warps), passes pointers to cooperative_map, then copies results out.
static CMAP_INPUT: [core::sync::atomic::AtomicU32; 256] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 256]
};

#[no_mangle]
pub unsafe extern "gpu-kernel" fn cooperative_map_test(result: *mut u32) {
    thread::gpu_main(|| {
        // Initialize input in global static (visible to all warps)
        for i in 0..256u32 {
            CMAP_INPUT[i as usize].store(i, core::sync::atomic::Ordering::Relaxed);
        }

        // cooperative_map: all warps double each element
        // No global atomics, no unsafe, no closure captures
        thread::cooperative_map(
            CMAP_INPUT.as_ptr() as *const u8,
            result as *mut u8,
            256,
            |args| {
                let src = args.src as *const u32;
                let dst = args.dst as *mut u32;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        let v = core::ptr::read_volatile(src.add(i));
                        core::ptr::write_volatile(dst.add(i), v * 2);
                    }
                    i += args.n_warps as usize;
                }
            },
        );
    });
}

/// Test: cooperative_reduce — all warps sum partitions of an array.
///
/// Input: static array [0..256], each warp sums its partition.
/// Output: result[0] = total sum = 0+1+2+...+255 = 32640
///
/// Launch with: block_dim=(128,1,1)
static CREDUCE_INPUT: [core::sync::atomic::AtomicU64; 256] = {
    const Z: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    [Z; 256]
};

#[no_mangle]
pub unsafe extern "gpu-kernel" fn cooperative_reduce_test(result: *mut u64) {
    thread::gpu_main(|| {
        // Initialize input
        for i in 0..256u64 {
            CREDUCE_INPUT[i as usize].store(i, core::sync::atomic::Ordering::Relaxed);
        }

        let total = thread::cooperative_reduce(
            CREDUCE_INPUT.as_ptr() as *const u8,
            256,
            |args| {
                let src = args.src as *const u64;
                let mut sum = 0u64;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        sum += core::ptr::read_volatile(src.add(i));
                    }
                    i += args.n_warps as usize;
                }
                sum
            },
        );

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, total);
            }
        }
    });
}

/// Test: cooperative_map_with_params — elementwise multiply by a scalar parameter.
///
/// Input: static array [0..256]
/// params[0] = scalar = 7
/// Output: result[i] = i * 7
///
/// Launch with: block_dim=(128,1,1)
static CEXT_INPUT: [core::sync::atomic::AtomicU32; 256] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 256]
};

#[no_mangle]
pub unsafe extern "gpu-kernel" fn cooperative_map_ext_test(result: *mut u32) {
    thread::gpu_main(|| {
        // Initialize input
        for i in 0..256u32 {
            CEXT_INPUT[i as usize].store(i, core::sync::atomic::Ordering::Relaxed);
        }

        // Multiply each element by scalar=7 passed via params[0]
        thread::cooperative_map_with_params(
            CEXT_INPUT.as_ptr() as *const u8,
            result as *mut u8,
            256,
            [7, 0, 0, 0],
            |args| {
                let src = args.src as *const u32;
                let dst = args.dst as *mut u32;
                let scale = args.params[0] as u32;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        let v = core::ptr::read_volatile(src.add(i));
                        core::ptr::write_volatile(dst.add(i), v * scale);
                    }
                    i += args.n_warps as usize;
                }
            },
        );
    });
}

/// Demo: extern "gpu-kernel" ABI — the native Rust GPU entry point.
///
/// This is identical to thread_spawn_test but uses extern "gpu-kernel" ABI.
///
/// Launch with: block_dim=(128,1,1)
/// Output: result[0] = 42, result[1] = 99
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
