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
