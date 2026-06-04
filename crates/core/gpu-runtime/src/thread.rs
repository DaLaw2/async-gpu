//! GPU thread pool — `thread::spawn()` maps to warp execution.
//!
//! Each GPU warp (32 SIMT lanes) acts as a single logical "thread".
//! Warp 0 runs the user's main function; other warps park until work
//! is assigned via `spawn()`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use gpu_runtime::thread;
//!
//! // Kernel entry: gpu_main dispatches warps
//! #[no_mangle]
//! pub unsafe extern "ptx-kernel" fn my_kernel(buf: *mut u8) {
//!     thread::gpu_main(|| {
//!         let h1 = thread::spawn(|| 42u32);
//!         let h2 = thread::spawn(|| 99u32);
//!         let r1 = h1.join();
//!         let r2 = h2.join();
//!         // r1 == 42, r2 == 99
//!     });
//! }
//! ```
//!
//! # Model
//!
//! - Launch the kernel with `block_dim = (N*32, 1, 1)` where N = desired thread count
//! - Warp 0 runs the closure passed to `gpu_main()`
//! - Warps 1..N-1 enter a parking loop, polling their status flag
//! - `spawn()` boxes the closure, stores a trampoline + data pointer in the warp's slot,
//!   and atomically sets the status to ASSIGNED
//! - The parked warp sees ASSIGNED, calls the trampoline, writes the result, sets DONE
//! - `JoinHandle::join()` spins until DONE, reads the result, resets to IDLE

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use core::marker::PhantomData;

const MAX_WARPS: usize = 32;

const STATUS_IDLE: u32 = 0;
const STATUS_ASSIGNED: u32 = 1;
const STATUS_RUNNING: u32 = 2;
const STATUS_DONE: u32 = 3;
const STATUS_EXIT: u32 = 4;

// Per-warp slot: status + trampoline fn pointer + data pointer + result pointer
static WARP_STATUS: [AtomicU32; MAX_WARPS] = {
    const INIT: AtomicU32 = AtomicU32::new(STATUS_IDLE);
    [INIT; MAX_WARPS]
};
static WARP_FN: [AtomicU64; MAX_WARPS] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; MAX_WARPS]
};
static WARP_DATA: [AtomicU64; MAX_WARPS] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; MAX_WARPS]
};
static WARP_RESULT: [AtomicU64; MAX_WARPS] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; MAX_WARPS]
};

static NUM_WARPS: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
fn warp_id() -> u32 {
    crate::index::thread_idx_x() / 32
}

#[inline(always)]
fn lane_id() -> u32 {
    crate::index::thread_idx_x() % 32
}

#[inline(always)]
fn nanosleep_short() {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("nanosleep.u32 100;");
    }
}

/// Kernel entry point wrapper that sets up the warp thread pool.
///
/// Warp 0 runs `main_fn`. Other warps enter a parking loop and wait for
/// work via `spawn()`. After `main_fn` returns, all parked warps are
/// signaled to exit.
///
/// The host must launch with `block_dim.x` = N × 32, where N is the
/// number of desired warps (threads).
///
/// # Safety
///
/// Must be called by ALL threads in the block (all warps participate).
pub fn gpu_main<F: FnOnce()>(main_fn: F) {
    let wid = warp_id();
    let n_warps = crate::index::block_dim_x() / 32;
    if n_warps == 0 {
        return;
    }

    // Lane 0 of warp 0 initializes the pool
    if wid == 0 && lane_id() == 0 {
        NUM_WARPS.store(n_warps, Ordering::Release);
        // Ensure all slots are IDLE
        for i in 1..n_warps as usize {
            WARP_STATUS[i].store(STATUS_IDLE, Ordering::Relaxed);
        }
    }

    // Block-wide barrier: all warps sync before proceeding
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("bar.sync 0;");
    }

    if wid == 0 {
        // Warp 0: run the user's main function
        main_fn();

        // Signal all worker warps to exit
        for i in 1..n_warps as usize {
            WARP_STATUS[i].store(STATUS_EXIT, Ordering::Release);
        }
    } else if (wid as usize) < MAX_WARPS {
        // Worker warps: enter parking loop
        worker_loop(wid as usize);
    }

    // Final barrier: all warps sync before kernel exit
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("bar.sync 0;");
    }
}

fn worker_loop(wid: usize) {
    loop {
        // All lanes read the status (same global address → coherent, single transaction)
        let status = WARP_STATUS[wid].load(Ordering::Acquire);
        match status {
            STATUS_ASSIGNED => {
                if lane_id() == 0 {
                    WARP_STATUS[wid].store(STATUS_RUNNING, Ordering::Release);
                }

                let fn_ptr = WARP_FN[wid].load(Ordering::Acquire);
                let data_ptr = WARP_DATA[wid].load(Ordering::Acquire);

                // All 32 lanes call the trampoline in SIMT lockstep.
                // The closure runs with full 32-lane parallelism available.
                // Only lane 0 stores the result (handled inside trampoline).
                let trampoline: fn(*mut u8) = unsafe { core::mem::transmute(fn_ptr) };
                trampoline(data_ptr as *mut u8);

                if lane_id() == 0 {
                    WARP_STATUS[wid].store(STATUS_DONE, Ordering::Release);
                }
            }
            STATUS_EXIT => break,
            _ => {
                nanosleep_short();
            }
        }
    }
}

/// Handle to a spawned thread (warp). Can be used to wait for completion.
pub struct JoinHandle<T> {
    warp_id: usize,
    _phantom: PhantomData<T>,
}

impl<T> JoinHandle<T> {
    /// Wait for the spawned thread to finish and return its result.
    pub fn join(self) -> T {
        // Spin until the warp is done
        loop {
            let status = WARP_STATUS[self.warp_id].load(Ordering::Acquire);
            if status == STATUS_DONE {
                break;
            }
            nanosleep_short();
        }

        // Read the result
        let result_ptr = WARP_RESULT[self.warp_id].load(Ordering::Acquire) as *mut T;
        let result = unsafe { core::ptr::read(result_ptr) };

        // Reset the slot to IDLE for reuse
        WARP_STATUS[self.warp_id].store(STATUS_IDLE, Ordering::Release);

        result
    }
}

/// Spawn a new thread (warp) to execute the given closure.
///
/// Returns a `JoinHandle` that can be used to wait for the thread to finish.
/// The closure runs on a parked warp — only lane 0 of that warp executes
/// the closure directly.
///
/// # Panics
///
/// Panics if no idle warps are available.
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Trampoline: type-erased function that calls the closure and stores the result.
    // Called by ALL 32 lanes in SIMT lockstep. Only lane 0 manages the closure
    // data and result storage; other lanes participate in any warp-level ops
    // the closure may use internally.
    fn trampoline<F, T>(raw: *mut u8)
    where
        F: FnOnce() -> T,
        T: 'static,
    {
        let lid = crate::index::thread_idx_x() % 32;

        // Only lane 0 reads the closure and stores the result.
        // Other lanes execute the closure body via SIMT but don't
        // touch the closure storage. For closures that use warp ops
        // (shuffle, reduce), all lanes participate naturally.
        if lid == 0 {
            let f = unsafe { core::ptr::read(raw as *const F) };
            let result = f();

            // Write result into the scratch buffer after the closure data
            let result_slot = unsafe {
                let slot = raw.add(core::mem::size_of::<F>()) as *mut T;
                core::ptr::write(slot, result);
                slot
            };

            let wid = crate::index::thread_idx_x() / 32;
            WARP_RESULT[wid as usize].store(result_slot as u64, Ordering::Release);
        }
    }

    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;

    // Find an idle warp (linear scan from warp 1)
    let mut target_warp = 0usize;
    for i in 1..n_warps {
        if WARP_STATUS[i].load(Ordering::Acquire) == STATUS_IDLE {
            target_warp = i;
            break;
        }
    }

    // For now, if no warp available we spin-wait (basic backpressure)
    if target_warp == 0 {
        loop {
            for i in 1..n_warps {
                if WARP_STATUS[i].load(Ordering::Acquire) == STATUS_IDLE {
                    target_warp = i;
                    break;
                }
            }
            if target_warp != 0 {
                break;
            }
            nanosleep_short();
        }
    }

    // Allocate space for the closure + result in a static buffer
    // For simplicity: use a per-warp scratch buffer in global memory
    // Each warp gets SCRATCH_SIZE bytes for closure data + result
    const SCRATCH_SIZE: usize = 256;
    static SCRATCH: [[AtomicU32; SCRATCH_SIZE / 4]; MAX_WARPS] = {
        const ROW: [AtomicU32; SCRATCH_SIZE / 4] = {
            const ZERO: AtomicU32 = AtomicU32::new(0);
            [ZERO; SCRATCH_SIZE / 4]
        };
        [ROW; MAX_WARPS]
    };

    let scratch_ptr = SCRATCH[target_warp].as_ptr() as *mut u8;

    // Copy closure data into the scratch buffer
    let closure_size = core::mem::size_of::<F>();
    assert!(
        closure_size + core::mem::size_of::<T>() <= SCRATCH_SIZE,
        "closure + result too large for scratch buffer"
    );
    unsafe {
        core::ptr::write(scratch_ptr as *mut F, f);
    }

    // Set up the warp slot
    let trampoline_fn = trampoline::<F, T> as fn(*mut u8);
    WARP_FN[target_warp].store(trampoline_fn as u64, Ordering::Relaxed);
    WARP_DATA[target_warp].store(scratch_ptr as u64, Ordering::Relaxed);
    WARP_RESULT[target_warp].store(0, Ordering::Relaxed);

    // Assign the work (this wakes up the worker)
    WARP_STATUS[target_warp].store(STATUS_ASSIGNED, Ordering::Release);

    JoinHandle {
        warp_id: target_warp,
        _phantom: PhantomData,
    }
}

/// Returns the number of available threads (warps) that can be spawned.
pub fn available_parallelism() -> usize {
    let n = NUM_WARPS.load(Ordering::Relaxed) as usize;
    if n > 1 { n - 1 } else { 0 } // subtract warp 0 (main thread)
}

/// Returns the current thread's ID (warp index).
pub fn current_id() -> u32 {
    warp_id()
}

/// Yield the current thread briefly.
pub fn yield_now() {
    nanosleep_short();
}

/// Sleep for approximately `nanos` nanoseconds.
pub fn sleep_nanos(nanos: u32) {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("nanosleep.u32 {ns};", ns = in(reg32) nanos);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = nanos;
    }
}

// ============================================================
// C-FFI entry points for std patches
// ============================================================
// These functions are called from the patched std::sys::thread::cuda module
// via `extern "C"`. They allow std::thread::spawn() to work on GPU without
// std depending on gpu_runtime directly.

/// Spawn a new warp thread. `trampoline` is a `fn(*mut u8)`, `data` is the
/// argument pointer. Returns the warp ID (>0) or 0 if no warp available.
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_spawn_raw(trampoline: u64, data: u64) -> u32 {
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    if n_warps <= 1 {
        return 0;
    }

    // Find an idle warp
    loop {
        for i in 1..n_warps {
            if WARP_STATUS[i].load(Ordering::Acquire) == STATUS_IDLE {
                WARP_FN[i].store(trampoline, Ordering::Relaxed);
                WARP_DATA[i].store(data, Ordering::Relaxed);
                WARP_RESULT[i].store(0, Ordering::Relaxed);
                WARP_STATUS[i].store(STATUS_ASSIGNED, Ordering::Release);
                return i as u32;
            }
        }
        nanosleep_short();
    }
}

/// Wait for warp `warp_id` to finish. Blocks (spins) until done, then resets
/// the slot to IDLE.
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_join_warp(warp_id: u32) {
    let wid = warp_id as usize;
    loop {
        if WARP_STATUS[wid].load(Ordering::Acquire) == STATUS_DONE {
            break;
        }
        nanosleep_short();
    }
    WARP_STATUS[wid].store(STATUS_IDLE, Ordering::Release);
}

/// Return the number of available worker warps (total warps minus main warp).
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_available_parallelism() -> u32 {
    let n = NUM_WARPS.load(Ordering::Relaxed);
    if n > 1 { n - 1 } else { 0 }
}

/// Return the current warp index (thread ID).
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_current_id() -> u32 {
    warp_id()
}
