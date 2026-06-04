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

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const MAX_WARPS: usize = 32;

const STATUS_IDLE: u32 = 0;
const STATUS_ASSIGNED: u32 = 1;
const STATUS_RUNNING: u32 = 2;
const STATUS_DONE: u32 = 3;
const STATUS_EXIT: u32 = 4;

#[allow(clippy::declare_interior_mutable_const)]
const ATOMIC_U32_ZERO: AtomicU32 = AtomicU32::new(STATUS_IDLE);
#[allow(clippy::declare_interior_mutable_const)]
const ATOMIC_U64_ZERO: AtomicU64 = AtomicU64::new(0);

static WARP_STATUS: [AtomicU32; MAX_WARPS] = [ATOMIC_U32_ZERO; MAX_WARPS];
static WARP_FN: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];
static WARP_DATA: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];
static WARP_RESULT: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];

static NUM_WARPS: AtomicU32 = AtomicU32::new(0);

// Per-warp scratch buffer for closure data + result (256 bytes each)
const SCRATCH_SIZE: usize = 256;
#[allow(clippy::declare_interior_mutable_const)]
const SCRATCH_ROW: [AtomicU32; SCRATCH_SIZE / 4] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU32 = AtomicU32::new(0);
    [ZERO; SCRATCH_SIZE / 4]
};
static SCRATCH: [[AtomicU32; SCRATCH_SIZE / 4]; MAX_WARPS] = [SCRATCH_ROW; MAX_WARPS];

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
/// Uses atomic polling instead of bar.sync for synchronization, so it
/// works correctly with patched std (where std init may cause warp
/// divergence before reaching the barrier).
pub fn gpu_main<F: FnOnce()>(main_fn: F) {
    let wid = warp_id();
    let n_warps = crate::index::block_dim_x() / 32;
    if n_warps == 0 {
        return;
    }

    if wid == 0 && lane_id() == 0 {
        NUM_WARPS.store(n_warps, Ordering::Release);
        for slot in WARP_STATUS.iter().skip(1).take(n_warps as usize - 1) {
            slot.store(STATUS_IDLE, Ordering::Relaxed);
        }
    }

    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("bar.sync 0;");
    }

    if wid == 0 {
        main_fn();

        if lane_id() == 0 {
            for slot in WARP_STATUS.iter().skip(1).take(n_warps as usize - 1) {
                slot.store(STATUS_EXIT, Ordering::Release);
            }
        }
    } else if (wid as usize) < MAX_WARPS {
        worker_loop(wid as usize);
    }

    // Final barrier: all warps sync before kernel exit
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("bar.sync 0;");
    }
}

/// Like `gpu_main` but uses atomic polling instead of `bar.sync`.
///
/// Use this when bar.sync is unreliable (e.g., std-compiled kernels where
/// std init may cause warp divergence before the barrier). Requires a FRESH
/// module (no stale globals from prior launches).
pub fn gpu_main_poll<F: FnOnce()>(main_fn: F) {
    let wid = warp_id();
    let n_warps = crate::index::block_dim_x() / 32;
    if n_warps == 0 {
        return;
    }

    if wid == 0 {
        if lane_id() == 0 {
            for slot in WARP_STATUS.iter().skip(1).take(n_warps as usize - 1) {
                slot.store(STATUS_IDLE, Ordering::Relaxed);
            }
            NUM_WARPS.store(n_warps, Ordering::Release);
        }

        main_fn();

        if lane_id() == 0 {
            for slot in WARP_STATUS.iter().skip(1).take(n_warps as usize - 1) {
                slot.store(STATUS_EXIT, Ordering::Release);
            }
        }
    } else if (wid as usize) < MAX_WARPS {
        if lane_id() == 0 {
            while NUM_WARPS.load(Ordering::Acquire) == 0 {
                nanosleep_short();
            }
        }
        worker_loop(wid as usize);
    }
}

fn worker_loop(wid: usize) {
    loop {
        let status = WARP_STATUS[wid].load(Ordering::Acquire);
        match status {
            STATUS_ASSIGNED | STATUS_COOPERATIVE => {
                if lane_id() == 0 {
                    WARP_STATUS[wid].store(STATUS_RUNNING, Ordering::Release);
                }

                let fn_ptr = WARP_FN[wid].load(Ordering::Acquire);
                let data_ptr = WARP_DATA[wid].load(Ordering::Acquire);

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

    // Find an idle warp (linear scan from warp 1), spin-wait if none available
    let target_warp = loop {
        let found = (1..n_warps).find(|&i| WARP_STATUS[i].load(Ordering::Acquire) == STATUS_IDLE);
        if let Some(w) = found {
            break w;
        }
        nanosleep_short();
    };

    // Use the module-level SCRATCH buffer for closure data + result

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
    WARP_FN[target_warp].store(trampoline_fn as usize as u64, Ordering::Relaxed);
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
    (NUM_WARPS.load(Ordering::Relaxed) as usize).saturating_sub(1)
}

/// Returns the current thread's ID (warp index).
pub fn current_id() -> u32 {
    warp_id()
}

/// Yield the current thread briefly.
pub fn yield_now() {
    nanosleep_short();
}

// ============================================================
// Cooperative Compute — all warps execute in data-parallel mode
// ============================================================

const STATUS_COOPERATIVE: u32 = 5;

/// Execute a closure cooperatively across all warps.
///
/// Called from the main thread (warp 0). Wakes all worker warps to execute
/// the same closure in parallel. Each warp uses `current_id()` and
/// `available_parallelism()` to determine its data partition.
///
/// After the closure returns on all warps, control returns to the main thread.
///
/// # Example
///
/// ```rust,ignore
/// thread::gpu_main(|| {
///     // Sequential: only warp 0
///     let data = read_file("input.bin");
///
///     // Cooperative: ALL warps participate
///     thread::cooperative(|| {
///         let wid = thread::current_id() as usize;
///         let n_warps = thread::available_parallelism() + 1;
///         for i in (wid..data.len()).step_by(n_warps) {
///             output[i] = data[i] * 2.0;
///         }
///     });
///
///     // Sequential: back to warp 0 only
///     write_file("output.bin", &output);
/// });
/// ```
/// # Safety
///
/// The closure must be safe to call from all warps simultaneously.
/// The caller must ensure proper data partitioning (no data races).
pub unsafe fn cooperative<F: Fn()>(f: &F) {
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    if n_warps <= 1 {
        f();
        return;
    }

    fn trampoline<F: Fn()>(raw: *mut u8) {
        let f = unsafe { &*(raw as *const F) };
        f();
    }

    // Copy closure to each worker's SCRATCH buffer (same mechanism as spawn).
    // This ensures each warp reads from its own known-good global memory.
    let closure_size = core::mem::size_of::<F>();
    assert!(
        closure_size <= SCRATCH_SIZE,
        "cooperative closure too large"
    );

    let trampoline_fn = trampoline::<F> as fn(*mut u8);

    if lane_id() == 0 {
        for i in 1..n_warps {
            let scratch_ptr = SCRATCH[i].as_ptr() as *mut u8;
            core::ptr::copy_nonoverlapping(f as *const F as *const u8, scratch_ptr, closure_size);
            WARP_FN[i].store(trampoline_fn as usize as u64, Ordering::Relaxed);
            WARP_DATA[i].store(scratch_ptr as u64, Ordering::Relaxed);
            WARP_STATUS[i].store(STATUS_COOPERATIVE, Ordering::Release);
        }
    }

    // Warp 0 also executes directly
    f();

    #[allow(clippy::needless_range_loop)]
    for i in 1..n_warps {
        loop {
            let s = WARP_STATUS[i].load(Ordering::Acquire);
            if s == STATUS_DONE {
                WARP_STATUS[i].store(STATUS_IDLE, Ordering::Release);
                break;
            }
            nanosleep_short();
        }
    }
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
    NUM_WARPS.load(Ordering::Relaxed).saturating_sub(1)
}

/// Return the current warp index (thread ID).
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_current_id() -> u32 {
    warp_id()
}
