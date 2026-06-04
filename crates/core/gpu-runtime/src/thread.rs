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
//! pub unsafe extern "gpu-kernel" fn my_kernel(buf: *mut u8) {
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

pub(crate) const MAX_WARPS: usize = 32;

pub(crate) const STATUS_IDLE: u32 = 0;
pub(crate) const STATUS_ASSIGNED: u32 = 1;
const STATUS_RUNNING: u32 = 2;
pub(crate) const STATUS_DONE: u32 = 3;
const STATUS_EXIT: u32 = 4;
/// Warp trapped (panic handler sets this before calling `trap;`).
/// Used by `BlockScope::join_all()` to detect dead warps.
pub(crate) const STATUS_TRAPPED: u32 = 6;

#[allow(clippy::declare_interior_mutable_const)]
const ATOMIC_U32_ZERO: AtomicU32 = AtomicU32::new(STATUS_IDLE);
#[allow(clippy::declare_interior_mutable_const)]
const ATOMIC_U64_ZERO: AtomicU64 = AtomicU64::new(0);

pub(crate) static WARP_STATUS: [AtomicU32; MAX_WARPS] = [ATOMIC_U32_ZERO; MAX_WARPS];
pub(crate) static WARP_FN: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];
pub(crate) static WARP_DATA: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];
pub(crate) static WARP_RESULT: [AtomicU64; MAX_WARPS] = [ATOMIC_U64_ZERO; MAX_WARPS];

pub(crate) static NUM_WARPS: AtomicU32 = AtomicU32::new(0);

/// Debug counter: incremented each time gpu_thread_spawn_raw is called.
static SPAWN_RAW_COUNT: AtomicU32 = AtomicU32::new(0);

/// Read the debug spawn counter (for testing).
#[unsafe(no_mangle)]
pub extern "C" fn gpu_thread_spawn_raw_count() -> u32 {
    SPAWN_RAW_COUNT.load(Ordering::Relaxed)
}

// Per-warp scratch buffer for closure data + result (256 bytes each)
pub(crate) const SCRATCH_SIZE: usize = 256;
#[allow(clippy::declare_interior_mutable_const)]
const SCRATCH_ROW: [AtomicU32; SCRATCH_SIZE / 4] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU32 = AtomicU32::new(0);
    [ZERO; SCRATCH_SIZE / 4]
};
pub(crate) static SCRATCH: [[AtomicU32; SCRATCH_SIZE / 4]; MAX_WARPS] = [SCRATCH_ROW; MAX_WARPS];

#[inline(always)]
pub(crate) fn warp_id() -> u32 {
    crate::index::thread_idx_x() / 32
}

#[inline(always)]
pub(crate) fn lane_id() -> u32 {
    crate::index::thread_idx_x() % 32
}

#[inline(always)]
pub(crate) fn nanosleep_short() {
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
        // Only lane 0 runs main_fn. std::thread::spawn and other
        // heap-allocating code is NOT SIMT-safe across lanes.
        if lane_id() == 0 {
            main_fn();

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

            // Only lane 0 runs main_fn. std::thread::spawn (and any
            // heap-allocating code) is NOT SIMT-safe: each lane would
            // get its own Box allocation, causing duplicate spawn
            // assignments and use-after-free on the trampoline data.
            main_fn();

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

pub(crate) const STATUS_COOPERATIVE: u32 = 5;

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

// ============================================================
// Cooperative Map — data-parallel transform without global atomics
// ============================================================

/// Global argument block for cooperative_map.
/// Written by warp 0, read by all warps via the trampoline.
/// Layout: [in_ptr: u64, out_ptr: u64, len: u64, fn_ptr: u64]
static COOP_MAP_ARGS: [AtomicU64; 4] = [ATOMIC_U64_ZERO; 4];

/// Arguments passed to the cooperative_map user function.
///
/// Each warp receives this struct with its partition info, allowing
/// data-parallel processing without closure captures or global atomics.
#[derive(Clone, Copy)]
pub struct CoopMapArgs {
    /// Pointer to input data (read-only).
    pub src: *const u8,
    /// Pointer to output data (write).
    pub dst: *mut u8,
    /// Total number of elements.
    pub len: usize,
    /// This warp's ID (0-based, includes warp 0).
    pub warp_id: u32,
    /// Total number of warps participating.
    pub n_warps: u32,
}

/// Execute a data-parallel map across all warps without closure captures.
///
/// This is the ergonomic alternative to `cooperative()`: instead of passing
/// data via global atomics and reading them inside a zero-capture closure,
/// the caller passes `(src, dst, len)` as explicit arguments. Each warp
/// receives a [`CoopMapArgs`] struct with its partition info.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::thread;
///
/// thread::gpu_main(|| {
///     let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
///     let mut output: Vec<f32> = vec![0.0; 4];
///
///     // All warps cooperatively double each element
///     thread::cooperative_map(
///         input.as_ptr() as *const u8,
///         output.as_mut_ptr() as *mut u8,
///         input.len(),
///         |args| {
///             let src = args.src as *const f32;
///             let dst = args.dst as *mut f32;
///             let mut i = args.warp_id as usize;
///             while i < args.len {
///                 unsafe {
///                     let v = core::ptr::read_volatile(src.add(i));
///                     core::ptr::write_volatile(dst.add(i), v * 2.0);
///                 }
///                 i += args.n_warps as usize;
///             }
///         },
///     );
///     // output == [2.0, 4.0, 6.0, 8.0]
/// });
/// ```
///
/// # Safety guarantees
///
/// - `src` and `dst` must be valid for the duration of the call
/// - The closure must partition work by `(warp_id, n_warps)` to avoid data races
/// - Unlike `cooperative()`, this function is safe to call (no `unsafe` block needed)
///   because it does not copy closure data across warp boundaries
pub fn cooperative_map(src: *const u8, dst: *mut u8, len: usize, f: fn(&CoopMapArgs)) {
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    let total = if n_warps == 0 { 1 } else { n_warps };

    // Warp 0 publishes the arguments to global memory
    if lane_id() == 0 {
        COOP_MAP_ARGS[0].store(src as u64, Ordering::Relaxed);
        COOP_MAP_ARGS[1].store(dst as u64, Ordering::Relaxed);
        COOP_MAP_ARGS[2].store(len as u64, Ordering::Relaxed);
        COOP_MAP_ARGS[3].store(f as usize as u64, Ordering::Relaxed);
    }

    if total <= 1 {
        // Single warp: just call directly
        let args = CoopMapArgs {
            src,
            dst,
            len,
            warp_id: 0,
            n_warps: 1,
        };
        if lane_id() == 0 {
            f(&args);
        }
        return;
    }

    // Trampoline: reads args from COOP_MAP_ARGS, calls user fn
    fn trampoline(_raw: *mut u8) {
        let lid = crate::index::thread_idx_x() % 32;
        if lid == 0 {
            let src = COOP_MAP_ARGS[0].load(Ordering::Acquire) as *const u8;
            let dst = COOP_MAP_ARGS[1].load(Ordering::Acquire) as *mut u8;
            let len = COOP_MAP_ARGS[2].load(Ordering::Acquire) as usize;
            let fn_ptr = COOP_MAP_ARGS[3].load(Ordering::Acquire);
            let f: fn(&CoopMapArgs) = unsafe { core::mem::transmute(fn_ptr) };

            let wid = crate::index::thread_idx_x() / 32;
            let n_warps = NUM_WARPS.load(Ordering::Acquire);

            let args = CoopMapArgs {
                src,
                dst,
                len,
                warp_id: wid,
                n_warps,
            };
            f(&args);
        }
    }

    let trampoline_fn = trampoline as fn(*mut u8);

    // Wake worker warps
    if lane_id() == 0 {
        for i in 1..total {
            WARP_FN[i].store(trampoline_fn as usize as u64, Ordering::Relaxed);
            WARP_DATA[i].store(0, Ordering::Relaxed);
            WARP_STATUS[i].store(STATUS_COOPERATIVE, Ordering::Release);
        }
    }

    // Warp 0 also participates
    if lane_id() == 0 {
        let args = CoopMapArgs {
            src,
            dst,
            len,
            warp_id: 0,
            n_warps: total as u32,
        };
        f(&args);
    }

    // Wait for all workers to finish
    #[allow(clippy::needless_range_loop)]
    for i in 1..total {
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

// ============================================================
// Cooperative Reduce — multi-warp reduction to a single value
// ============================================================

/// Global argument block for cooperative_reduce.
/// Layout: [src_ptr: u64, len: u64, fn_ptr: u64, accumulator: u64]
static COOP_REDUCE_ARGS: [AtomicU64; 4] = [ATOMIC_U64_ZERO; 4];

/// Arguments passed to the cooperative_reduce user function.
///
/// Each warp receives this struct with partition info. The user function
/// returns a `u64` partial result (its warp's contribution).
#[derive(Clone, Copy)]
pub struct CoopReduceArgs {
    /// Pointer to input data (read-only).
    pub src: *const u8,
    /// Total number of elements.
    pub len: usize,
    /// This warp's ID (0-based, includes warp 0).
    pub warp_id: u32,
    /// Total number of warps participating.
    pub n_warps: u32,
}

/// Execute a multi-warp reduction, returning a single combined value.
///
/// Each warp runs the user function on its partition and returns a `u64`
/// partial result. Warp 0 collects all partial results from WARP_RESULT
/// slots and sums them to produce the final value.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::thread;
///
/// thread::gpu_main(|| {
///     let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
///
///     let total = thread::cooperative_reduce(
///         data.as_ptr() as *const u8,
///         data.len(),
///         |args| {
///             let src = args.src as *const u64;
///             let mut sum = 0u64;
///             let mut i = args.warp_id as usize;
///             while i < args.len {
///                 sum += unsafe { core::ptr::read_volatile(src.add(i)) };
///                 i += args.n_warps as usize;
///             }
///             sum
///         },
///     );
///     // total == 36
/// });
/// ```
///
/// # Safety guarantees
///
/// - `src` must be valid for the duration of the call
/// - The user function must return a partial result that can be summed
/// - No closure captures (function pointer, not closure)
pub fn cooperative_reduce(src: *const u8, len: usize, f: fn(&CoopReduceArgs) -> u64) -> u64 {
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    let total = if n_warps == 0 { 1 } else { n_warps };

    // Warp 0 publishes the arguments to global memory
    if lane_id() == 0 {
        COOP_REDUCE_ARGS[0].store(src as u64, Ordering::Relaxed);
        COOP_REDUCE_ARGS[1].store(len as u64, Ordering::Relaxed);
        COOP_REDUCE_ARGS[2].store(f as usize as u64, Ordering::Relaxed);
        // Reset accumulator
        COOP_REDUCE_ARGS[3].store(0, Ordering::Relaxed);
    }

    if total <= 1 {
        // Single warp: just call directly
        let args = CoopReduceArgs {
            src,
            len,
            warp_id: 0,
            n_warps: 1,
        };
        return if lane_id() == 0 { f(&args) } else { 0 };
    }

    // Trampoline: reads args, calls user fn, stores result in WARP_RESULT
    fn reduce_trampoline(_raw: *mut u8) {
        let lid = crate::index::thread_idx_x() % 32;
        if lid == 0 {
            let src = COOP_REDUCE_ARGS[0].load(Ordering::Acquire) as *const u8;
            let len = COOP_REDUCE_ARGS[1].load(Ordering::Acquire) as usize;
            let fn_ptr = COOP_REDUCE_ARGS[2].load(Ordering::Acquire);
            let f: fn(&CoopReduceArgs) -> u64 = unsafe { core::mem::transmute(fn_ptr) };

            let wid = crate::index::thread_idx_x() / 32;
            let n_warps = NUM_WARPS.load(Ordering::Acquire);

            let args = CoopReduceArgs {
                src,
                len,
                warp_id: wid,
                n_warps,
            };
            let partial = f(&args);

            // Store partial result in WARP_RESULT for warp 0 to collect
            WARP_RESULT[wid as usize].store(partial, Ordering::Release);
        }
    }

    let trampoline_fn = reduce_trampoline as fn(*mut u8);

    // Wake worker warps
    if lane_id() == 0 {
        for i in 1..total {
            WARP_FN[i].store(trampoline_fn as usize as u64, Ordering::Relaxed);
            WARP_DATA[i].store(0, Ordering::Relaxed);
            WARP_STATUS[i].store(STATUS_COOPERATIVE, Ordering::Release);
        }
    }

    // Warp 0 also computes its partial result
    let warp0_partial = if lane_id() == 0 {
        let args = CoopReduceArgs {
            src,
            len,
            warp_id: 0,
            n_warps: total as u32,
        };
        f(&args)
    } else {
        0
    };

    // Wait for all workers to finish, then collect their results
    let mut combined = warp0_partial;
    #[allow(clippy::needless_range_loop)]
    for i in 1..total {
        loop {
            let s = WARP_STATUS[i].load(Ordering::Acquire);
            if s == STATUS_DONE {
                // Collect this warp's partial result
                combined += WARP_RESULT[i].load(Ordering::Acquire);
                WARP_STATUS[i].store(STATUS_IDLE, Ordering::Release);
                break;
            }
            nanosleep_short();
        }
    }

    combined
}

// ============================================================
// Cooperative Map with Params — extra user-defined u64 parameters
// ============================================================

/// Global argument block for cooperative_map_with_params.
/// Layout: [src_ptr: u64, dst_ptr: u64, len: u64, fn_ptr: u64, p0: u64, p1: u64, p2: u64, p3: u64]
static COOP_MAP_EXT_ARGS: [AtomicU64; 8] = [ATOMIC_U64_ZERO; 8];

/// Arguments passed to cooperative_map_with_params user function.
///
/// Extends [`CoopMapArgs`] with up to 4 user-defined u64 parameters.
/// These can carry scalars, matrix dimensions, stride values, etc.
#[derive(Clone, Copy)]
pub struct CoopMapExtArgs {
    /// Pointer to input data (read-only).
    pub src: *const u8,
    /// Pointer to output data (write).
    pub dst: *mut u8,
    /// Total number of elements.
    pub len: usize,
    /// This warp's ID (0-based, includes warp 0).
    pub warp_id: u32,
    /// Total number of warps participating.
    pub n_warps: u32,
    /// User-defined parameters (up to 4).
    pub params: [u64; 4],
}

/// Execute a data-parallel map with extra user-defined parameters.
///
/// Like [`cooperative_map`], but passes up to 4 additional `u64` parameters
/// to the user function. Useful for operations that need extra context
/// (e.g., scalar multiplier, matrix dimensions M/K/N, stride values).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::thread;
///
/// thread::gpu_main(|| {
///     let input: Vec<u32> = vec![1, 2, 3, 4];
///     let mut output: Vec<u32> = vec![0; 4];
///
///     // Multiply each element by scalar 10
///     thread::cooperative_map_with_params(
///         input.as_ptr() as *const u8,
///         output.as_mut_ptr() as *mut u8,
///         input.len(),
///         [10, 0, 0, 0],  // params[0] = scalar
///         |args| {
///             let src = args.src as *const u32;
///             let dst = args.dst as *mut u32;
///             let scale = args.params[0] as u32;
///             let mut i = args.warp_id as usize;
///             while i < args.len {
///                 unsafe {
///                     let v = core::ptr::read_volatile(src.add(i));
///                     core::ptr::write_volatile(dst.add(i), v * scale);
///                 }
///                 i += args.n_warps as usize;
///             }
///         },
///     );
///     // output == [10, 20, 30, 40]
/// });
/// ```
pub fn cooperative_map_with_params(
    src: *const u8,
    dst: *mut u8,
    len: usize,
    params: [u64; 4],
    f: fn(&CoopMapExtArgs),
) {
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    let total = if n_warps == 0 { 1 } else { n_warps };

    // Warp 0 publishes the arguments to global memory
    if lane_id() == 0 {
        COOP_MAP_EXT_ARGS[0].store(src as u64, Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[1].store(dst as u64, Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[2].store(len as u64, Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[3].store(f as usize as u64, Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[4].store(params[0], Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[5].store(params[1], Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[6].store(params[2], Ordering::Relaxed);
        COOP_MAP_EXT_ARGS[7].store(params[3], Ordering::Relaxed);
    }

    if total <= 1 {
        let args = CoopMapExtArgs {
            src,
            dst,
            len,
            warp_id: 0,
            n_warps: 1,
            params,
        };
        if lane_id() == 0 {
            f(&args);
        }
        return;
    }

    // Trampoline: reads args from COOP_MAP_EXT_ARGS, calls user fn
    fn ext_trampoline(_raw: *mut u8) {
        let lid = crate::index::thread_idx_x() % 32;
        if lid == 0 {
            let src = COOP_MAP_EXT_ARGS[0].load(Ordering::Acquire) as *const u8;
            let dst = COOP_MAP_EXT_ARGS[1].load(Ordering::Acquire) as *mut u8;
            let len = COOP_MAP_EXT_ARGS[2].load(Ordering::Acquire) as usize;
            let fn_ptr = COOP_MAP_EXT_ARGS[3].load(Ordering::Acquire);
            let f: fn(&CoopMapExtArgs) = unsafe { core::mem::transmute(fn_ptr) };

            let wid = crate::index::thread_idx_x() / 32;
            let n_warps = NUM_WARPS.load(Ordering::Acquire);

            let params = [
                COOP_MAP_EXT_ARGS[4].load(Ordering::Acquire),
                COOP_MAP_EXT_ARGS[5].load(Ordering::Acquire),
                COOP_MAP_EXT_ARGS[6].load(Ordering::Acquire),
                COOP_MAP_EXT_ARGS[7].load(Ordering::Acquire),
            ];

            let args = CoopMapExtArgs {
                src,
                dst,
                len,
                warp_id: wid,
                n_warps,
                params,
            };
            f(&args);
        }
    }

    let trampoline_fn = ext_trampoline as fn(*mut u8);

    // Wake worker warps
    if lane_id() == 0 {
        for i in 1..total {
            WARP_FN[i].store(trampoline_fn as usize as u64, Ordering::Relaxed);
            WARP_DATA[i].store(0, Ordering::Relaxed);
            WARP_STATUS[i].store(STATUS_COOPERATIVE, Ordering::Release);
        }
    }

    // Warp 0 also participates
    if lane_id() == 0 {
        let args = CoopMapExtArgs {
            src,
            dst,
            len,
            warp_id: 0,
            n_warps: total as u32,
            params,
        };
        f(&args);
    }

    // Wait for all workers to finish
    #[allow(clippy::needless_range_loop)]
    for i in 1..total {
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
    SPAWN_RAW_COUNT.fetch_add(1, Ordering::Relaxed);
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    if n_warps <= 1 {
        return 0;
    }

    // Find an idle warp. We write fn/data first, then publish
    // STATUS_ASSIGNED with a release store so the worker sees consistent data.
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
