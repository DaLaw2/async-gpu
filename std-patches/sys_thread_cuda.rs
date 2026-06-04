//! CUDA/GPU thread implementation for `std::sys::thread`.
//!
//! Maps std::thread::spawn() to GPU warp execution via the gpu_runtime
//! thread pool. Each "thread" is a GPU warp (32 SIMT lanes).
//!
//! The gpu_runtime::thread module manages the warp pool and provides
//! C-FFI entry points that this module calls via `extern "C"`.
//! The warp pool must be initialized via `gpu_runtime::thread::gpu_main()`
//! before std::thread::spawn() can be used.

use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::thread::ThreadInit;
use crate::time::Duration;

extern "C" {
    fn gpu_thread_spawn_raw(trampoline: u64, data: u64) -> u32;
    fn gpu_thread_join_warp(warp_id: u32);
    fn gpu_thread_available_parallelism() -> u32;
    fn gpu_thread_current_id() -> u32;
}

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

pub struct Thread {
    warp_id: u32,
}

impl Thread {
    /// Spawn a new GPU thread (warp).
    ///
    /// The `init` box contains the thread handle and the closure to run.
    /// We box it again to get a stable pointer, pass it to the warp pool,
    /// and the target warp will call the trampoline which initializes the
    /// thread and runs the closure.
    pub unsafe fn new(_stack: usize, init: Box<ThreadInit>) -> io::Result<Thread> {
        // Trampoline function called by the warp worker.
        // Receives a raw pointer to Box<ThreadInit>.
        extern "C" fn thread_trampoline(data: *mut u8) {
            unsafe {
                let init = Box::from_raw(data as *mut ThreadInit);
                // init() sets the current thread and returns the closure
                let main = init.init();
                main();
            }
        }

        let data_ptr = Box::into_raw(init) as u64;
        let trampoline_ptr = thread_trampoline as extern "C" fn(*mut u8) as u64;

        let warp_id = unsafe { gpu_thread_spawn_raw(trampoline_ptr, data_ptr) };
        if warp_id == 0 {
            // No warp available — shouldn't happen if gpu_main() was called
            // with enough warps. The raw function spins, so this branch is
            // unreachable in practice.
            return Err(io::Error::new(
                io::ErrorKind::ResourceBusy,
                "no GPU warp available for thread::spawn",
            ));
        }

        Ok(Thread { warp_id })
    }

    pub fn join(self) {
        unsafe {
            gpu_thread_join_warp(self.warp_id);
        }
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    let n = unsafe { gpu_thread_available_parallelism() } as usize;
    NonZero::new(n).ok_or(io::Error::UNKNOWN_THREAD_COUNT)
}

pub fn current_os_id() -> Option<u64> {
    Some(unsafe { gpu_thread_current_id() } as u64)
}

pub fn yield_now() {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("nanosleep.u32 100;");
    }
}

pub fn set_name(_name: &CStr) {
    // GPU threads don't have OS-level names
}

pub fn sleep(dur: Duration) {
    // Convert duration to nanoseconds (clamped to u32::MAX ≈ 4.3 seconds)
    let nanos = dur.as_nanos().min(u32::MAX as u128) as u32;
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        core::arch::asm!("nanosleep.u32 {ns};", ns = in(reg32) nanos);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = nanos;
    }
}
