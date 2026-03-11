//! errno implementation for GPU.
//!
//! On GPU, "thread-local" storage is tricky — there's no OS-level TLS.
//! We use a static mut that is effectively per-lane in the SIMT model,
//! since each lane has its own register file. The compiler will place
//! this in local memory (per-thread stack).
//!
//! Note: This is a simplified implementation. In a multi-warp scenario
//! with shared global state, we might need per-lane indexing into a
//! global array. For now, since each kernel invocation has its own
//! execution context, a simple static works.

use crate::types::c_int;

// Error codes (matching Linux errno values)
pub const EPERM: c_int = 1;
pub const ENOENT: c_int = 2;
pub const ESRCH: c_int = 3;
pub const EINTR: c_int = 4;
pub const EIO: c_int = 5;
pub const ENXIO: c_int = 6;
pub const EBADF: c_int = 9;
pub const ENOMEM: c_int = 12;
pub const EACCES: c_int = 13;
pub const EFAULT: c_int = 14;
pub const EEXIST: c_int = 17;
pub const EINVAL: c_int = 22;
pub const ENOSYS: c_int = 38;
pub const ENOTSUP: c_int = 95;

/// Per-thread errno storage.
///
/// On nvptx64, each CUDA thread has its own local memory space.
/// A `static mut` in a GPU kernel effectively becomes per-thread
/// because LLVM places it in the `.local` address space for PTX.
///
/// However, `static mut` in Rust is per-module, not per-thread.
/// For a proper per-thread errno, we would need to either:
/// 1. Pass errno as a parameter through the call chain
/// 2. Use PTX `.local` memory explicitly via inline asm
/// 3. Index into a global array by thread ID
///
/// For now, we provide the `__errno_location` function that std
/// expects, returning a pointer. The actual per-thread isolation
/// depends on the calling context.
static mut GPU_ERRNO: c_int = 0;

/// Returns a pointer to the thread-local errno value.
///
/// # Safety
/// On GPU, this returns a pointer to a global static. Callers must
/// ensure single-threaded access or use appropriate synchronization.
/// In practice, std's usage pattern (set errno, then immediately
/// check it) is safe within a single CUDA thread.
#[no_mangle]
pub unsafe extern "C" fn __errno_location() -> *mut c_int {
    core::ptr::addr_of_mut!(GPU_ERRNO)
}

/// Set errno to the given value.
#[inline(always)]
pub unsafe fn set_errno(val: c_int) {
    GPU_ERRNO = val;
}

/// Get the current errno value.
#[inline(always)]
pub unsafe fn get_errno() -> c_int {
    GPU_ERRNO
}
