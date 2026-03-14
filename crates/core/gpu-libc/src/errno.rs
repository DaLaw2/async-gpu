//! Per-thread errno implementation for GPU.
//!
//! Uses a thread-ID indexed array in global memory so that each CUDA
//! thread has its own errno value. Supports up to `MAX_GPU_THREADS`
//! concurrent threads (configurable, default 1024 = one block max).
//!
//! The errno array is statically allocated. For launches with more
//! threads than `MAX_GPU_THREADS`, threads beyond the limit share
//! errno slot 0 (graceful degradation, not UB).

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

/// Maximum number of concurrent GPU threads with independent errno.
/// Default: 1024 (one full block). Threads beyond this share slot 0.
const MAX_GPU_THREADS: usize = 1024;

/// Per-thread errno storage array indexed by flat thread ID.
static mut ERRNO_ARRAY: [c_int; MAX_GPU_THREADS] = [0; MAX_GPU_THREADS];

/// Read the flat thread index within the current block via inline PTX.
/// Returns `threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`.
#[inline(always)]
fn thread_id_in_block() -> u32 {
    let tid_x: u32;
    let tid_y: u32;
    let tid_z: u32;
    let ntid_x: u32;
    let ntid_y: u32;
    unsafe {
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid_x);
        core::arch::asm!("mov.u32 {}, %tid.y;", out(reg32) tid_y);
        core::arch::asm!("mov.u32 {}, %tid.z;", out(reg32) tid_z);
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid_x);
        core::arch::asm!("mov.u32 {}, %ntid.y;", out(reg32) ntid_y);
    }
    tid_x + tid_y * ntid_x + tid_z * ntid_x * ntid_y
}

/// Get the errno array index for the current thread.
/// Clamps to MAX_GPU_THREADS-1 to prevent out-of-bounds access.
#[inline(always)]
fn errno_index() -> usize {
    let tid = thread_id_in_block() as usize;
    if tid < MAX_GPU_THREADS {
        tid
    } else {
        0 // graceful fallback for oversized launches
    }
}

/// Returns a pointer to the per-thread errno value.
///
/// Each CUDA thread gets its own errno slot, indexed by thread ID
/// within the block. This makes errno thread-safe for up to 1024
/// concurrent threads per block.
#[no_mangle]
pub unsafe extern "C" fn __errno_location() -> *mut c_int {
    let idx = errno_index();
    core::ptr::addr_of_mut!(ERRNO_ARRAY[idx])
}

/// Set errno to the given value for the current thread.
#[inline(always)]
pub unsafe fn set_errno(val: c_int) {
    let idx = errno_index();
    ERRNO_ARRAY[idx] = val;
}

/// Get the current errno value for the current thread.
#[inline(always)]
pub unsafe fn get_errno() -> c_int {
    let idx = errno_index();
    ERRNO_ARRAY[idx]
}
