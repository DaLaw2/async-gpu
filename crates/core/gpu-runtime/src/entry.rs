//! Implicit hostcall injection via device global.
//!
//! The host writes the hostcall buffer device pointer to `__HOSTCALL_BUF`
//! (a device global) via `cuModuleGetGlobal_v2` + `cuMemcpyHtoD` before
//! kernel launch. The kernel calls `auto_init()` at entry to read the
//! pointer and initialize all subsystems (stdio, panic, libc I/O).
//!
//! This enables zero-parameter kernel entry:
//!
//! ```rust,ignore
//! #[no_mangle]
//! pub unsafe extern "gpu-kernel" fn my_kernel() {
//!     gpu_runtime::entry::auto_init();
//!     gpu_runtime::thread::gpu_main_poll(|| {
//!         println!("Hello from zero-param kernel!");
//!     });
//! }
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

/// Canonical device global for hostcall buffer injection.
///
/// The host writes the hostcall session's device pointer here via
/// `cuModuleGetGlobal_v2` + `cuMemcpyHtoD` after module load but
/// before kernel launch. The kernel reads it during `auto_init()`.
///
/// # Symbol visibility
///
/// `#[no_mangle]` ensures the symbol appears as `__HOSTCALL_BUF` in PTX,
/// discoverable by `cuModuleGetGlobal_v2`. `#[used]` prevents the linker
/// from stripping it as unused.
#[no_mangle]
#[used]
pub static __HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);

/// Read the hostcall buffer pointer from the device global.
///
/// Returns the pointer as `*mut u8`, or null if the host did not inject it.
#[inline(always)]
pub fn hostcall_buf_ptr() -> *mut u8 {
    __HOSTCALL_BUF.load(Ordering::Relaxed) as *mut u8
}

/// Auto-initialize all GPU subsystems from the device global.
///
/// Reads `__HOSTCALL_BUF` and initializes:
/// - Panic handler (`gpu_panic_init`)
///
/// For kernels that use `gpu-kernel-test` (with patched std), the
/// stdio and libc subsystems should be initialized by the kernel
/// crate's own `stdio_init` and `gpu_libc_io_init` calls, which
/// can also read from `__HOSTCALL_BUF` via this module.
///
/// # Safety
///
/// Must be called after the host has written the hostcall pointer
/// to `__HOSTCALL_BUF` (i.e., at kernel entry, before any hostcall
/// operations).
#[inline(always)]
pub unsafe fn auto_init() {
    let buf = hostcall_buf_ptr();
    if !buf.is_null() {
        crate::panic::gpu_panic_init(buf);
    }
}
