// Shared GPU kernel core — helpers, basic asm/atomic kernels, and math kernels.
//
// This crate provides:
// - `helpers` module (pub): shared hostcall, compute, and I/O helpers
// - `basic` module: asm/atomic test kernels
// - `compute_math` module: f32 math validation kernels
//
// Other kernel crates depend on this crate for `helpers` via:
//   `use gpu_kernel_core::helpers::bar_sync;`

#![no_main]
#![feature(restricted_std)]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

// Declare dynamic shared memory symbol at module level (PTX).
// This emits `.extern .shared .align 4 .b8 dynamic_smem[];`
// so that kernels can reference it via inline asm.
#[cfg(target_arch = "nvptx64")]
core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];");

// === Public modules ===
pub mod helpers;

// === Kernel modules ===
mod basic;
mod compute_math;

// Force-link stdio symbols from gpu-runtime. These are called by the patched
// std PAL via `extern "C"` blocks, so LTO would strip them without this anchor.
#[used]
static _KEEP_STDOUT: unsafe fn(*const u8, usize) -> usize = gpu_runtime::stdio::gpu_stdout_write;
#[used]
static _KEEP_STDIN: unsafe fn(*mut u8, usize) -> usize = gpu_runtime::stdio::gpu_stdin_read;

/// Auto-initialize stdio from the `__HOSTCALL_BUF` device global.
///
/// The host writes the hostcall pointer to the device global via
/// `cuModuleGetGlobal_v2` + `cuMemcpyHtoD` before launch. This function
/// reads it and initializes all subsystems (stdio, panic, libc I/O).
///
/// Returns the hostcall buffer pointer (for use by caller), or null if
/// the host did not inject it.
pub fn stdio_auto_init() -> *mut u8 {
    let buf = gpu_runtime::entry::hostcall_buf_ptr();
    if !buf.is_null() {
        gpu_runtime::stdio::stdio_init(buf);
        unsafe {
            gpu_runtime::panic::gpu_panic_init(buf);
            gpu_libc::gpu_libc_io_init(buf);
        }
    }
    buf
}
