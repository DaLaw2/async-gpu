// GPU ML/HPC compute kernels — GEMM, transformer, CNN, search, physics, etc.
//
// This crate contains compute-heavy kernels that do not require std I/O.
// All kernel entry points use `#[no_mangle] pub unsafe extern "gpu-kernel"`.
// Shared helpers are imported from `gpu_kernel_core::helpers`.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

// Re-export gpu-kernel-core so its kernel symbols are linked into this crate's cdylib.
extern crate gpu_kernel_core;

// Declare dynamic shared memory symbol at module level (PTX).
// This emits `.extern .shared .align 4 .b8 dynamic_smem[];`
// so that kernels can reference it via inline asm.
#[cfg(target_arch = "nvptx64")]
core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];");

// === Compute kernel modules ===
mod compute_cnn;
mod compute_demo;
mod compute_fused;
mod compute_gemm;
#[cfg(feature = "sm_80")]
mod compute_mma;
mod compute_persistent;
mod compute_physics;
mod compute_search;
mod compute_transformer;

// Force-link stdio symbols from gpu-runtime. These are called by the patched
// std PAL via `extern "C"` blocks, so LTO would strip them without this anchor.
#[used]
static _KEEP_STDOUT: unsafe fn(*const u8, usize) -> usize = gpu_runtime::stdio::gpu_stdout_write;
#[used]
static _KEEP_STDIN: unsafe fn(*mut u8, usize) -> usize = gpu_runtime::stdio::gpu_stdin_read;
