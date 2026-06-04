#![no_std]
#![feature(abi_ptx)]
#![cfg_attr(feature = "gpu_kernel_abi", feature(abi_gpu_kernel))]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

// Install the gpu-runtime panic handler (sends panic message via hostcall)
gpu_runtime::panic_handler!();

// Declare dynamic shared memory symbol at module level (PTX).
// This emits `.extern .shared .align 4 .b8 dynamic_smem[];`
// so that kernels can reference it via inline asm.
#[cfg(target_arch = "nvptx64")]
core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];");

mod helpers;

mod basic;
mod compute_cnn;
mod compute_demo;
mod compute_fused;
mod compute_persistent;
mod compute_physics;
mod compute_gemm;
mod compute_math;
#[cfg(feature = "sm_80")]
mod compute_mma;
mod compute_search;
mod compute_transformer;
mod hostcall_kernels;
mod hybrid;
mod pipeline;
mod thread_test;
mod warp;
