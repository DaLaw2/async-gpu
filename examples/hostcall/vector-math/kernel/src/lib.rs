//! Vector Math — GPU kernels for pure compute operations.
//!
//! Three kernels demonstrating different compute patterns:
//! 1. `saxpy` — scalar-vector multiply-add: y = a*x + y
//! 2. `dot_partial` — parallel partial reduction (each block sums its chunk)
//! 3. `softmax_inplace` — numerically stable softmax (host-assisted max/sum)

#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("trap;");
    }
    loop {}
}

// ================================================================
// Kernel 1: SAXPY — y[i] = a * x[i] + y[i]
// ================================================================

#[no_mangle]
pub unsafe extern "gpu-kernel" fn saxpy(x: *const f32, y: *mut f32, a: f32, n: u32) {
    let idx = nvptx::_block_idx_x() as u32 * nvptx::_block_dim_x() as u32
        + nvptx::_thread_idx_x() as u32;
    if idx < n {
        let val = *x.add(idx as usize);
        let cur = *y.add(idx as usize);
        *y.add(idx as usize) = a * val + cur;
    }
}

// ================================================================
// Kernel 2: Element-wise multiply — out[i] = x[i] * y[i]
// ================================================================

/// Element-wise multiply: out[i] = x[i] * y[i].
/// This is the first step of a dot product: GPU does element-wise,
/// host sums the result (demonstrating CPU-GPU cooperation).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn elementwise_mul(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: u32,
) {
    let idx = nvptx::_block_idx_x() as u32 * nvptx::_block_dim_x() as u32
        + nvptx::_thread_idx_x() as u32;
    if idx < n {
        *out.add(idx as usize) = *x.add(idx as usize) * *y.add(idx as usize);
    }
}

// ================================================================
// Kernel 3: Softmax pass 1 — compute exp(x - max_val)
// ================================================================

/// Computes out[i] = exp(input[i] - max_val) for each element.
/// The host provides the pre-computed max_val for numerical stability.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn softmax_exp(
    input: *const f32,
    output: *mut f32,
    max_val: f32,
    n: u32,
) {
    let idx = nvptx::_block_idx_x() as u32 * nvptx::_block_dim_x() as u32
        + nvptx::_thread_idx_x() as u32;
    if idx < n {
        let x = *input.add(idx as usize);
        *output.add(idx as usize) = gpu_exp(x - max_val);
    }
}

// ================================================================
// Kernel 4: Softmax pass 2 — normalize by sum
// ================================================================

/// Computes output[i] = input[i] / sum for each element.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn softmax_normalize(
    data: *mut f32,
    sum: f32,
    n: u32,
) {
    let idx = nvptx::_block_idx_x() as u32 * nvptx::_block_dim_x() as u32
        + nvptx::_thread_idx_x() as u32;
    if idx < n {
        let val = *data.add(idx as usize);
        *data.add(idx as usize) = val / sum;
    }
}

/// Approximate exp(x) using PTX ex2.approx + scaling.
/// exp(x) = 2^(x * log2(e))
#[inline(always)]
unsafe fn gpu_exp(x: f32) -> f32 {
    let log2_e: f32 = 1.442695041;
    let scaled = x * log2_e;
    let result: f32;
    core::arch::asm!(
        "ex2.approx.ftz.f32 {out}, {inp};",
        inp = in(reg32) scaled,
        out = out(reg32) result,
    );
    result
}
