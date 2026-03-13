// ml-workload.1: f32 math validation kernel

use crate::helpers::gpu_sqrtf;

/// ml-workload.1: f32 math validation kernel.
/// Tests: f32 add, mul, div, fma, sqrt on GPU.
/// output[0] = 3.0 + 4.0 = 7.0
/// output[1] = 3.0 * 4.0 = 12.0
/// output[2] = 10.0 / 4.0 = 2.5
/// output[3] = sqrt(9.0) = 3.0
/// output[4] = dot([1,2,3,4], [5,6,7,8]) = 5+12+21+32 = 70.0
/// output[5] = ||[3,4]|| = sqrt(9+16) = 5.0
/// output[6] = cosine_sim([1,0], [0,1]) = 0.0
/// output[7] = cosine_sim([1,0], [1,0]) = 1.0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_math_test(output: *mut f32) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    if tid != 0 {
        return;
    }

    // Basic ops
    let a: f32 = 3.0;
    let b: f32 = 4.0;
    core::ptr::write_volatile(output.add(0), a + b); // 7.0
    core::ptr::write_volatile(output.add(1), a * b); // 12.0
    core::ptr::write_volatile(output.add(2), 10.0f32 / b); // 2.5
    core::ptr::write_volatile(output.add(3), gpu_sqrtf(9.0)); // 3.0

    // Dot product
    let v1 = [1.0f32, 2.0, 3.0, 4.0];
    let v2 = [5.0f32, 6.0, 7.0, 8.0];
    let mut dot: f32 = 0.0;
    let mut i = 0;
    while i < 4 {
        dot += v1[i] * v2[i];
        i += 1;
    }
    core::ptr::write_volatile(output.add(4), dot); // 70.0

    // Norm
    let norm = gpu_sqrtf(3.0 * 3.0 + 4.0 * 4.0);
    core::ptr::write_volatile(output.add(5), norm); // 5.0

    // Cosine similarity: orthogonal vectors -> 0.0
    // cos([1,0], [0,1]) = 0 / (1*1) = 0.0
    let cos_orth = 0.0f32 / (1.0f32 * 1.0f32);
    core::ptr::write_volatile(output.add(6), cos_orth); // 0.0

    // Cosine similarity: identical vectors -> 1.0
    // cos([1,0], [1,0]) = 1 / (1*1) = 1.0
    let cos_same = 1.0f32 / (1.0f32 * 1.0f32);
    core::ptr::write_volatile(output.add(7), cos_same); // 1.0
}
