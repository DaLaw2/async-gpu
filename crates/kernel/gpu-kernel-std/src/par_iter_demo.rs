// Parallel iterator demo — showcases GpuParallelIterator API on GPU.
//
// Four kernels demonstrate the par_iter combinator chains:
// 1. map + collect_into:   output[i] = input[i] * 2.0 + 1.0
// 2. map + sum (fold):     sum of input[i]^2
// 3. enumerate + map + collect_into: output[i] = input[i] + i as f32
// 4. zip + map + collect_into: output[i] = a[i] + b[i]
//
// Each kernel uses block_scope + spawn_all for warp-parallel execution.
// The par_iter chain fuses at compile time — no intermediate buffers.

use gpu_runtime::par_iter::{par_iter, GpuParallelIterator, GpuSlice, GpuSliceMut};
use gpu_runtime::scope::init_shared_mem_allocator;
use gpu_runtime::thread;

// ============================================================
// Demo 1: map + collect_into
// ============================================================
//
// Applies f(x) = x * 2.0 + 1.0 to each element.
// Warp-striped: each warp processes elements stride n_warps apart.
//
// Expected: output[i] = input[i] * 2.0 + 1.0
//
// # Arguments
// * `input`  - N f32 input values
// * `output` - N f32 output values (pre-allocated)
// * `n`      - number of elements
// * `status` - 1 u32 done flag
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: map(|x| x * 2.0 + 1.0).collect_into()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_map_collect(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };
        let dst = unsafe { GpuSliceMut::from_raw_parts(output, len) };

        // Fused chain: compiles to a single loop body per warp.
        // x * 2.0 + 1.0 — two operations, zero intermediate buffers.
        src.par_iter()
            .map(|x: f32| x * 2.0 + 1.0)
            .collect_into(dst);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(status, 1);
            }
        }
    });
}

// ============================================================
// Demo 2: map + sum (reduction)
// ============================================================
//
// Computes sum of squares: sum(input[i]^2).
// Each warp reduces its partition; warp 0 combines partials.
//
// Expected: result[0] = sum of input[i]^2 (as f32 bits)
//
// # Arguments
// * `input`  - N f32 input values
// * `n`      - number of elements
// * `result` - output: [sum_bits_lo, sum_bits_hi, done_flag]
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: map(|x| x * x).sum()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_map_sum(
    input: *const f32,
    n: u32,
    result: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };

        // Fused chain: map(square) + fold(+, 0.0) via .sum().
        // Each warp reduces its partition, warp 0 combines.
        let total: f32 = src.par_iter().map(|x: f32| x * x).sum();

        if gpu_runtime::index::thread_idx_x() == 0 {
            // Write f32 result as u32 bits for safe transport.
            let bits = total.to_bits();
            unsafe {
                core::ptr::write_volatile(result, bits);
                core::ptr::write_volatile(result.add(1), 1); // done flag
            }
        }
    });
}

// ============================================================
// Demo 3: enumerate + map + collect_into
// ============================================================
//
// Adds the element index to each value: output[i] = input[i] + i as f32.
// Tests the enumerate adapter which pairs (index, element).
//
// Expected: output[i] = input[i] + i as f32
//
// # Arguments
// * `input`  - N f32 input values
// * `output` - N f32 output values (pre-allocated)
// * `n`      - number of elements
// * `status` - 1 u32 done flag
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: enumerate().map(|(i, x)| x + i as f32).collect_into()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_enumerate_collect(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };
        let dst = unsafe { GpuSliceMut::from_raw_parts(output, len) };

        // Fused chain: enumerate pairs (index, element), map adds them.
        // The closure receives (usize, f32) and produces f32.
        src.par_iter()
            .enumerate()
            .map(|(i, x): (usize, f32)| x + i as f32)
            .collect_into(dst);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(status, 1);
            }
        }
    });
}

// ============================================================
// Demo 4: zip + map + collect_into
// ============================================================
//
// Element-wise addition of two arrays: output[i] = a[i] + b[i].
// Tests the zip adapter which pairs elements from two iterators.
//
// Expected: output[i] = a[i] + b[i]
//
// # Arguments
// * `a`      - N f32 input values (first array)
// * `b`      - N f32 input values (second array)
// * `output` - N f32 output values (pre-allocated)
// * `n`      - number of elements
// * `status` - 1 u32 done flag
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: zip(a, b).map(|(x, y)| x + y).collect_into()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_zip_collect(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let src_a = unsafe { GpuSlice::from_raw_parts(a, len) };
        let src_b = unsafe { GpuSlice::from_raw_parts(b, len) };
        let dst = unsafe { GpuSliceMut::from_raw_parts(output, len) };

        // Fused chain: zip pairs elements from two iterators,
        // map adds the pair. Zero intermediate buffers.
        par_iter(&src_a)
            .zip(par_iter(&src_b))
            .map(|(x, y): (f32, f32)| x + y)
            .collect_into(dst);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(status, 1);
            }
        }
    });
}
