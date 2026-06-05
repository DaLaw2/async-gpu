// Parallel iterator demo — showcases GpuParallelIterator API on GPU.
//
// Six kernels demonstrate the par_iter combinator chains:
// 1. map + collect_into:   output[i] = input[i] * 2.0 + 1.0
// 2. map + sum (fold):     sum of input[i]^2
// 3. enumerate + map + collect_into: output[i] = input[i] + i as f32
// 4. zip + map + collect_into: output[i] = a[i] + b[i]
// 5. filter + collect_count: filter even indices, collect, return count
// 6. filter + map + sum:   filter(> threshold).map(square).sum()
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

// ============================================================
// Demo 5: filter + collect_count
// ============================================================
//
// Filters even-indexed elements and collects them into an output buffer.
// Uses `collect_count` to get the number of elements written.
//
// Expected: output contains input[0], input[2], input[4], ...
//           result[0] = count of even-indexed elements (= ceil(n/2))
//           result[1] = done flag
//
// # Arguments
// * `input`  - N f32 input values
// * `output` - N f32 output buffer (pre-allocated, worst-case size)
// * `n`      - number of elements
// * `result` - output: [count, done_flag]
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: enumerate().filter(even index).collect_count()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_filter_collect(
    input: *const f32,
    output: *mut f32,
    n: u32,
    result: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };
        let dst = unsafe { GpuSliceMut::from_raw_parts(output, len) };

        // Filter even indices: enumerate, keep elements at even positions,
        // map back to just the value, then collect_count.
        let count = src
            .par_iter()
            .enumerate()
            .filter(|(_i, _x): &(usize, f32)| _i % 2 == 0)
            .map(|(_i, x): (usize, f32)| x)
            .collect_count(dst);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, count as u32);
                core::ptr::write_volatile(result.add(1), 1); // done flag
            }
        }
    });
}

// ============================================================
// Demo 6: filter + map + sum (fused filter-map-reduce)
// ============================================================
//
// Filters elements greater than a threshold, squares them, and sums.
// Demonstrates the full filter().map().sum() fusion chain.
//
// Expected: result[0] = sum of (x*x) for all x in input where x > threshold
//           result[1] = done flag
//
// # Arguments
// * `input`     - N f32 input values
// * `n`         - number of elements
// * `threshold` - f32 threshold (as u32 bits)
// * `result`    - output: [sum_bits, done_flag]
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Par_iter demo: filter(|x| x > threshold).map(|x| x * x).sum()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_filter_map_sum(
    input: *const f32,
    n: u32,
    threshold_bits: u32,
    result: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let threshold = f32::from_bits(threshold_bits);
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };

        // Fused chain: filter(> threshold) → map(square) → sum.
        // The filter predicate and map function are inlined into a single
        // loop body per warp — no intermediate buffers.
        let total: f32 = src
            .par_iter()
            .filter(move |x: &f32| *x > threshold)
            .map(|x: f32| x * x)
            .sum();

        if gpu_runtime::index::thread_idx_x() == 0 {
            let bits = total.to_bits();
            unsafe {
                core::ptr::write_volatile(result, bits);
                core::ptr::write_volatile(result.add(1), 1); // done flag
            }
        }
    });
}

// ============================================================
// Demo 7: chained map + collect_into (two separate .map() calls)
// ============================================================
//
// Applies f(x) = (x * 2.0) then g(x) = (x + 1.0) via TWO separate
// .map() calls chained at the type level. The Rust compiler fuses
// GpuMap<GpuMap<GpuParIter<f32>, C1>, C2> into a single inline
// get_unchecked() call: g(f(read_volatile(ptr)))
//
// This is the core zero-intermediate-buffer proof: two explicit .map()
// adapters produce ZERO intermediate buffers at runtime because Rust
// monomorphization + LLVM inlining composes them into a single expression.
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

/// Chained iterator fusion: map(|x| x * 2.0).map(|x| x + 1.0).collect_into()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_chained_map_collect(
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

        // KEY: Two separate .map() calls, NOT a single closure.
        // At compile time this creates GpuMap<GpuMap<GpuParIter<f32>, C1>, C2>.
        // LLVM inlines the nested get_unchecked() chain into:
        //   let x = read_volatile(ptr.add(i));
        //   write_volatile(out.add(i), (x * 2.0) + 1.0);
        // Zero intermediate buffers, zero heap allocation.
        src.par_iter()
            .map(|x: f32| x * 2.0)
            .map(|x: f32| x + 1.0)
            .collect_into(dst);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(status, 1);
            }
        }
    });
}

// ============================================================
// Demo 8: map + filter + count (chained map-into-filter)
// ============================================================
//
// Squares each element, then counts how many exceed a threshold.
// Demonstrates: GpuFilter<GpuMap<GpuParIter<f32>, C1>, C2>.count()
//
// The .map() fuses with the filter predicate: each warp computes
// x*x and checks > threshold in a single pass, no intermediate buffer.
//
// Expected: result[0] = count of elements where input[i]^2 > threshold
//           result[1] = done flag
//
// # Arguments
// * `input`          - N f32 input values
// * `n`              - number of elements
// * `threshold_bits` - f32 threshold as u32 bits
// * `result`         - output: [count, done_flag]
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Chained fusion: map(|x| x * x).filter(|x| *x > threshold).count()
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_map_filter_count(
    input: *const f32,
    n: u32,
    threshold_bits: u32,
    result: *mut u32,
) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(512);
        }

        let len = n as usize;
        let threshold = f32::from_bits(threshold_bits);
        let src = unsafe { GpuSlice::from_raw_parts(input, len) };

        // Chain: map(square) → filter(> threshold) → count.
        // The map is fused into the filter's iteration — get_unchecked(i)
        // returns x*x, then the filter predicate checks > threshold.
        // Zero intermediate buffers.
        let count = src
            .par_iter()
            .map(|x: f32| x * x)
            .filter(move |x: &f32| *x > threshold)
            .count();

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, count as u32);
                core::ptr::write_volatile(result.add(1), 1); // done flag
            }
        }
    });
}

// ============================================================
// Demo 9: triple map + sum (deep fusion chain)
// ============================================================
//
// Three separate .map() calls chained then reduced via .sum().
// Demonstrates deeper fusion: GpuMap<GpuMap<GpuMap<GpuParIter>>>.
//
// Pipeline: x → x + 1.0 → x * 3.0 → x - 0.5 → sum
// Equivalent to: sum( (input[i] + 1.0) * 3.0 - 0.5 )
//
// Expected: result[0] = sum_bits (f32 as u32)
//           result[1] = done flag
//
// # Arguments
// * `input`  - N f32 input values
// * `n`      - number of elements
// * `result` - output: [sum_bits, done_flag]
//
// # Launch config
// * Grid: (1, 1, 1)
// * Block: (128, 1, 1) — 4 warps
// * Shared memory: 512 bytes

/// Deep fusion: map().map().map().sum() — three chained maps
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_triple_map_sum(
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

        // Three chained maps: all fused into a single expression per element.
        // GpuMap<GpuMap<GpuMap<GpuParIter<f32>, C1>, C2>, C3>
        // → get_unchecked(i) = ((read_volatile(ptr.add(i)) + 1.0) * 3.0) - 0.5
        let total: f32 = src
            .par_iter()
            .map(|x: f32| x + 1.0)
            .map(|x: f32| x * 3.0)
            .map(|x: f32| x - 0.5)
            .sum();

        if gpu_runtime::index::thread_idx_x() == 0 {
            let bits = total.to_bits();
            unsafe {
                core::ptr::write_volatile(result, bits);
                core::ptr::write_volatile(result.add(1), 1); // done flag
            }
        }
    });
}
