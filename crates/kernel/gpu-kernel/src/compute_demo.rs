//! Multi-stage compute pipeline demo — showcases async GPU's advantage over raw CUDA.
//!
//! This kernel runs a multi-stage compute pipeline entirely on the GPU:
//! 1. Generate test data (math intrinsics)
//! 2. Softmax normalization (warp-cooperative)
//! 3. GELU activation (element-wise)
//! 4. Warp reduction (sum across lanes)
//! 5. Convergence check (GPU-autonomous, no host roundtrip)
//! 6. Write results + timing
//!
//! In raw CUDA, each stage would require a separate kernel launch (5-20μs overhead each).
//! Here, all stages run in a single launch with zero inter-stage overhead.

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx;

// ============================================================
// Separate stage kernels for multi-launch benchmark comparison
// (simulating the CUDA way: one kernel per compute stage)
// ============================================================

/// Stage kernel: warp softmax (one stage only).
/// For multi-launch benchmark comparison.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bench_stage_softmax(data: *mut f32, status: *mut u32) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{index, nn};
        let tid = index::thread_idx_x();
        let val = *data.add(tid as usize);
        let result = nn::warp_softmax_f32(val);
        *data.add(tid as usize) = result;
        if tid == 0 {
            core::ptr::write_volatile(status, 1);
        }
    }
}

/// Stage kernel: element-wise GELU activation (one stage only).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bench_stage_gelu(data: *mut f32, status: *mut u32) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{index, nn};
        let tid = index::thread_idx_x();
        let val = *data.add(tid as usize);
        *data.add(tid as usize) = nn::gelu_f32(val);
        if tid == 0 {
            core::ptr::write_volatile(status, 1);
        }
    }
}

/// Stage kernel: warp reduction sum + write result.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bench_stage_reduce(
    data: *const f32,
    result: *mut f32,
    status: *mut u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{index, warp};
        let tid = index::thread_idx_x();
        let val = *data.add(tid as usize);
        let sum = warp::reduce_sum_f32(val);
        if tid == 0 {
            core::ptr::write_volatile(result, sum);
            core::ptr::write_volatile(status, 1);
        }
    }
}

/// Multi-stage compute pipeline with GPU-autonomous convergence.
///
/// # Arguments
/// * `output` - 32 f32 results (one per lane)
/// * `status` - 4 u32 values: [iterations, elapsed_nanos_lo, elapsed_nanos_hi, done_flag]
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (32, 1, 1) — one full warp
/// * Shared memory: 0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_pipeline_demo(
    output: *mut f32,
    status: *mut u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{index, math, nn, warp};

        let tid = index::thread_idx_x();
        let start = index::clock_nanos();

        // ── Stage 1: Generate test data using math intrinsics ──
        // Each lane gets a unique value derived from sin + cos
        let x = math::sin_f32(tid as f32 * 0.3) + math::cos_f32(tid as f32 * 0.17) + 1.0;

        // ── Iterative compute pipeline (GPU-autonomous) ──
        let mut val = x;
        let mut iterations = 0u32;
        let target_sum = 16.0f32;
        let epsilon = 0.05f32;

        loop {
            // Stage 2: Warp softmax — normalize values across all 32 lanes
            val = nn::warp_softmax_f32(val);

            // Stage 3: GELU activation — nonlinear transform
            val = nn::gelu_f32(val);

            // Stage 4: Warp reduction — sum all 32 lanes
            let sum = warp::reduce_sum_f32(val);

            iterations += 1;

            // Stage 5: Convergence check — entirely on GPU, zero host roundtrip
            let diff = math::abs_f32(sum - target_sum);
            if diff < epsilon || iterations >= 100 {
                break;
            }

            // Adjust toward target for next iteration
            // Scale each lane's value proportionally
            if sum > 0.0 {
                val = val * (target_sum / sum);
            } else {
                val = target_sum / 32.0;
            }
        }

        // ── Stage 6: Write final results ──
        core::ptr::write_volatile(output.add(tid as usize), val);

        // ── Stage 7: Write timing + stats (thread 0 only) ──
        if tid == 0 {
            let elapsed = index::clock_nanos() - start;
            core::ptr::write_volatile(status, iterations);
            core::ptr::write_volatile(status.add(1), elapsed as u32);
            core::ptr::write_volatile(status.add(2), (elapsed >> 32) as u32);
            core::ptr::write_volatile(status.add(3), 1); // done flag
        }
    }
}

/// Block-level softmax demo — uses shared memory for block-wide reduction.
///
/// Demonstrates block-level compute utilities (block::reduce_max_f32, block::reduce_sum_f32).
/// Each thread computes softmax(input[tid]) across the entire block.
///
/// # Arguments
/// * `input` - N f32 input values
/// * `output` - N f32 softmax output values
/// * `n` - number of elements (must equal block_dim_x, power of 2)
/// * `status` - 1 u32 done flag
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (N, 1, 1) — N must be power of 2, ≤ 1024
/// * Shared memory: N * 4 bytes
#[no_mangle]
pub unsafe extern "ptx-kernel" fn block_softmax_demo(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{block, index, math};

        let tid = index::thread_idx_x();
        if tid >= n {
            return;
        }

        let x = *input.add(tid as usize);

        // Phase 1: Find max across block (numerically stable softmax)
        let max_val = block::reduce_max_f32(x, tid, n, 0);

        // Phase 2: Compute exp(x - max)
        let exp_val = math::exp_f32(x - max_val);

        // Phase 3: Sum all exp values across block
        let sum = block::reduce_sum_f32(exp_val, tid, n, 0);

        // Phase 4: Normalize
        *output.add(tid as usize) = exp_val / sum;

        if tid == 0 {
            core::ptr::write_volatile(status, 1);
        }
    }
}

/// Warp layer normalization demo — demonstrates nn::warp_layer_norm_f32.
///
/// # Arguments
/// * `input` - 32 f32 input values
/// * `gamma` - 32 f32 scale parameters
/// * `beta` - 32 f32 shift parameters
/// * `output` - 32 f32 normalized output values
/// * `status` - 1 u32 done flag
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (32, 1, 1) — one full warp
/// * Shared memory: 0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_layer_norm_demo(
    input: *const f32,
    gamma: *const f32,
    beta: *const f32,
    output: *mut f32,
    status: *mut u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        use gpu_runtime::{index, nn};

        let tid = index::thread_idx_x();
        if tid >= 32 {
            return;
        }

        let x = *input.add(tid as usize);
        let g = *gamma.add(tid as usize);
        let b = *beta.add(tid as usize);

        let result = nn::warp_layer_norm_f32(x, g, b);

        *output.add(tid as usize) = result;

        if tid == 0 {
            core::ptr::write_volatile(status, 1);
        }
    }
}
