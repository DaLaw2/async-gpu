// GEMM variants: tiled GEMM, softmax, multi-tile, multi-warp, multi-block, full GEMM, f32 GEMM.

use crate::helpers::{bar_sync, get_dynamic_smem_ptr, gpu_exp_f32};
use core::arch::nvptx;

// ============================================================
// gpu-compute.5: Tiled GEMM — MMA + shared memory pipeline
// ============================================================

/// gpu-compute.5: Tiled GEMM combining Tensor Core MMA + shared memory.
///
/// Demonstrates the full pipeline:
///   global memory -> shared memory -> MMA fragment registers -> MMA -> global memory
///
/// Computes D[16x8] = A[16x16] x B[16x8] + C (C=0).
/// A and B are f16, D is f32. Uses a single MMA tile (m16n8k16).
///
/// Test uses all-1.0 matrices: every element of D should be 16.0
/// (sum of 16 products of 1.0 x 1.0).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_tiled_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        // Step 1: Load A and B from global to shared memory
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem;
        let b_smem = smem.add(128);

        // 32 threads load 128 u32s of A (4 each)
        for i in 0..4u32 {
            let idx = (tid * 4 + i) as usize;
            *a_smem.add(idx) = *a_global.add(idx);
        }
        // 32 threads load 64 u32s of B (2 each)
        for i in 0..2u32 {
            let idx = (tid * 2 + i) as usize;
            *b_smem.add(idx) = *b_global.add(idx);
        }

        bar_sync();

        // Step 2: Load MMA fragments from shared memory.
        let a0 = *a_smem.add(0);
        let a1 = *a_smem.add(1);
        let a2 = *a_smem.add(2);
        let a3 = *a_smem.add(3);
        let b0 = *b_smem.add(0);
        let b1 = *b_smem.add(1);

        // C = 0 (f32 accumulator)
        let c0: u32 = 0;
        let c1: u32 = 0;
        let c2: u32 = 0;
        let c3: u32 = 0;

        // Step 3: Execute MMA
        let d0: u32;
        let d1: u32;
        let d2: u32;
        let d3: u32;
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );

        // Step 4: Write D fragments to global memory (thread-indexed layout)
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = d0;
        *d_global.add(out_base + 1) = d1;
        *d_global.add(out_base + 2) = d2;
        *d_global.add(out_base + 3) = d3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

/// gpu-compute.6: Softmax with shared memory reduction.
///
/// Computes softmax(x) for a vector of N f32 values (N <= 32, one per thread):
///   1. Find max via shared memory parallel reduction
///   2. Compute exp(x - max) per thread
///   3. Sum exp values via shared memory parallel reduction
///   4. Divide each exp by sum -> softmax output
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_softmax(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid >= n {
        return;
    }

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut f32;
        let x = *input.add(tid as usize);

        // Step 1: Find max via shared memory reduction
        *smem.add(tid as usize) = x;
        bar_sync();

        let mut stride = n / 2;
        while stride > 0 {
            if tid < stride {
                let a = *smem.add(tid as usize);
                let b = *smem.add((tid + stride) as usize);
                if b > a {
                    *smem.add(tid as usize) = b;
                }
            }
            bar_sync();
            stride /= 2;
        }
        let max_val = *smem.add(0);
        bar_sync();

        // Step 2: Compute exp(x - max) per thread
        let exp_val = gpu_exp_f32(x - max_val);
        *smem.add(tid as usize) = exp_val;
        bar_sync();

        // Step 3: Sum via shared memory reduction
        stride = n / 2;
        while stride > 0 {
            if tid < stride {
                let a = *smem.add(tid as usize);
                let b = *smem.add((tid + stride) as usize);
                *smem.add(tid as usize) = a + b;
            }
            bar_sync();
            stride /= 2;
        }
        let sum = *smem.add(0);
        bar_sync();

        // Step 4: Normalize
        *output.add(tid as usize) = exp_val / sum;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-pipeline.2: Multi-tile K-accumulation GEMM loop
// ============================================================

/// Multi-tile GEMM: D = A(16xK) x B(Kx8) with K-dimension tiling.
///
/// Loops over K in tiles of 16, accumulating MMA results in f32 registers.
/// A is row-major f16x2 packed [16][K/2] u32, B is row-major f16x2 packed [K][4] u32.
/// D output is 16x8 f32 in thread-indexed layout (128 u32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_multi_tile_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    k_tiles: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [16][8] = 128 u32 per tile
        let b_smem = smem.add(128); // [16][4] = 64 u32 per tile

        let group = tid / 4;
        let lane = tid % 4;
        let k_half = k_tiles * 8; // K/2 = packed u32 count per row of A

        // Initialize accumulator to zero
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Load A tile: 32 threads load 128 u32 (4 each)
            // A_tile[row][col_packed] = A_full[row][t*8 + col_packed]
            let mut i = 0u32;
            while i < 4 {
                let smem_idx = tid * 4 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }

            // Load B tile: 32 threads load 64 u32 (2 each)
            // B_tile[row][col_packed] = B_full[t*16 + row][col_packed]
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 4;
                let col_packed = smem_idx % 4;
                let global_idx = (t * 16 + row) * 4 + col_packed;
                *b_smem.add(smem_idx as usize) = *b_global.add(global_idx as usize);
                i += 1;
            }

            bar_sync();

            // Load MMA fragments from shared memory (same mapping as gpu-pipeline.1)
            let a0 = *a_smem.add((group * 8 + lane) as usize);
            let a1 = *a_smem.add(((group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add((group * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);

            let b0 = *b_smem.add((group * 4 + lane) as usize);
            let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

            // MMA: D = A*B + C (accumulate across tiles)
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            // Feed D back as C for next iteration
            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync(); // Ensure all threads done before overwriting smem
            t += 1;
        }

        // Write final accumulated D fragments to output
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = c0;
        *d_global.add(out_base + 1) = c1;
        *d_global.add(out_base + 2) = c2;
        *d_global.add(out_base + 3) = c3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-pipeline.3: End-to-end GEMM + softmax pipeline
// ============================================================

/// Autonomous GEMM + softmax pipeline: output = softmax(A x B, per row).
///
/// Phase 1: Multi-tile GEMM (reuses gpu-pipeline.2 pattern)
/// Phase 2: Write GEMM output to shared memory in matrix order
/// Phase 3: Per-row softmax (16 threads, 1 row each, 8 elements)
///
/// This demonstrates GPU-autonomous multi-step compute: the host launches once,
/// and the GPU executes the entire GEMM -> softmax pipeline without intervention.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_gemm_softmax_pipeline(
    a_global: *const u32,
    b_global: *const u32,
    softmax_output: *mut f32,
    k_tiles: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // 128 u32 for A tile
        let b_smem = smem.add(128); // 64 u32 for B tile

        let group = tid / 4;
        let lane = tid % 4;
        let k_half = k_tiles * 8;

        // === Phase 1: Multi-tile GEMM ===
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            let mut i = 0u32;
            while i < 4 {
                let smem_idx = tid * 4 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 4;
                let col_packed = smem_idx % 4;
                let global_idx = (t * 16 + row) * 4 + col_packed;
                *b_smem.add(smem_idx as usize) = *b_global.add(global_idx as usize);
                i += 1;
            }
            bar_sync();

            let a0 = *a_smem.add((group * 8 + lane) as usize);
            let a1 = *a_smem.add(((group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add((group * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);
            let b0 = *b_smem.add((group * 4 + lane) as usize);
            let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;
            bar_sync();
            t += 1;
        }

        // === Phase 2: Write GEMM output to shared memory in matrix order ===
        // Fragment mapping: d0=D[lane*2][group], d1=D[lane*2+1][group],
        //                   d2=D[lane*2+8][group], d3=D[lane*2+9][group]
        let d_smem = smem as *mut f32; // reuse shared memory (128 f32 fits in 192 u32)
        *d_smem.add((lane * 2 * 8 + group) as usize) = f32::from_bits(c0);
        *d_smem.add(((lane * 2 + 1) * 8 + group) as usize) = f32::from_bits(c1);
        *d_smem.add(((lane * 2 + 8) * 8 + group) as usize) = f32::from_bits(c2);
        *d_smem.add(((lane * 2 + 9) * 8 + group) as usize) = f32::from_bits(c3);
        bar_sync();

        // === Phase 3: Per-row softmax (threads 0-15 each handle one row) ===
        if tid < 16 {
            let row_base = (tid * 8) as usize;

            // Find max in this row
            let mut max_val = *d_smem.add(row_base);
            let mut j = 1usize;
            while j < 8 {
                let v = *d_smem.add(row_base + j);
                if v > max_val {
                    max_val = v;
                }
                j += 1;
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            let mut exp_vals = [0.0f32; 8];
            j = 0;
            while j < 8 {
                let e = gpu_exp_f32(*d_smem.add(row_base + j) - max_val);
                exp_vals[j] = e;
                sum += e;
                j += 1;
            }

            // Normalize and write to global output
            j = 0;
            while j < 8 {
                *softmax_output.add(row_base + j) = exp_vals[j] / sum;
                j += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, softmax_output, k_tiles);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gemm-scale.1: Multi-warp output tiling
// ============================================================

/// Multi-warp GEMM: D(32x16) = A(32xK) x B(Kx16), 4 warps in 2x2 layout.
///
/// 128 threads (4 warps), each warp computes a 16x8 MMA tile.
/// Warp layout: warp_m = warp_id/2 (0..1), warp_n = warp_id%2 (0..1).
/// Shared memory per K-tile: A[32][8] + B[16][8] = 384 u32.
/// A is row-major f16x2 packed [32][K/2] u32.
/// B is row-major f16x2 packed [K][8] u32 (N=16 -> 8 packed per row).
/// D is row-major f32 [32][16].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_warp_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        // Warp arrangement: 2x2 (2 in M, 2 in N)
        let warp_m = warp_id / 2; // 0 or 1
        let warp_n = warp_id % 2; // 0 or 1

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32 (col-major packed)

        let k_half = k_tiles * 8; // K/2 = packed u32 per row of A
        let k_half_cm = k_tiles * 8; // K/2 = packed u32 per column of B (col-major)

        // Initialize accumulator
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32, 128 threads -> 2 each
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: [N][8] col-major packed, 128 threads -> 1 each
            // B_cm layout: b_global[col * k_half_cm + k_pair]
            // = pack(B[k_pair*2][col], B[k_pair*2+1][col])
            if tid < 128 {
                let col = tid / 8; // N column (0..15)
                let k_pair = tid % 8; // row pair within tile (0..7)
                let global_idx = col * k_half_cm + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_global.add(global_idx as usize);
            }

            bar_sync();

            // Load A fragments for this warp's M-slice (warp_m * 16)
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            // Load B fragments for this warp's N-slice (col-major packed)
            // b_smem layout: [N][8], col = warp_n*8+group, k_pair = lane
            // b0 = pack(B[lane*2][col], B[lane*2+1][col])
            // b1 = pack(B[lane*2+8][col], B[lane*2+9][col])
            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // MMA: D = A*B + C
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output in row-major [32][16] f32
        // Correct MMA fragment mapping (m16n8k16.row.col):
        //   d0->D[group][lane*2], d1->D[group][lane*2+1],
        //   d2->D[group+8][lane*2], d3->D[group+8][lane*2+1]
        let r0 = warp_m * 16 + group;
        let r2 = warp_m * 16 + group + 8;
        let c0_idx = warp_n * 8 + lane * 2;
        let c1_idx = c0_idx + 1;

        *d_global.add((r0 * n_cols + c0_idx) as usize) = f32::from_bits(c0);
        *d_global.add((r0 * n_cols + c1_idx) as usize) = f32::from_bits(c1);
        *d_global.add((r2 * n_cols + c0_idx) as usize) = f32::from_bits(c2);
        *d_global.add((r2 * n_cols + c1_idx) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// Multi-block GEMM (gemm-scale.2)
// ============================================================

/// Multi-block GEMM: D(MxN) = A(MxK) x B(KxN), M = num_blocks * 32, N = 16.
///
/// Each block: 128 threads (4 warps), computes D[block_m*32..(block_m+1)*32][0..15].
/// grid_dim = (M/32, 1, 1), block_dim = (128, 1, 1).
/// A is row-major f16x2 packed [M][K/2] u32.
/// B is column-major f16x2 packed [N][K/2] u32.
/// D is row-major f32 [M][N].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_block_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    m_rows: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32; // which 32-row block

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        // Warp arrangement: 2x2 (2 in M, 2 in N)
        let warp_m = warp_id / 2; // 0 or 1
        let warp_n = warp_id % 2; // 0 or 1

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32 (col-major packed)

        let k_half = k_tiles * 8; // K/2 = packed u32 per row of A

        // Offset A by block_m * 32 rows
        let a_block = a_global.add((block_m * 32 * k_half) as usize);

        // Initialize accumulator
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32, 128 threads -> 2 each
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_block.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: [N][8] col-major packed, 128 threads -> 1 each
            // B is shared across all blocks — no offset needed
            if tid < 128 {
                let col = tid / 8; // N column (0..15)
                let k_pair = tid % 8; // row pair within tile (0..7)
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_global.add(global_idx as usize);
            }

            bar_sync();

            // Load A fragments for this warp's M-slice (warp_m * 16)
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            // Load B fragments for this warp's N-slice (col-major packed)
            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // MMA: D = A*B + C
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output — offset by block_m * 32 rows in global D
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let c0_idx = warp_n * 8 + lane * 2;
        let c1_idx = c0_idx + 1;

        *d_global.add((global_r0 * n_cols + c0_idx) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + c1_idx) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + c0_idx) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + c1_idx) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols, m_rows);
    }

    // All blocks atomically increment status; host checks for num_blocks
    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Full GEMM with 2D tiling (gemm-scale.3)
// ============================================================

/// Full GEMM: D(MxN) = A(MxK) x B(KxN), arbitrary M/N multiples of 32/16.
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Each block: 128 threads (4 warps), computes D[bm*32..(bm+1)*32][bn*16..(bn+1)*16].
/// A is row-major f16x2 packed [M][K/2] u32.
/// B is column-major f16x2 packed [N][K/2] u32.
/// D is row-major f32 [M][N].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_half = k_tiles * 8; // K/2

        // A block offset: rows [block_m*32 .. block_m*32+32]
        let a_block = a_global.add((block_m * 32 * k_half) as usize);
        // B block offset: columns [block_n*16 .. block_n*16+16]
        // B is col-major packed: [N][K/2], so column offset = block_n * 16 * k_half
        let b_block = b_global.add((block_n * 16 * k_half) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_block.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: 16 columns x 8 k-pairs = 128 u32
            if tid < 128 {
                let col = tid / 8; // local column within this 16-col block
                let k_pair = tid % 8;
                // b_block already offset to the right 16 columns
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_block.add(global_idx as usize);
            }

            bar_sync();

            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output with global row/col offsets
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Full GEMM with f32 A input (precision-fix.2)
// ============================================================

/// Full GEMM with f32 activation input: D(MxN) = A(MxK) x B(KxN).
///
/// Same as `full_gemm` but A is row-major f32 [M][K] instead of packed f16x2.
/// The kernel converts f32->f16 per-tile in shared memory, eliminating the
/// separate f32_to_f16x2_pack kernel launch and global memory f16 roundtrip.
///
/// B is still column-major f16x2 packed [N][K/2] u32 (weights, packed once).
/// D is row-major f32 [M][N].
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Shared memory: (256 + 128) * 4 = 1536 bytes (same as full_gemm).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm_f32in(
    a_global: *const f32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32 (f16x2 packed in smem)
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_full = k_tiles * 16; // K (number of f32 elements per A row)
        let k_half = k_tiles * 8; // K/2 (number of u32 per B column)

        // A block offset: rows [block_m*32 .. block_m*32+32], f32 layout
        let a_block = a_global.add((block_m * 32 * k_full) as usize);
        // B block offset: columns [block_n*16 .. block_n*16+16]
        let b_block = b_global.add((block_n * 16 * k_half) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: read f32 pairs, convert to f16x2, store in smem
            // smem layout: [32][8] u32, each u32 = 2 packed f16 values
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                // Two f32 values that map to this packed u32
                let k_base = t * 16 + col_packed * 2;
                let v0 = *a_block.add((row * k_full + k_base) as usize);
                let v1 = *a_block.add((row * k_full + k_base + 1) as usize);
                // Convert f32 -> f16 via PTX cvt
                let h0: u32;
                let h1: u32;
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h0,
                    f = in(reg32) v0,
                );
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h1,
                    f = in(reg32) v1,
                );
                // Pack: lo | (hi << 16)
                let packed = (h0 & 0xFFFF) | (h1 << 16);
                *a_smem.add(smem_idx as usize) = packed;
                i += 1;
            }

            // Cooperative load B tile: 16 columns x 8 k-pairs = 128 u32 (same as full_gemm)
            if tid < 128 {
                let col = tid / 8;
                let k_pair = tid % 8;
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_block.add(global_idx as usize);
            }

            bar_sync();

            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output with global row/col offsets (same as full_gemm)
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// BF16 MMA GEMM kernel (mixed-precision.1)
// ============================================================

/// Full GEMM using BF16 Tensor Core MMA with f32 inputs.
///
/// Both A and B are f32 — converted to BF16 on-the-fly in shared memory.
/// Uses mma.sync.aligned.m16n8k16 with BF16 inputs and f32 accumulator.
/// BF16 has 8-bit exponent (same range as f32) + 7-bit mantissa, providing
/// better dynamic range than f16 (5-bit exponent) at the cost of slightly
/// less precision than f16 (7 vs 10 mantissa bits).
///
/// A: row-major f32 [M][K]
/// B: column-major f32 [N][K]
/// D: row-major f32 [M][N]
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Shared memory: (256 + 128) * 4 = 1536 bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm_bf16(
    a_global: *const f32,
    b_global: *const f32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32 (bf16x2 packed)
        let b_smem = smem.add(256); // [16][8] = 128 u32 (bf16x2 packed)

        let k_full = k_tiles * 16; // K elements per row/column

        let a_block = a_global.add((block_m * 32 * k_full) as usize);
        let b_block = b_global.add((block_n * 16 * k_full) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: read f32 pairs, convert to bf16x2, store in smem
            // 128 threads handle 256 packed u32 = 2 iterations per thread
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let k_base = t * 16 + col_packed * 2;
                let v0 = *a_block.add((row * k_full + k_base) as usize);
                let v1 = *a_block.add((row * k_full + k_base + 1) as usize);
                let packed: u32;
                core::arch::asm!(
                    "cvt.rn.bf16x2.f32 {p}, {f1}, {f0};",
                    p = out(reg32) packed,
                    f0 = in(reg32) v0,
                    f1 = in(reg32) v1,
                );
                *a_smem.add(smem_idx as usize) = packed;
                i += 1;
            }

            // Cooperative load B tile: read f32 pairs from column-major B, convert to bf16x2
            if tid < 128 {
                let col = tid / 8;
                let k_pair = tid % 8;
                let k_base = t * 16 + k_pair * 2;
                let v0 = *b_block.add((col * k_full + k_base) as usize);
                let v1 = *b_block.add((col * k_full + k_base + 1) as usize);
                let packed: u32;
                core::arch::asm!(
                    "cvt.rn.bf16x2.f32 {p}, {f1}, {f0};",
                    p = out(reg32) packed,
                    f0 = in(reg32) v0,
                    f1 = in(reg32) v1,
                );
                *b_smem.add(tid as usize) = packed;
            }

            bar_sync();

            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // BF16 MMA: D = A*B + C, accumulating across k-tiles
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// TF32 Tensor Core GEMM kernel (tf32-mma.1)
// ============================================================

/// Full GEMM using TF32 Tensor Core MMA with f32 inputs.
///
/// Both A and B are f32 — loaded directly into shared memory as f32.
/// The GPU hardware internally truncates to TF32 (10-bit mantissa, 8-bit exponent)
/// when executing the MMA instruction. No explicit conversion needed.
///
/// Uses mma.sync.aligned.m16n8k8 with TF32 inputs and f32 accumulator.
/// TF32 has the same 10-bit mantissa as f16 but with f32 exponent range (8-bit),
/// avoiding overflow/underflow issues that plague f16.
///
/// A: row-major f32 [M][K]
/// B: column-major f32 [N][K]
/// D: row-major f32 [M][N]
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Shared memory: (256 + 128) * 4 = 1536 bytes (A[32][8] + B[16][8] f32).
/// k_tiles = K / 8 (MMA k-dimension is 8 for TF32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm_tf32(
    a_global: *const f32,
    b_global: *const f32,
    d_global: *mut f32,
    k_tiles: u32, // K / 8
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 f32 values
        let b_smem = smem.add(256); // [16][8] = 128 f32 values

        let k_full = k_tiles * 8; // K elements per row/column

        let a_block = a_global.add((block_m * 32 * k_full) as usize);
        let b_block = b_global.add((block_n * 16 * k_full) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: 128 threads load 256 f32 (2 per thread)
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col = smem_idx % 8;
                let v = *a_block.add((row * k_full + t * 8 + col) as usize);
                *a_smem.add(smem_idx as usize) = v.to_bits();
                i += 1;
            }

            // Cooperative load B tile: 128 threads load 128 f32 (1 per thread)
            if tid < 128 {
                let col = tid / 8;
                let k_idx = tid % 8;
                let v = *b_block.add((col * k_full + t * 8 + k_idx) as usize);
                *b_smem.add(tid as usize) = v.to_bits();
            }

            bar_sync();

            // Load MMA fragments from shared memory
            // A: a[i] = A_smem[(warp_row + group/group+8) * 8 + lane/lane+4]
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            // B: b[i] = B_smem[col * 8 + lane/lane+4]
            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // TF32 MMA: D = A*B + C, accumulating across k-tiles
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );
            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();

            t += 1;
        }

        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Split-K f16 MMA GEMM kernel (mma-splitk.2)
// ============================================================

/// Split-K GEMM: D(MxN) = A(MxK) x B(KxN) with K-dimension partitioning.
///
/// Partitions K across grid.z thread blocks. Each z-slice computes a partial
/// result for its K chunk, then atomically adds to the output buffer.
/// This limits per-block accumulation error and improves SM utilization.
///
/// A: [M, K] row-major f32 (converted to f16 per-tile in shared memory).
/// B: [N, K/2] column-major f16x2 packed u32.
/// D: [M, N] row-major f32 — MUST be zero-initialized before launch.
///
/// grid_dim = (M/32, N/16, split_k), block_dim = (128, 1, 1).
/// shared_mem_bytes = (256 + 128) * 4 = 1536.
///
/// z=0 writes directly (no atomic), z>0 uses atomicAdd to reduce contention.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm_splitk(
    a_global: *const f32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    split_k: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);
        let block_z: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.z;", r = out(reg32) block_z);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_full = k_tiles * 16; // K in f32 elements
        let k_half = k_tiles * 8; // K/2 in packed u32

        // Compute K range for this z-slice
        let tiles_per_split = (k_tiles + split_k - 1) / split_k;
        let t_start = block_z * tiles_per_split;
        let t_end = if t_start + tiles_per_split < k_tiles {
            t_start + tiles_per_split
        } else {
            k_tiles
        };

        // Skip if this z-slice has no work (can happen with uneven division)
        if t_start >= k_tiles {
            if tid == 0 {
                let _prev: u32;
                core::arch::asm!(
                    "atom.global.add.u32 {prev}, [{addr}], 1;",
                    prev = out(reg32) _prev,
                    addr = in(reg64) status,
                );
            }
            return;
        }

        // A block offset: rows [block_m*32 .. block_m*32+32]
        let a_block = a_global.add((block_m * 32 * k_full) as usize);
        // B block offset: columns [block_n*16 .. block_n*16+16]
        let b_block = b_global.add((block_n * 16 * k_half) as usize);

        // Initialize accumulator to zero for this partial sum
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        // Iterate over this z-slice's K range only
        let mut t = t_start;
        while t < t_end {
            // Cooperative load A tile: read f32 pairs, convert to f16x2
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let k_base = t * 16 + col_packed * 2;
                let v0 = *a_block.add((row * k_full + k_base) as usize);
                let v1 = *a_block.add((row * k_full + k_base + 1) as usize);
                let h0: u32;
                let h1: u32;
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h0,
                    f = in(reg32) v0,
                );
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h1,
                    f = in(reg32) v1,
                );
                let packed = (h0 & 0xFFFF) | (h1 << 16);
                *a_smem.add(smem_idx as usize) = packed;
                i += 1;
            }

            // Cooperative load B tile: 16 columns x 8 k-pairs = 128 u32
            if tid < 128 {
                let col = tid / 8;
                let k_pair = tid % 8;
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_block.add(global_idx as usize);
            }

            bar_sync();

            // Load A/B fragments and execute MMA
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write partial results to output
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        let idx0 = (global_r0 * n_cols + global_c0) as usize;
        let idx1 = (global_r0 * n_cols + global_c1) as usize;
        let idx2 = (global_r2 * n_cols + global_c0) as usize;
        let idx3 = (global_r2 * n_cols + global_c1) as usize;

        // All z-slices use atomicAdd (output is zero-initialized).
        // Using direct write for z=0 would race with z>0's atomicAdds.
        let _old0: u32;
        let _old1: u32;
        let _old2: u32;
        let _old3: u32;
        core::arch::asm!(
            "atom.global.add.f32 {old}, [{addr}], {val};",
            old = out(reg32) _old0,
            addr = in(reg64) d_global.add(idx0),
            val = in(reg32) c0,
        );
        core::arch::asm!(
            "atom.global.add.f32 {old}, [{addr}], {val};",
            old = out(reg32) _old1,
            addr = in(reg64) d_global.add(idx1),
            val = in(reg32) c1,
        );
        core::arch::asm!(
            "atom.global.add.f32 {old}, [{addr}], {val};",
            old = out(reg32) _old2,
            addr = in(reg64) d_global.add(idx2),
            val = in(reg32) c2,
        );
        core::arch::asm!(
            "atom.global.add.f32 {old}, [{addr}], {val};",
            old = out(reg32) _old3,
            addr = in(reg64) d_global.add(idx3),
            val = in(reg32) c3,
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols, split_k);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Pure f32 GEMM kernel (full-inference.6)
// ============================================================

/// Tiled f32 GEMM without Tensor Cores: D = A x B.
///
/// A: [M, K] row-major f32.
/// B: [K, N] column-major f32 (stored as b[col * K + row]).
/// D: [M, N] row-major f32.
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// shared_mem_bytes = (32*16 + 16*16) * 4 = 3072.
///
/// Each block computes a 32x16 output tile. Each thread computes 4 output
/// elements (2 rows x 2 cols). K dimension is tiled in chunks of 16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_f32(
    a_global: *const f32,
    b_global: *const f32,
    d_global: *mut f32,
    k_dim: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let smem = get_dynamic_smem_ptr() as *mut f32;
        let a_smem = smem; // [32][16] = 512 f32
        let b_smem = smem.add(512); // [16][16] = 256 f32

        // Thread mapping: 128 threads -> 32x16 output = 4 per thread
        // tid / 8 = row_pair (0..15), handles rows row_pair*2, row_pair*2+1
        // tid % 8 = col_pair (0..7), handles cols col_pair*2, col_pair*2+1
        let row_pair = tid / 8;
        let col_pair = tid % 8;
        let r0 = row_pair * 2;
        let r1 = r0 + 1;
        let c0 = col_pair * 2;
        let c1 = c0 + 1;

        let global_r0 = block_m * 32 + r0;
        let global_r1 = block_m * 32 + r1;
        let global_c0 = block_n * 16 + c0;
        let global_c1 = block_n * 16 + c1;

        // A block starts at row (block_m * 32)
        let a_block = a_global.add((block_m * 32 * k_dim) as usize);
        // B block starts at col (block_n * 16)
        let b_block = b_global.add((block_n * 16 * k_dim) as usize);

        let mut acc00: f32 = 0.0;
        let mut acc01: f32 = 0.0;
        let mut acc10: f32 = 0.0;
        let mut acc11: f32 = 0.0;

        let k_tiles = (k_dim + 15) / 16;
        let mut t = 0u32;
        while t < k_tiles {
            let k_base = t * 16;

            // Cooperative load A tile: 32x16 = 512 f32, 128 threads x 4 elements each
            let mut i = 0u32;
            while i < 4 {
                let flat = tid * 4 + i;
                let row = flat / 16;
                let col = flat % 16;
                let k_idx = k_base + col;
                let val = if k_idx < k_dim {
                    *a_block.add((row * k_dim + k_idx) as usize)
                } else {
                    0.0
                };
                *a_smem.add(flat as usize) = val;
                i += 1;
            }

            // Cooperative load B tile: 16x16 = 256 f32, 128 threads x 2 elements each
            let mut j = 0u32;
            while j < 2 {
                let flat = tid * 2 + j;
                let col = flat / 16; // B column within this tile (0..15)
                let k_row = flat % 16; // K row within tile
                let k_idx = k_base + k_row;
                let b_col = block_n * 16 + col;
                let val = if k_idx < k_dim && b_col < n_cols {
                    // B is column-major: b[col * K + k]
                    *b_global.add((b_col * k_dim + k_idx) as usize)
                } else {
                    0.0
                };
                // Store in shared mem as b_smem[col][k_row] = b_smem[col * 16 + k_row]
                *b_smem.add(flat as usize) = val;
                j += 1;
            }

            bar_sync();

            // Compute: each thread accumulates 4 output elements over K=16
            let tile_k = if k_base + 16 <= k_dim {
                16
            } else {
                k_dim - k_base
            };
            let mut k = 0u32;
            while k < tile_k {
                let a_r0 = *a_smem.add((r0 * 16 + k) as usize);
                let a_r1 = *a_smem.add((r1 * 16 + k) as usize);
                let b_c0 = *b_smem.add((c0 * 16 + k) as usize);
                let b_c1 = *b_smem.add((c1 * 16 + k) as usize);

                // Use FMA for better precision
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc00,
                    a = in(reg32) a_r0,
                    b = in(reg32) b_c0,
                    c = in(reg32) acc00,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc01,
                    a = in(reg32) a_r0,
                    b = in(reg32) b_c1,
                    c = in(reg32) acc01,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc10,
                    a = in(reg32) a_r1,
                    b = in(reg32) b_c0,
                    c = in(reg32) acc10,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc11,
                    a = in(reg32) a_r1,
                    b = in(reg32) b_c1,
                    c = in(reg32) acc11,
                );

                k += 1;
            }

            bar_sync();
            t += 1;
        }

        // Write output
        *d_global.add((global_r0 * n_cols + global_c0) as usize) = acc00;
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = acc01;
        *d_global.add((global_r1 * n_cols + global_c0) as usize) = acc10;
        *d_global.add((global_r1 * n_cols + global_c1) as usize) = acc11;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_dim, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// High-performance f32 GEMM v2: 128×128 tile, 8×8 register blocking
// ============================================================

/// High-performance tiled f32 GEMM v2: D = A × B.
///
/// A: [M, K] row-major f32.
/// B: [K, N] row-major f32.  (NO transpose needed — both row-major)
/// D: [M, N] row-major f32.
///
/// CTA tile: 128×64, BK=8.
/// 256 threads, each thread computes 4×8 = 32 output elements.
/// A stored transposed in smem with padding for bank conflict avoidance.
/// Double-buffered shared memory.
///
/// grid_dim = (ceil(M/128), ceil(N/64), 1), block_dim = (256, 1, 1).
/// shared_mem_bytes = 2 * (8 * 132 + 8 * 64) * 4 = 11264.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_f32_v2(
    a_global: *const f32,
    b_global: *const f32,
    d_global: *mut f32,
    m_dim: u32,
    n_dim: u32,
    k_dim: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m: u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.x;", r = out(reg32) block_m);
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        // Tile constants: 128×64 CTA tile, BK=8
        const BM: u32 = 128;
        const BN: u32 = 64;
        const BK: u32 = 8;
        const A_STRIDE: u32 = BM + 4; // 132 — padding avoids bank conflicts
        const STAGE_FLOATS: u32 = BK * A_STRIDE + BK * BN; // 8*132 + 8*64 = 1056+512 = 1568

        let smem = get_dynamic_smem_ptr() as *mut f32;

        // Thread mapping: 256 threads as 32×8, each computes 4×8 = 32 outputs
        let thread_row = tid / 8; // 0..31
        let thread_col = tid % 8; // 0..7

        // 32 named accumulators (4 rows × 8 cols) — prevents register spilling
        let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
        let mut c04: f32 = 0.0; let mut c05: f32 = 0.0; let mut c06: f32 = 0.0; let mut c07: f32 = 0.0;
        let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
        let mut c14: f32 = 0.0; let mut c15: f32 = 0.0; let mut c16: f32 = 0.0; let mut c17: f32 = 0.0;
        let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
        let mut c24: f32 = 0.0; let mut c25: f32 = 0.0; let mut c26: f32 = 0.0; let mut c27: f32 = 0.0;
        let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;
        let mut c34: f32 = 0.0; let mut c35: f32 = 0.0; let mut c36: f32 = 0.0; let mut c37: f32 = 0.0;

        let a_base_row = block_m * BM;
        let b_base_col = block_n * BN;
        let k_tiles = (k_dim + BK - 1) / BK;

        // Load first tile into buffer 0
        let mut buf: u32 = 0;
        gemm_v2_load_tile_128x64(
            smem, buf, STAGE_FLOATS, A_STRIDE, a_global, b_global, a_base_row, b_base_col, 0,
            m_dim, n_dim, k_dim, tid,
        );
        bar_sync();

        let mut t: u32 = 0;
        while t < k_tiles {
            // Prefetch next tile
            if t + 1 < k_tiles {
                let next_buf = 1 - buf;
                gemm_v2_load_tile_128x64(
                    smem, next_buf, STAGE_FLOATS, A_STRIDE, a_global, b_global, a_base_row,
                    b_base_col, (t + 1) * BK, m_dim, n_dim, k_dim, tid,
                );
            }

            let a_smem = (buf * STAGE_FLOATS) as usize;
            let b_smem = (buf * STAGE_FLOATS + BK * A_STRIDE) as usize;

            let a_row_base = (thread_row * 4) as usize;
            let b_col_base = (thread_col * 8) as usize;

            let mut k: u32 = 0;
            while k < BK {
                if t * BK + k >= k_dim {
                    break;
                }

                // Load 4 A values
                let ao = a_smem + (k * A_STRIDE) as usize + a_row_base;
                let a0 = *smem.add(ao);
                let a1 = *smem.add(ao + 1);
                let a2 = *smem.add(ao + 2);
                let a3 = *smem.add(ao + 3);

                // Load 8 B values
                let bo = b_smem + (k * BN) as usize + b_col_base;
                let b0 = *smem.add(bo);
                let b1 = *smem.add(bo + 1);
                let b2 = *smem.add(bo + 2);
                let b3 = *smem.add(bo + 3);
                let b4 = *smem.add(bo + 4);
                let b5 = *smem.add(bo + 5);
                let b6 = *smem.add(bo + 6);
                let b7 = *smem.add(bo + 7);

                // 32 FMAs (4×8 outer product)
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c00, a = in(reg32) a0, b = in(reg32) b0, c = in(reg32) c00);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c01, a = in(reg32) a0, b = in(reg32) b1, c = in(reg32) c01);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c02, a = in(reg32) a0, b = in(reg32) b2, c = in(reg32) c02);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c03, a = in(reg32) a0, b = in(reg32) b3, c = in(reg32) c03);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c04, a = in(reg32) a0, b = in(reg32) b4, c = in(reg32) c04);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c05, a = in(reg32) a0, b = in(reg32) b5, c = in(reg32) c05);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c06, a = in(reg32) a0, b = in(reg32) b6, c = in(reg32) c06);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c07, a = in(reg32) a0, b = in(reg32) b7, c = in(reg32) c07);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c10, a = in(reg32) a1, b = in(reg32) b0, c = in(reg32) c10);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c11, a = in(reg32) a1, b = in(reg32) b1, c = in(reg32) c11);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c12, a = in(reg32) a1, b = in(reg32) b2, c = in(reg32) c12);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c13, a = in(reg32) a1, b = in(reg32) b3, c = in(reg32) c13);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c14, a = in(reg32) a1, b = in(reg32) b4, c = in(reg32) c14);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c15, a = in(reg32) a1, b = in(reg32) b5, c = in(reg32) c15);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c16, a = in(reg32) a1, b = in(reg32) b6, c = in(reg32) c16);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c17, a = in(reg32) a1, b = in(reg32) b7, c = in(reg32) c17);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c20, a = in(reg32) a2, b = in(reg32) b0, c = in(reg32) c20);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c21, a = in(reg32) a2, b = in(reg32) b1, c = in(reg32) c21);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c22, a = in(reg32) a2, b = in(reg32) b2, c = in(reg32) c22);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c23, a = in(reg32) a2, b = in(reg32) b3, c = in(reg32) c23);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c24, a = in(reg32) a2, b = in(reg32) b4, c = in(reg32) c24);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c25, a = in(reg32) a2, b = in(reg32) b5, c = in(reg32) c25);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c26, a = in(reg32) a2, b = in(reg32) b6, c = in(reg32) c26);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c27, a = in(reg32) a2, b = in(reg32) b7, c = in(reg32) c27);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c30, a = in(reg32) a3, b = in(reg32) b0, c = in(reg32) c30);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c31, a = in(reg32) a3, b = in(reg32) b1, c = in(reg32) c31);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c32, a = in(reg32) a3, b = in(reg32) b2, c = in(reg32) c32);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c33, a = in(reg32) a3, b = in(reg32) b3, c = in(reg32) c33);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c34, a = in(reg32) a3, b = in(reg32) b4, c = in(reg32) c34);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c35, a = in(reg32) a3, b = in(reg32) b5, c = in(reg32) c35);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c36, a = in(reg32) a3, b = in(reg32) b6, c = in(reg32) c36);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) c37, a = in(reg32) a3, b = in(reg32) b7, c = in(reg32) c37);

                k += 1;
            }

            bar_sync();
            buf = 1 - buf;
            t += 1;
        }

        // Write 4×8 output
        let out_r = a_base_row + thread_row * 4;
        let out_c = b_base_col + thread_col * 8;

        macro_rules! write_if {
            ($r:expr, $c:expr, $val:expr) => {
                if $r < m_dim && $c < n_dim {
                    *d_global.add(($r * n_dim + $c) as usize) = $val;
                }
            };
        }
        write_if!(out_r, out_c, c00); write_if!(out_r, out_c+1, c01); write_if!(out_r, out_c+2, c02); write_if!(out_r, out_c+3, c03);
        write_if!(out_r, out_c+4, c04); write_if!(out_r, out_c+5, c05); write_if!(out_r, out_c+6, c06); write_if!(out_r, out_c+7, c07);
        write_if!(out_r+1, out_c, c10); write_if!(out_r+1, out_c+1, c11); write_if!(out_r+1, out_c+2, c12); write_if!(out_r+1, out_c+3, c13);
        write_if!(out_r+1, out_c+4, c14); write_if!(out_r+1, out_c+5, c15); write_if!(out_r+1, out_c+6, c16); write_if!(out_r+1, out_c+7, c17);
        write_if!(out_r+2, out_c, c20); write_if!(out_r+2, out_c+1, c21); write_if!(out_r+2, out_c+2, c22); write_if!(out_r+2, out_c+3, c23);
        write_if!(out_r+2, out_c+4, c24); write_if!(out_r+2, out_c+5, c25); write_if!(out_r+2, out_c+6, c26); write_if!(out_r+2, out_c+7, c27);
        write_if!(out_r+3, out_c, c30); write_if!(out_r+3, out_c+1, c31); write_if!(out_r+3, out_c+2, c32); write_if!(out_r+3, out_c+3, c33);
        write_if!(out_r+3, out_c+4, c34); write_if!(out_r+3, out_c+5, c35); write_if!(out_r+3, out_c+6, c36); write_if!(out_r+3, out_c+7, c37);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, m_dim, n_dim, k_dim);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

/// Load one A+B tile into shared memory for gemm_f32_v2 (128×64 version).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_v2_load_tile_128x64(
    smem: *mut f32,
    buf: u32,
    stage_floats: u32,
    a_stride: u32,
    a_global: *const f32,
    b_global: *const f32,
    a_base_row: u32,
    b_base_col: u32,
    k_start: u32,
    m_dim: u32,
    n_dim: u32,
    k_dim: u32,
    tid: u32,
) {
    let a_smem_base = (buf * stage_floats) as usize;
    let b_smem_base = (buf * stage_floats + 8 * a_stride) as usize;

    const BM: u32 = 128;
    const BN: u32 = 64;
    const BK: u32 = 8;

    // Load A: 128×8 = 1024 elements, 256 threads → 4 elements each
    let mut i: u32 = 0;
    while i < 4 {
        let flat = tid * 4 + i;
        let a_row = flat / BK;
        let a_k = flat % BK;

        let global_row = a_base_row + a_row;
        let global_k = k_start + a_k;

        let val = if global_row < m_dim && global_k < k_dim {
            *a_global.add((global_row * k_dim + global_k) as usize)
        } else {
            0.0
        };

        // Store transposed: smem[k * A_STRIDE + row]
        *smem.add(a_smem_base + (a_k * a_stride + a_row) as usize) = val;
        i += 1;
    }

    // Load B: 8×64 = 512 elements, 256 threads → 2 elements each
    let mut j: u32 = 0;
    while j < 2 {
        let flat = tid * 2 + j;
        let b_k = flat / BN;
        let b_col = flat % BN;

        let global_k = k_start + b_k;
        let global_col = b_base_col + b_col;

        let val = if global_k < k_dim && global_col < n_dim {
            *b_global.add((global_k * n_dim + global_col) as usize)
        } else {
            0.0
        };

        *smem.add(b_smem_base + (b_k * BN + b_col) as usize) = val;
        j += 1;
    }
}

// ============================================================
// MMA Fragment Diagnostic Kernel
// ============================================================

/// Diagnostic kernel that dumps MMA fragment values and result for debugging.
///
/// Single MMA tile (m16n8k16): A[16x16] (f32, GPU converts to f16) x B[16x8] (f16x2) = D[16x8]
/// Uses same fragment mapping as full_gemm_f32in but also writes fragment values
/// and per-thread MMA outputs to a debug buffer.
///
/// debug_buf layout (u32, total 384 entries):
///   [0..31]:    a0 for each thread in warp 0
///   [32..63]:   a1 for each thread in warp 0
///   [64..95]:   a2 for each thread in warp 0
///   [96..127]:  a3 for each thread in warp 0
///   [128..159]: b0 for each thread in warp 0
///   [160..191]: b1 for each thread in warp 0
///   [192..223]: d0 for each thread in warp 0
///   [224..255]: d1 for each thread in warp 0
///   [256..287]: d2 for each thread in warp 0
///   [288..319]: d3 for each thread in warp 0
///   [320..351]: smem_a[0..31] (first 32 entries of a shared memory)
///   [352..383]: smem_b[0..31] (first 32 entries of b shared memory)
///
/// grid_dim = (1, 1, 1), block_dim = (128, 1, 1), shared_mem = (256+128)*4
#[no_mangle]
pub unsafe extern "ptx-kernel" fn mma_diag(
    a_global: *const f32,
    b_global: *const u32,
    d_global: *mut f32,
    debug_buf: *mut u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_full = 16u32; // K = 16 (single tile)
        let k_half = 8u32; // K/2

        // Cooperative load A tile: read f32 pairs, convert to f16x2
        let mut i = 0u32;
        while i < 2 {
            let smem_idx = tid * 2 + i;
            let row = smem_idx / 8;
            let col_packed = smem_idx % 8;
            let k_base = col_packed * 2;
            let v0 = *a_global.add((row * k_full + k_base) as usize);
            let v1 = *a_global.add((row * k_full + k_base + 1) as usize);
            let h0: u32;
            let h1: u32;
            core::arch::asm!(
                "cvt.rn.f16.f32 {h}, {f};",
                h = out(reg32) h0,
                f = in(reg32) v0,
            );
            core::arch::asm!(
                "cvt.rn.f16.f32 {h}, {f};",
                h = out(reg32) h1,
                f = in(reg32) v1,
            );
            let packed = (h0 & 0xFFFF) | (h1 << 16);
            *a_smem.add(smem_idx as usize) = packed;
            i += 1;
        }

        // Cooperative load B tile
        if tid < 128 {
            let col = tid / 8;
            let k_pair = tid % 8;
            let global_idx = col * k_half + k_pair;
            *b_smem.add(tid as usize) = *b_global.add(global_idx as usize);
        }

        bar_sync();

        // Dump shared memory (warp 0 thread 0 dumps first 32 entries of each)
        if tid == 0 {
            let mut j = 0u32;
            while j < 32 {
                *debug_buf.add((320 + j) as usize) = *a_smem.add(j as usize);
                *debug_buf.add((352 + j) as usize) = *b_smem.add(j as usize);
                j += 1;
            }
        }

        // Load fragments — MMA m16n8k16 register order: a0=(row_lo,k_lo), a1=(row_hi,k_lo), a2=(row_lo,k_hi), a3=(row_hi,k_hi)
        let a_off = warp_m * 16;
        let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
        let a1 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
        let a2 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
        let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

        let b_col = warp_n * 8 + group;
        let b0 = *b_smem.add((b_col * 8 + lane) as usize);
        let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

        // Dump fragment values for warp 0
        if warp_id == 0 {
            *debug_buf.add(local_tid as usize) = a0;
            *debug_buf.add((32 + local_tid) as usize) = a1;
            *debug_buf.add((64 + local_tid) as usize) = a2;
            *debug_buf.add((96 + local_tid) as usize) = a3;
            *debug_buf.add((128 + local_tid) as usize) = b0;
            *debug_buf.add((160 + local_tid) as usize) = b1;
        }

        // Execute MMA
        let c0: u32 = 0;
        let c1: u32 = 0;
        let c2: u32 = 0;
        let c3: u32 = 0;
        let d0: u32;
        let d1: u32;
        let d2: u32;
        let d3: u32;
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );

        // Dump MMA output for warp 0
        if warp_id == 0 {
            *debug_buf.add((192 + local_tid) as usize) = d0;
            *debug_buf.add((224 + local_tid) as usize) = d1;
            *debug_buf.add((256 + local_tid) as usize) = d2;
            *debug_buf.add((288 + local_tid) as usize) = d3;
        }

        // Write output (same as full_gemm_f32in)
        let global_r0 = warp_m * 16 + group;
        let global_r2 = warp_m * 16 + group + 8;
        let global_c0 = warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(d0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(d1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(d2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(d3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, debug_buf, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// INT8 GEMM via dp4a — 4x INT8 dot product per instruction
// ============================================================

/// INT8 GEMM: C[M,N] = A_int8[M,K] × B_int8[K,N] using dp4a.
///
/// A is row-major [M, K/4] packed u32, B is column-major [N, K/4] packed u32.
/// Output C: [M, N] as i32 (INT32 accumulation).
///
/// Each thread computes one C[row, col] by accumulating K/4 dp4a operations.
///
/// Grid: (ceil(M*N / 256), 1, 1), Block: (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn int8_gemm_dp4a(
    a_packed: *const u32,
    b_packed: *const u32,
    c_out: *mut i32,
    m: u32,
    n: u32,
    k_div4: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = m * n;

        if global_id < total {
            let row = global_id / n;
            let col = global_id % n;

            let mut acc: u32 = 0;
            let a_row = a_packed.add((row * k_div4) as usize);
            let b_col = b_packed.add((col * k_div4) as usize);

            for k in 0..k_div4 {
                let a_val = *a_row.add(k as usize);
                let b_val = *b_col.add(k as usize);
                core::arch::asm!(
                    "dp4a.s32.s32 {out}, {a}, {b}, {c};",
                    out = out(reg32) acc,
                    a = in(reg32) a_val,
                    b = in(reg32) b_val,
                    c = in(reg32) acc,
                );
            }

            *c_out.add((row * n + col) as usize) = acc as i32;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_packed, b_packed, c_out, m, n, k_div4);
    }

    if tid == 0 {
        *status = 0;
    }
}

/// Dequantize INT32 GEMM result to f32.
///
/// out_f32[i] = c_int32[i] * scale_a * scale_b[col]
///
/// Grid: (ceil(n_total/256), 1, 1), Block: (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn int8_dequantize(
    c_int32: *const i32,
    out_f32: *mut f32,
    scale_a: f32,
    scale_b: *const f32,
    n_total: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n_total {
            let col = global_id % n_cols;
            let val = *c_int32.add(global_id as usize);
            let sb = *scale_b.add(col as usize);
            *out_f32.add(global_id as usize) = val as f32 * scale_a * sb;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (c_int32, out_f32, scale_a, scale_b, n_total, n_cols);
    }

    if tid == 0 {
        *status = 0;
    }
}

// ============================================================
// INT4 (W4A16) GEMM — 4-bit weights, f32 activations
// ============================================================

/// INT4 dequantize-on-the-fly GEMM: C[M,N] = A_f32[M,K] × dequant(W_int4[K,N]).
///
/// W_packed: [K/8, N] u32 (8 INT4 values per u32 along K, unsigned [0,15], zero_point=8).
/// scales: [K/group_size, N] f32.
///
/// Grid: (ceil(M*N / 256), 1, 1), Block: (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn int4_gemm_w4a16(
    a: *const f32,
    w_packed: *const u32,
    scales: *const f32,
    c_out: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = m * n;

        if global_id < total {
            let row = global_id / n;
            let col = global_id % n;
            let k_packed = k / 8;
            let mut acc: f32 = 0.0;

            for kp in 0..k_packed {
                let packed = *w_packed.add((kp * n + col) as usize);

                for bit in 0..8u32 {
                    let ki = kp * 8 + bit;
                    if ki >= k {
                        break;
                    }
                    let nibble = (packed >> (bit * 4)) & 0xF;
                    let w_signed = nibble as i32 - 8;
                    let group = ki / group_size;
                    let scale = *scales.add((group * n + col) as usize);
                    let w_f32 = w_signed as f32 * scale;
                    let a_val = *a.add((row * k + ki) as usize);

                    core::arch::asm!(
                        "fma.rn.f32 {d}, {a}, {b}, {c};",
                        d = out(reg32) acc,
                        a = in(reg32) a_val,
                        b = in(reg32) w_f32,
                        c = in(reg32) acc,
                    );
                }
            }

            *c_out.add((row * n + col) as usize) = acc;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, w_packed, scales, c_out, m, n, k, group_size);
    }

    if tid == 0 {
        *status = 0;
    }
}
