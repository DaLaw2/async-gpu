// Fused GEMM + bias + activation kernels.
// Eliminates 2 kernel launches (bias_add + activation) per Linear layer.

use crate::helpers::{bar_sync, get_dynamic_smem_ptr, gpu_exp_f32};
use core::arch::nvptx;

/// Fused GEMM + bias + GELU.
///
/// D[i,j] = GELU(sum_k(A[i,k] * B[j,k]) + bias[j])
///
/// A: [M_pad, K_pad] row-major, B: [N_pad, K_pad] col-major (pre-transposed).
/// bias: [N_pad]. D: [M_pad, N_pad] row-major.
///
/// Same tiling as gemm_f32: 32×16 output tile, 128 threads, 16-wide K tiles.
/// Grid: (M_pad/32, N_pad/16, 1), Block: (128, 1, 1), Shared: (32*16 + 16*16)*4 bytes.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn gemm_bias_gelu(
    a_global: *const f32,
    b_global: *const f32,
    bias: *const f32,
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
        let a_smem = smem;
        let b_smem = smem.add(512);

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

        let a_block = a_global.add((block_m * 32 * k_dim) as usize);
        let b_block = b_global.add((block_n * 16 * k_dim) as usize);

        let mut acc00: f32 = 0.0;
        let mut acc01: f32 = 0.0;
        let mut acc10: f32 = 0.0;
        let mut acc11: f32 = 0.0;

        let k_tiles = (k_dim + 15) / 16;
        let mut t = 0u32;
        while t < k_tiles {
            let k_base = t * 16;

            // Load A tile [32][16] into shared memory
            // 128 threads, 512 elements = 4 per thread
            for i in 0..4u32 {
                let elem = tid * 4 + i;
                let row = elem / 16;
                let col = elem % 16;
                let g_row = block_m * 32 + row;
                let g_col = k_base + col;
                *a_smem.add(elem as usize) = *a_global.add((g_row * k_dim + g_col) as usize);
            }

            // Load B tile [16][16] into shared memory
            // 128 threads, 256 elements = 2 per thread
            for i in 0..2u32 {
                let elem = tid * 2 + i;
                let row = elem / 16;
                let col = elem % 16;
                let g_row = block_n * 16 + row;
                let g_col = k_base + col;
                *b_smem.add(elem as usize) = *b_global.add((g_row * k_dim + g_col) as usize);
            }

            bar_sync();

            // Compute 4 output elements
            let mut k = 0u32;
            while k < 16 {
                let a_r0 = *a_smem.add((r0 * 16 + k) as usize);
                let a_r1 = *a_smem.add((r1 * 16 + k) as usize);
                let b_c0 = *b_smem.add((c0 * 16 + k) as usize);
                let b_c1 = *b_smem.add((c1 * 16 + k) as usize);

                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc00, a = in(reg32) a_r0, b = in(reg32) b_c0, c = in(reg32) acc00,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc01, a = in(reg32) a_r0, b = in(reg32) b_c1, c = in(reg32) acc01,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc10, a = in(reg32) a_r1, b = in(reg32) b_c0, c = in(reg32) acc10,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc11, a = in(reg32) a_r1, b = in(reg32) b_c1, c = in(reg32) acc11,
                );

                k += 1;
            }

            bar_sync();
            t += 1;
        }

        // Fused: add bias + GELU
        let b0 = *bias.add(global_c0 as usize);
        let b1 = *bias.add(global_c1 as usize);
        acc00 += b0;
        acc01 += b1;
        acc10 += b0;
        acc11 += b1;

        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Simplified: use sigmoid approximation for speed
        // GELU(x) ≈ x * sigmoid(1.702 * x)
        acc00 = gelu_approx(acc00);
        acc01 = gelu_approx(acc01);
        acc10 = gelu_approx(acc10);
        acc11 = gelu_approx(acc11);

        // Write output
        *d_global.add((global_r0 * n_cols + global_c0) as usize) = acc00;
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = acc01;
        *d_global.add((global_r1 * n_cols + global_c0) as usize) = acc10;
        *d_global.add((global_r1 * n_cols + global_c1) as usize) = acc11;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, bias, d_global, k_dim, n_cols);
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

/// Fused GEMM + bias + ReLU.
///
/// Same structure as gemm_bias_gelu but with max(0, x) activation.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn gemm_bias_relu(
    a_global: *const f32,
    b_global: *const f32,
    bias: *const f32,
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
        let a_smem = smem;
        let b_smem = smem.add(512);

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

        let a_block = a_global.add((block_m * 32 * k_dim) as usize);
        let b_block = b_global.add((block_n * 16 * k_dim) as usize);

        let mut acc00: f32 = 0.0;
        let mut acc01: f32 = 0.0;
        let mut acc10: f32 = 0.0;
        let mut acc11: f32 = 0.0;

        let k_tiles = (k_dim + 15) / 16;
        let mut t = 0u32;
        while t < k_tiles {
            let k_base = t * 16;

            for i in 0..4u32 {
                let elem = tid * 4 + i;
                let row = elem / 16;
                let col = elem % 16;
                let g_row = block_m * 32 + row;
                let g_col = k_base + col;
                *a_smem.add(elem as usize) = *a_global.add((g_row * k_dim + g_col) as usize);
            }

            for i in 0..2u32 {
                let elem = tid * 2 + i;
                let row = elem / 16;
                let col = elem % 16;
                let g_row = block_n * 16 + row;
                let g_col = k_base + col;
                *b_smem.add(elem as usize) = *b_global.add((g_row * k_dim + g_col) as usize);
            }

            bar_sync();

            let mut k = 0u32;
            while k < 16 {
                let a_r0 = *a_smem.add((r0 * 16 + k) as usize);
                let a_r1 = *a_smem.add((r1 * 16 + k) as usize);
                let b_c0 = *b_smem.add((c0 * 16 + k) as usize);
                let b_c1 = *b_smem.add((c1 * 16 + k) as usize);

                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc00, a = in(reg32) a_r0, b = in(reg32) b_c0, c = in(reg32) acc00);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc01, a = in(reg32) a_r0, b = in(reg32) b_c1, c = in(reg32) acc01);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc10, a = in(reg32) a_r1, b = in(reg32) b_c0, c = in(reg32) acc10);
                core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc11, a = in(reg32) a_r1, b = in(reg32) b_c1, c = in(reg32) acc11);

                k += 1;
            }

            bar_sync();
            t += 1;
        }

        // Fused: add bias + ReLU
        let b0 = *bias.add(global_c0 as usize);
        let b1 = *bias.add(global_c1 as usize);
        acc00 = (acc00 + b0).max(0.0);
        acc01 = (acc01 + b1).max(0.0);
        acc10 = (acc10 + b0).max(0.0);
        acc11 = (acc11 + b1).max(0.0);

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = acc00;
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = acc01;
        *d_global.add((global_r1 * n_cols + global_c0) as usize) = acc10;
        *d_global.add((global_r1 * n_cols + global_c1) as usize) = acc11;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, bias, d_global, k_dim, n_cols);
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

/// GELU approximation: x * sigmoid(1.702 * x).
#[inline(always)]
unsafe fn gelu_approx(x: f32) -> f32 {
    let sx = 1.702 * x;
    let neg_sx = -sx;
    let exp_neg = gpu_exp_f32(neg_sx);
    let sigmoid = 1.0 / (1.0 + exp_neg);
    x * sigmoid
}
