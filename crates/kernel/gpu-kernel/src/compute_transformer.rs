// Transformer building blocks: LayerNorm, GELU, attention, flash attention,
// embedding lookup, bias add, elementwise add, QKV split, concat heads,
// f32-to-f16x2 pack, zero pad.

use crate::helpers::{bar_sync, get_dynamic_smem_ptr, gpu_exp_f32, gpu_sqrtf};
use core::arch::nvptx;

// ============================================================
// LayerNorm kernel (transformer-layer.1)
// ============================================================

/// Warp-level butterfly reduction: sum across all 32 lanes.
#[inline(always)]
unsafe fn warp_reduce_sum_f32(mut val: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: f32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            val += other;
            offset /= 2;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {}
    val
}

/// LayerNorm: y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// grid_dim = (num_rows, 1, 1), block_dim = (32, 1, 1).
/// Each block (1 warp) processes one row of d_model elements.
/// Input/output: f32 [num_rows][d_model].
/// gamma, beta: f32 [d_model].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_norm(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    d_model: u32,
    eps: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let row = nvptx::_block_idx_x() as u32;
        let row_ptr = input.add((row * d_model) as usize);
        let out_ptr = output.add((row * d_model) as usize);
        let elems_per_thread = d_model / 32;

        // Phase 1: Compute mean — each thread sums its elements
        let mut local_sum: f32 = 0.0;
        let mut i = 0u32;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            local_sum += *row_ptr.add(idx as usize);
            i += 1;
        }
        let total_sum = warp_reduce_sum_f32(local_sum);
        let mean = total_sum / d_model as f32;

        // Phase 2: Compute variance — each thread sums squared deviations
        let mut local_var: f32 = 0.0;
        i = 0;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            let diff = *row_ptr.add(idx as usize) - mean;
            local_var += diff * diff;
            i += 1;
        }
        let total_var = warp_reduce_sum_f32(local_var);
        let var = total_var / d_model as f32;
        let inv_std = 1.0 / gpu_sqrtf(var + eps);

        // Phase 3: Normalize and apply affine
        i = 0;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            let x = *row_ptr.add(idx as usize);
            let g = *gamma.add(idx as usize);
            let b = *beta.add(idx as usize);
            *out_ptr.add(idx as usize) = g * (x - mean) * inv_std + b;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, gamma, beta, d_model, eps);
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
// High-performance LayerNorm v2 — single-pass Welford + 256 threads
// ============================================================

/// High-performance LayerNorm: single-pass Welford's algorithm, 256 threads.
///
/// Uses Welford's online algorithm to compute mean and variance in a single pass,
/// then normalizes and applies affine transform.
///
/// grid_dim = (num_rows, 1, 1), block_dim = (256, 1, 1).
/// shared_mem_bytes = 256 * 2 * 4 = 2048 (for partial sums).
///
/// For GPT-2: d_model=768, each of 256 threads handles 3 elements.
/// Phase 1: Single-pass mean+M2 via Welford, then warp+block reduction in smem.
/// Phase 2: Normalize + affine in single pass.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_norm_v2(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    d_model: u32,
    eps: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let row = nvptx::_block_idx_x() as u32;
        let row_ptr = input.add((row * d_model) as usize);
        let out_ptr = output.add((row * d_model) as usize);
        let smem = get_dynamic_smem_ptr() as *mut f32;

        // Phase 1: Single-pass — compute sum and sum of squares
        // Each thread accumulates partial sums over its assigned elements
        let mut local_sum: f32 = 0.0;
        let mut local_sq_sum: f32 = 0.0;

        // Strided access: thread i reads elements i, i+256, i+512, ...
        // This gives coalesced memory access (consecutive threads read consecutive elements)
        let mut idx = tid;
        while idx < d_model {
            let x = *row_ptr.add(idx as usize);
            local_sum += x;
            local_sq_sum += x * x;
            idx += 256;
        }

        // Warp-level reduction for sum and sq_sum
        let warp_sum = warp_reduce_sum_f32(local_sum);
        let warp_sq_sum = warp_reduce_sum_f32(local_sq_sum);

        // Block-level reduction via shared memory
        let warp_id = tid / 32;
        let lane_id = tid % 32;
        // Store warp results (lane 0 of each warp writes to smem)
        if lane_id == 0 {
            *smem.add(warp_id as usize) = warp_sum; // smem[0..7] = sums
            *smem.add((warp_id + 8) as usize) = warp_sq_sum; // smem[8..15] = sq_sums
        }
        bar_sync();

        // Thread 0 reduces across warps (serial — only 8 values)
        if tid == 0 {
            let mut total_sum: f32 = 0.0;
            let mut total_sq_sum: f32 = 0.0;
            let mut w: u32 = 0;
            while w < 8 {
                total_sum += *smem.add(w as usize);
                total_sq_sum += *smem.add((w + 8) as usize);
                w += 1;
            }
            let m = total_sum / d_model as f32;
            let var = total_sq_sum / d_model as f32 - m * m;
            *smem.add(16) = m;
            *smem.add(17) = 1.0 / gpu_sqrtf(var + eps);
        }
        bar_sync();
        let mean = *smem.add(16);
        let inv_std = *smem.add(17);

        // Phase 2: Normalize + affine (single pass, coalesced access)
        idx = tid;
        while idx < d_model {
            let x = *row_ptr.add(idx as usize);
            let g = *gamma.add(idx as usize);
            let b = *beta.add(idx as usize);
            *out_ptr.add(idx as usize) = g * (x - mean) * inv_std + b;
            idx += 256;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, gamma, beta, d_model, eps);
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
// GELU kernel (transformer-layer.2)
// ============================================================

/// GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
/// Each thread processes one element.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gelu_forward(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
            let sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/pi)
            let coeff: f32 = 0.044715;
            let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            // tanh via exp: tanh(z) = 1 - 2/(exp(2z)+1)
            // Clamp to prevent exp overflow: tanh(10) = 1.0 in f32
            let tanh_val = if inner > 10.0 {
                1.0f32
            } else if inner < -10.0 {
                -1.0f32
            } else {
                let exp_2z = gpu_exp_f32(2.0 * inner);
                (exp_2z - 1.0) / (exp_2z + 1.0)
            };
            let result = x * 0.5 * (1.0 + tanh_val);
            *output.add(global_id as usize) = result;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
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
// Scaled Dot-Product Attention kernel (transformer-layer.3)
// ============================================================

/// Single-head scaled dot-product attention (f32, small seq):
/// output[seq][d_head] = softmax(Q[seq][d_head] x K^T[d_head][seq] / sqrt(d_head)) x V[seq][d_head]
///
/// grid_dim = (n_heads, 1, 1), block_dim = (32, 1, 1).
/// Each block (1 warp) processes one attention head.
/// Q, K, V are laid out as [n_heads][seq_len][d_head] f32 (already projected & split).
/// Output: [n_heads][seq_len][d_head] f32.
/// Constraint: seq_len <= 32 (one thread per query position).
/// causal_mask: 0 = no mask (bidirectional), 1 = causal (mask future positions).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_head(
    q_global: *const f32, // [n_heads][seq_len][d_head]
    k_global: *const f32, // [n_heads][seq_len][d_head]
    v_global: *const f32, // [n_heads][seq_len][d_head]
    out_global: *mut f32, // [n_heads][seq_len][d_head]
    seq_len: u32,
    d_head: u32,
    causal_mask: u32, // 0 = no mask, 1 = causal
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let head_offset = head * seq_len * d_head;

        if tid >= seq_len {
            // Still participate in warp-level operations, but skip writes.
            // For simplicity, just return — seq_len should equal 32 or be < 32.
            return;
        }

        let q_head = q_global.add(head_offset as usize);
        let k_head = k_global.add(head_offset as usize);
        let v_head = v_global.add(head_offset as usize);
        let out_head = out_global.add(head_offset as usize);

        // This thread handles query row `tid` (one row of seq_len).
        // Step 1: Compute attention scores = Q[tid] . K[j]^T / sqrt(d_head)
        //         for j = 0..seq_len
        let scale = 1.0 / gpu_sqrtf(d_head as f32);
        let smem = get_dynamic_smem_ptr() as *mut f32;
        // smem layout: [seq_len * seq_len] for scores
        // With seq=32, that's 32*32 = 1024 f32 = 4KB — fits.

        // Compute scores for this query row
        let scores = smem.add((tid * seq_len) as usize); // my row in score matrix
        let mut j = 0u32;
        while j < seq_len {
            // Apply causal mask: positions j > tid get -inf (will become 0 after softmax)
            if causal_mask != 0 && j > tid {
                // Use a large negative value instead of -inf to avoid NaN in exp
                *scores.add(j as usize) = -1.0e38_f32;
            } else {
                let mut dot: f32 = 0.0;
                let mut d = 0u32;
                while d < d_head {
                    dot += *q_head.add((tid * d_head + d) as usize)
                        * *k_head.add((j * d_head + d) as usize);
                    d += 1;
                }
                *scores.add(j as usize) = dot * scale;
            }
            j += 1;
        }

        // Step 2: Softmax over scores[tid][0..seq_len]
        // Find max (only over non-masked positions, but -1e38 won't be max)
        let mut max_val: f32 = *scores.add(0);
        j = 1;
        while j < seq_len {
            let v = *scores.add(j as usize);
            if v > max_val {
                max_val = v;
            }
            j += 1;
        }
        // Exp and sum
        let mut sum_exp: f32 = 0.0;
        j = 0;
        while j < seq_len {
            let e = gpu_exp_f32(*scores.add(j as usize) - max_val);
            *scores.add(j as usize) = e;
            sum_exp += e;
            j += 1;
        }
        // Normalize
        let inv_sum = 1.0 / sum_exp;
        j = 0;
        while j < seq_len {
            *scores.add(j as usize) = *scores.add(j as usize) * inv_sum;
            j += 1;
        }

        // Step 3: Output = attention_weights x V
        // out[tid][d] = sum_j weights[tid][j] * V[j][d]
        let mut d = 0u32;
        while d < d_head {
            let mut acc: f32 = 0.0;
            j = 0;
            while j < seq_len {
                acc += *scores.add(j as usize) * *v_head.add((j * d_head + d) as usize);
                j += 1;
            }
            *out_head.add((tid * d_head + d) as usize) = acc;
            d += 1;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (
            q_global,
            k_global,
            v_global,
            out_global,
            seq_len,
            d_head,
            causal_mask,
        );
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
// FlashAttention kernel (attention-scale.3)
// ============================================================

/// FlashAttention forward pass — tiled attention for arbitrary seq_len.
///
/// Uses online softmax to avoid materializing the O(seq^2) score matrix.
/// Processes K/V in tiles of B_C=32 columns, streaming from global memory.
///
/// Layout: Q/K/V/out are [n_heads, seq_len, d_head] row-major f32.
/// d_head MUST be 64 (GPT-2 small).
///
/// grid_dim = (n_heads, ceil(seq_len/32), 1)
/// block_dim = (32, 1, 1)  — one warp
/// Shared memory: k_tile[32][64] + v_tile[32][64] = 16384 bytes
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_attention(
    q_global: *const f32,
    k_global: *const f32,
    v_global: *const f32,
    out_global: *mut f32,
    seq_len: u32,
    d_head: u32,
    causal_mask: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let q_tile_idx: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) q_tile_idx);

        let my_row = q_tile_idx * 32 + tid; // global Q row index

        // Head offset in global arrays
        let head_off = (head * seq_len * d_head) as usize;
        let q_head = q_global.add(head_off);
        let k_head = k_global.add(head_off);
        let v_head = v_global.add(head_off);
        let out_head = out_global.add(head_off);

        let scale = 1.0 / gpu_sqrtf(d_head as f32);

        // Shared memory: k_tile[32][64] + v_tile[32][64]
        let smem = get_dynamic_smem_ptr() as *mut f32;
        let k_tile = smem; // 32 * 64 = 2048 f32
        let v_tile = smem.add(2048); // 32 * 64 = 2048 f32

        // Load Q row into registers (thread-private, loaded once)
        let mut q_row: [f32; 64] = [0.0; 64];
        if my_row < seq_len {
            let mut d = 0u32;
            while d < d_head {
                q_row[d as usize] = *q_head.add((my_row * d_head + d) as usize);
                d += 1;
            }
        }

        // Initialize online softmax state
        let mut m: f32 = -1.0e38; // running max
        let mut l: f32 = 0.0; // running sum of exp(score - m)

        // Output accumulator (unnormalized): o_acc[d] = sum of P_ij * V_j[d]
        let mut o_acc: [f32; 64] = [0.0; 64];

        // Number of KV tiles
        let n_kv_tiles = (seq_len + 31) / 32;

        let mut t = 0u32;
        while t < n_kv_tiles {
            let kv_col_start = t * 32;

            // Early exit for causal: if entire tile is above diagonal for ALL rows in Q tile
            if causal_mask != 0 && kv_col_start > q_tile_idx * 32 + 31 {
                // All KV columns > all Q rows -> fully masked, skip
                // Remaining tiles are even further right, so break
                break;
            }

            let tile_size = if kv_col_start + 32 <= seq_len {
                32u32
            } else {
                seq_len - kv_col_start
            };

            // Cooperative load K tile: 32 threads load 32 rows of K, each thread loads 1 row
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *k_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *k_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            // Cooperative load V tile: same pattern
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *v_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *v_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            crate::helpers::bar_sync();

            if my_row < seq_len {
                // Compute scores for this row against tile columns
                // and perform online softmax update
                let mut tile_max: f32 = -1.0e38;
                let mut scores: [f32; 32] = [0.0; 32]; // temp scores for this tile

                let mut c = 0u32;
                while c < tile_size {
                    let kv_col = kv_col_start + c;

                    if causal_mask != 0 && kv_col > my_row {
                        // Masked position
                        scores[c as usize] = -1.0e38;
                    } else {
                        // Dot product: Q[my_row] . K[kv_col]
                        let mut dot: f32 = 0.0;
                        let mut d = 0u32;
                        while d < d_head {
                            dot += q_row[d as usize] * *k_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                        let s = dot * scale;
                        scores[c as usize] = s;
                        if s > tile_max {
                            tile_max = s;
                        }
                    }
                    c += 1;
                }

                // Pad remaining scores (if tile_size < 32) with -inf
                c = tile_size;
                while c < 32 {
                    scores[c as usize] = -1.0e38;
                    c += 1;
                }

                // Online softmax update
                let m_new = if tile_max > m { tile_max } else { m };

                // Compute exp(scores - m_new) and their sum
                let mut row_sum: f32 = 0.0;
                let mut exp_scores: [f32; 32] = [0.0; 32];
                c = 0;
                while c < tile_size {
                    let e = gpu_exp_f32(scores[c as usize] - m_new);
                    exp_scores[c as usize] = e;
                    row_sum += e;
                    c += 1;
                }

                // Correction factor for old accumulator
                let correction = gpu_exp_f32(m - m_new);

                // Rescale old output accumulator
                let mut d = 0u32;
                while d < d_head {
                    o_acc[d as usize] = o_acc[d as usize] * correction;
                    d += 1;
                }

                // Accumulate: o_acc += P_tile x V_tile (for this row)
                // o_acc[d] += sum_c exp_scores[c] * V_tile[c][d]
                c = 0;
                while c < tile_size {
                    let p = exp_scores[c as usize];
                    if p > 0.0 {
                        let mut d = 0u32;
                        while d < d_head {
                            o_acc[d as usize] += p * *v_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                    }
                    c += 1;
                }

                // Update running stats
                l = l * correction + row_sum;
                m = m_new;
            }

            crate::helpers::bar_sync();
            t += 1;
        }

        // Final normalization: o_acc /= l
        if my_row < seq_len && l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0u32;
            while d < d_head {
                *out_head.add((my_row * d_head + d) as usize) = o_acc[d as usize] * inv_l;
                d += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (
            q_global,
            k_global,
            v_global,
            out_global,
            seq_len,
            d_head,
            causal_mask,
        );
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

/// FlashAttention with separate Q length and KV length — for KV-cached generation.
///
// ============================================================
// Flash Attention V2 — register-blocked score computation
// ============================================================

/// Improved flash attention with register-blocked dot products.
///
/// Same FlashAttention-1 algorithm (online softmax with rescaling), but:
/// - 128 threads per block (4 warps) instead of 32
/// - Each thread computes 4 Q rows worth of scores per KV tile
/// - Score dot products use 4-way parallel accumulation
///
/// Q,K,V: [n_heads * seq_len, d_head] head-major layout.
/// grid = (n_heads, ceil(seq_len/32), 1), block = (32, 1, 1).
/// BUT: we keep 32 threads for now and optimize the inner loop.
///
/// Key optimization: process 4 K-rows per inner iteration to improve ILP.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_attention_v2(
    q_global: *const f32,
    k_global: *const f32,
    v_global: *const f32,
    out_global: *mut f32,
    seq_len: u32,
    d_head: u32,
    causal_mask: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let q_tile_idx: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) q_tile_idx);

        let my_row = q_tile_idx * 32 + tid;

        let head_off = (head * seq_len * d_head) as usize;
        let q_head = q_global.add(head_off);
        let k_head = k_global.add(head_off);
        let v_head = v_global.add(head_off);
        let out_head = out_global.add(head_off);

        let scale = 1.0 / gpu_sqrtf(d_head as f32);

        let smem = get_dynamic_smem_ptr() as *mut f32;
        let k_tile = smem;
        let v_tile = smem.add(2048);

        // Load Q row into registers
        let mut q_row: [f32; 64] = [0.0; 64];
        if my_row < seq_len {
            let mut d = 0u32;
            while d < d_head {
                q_row[d as usize] = *q_head.add((my_row * d_head + d) as usize);
                d += 1;
            }
        }

        let mut m: f32 = -1.0e38;
        let mut l: f32 = 0.0;
        let mut o_acc: [f32; 64] = [0.0; 64];

        let n_kv_tiles = (seq_len + 31) / 32;

        let mut t = 0u32;
        while t < n_kv_tiles {
            let kv_col_start = t * 32;

            if causal_mask != 0 && kv_col_start > q_tile_idx * 32 + 31 {
                break;
            }

            let tile_size = if kv_col_start + 32 <= seq_len {
                32u32
            } else {
                seq_len - kv_col_start
            };

            // Load K and V tiles cooperatively
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *k_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *k_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
                d = 0;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *v_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *v_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }
            bar_sync();

            if my_row < seq_len {
                // Compute scores and online softmax update
                let mut tile_max: f32 = -1.0e38;
                let mut scores: [f32; 32] = [0.0; 32];

                // Score computation: process 4 d-elements at a time for ILP
                let mut c = 0u32;
                while c < tile_size {
                    let kv_col = kv_col_start + c;
                    if causal_mask != 0 && kv_col > my_row {
                        scores[c as usize] = -1.0e38;
                    } else {
                        // Dot product with 4-way unrolling for ILP
                        let mut dot: f32 = 0.0;
                        let k_off = (c * d_head) as usize;
                        let mut d = 0u32;
                        while d + 3 < d_head {
                            let q0 = q_row[d as usize];
                            let q1 = q_row[(d + 1) as usize];
                            let q2 = q_row[(d + 2) as usize];
                            let q3 = q_row[(d + 3) as usize];
                            let k0 = *k_tile.add(k_off + d as usize);
                            let k1 = *k_tile.add(k_off + (d + 1) as usize);
                            let k2 = *k_tile.add(k_off + (d + 2) as usize);
                            let k3 = *k_tile.add(k_off + (d + 3) as usize);
                            // 4 independent FMAs — GPU can pipeline these
                            let mut d0: f32;
                            let mut d1: f32;
                            let mut d2: f32;
                            let mut d3: f32;
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) d0, a = in(reg32) q0, b = in(reg32) k0, c = in(reg32) dot);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) d1, a = in(reg32) q1, b = in(reg32) k1, c = in(reg32) 0.0f32);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) d2, a = in(reg32) q2, b = in(reg32) k2, c = in(reg32) 0.0f32);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) d3, a = in(reg32) q3, b = in(reg32) k3, c = in(reg32) 0.0f32);
                            dot = d0 + d1 + d2 + d3;
                            d += 4;
                        }
                        while d < d_head {
                            dot += q_row[d as usize] * *k_tile.add(k_off + d as usize);
                            d += 1;
                        }
                        let s = dot * scale;
                        scores[c as usize] = s;
                        if s > tile_max {
                            tile_max = s;
                        }
                    }
                    c += 1;
                }

                // Online softmax update
                let m_new = if tile_max > m { tile_max } else { m };

                let mut row_sum: f32 = 0.0;
                let mut exp_scores: [f32; 32] = [0.0; 32];
                c = 0;
                while c < tile_size {
                    let e = gpu_exp_f32(scores[c as usize] - m_new);
                    exp_scores[c as usize] = e;
                    row_sum += e;
                    c += 1;
                }

                let correction = gpu_exp_f32(m - m_new);

                // Rescale old output + accumulate new with 4-way unrolling
                let mut d = 0u32;
                while d + 3 < d_head {
                    o_acc[d as usize] = o_acc[d as usize] * correction;
                    o_acc[(d + 1) as usize] = o_acc[(d + 1) as usize] * correction;
                    o_acc[(d + 2) as usize] = o_acc[(d + 2) as usize] * correction;
                    o_acc[(d + 3) as usize] = o_acc[(d + 3) as usize] * correction;
                    d += 4;
                }
                while d < d_head {
                    o_acc[d as usize] *= correction;
                    d += 1;
                }

                // P × V accumulation
                c = 0;
                while c < tile_size {
                    let p = exp_scores[c as usize];
                    if p > 1.0e-30 {
                        let v_off = (c * d_head) as usize;
                        d = 0;
                        while d + 3 < d_head {
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) o_acc[d as usize], a = in(reg32) p, b = in(reg32) *v_tile.add(v_off + d as usize), c = in(reg32) o_acc[d as usize]);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) o_acc[(d+1) as usize], a = in(reg32) p, b = in(reg32) *v_tile.add(v_off + (d+1) as usize), c = in(reg32) o_acc[(d+1) as usize]);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) o_acc[(d+2) as usize], a = in(reg32) p, b = in(reg32) *v_tile.add(v_off + (d+2) as usize), c = in(reg32) o_acc[(d+2) as usize]);
                            core::arch::asm!("fma.rn.f32 {d}, {a}, {b}, {c};", d = out(reg32) o_acc[(d+3) as usize], a = in(reg32) p, b = in(reg32) *v_tile.add(v_off + (d+3) as usize), c = in(reg32) o_acc[(d+3) as usize]);
                            d += 4;
                        }
                        while d < d_head {
                            o_acc[d as usize] += p * *v_tile.add(v_off + d as usize);
                            d += 1;
                        }
                    }
                    c += 1;
                }

                l = l * correction + row_sum;
                m = m_new;
            }

            bar_sync();
            t += 1;
        }

        // Write output
        if my_row < seq_len && l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0u32;
            while d < d_head {
                *out_head.add((my_row * d_head + d) as usize) = o_acc[d as usize] * inv_l;
                d += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (q_global, k_global, v_global, out_global, seq_len, d_head, causal_mask);
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

/// Identical algorithm to `flash_attention`, but Q and K/V can have different lengths.
/// Q layout: [n_heads, q_len, d_head] — typically q_len=1 for autoregressive generation
/// K/V layout: [n_heads, kv_len, d_head] — the full KV cache (all previous + current token)
/// Out layout: [n_heads, q_len, d_head]
///
/// grid_dim = (n_heads, ceil(q_len/32), 1)
/// block_dim = (32, 1, 1)
/// Shared memory: k_tile[32][64] + v_tile[32][64] = 16384 bytes
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_attention_kv(
    q_global: *const f32,
    k_global: *const f32,
    v_global: *const f32,
    out_global: *mut f32,
    q_len: u32,
    kv_len: u32,
    d_head: u32,
    causal_mask: u32,
    // For causal masking with KV cache: the Q row offset in the full sequence.
    // e.g., if we've cached 10 tokens and are generating token 11, q_offset=10
    // so that Q row 0 maps to global position 10 for masking purposes.
    q_offset: u32,
    // Stride for K/V head offsets. When K/V come from a pre-allocated cache with
    // max_seq slots, kv_stride = max_seq even though kv_len < max_seq.
    // Set kv_stride = kv_len when K/V are packed contiguously.
    kv_stride: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let q_tile_idx: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) q_tile_idx);

        let my_row = q_tile_idx * 32 + tid; // local Q row index (0..q_len-1)
        let my_global_row = my_row + q_offset; // global position for causal masking

        // Head offsets: Q uses q_len stride, K/V use kv_stride stride
        let q_head = q_global.add((head * q_len * d_head) as usize);
        let k_head = k_global.add((head * kv_stride * d_head) as usize);
        let v_head = v_global.add((head * kv_stride * d_head) as usize);
        let out_head = out_global.add((head * q_len * d_head) as usize);

        let scale = 1.0 / gpu_sqrtf(d_head as f32);

        let smem = get_dynamic_smem_ptr() as *mut f32;
        let k_tile = smem;
        let v_tile = smem.add(2048);

        // Load Q row into registers
        let mut q_row: [f32; 64] = [0.0; 64];
        if my_row < q_len {
            let mut d = 0u32;
            while d < d_head {
                q_row[d as usize] = *q_head.add((my_row * d_head + d) as usize);
                d += 1;
            }
        }

        let mut m: f32 = -1.0e38;
        let mut l: f32 = 0.0;
        let mut o_acc: [f32; 64] = [0.0; 64];

        let n_kv_tiles = (kv_len + 31) / 32;

        let mut t = 0u32;
        while t < n_kv_tiles {
            let kv_col_start = t * 32;

            // Causal early exit: if entire KV tile is after all Q positions
            if causal_mask != 0 && kv_col_start > q_offset + q_tile_idx * 32 + 31 {
                break;
            }

            let tile_size = if kv_col_start + 32 <= kv_len {
                32u32
            } else {
                kv_len - kv_col_start
            };

            // Cooperative load K tile
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < kv_len {
                        *k_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *k_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            // Cooperative load V tile
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < kv_len {
                        *v_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *v_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            crate::helpers::bar_sync();

            if my_row < q_len {
                let mut tile_max: f32 = -1.0e38;
                let mut scores: [f32; 32] = [0.0; 32];

                let mut c = 0u32;
                while c < tile_size {
                    let kv_col = kv_col_start + c;

                    if causal_mask != 0 && kv_col > my_global_row {
                        scores[c as usize] = -1.0e38;
                    } else {
                        let mut dot: f32 = 0.0;
                        let mut d = 0u32;
                        while d < d_head {
                            dot += q_row[d as usize] * *k_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                        let s = dot * scale;
                        scores[c as usize] = s;
                        if s > tile_max {
                            tile_max = s;
                        }
                    }
                    c += 1;
                }

                c = tile_size;
                while c < 32 {
                    scores[c as usize] = -1.0e38;
                    c += 1;
                }

                let m_new = if tile_max > m { tile_max } else { m };

                let mut row_sum: f32 = 0.0;
                let mut exp_scores: [f32; 32] = [0.0; 32];
                c = 0;
                while c < tile_size {
                    let e = gpu_exp_f32(scores[c as usize] - m_new);
                    exp_scores[c as usize] = e;
                    row_sum += e;
                    c += 1;
                }

                let correction = gpu_exp_f32(m - m_new);

                let mut d = 0u32;
                while d < d_head {
                    o_acc[d as usize] = o_acc[d as usize] * correction;
                    d += 1;
                }

                c = 0;
                while c < tile_size {
                    let p = exp_scores[c as usize];
                    if p > 0.0 {
                        let mut d = 0u32;
                        while d < d_head {
                            o_acc[d as usize] += p * *v_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                    }
                    c += 1;
                }

                l = l * correction + row_sum;
                m = m_new;
            }

            crate::helpers::bar_sync();
            t += 1;
        }

        // Final normalization
        if my_row < q_len && l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0u32;
            while d < d_head {
                *out_head.add((my_row * d_head + d) as usize) = o_acc[d as usize] * inv_l;
                d += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (
            q_global, k_global, v_global, out_global, q_len, kv_len, d_head, causal_mask,
            q_offset, kv_stride,
        );
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
// Token + positional embedding kernel (full-inference.1)
// ============================================================

/// Embedding lookup + addition: out[pos, d] = wte[token_ids[pos], d] + wpe[pos, d]
///
/// wte: [vocab_size, d_model] f32 (token embedding table)
/// wpe: [max_seq, d_model] f32 (positional embedding table)
/// token_ids: [seq_len] u32
/// out: [seq_len, d_model] f32
///
/// grid_dim = (ceil(seq_len * d_model / 256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embedding_lookup(
    wte: *const f32,
    wpe: *const f32,
    token_ids: *const u32,
    out: *mut f32,
    seq_len: u32,
    d_model: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = seq_len * d_model;

        if global_id < total {
            let pos = global_id / d_model;
            let d = global_id % d_model;

            let token_id = *token_ids.add(pos as usize);
            let tok_emb = *wte.add((token_id * d_model + d) as usize);
            let pos_emb = *wpe.add((pos * d_model + d) as usize);

            *out.add(global_id as usize) = tok_emb + pos_emb;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (wte, wpe, token_ids, out, seq_len, d_model);
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
// Bias-add kernel (transformer-layer.4 helper)
// ============================================================

/// Add bias to a 2D matrix in-place: data[i][j] += bias[j]
///
/// grid_dim = (ceil(total/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bias_add(
    data: *mut f32,
    bias: *const f32,
    n_cols: u32,
    total: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < total {
            let col = global_id % n_cols;
            let val = *data.add(global_id as usize) + *bias.add(col as usize);
            *data.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (data, bias, n_cols, total);
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
// Elementwise add kernel (residual connection helper)
// ============================================================

/// a[i] += b[i] (in-place residual add)
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn elementwise_add(a: *mut f32, b: *const f32, n: u32) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        if global_id < n {
            let val = *a.add(global_id as usize) + *b.add(global_id as usize);
            *a.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, n);
    }
}

// ============================================================
// Vectorized elementwise_add V2 — 4 elements per thread, coalesced
// ============================================================

/// Vectorized elementwise add: a[i] += b[i], 4 elements per thread.
///
/// grid_dim = (ceil(n/1024), 1, 1), block_dim = (256, 1, 1).
/// Each thread handles 4 consecutive elements for better memory coalescing.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn elementwise_add_v2(
    a: *mut f32,
    b: *const f32,
    n: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let base = (block_x * 256 + tid) * 4;

        if base + 3 < n {
            // Fast path: all 4 elements in bounds
            let a0 = *a.add(base as usize);
            let a1 = *a.add((base + 1) as usize);
            let a2 = *a.add((base + 2) as usize);
            let a3 = *a.add((base + 3) as usize);
            let b0 = *b.add(base as usize);
            let b1 = *b.add((base + 1) as usize);
            let b2 = *b.add((base + 2) as usize);
            let b3 = *b.add((base + 3) as usize);
            *a.add(base as usize) = a0 + b0;
            *a.add((base + 1) as usize) = a1 + b1;
            *a.add((base + 2) as usize) = a2 + b2;
            *a.add((base + 3) as usize) = a3 + b3;
        } else {
            // Tail: check each element
            let mut i = 0u32;
            while i < 4 {
                let idx = base + i;
                if idx < n {
                    *a.add(idx as usize) = *a.add(idx as usize) + *b.add(idx as usize);
                }
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, n);
    }
}

// ============================================================
// Vectorized GELU V2 — fast sigmoid approximation, 4 elements per thread
// ============================================================

/// Fast GELU: y = x * sigmoid(1.702 * x), 4 elements per thread.
///
/// Uses the SiLU-like approximation instead of tanh formula.
/// grid_dim = (ceil(n/1024), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gelu_forward_v2(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let base = (block_x * 256 + tid) * 4;

        // Fast GELU: x * sigmoid(1.702 * x) where sigmoid(z) = 1/(1+exp(-z))
        let coeff: f32 = 1.702;

        if base + 3 < n {
            let x0 = *input.add(base as usize);
            let x1 = *input.add((base + 1) as usize);
            let x2 = *input.add((base + 2) as usize);
            let x3 = *input.add((base + 3) as usize);

            // sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
            let s0 = 1.0 / (1.0 + gpu_exp_f32(-coeff * x0));
            let s1 = 1.0 / (1.0 + gpu_exp_f32(-coeff * x1));
            let s2 = 1.0 / (1.0 + gpu_exp_f32(-coeff * x2));
            let s3 = 1.0 / (1.0 + gpu_exp_f32(-coeff * x3));

            *output.add(base as usize) = x0 * s0;
            *output.add((base + 1) as usize) = x1 * s1;
            *output.add((base + 2) as usize) = x2 * s2;
            *output.add((base + 3) as usize) = x3 * s3;
        } else {
            let mut i = 0u32;
            while i < 4 {
                let idx = base + i;
                if idx < n {
                    let x = *input.add(idx as usize);
                    let s = 1.0 / (1.0 + gpu_exp_f32(-coeff * x));
                    *output.add(idx as usize) = x * s;
                }
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
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
// Vectorized elementwise_add V3 — explicit PTX float4 loads
// ============================================================

/// Vectorized elementwise add with PTX float4 loads: a[i] += b[i].
///
/// Uses ld.global.v4.f32 for 128-bit coalesced loads (4 elements per instruction).
/// Each thread processes 4 elements. n MUST be divisible by 4.
///
/// grid_dim = (ceil(n/1024), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn elementwise_add_v3(
    a: *mut f32,
    b: *const f32,
    n: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let base = (block_x * 256 + tid) * 4;

        if base + 3 < n {
            // Float4 load from a
            let a_ptr = a.add(base as usize) as *const u64;
            let b_ptr = b.add(base as usize) as *const u64;

            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;

            core::arch::asm!(
                "ld.global.v4.f32 {{{a0}, {a1}, {a2}, {a3}}}, [{addr}];",
                a0 = out(reg32) a0, a1 = out(reg32) a1,
                a2 = out(reg32) a2, a3 = out(reg32) a3,
                addr = in(reg64) a.add(base as usize),
            );
            core::arch::asm!(
                "ld.global.v4.f32 {{{b0}, {b1}, {b2}, {b3}}}, [{addr}];",
                b0 = out(reg32) b0, b1 = out(reg32) b1,
                b2 = out(reg32) b2, b3 = out(reg32) b3,
                addr = in(reg64) b.add(base as usize),
            );

            let r0 = a0 + b0;
            let r1 = a1 + b1;
            let r2 = a2 + b2;
            let r3 = a3 + b3;

            core::arch::asm!(
                "st.global.v4.f32 [{addr}], {{{r0}, {r1}, {r2}, {r3}}};",
                addr = in(reg64) a.add(base as usize),
                r0 = in(reg32) r0, r1 = in(reg32) r1,
                r2 = in(reg32) r2, r3 = in(reg32) r3,
            );
        } else {
            // Tail: scalar fallback
            let mut i = 0u32;
            while i < 4 {
                let idx = base + i;
                if idx < n {
                    *a.add(idx as usize) = *a.add(idx as usize) + *b.add(idx as usize);
                }
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, n);
    }
}

// ============================================================
// QKV split + transpose kernel (transformer-layer.6 helper)
// ============================================================

/// Split QKV from [seq, 3*d_model] into Q, K, V as [n_heads][seq][d_head].
///
/// input: [seq_len, 3*d_model] f32 (row-major)
/// q/k/v_out: [n_heads, seq_len, d_head] f32 (head-major)
///
/// grid_dim = (ceil(total_out/256), 1, 1), block_dim = (256, 1, 1).
/// total_out = n_heads * seq_len * d_head (per output tensor).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn split_qkv(
    input: *const f32, // [seq_len, 3*d_model]
    q_out: *mut f32,   // [n_heads, seq_len, d_head]
    k_out: *mut f32,   // [n_heads, seq_len, d_head]
    v_out: *mut f32,   // [n_heads, seq_len, d_head]
    seq_len: u32,
    n_heads: u32,
    d_head: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = n_heads * seq_len * d_head;

        if global_id < total {
            // Decode [head][seq][d] indices
            let d = global_id % d_head;
            let seq = (global_id / d_head) % seq_len;
            let head = global_id / (d_head * seq_len);

            let d_model = n_heads * d_head;
            // Input layout: [seq, 3*d_model], Q at [0..d_model], K at [d_model..2*d_model], V at [2*d_model..3*d_model]
            let base = seq * 3 * d_model + head * d_head + d;
            let q_val = *input.add(base as usize);
            let k_val = *input.add((base + d_model) as usize);
            let v_val = *input.add((base + 2 * d_model) as usize);

            let out_idx = global_id as usize;
            *q_out.add(out_idx) = q_val;
            *k_out.add(out_idx) = k_val;
            *v_out.add(out_idx) = v_val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, q_out, k_out, v_out, seq_len, n_heads, d_head);
    }
}

// ============================================================
// Attention concat kernel (transformer-layer.6 helper)
// ============================================================

/// Concat attention output from [n_heads][seq][d_head] -> [seq][d_model].
///
/// grid_dim = (ceil(total/256), 1, 1), block_dim = (256, 1, 1).
/// total = seq_len * d_model.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn concat_heads(
    input: *const f32, // [n_heads][seq_len][d_head]
    output: *mut f32,  // [seq_len][d_model]
    seq_len: u32,
    n_heads: u32,
    d_head: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let d_model = n_heads * d_head;
        let total = seq_len * d_model;

        if global_id < total {
            // Decode [seq][d_model] -> [seq][head][d]
            let d = global_id % d_head;
            let col = global_id % d_model;
            let seq = global_id / d_model;
            let head = col / d_head;

            // Input index: head * seq_len * d_head + seq * d_head + d
            let in_idx = head * seq_len * d_head + seq * d_head + d;
            *output.add(global_id as usize) = *input.add(in_idx as usize);
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, seq_len, n_heads, d_head);
    }
}

// ============================================================
// f32 -> f16x2 pack kernel (transformer-layer.4 helper)
// ============================================================

/// Pack f32 row-major [M][K] -> f16x2 row-major [M][K/2] u32.
/// K must be even. Each thread packs one pair.
///
/// grid_dim = (ceil(total_pairs/256), 1, 1), block_dim = (256, 1, 1).
/// total_pairs = M * K / 2.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_to_f16x2_pack(
    input: *const f32,
    output: *mut u32,
    total_pairs: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < total_pairs {
            let idx = global_id * 2;
            let v0 = *input.add(idx as usize);
            let v1 = *input.add((idx + 1) as usize);
            // f32 -> f16 via PTX cvt instruction
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
            *output.add(global_id as usize) = packed;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, total_pairs);
    }
}

// ============================================================
// Zero-pad kernel (full-inference.5.1)
// ============================================================

/// Zero out elements at index >= start_offset.
///
/// Used to clear padded rows in activation buffers, preventing NaN
/// propagation from padded positions through GEMM shared memory tiles.
///
/// grid_dim = (ceil(total_elems / 256), 1, 1), block_dim = (256, 1, 1).
/// start_offset: first element to zero (e.g., actual_seq * d_model).
// ============================================================
// KV cache append kernel (kv-cache.3)
// ============================================================

/// Copies a single token's K or V data from a padded source buffer into the KV cache.
///
/// Source layout: [n_heads, src_seq_stride, d_head] — we read row 0 of each head.
/// Cache layout: [n_heads, max_seq, d_head] — we write at `write_pos` of each head.
///
/// Launch with grid = (n_heads * d_head).div_ceil(256), block = 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn kv_cache_append(
    src: *const f32,
    cache: *mut f32,
    n_heads: u32,
    src_seq_stride: u32,
    max_seq: u32,
    d_head: u32,
    write_pos: u32,
    _status: *mut u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        let gid = nvptx::_block_idx_x() as u32 * 256 + nvptx::_thread_idx_x() as u32;
        let total = n_heads * d_head;
        if gid < total {
            let head = gid / d_head;
            let d = gid % d_head;
            let src_idx = head * src_seq_stride * d_head + d; // row 0 of each head
            let dst_idx = head * max_seq * d_head + write_pos * d_head + d;
            *cache.add(dst_idx as usize) = *src.add(src_idx as usize);
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (src, cache, n_heads, src_seq_stride, max_seq, d_head, write_pos, _status);
    }
}

/// total_elems: total buffer size.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn zero_pad(buffer: *mut f32, start_offset: u32, total_elems: u32) {
    #[cfg(target_arch = "nvptx64")]
    {
        let gid = nvptx::_block_idx_x() as u32 * 256 + nvptx::_thread_idx_x() as u32;
        let idx = start_offset + gid;
        if idx < total_elems {
            *buffer.add(idx as usize) = 0.0;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (buffer, start_offset, total_elems);
    }
}

// ============================================================
// Backward kernels for autograd
// ============================================================

/// GELU backward: d_input[i] = d_output[i] * gelu'(input[i]).
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gelu_backward(
    d_output: *const f32,
    input: *const f32,
    d_input: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            let dy = *d_output.add(global_id as usize);

            let sqrt_2_over_pi: f32 = 0.7978845608;
            let coeff: f32 = 0.044715;
            let z = sqrt_2_over_pi * (x + coeff * x * x * x);

            let tanh_z = if z > 10.0 {
                1.0f32
            } else if z < -10.0 {
                -1.0f32
            } else {
                let exp_2z = gpu_exp_f32(2.0 * z);
                (exp_2z - 1.0) / (exp_2z + 1.0)
            };

            let sech2_z = 1.0 - tanh_z * tanh_z;
            let dz_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
            let gelu_grad = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech2_z * dz_dx;

            *d_input.add(global_id as usize) = dy * gelu_grad;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (d_output, input, d_input, n);
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

/// SiLU backward: d_input[i] = d_output[i] * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))).
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn silu_backward(
    d_output: *const f32,
    input: *const f32,
    d_input: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            let dy = *d_output.add(global_id as usize);

            let sig = 1.0 / (1.0 + gpu_exp_f32(-x));
            let silu_grad = sig + x * sig * (1.0 - sig);
            *d_input.add(global_id as usize) = dy * silu_grad;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (d_output, input, d_input, n);
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

/// Sigmoid backward: d_input[i] = d_output[i] * sigmoid(x) * (1 - sigmoid(x)).
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sigmoid_backward(
    d_output: *const f32,
    input: *const f32,
    d_input: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            let dy = *d_output.add(global_id as usize);

            let sig = 1.0 / (1.0 + gpu_exp_f32(-x));
            *d_input.add(global_id as usize) = dy * sig * (1.0 - sig);
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (d_output, input, d_input, n);
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

/// ReLU backward: d_input[i] = d_output[i] * (input[i] > 0 ? 1 : 0).
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn relu_backward(
    d_output: *const f32,
    input: *const f32,
    d_input: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            let dy = *d_output.add(global_id as usize);
            let grad = if x > 0.0 { dy } else { 0.0 };
            *d_input.add(global_id as usize) = grad;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (d_output, input, d_input, n);
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

/// Bias add backward: d_bias[j] = sum_i(d_output[i][j]).
///
/// grid_dim = (ceil(n_cols/256), 1, 1), block_dim = (256, 1, 1).
/// Simple serial sum per column (n_rows is small for typical training).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bias_add_backward(
    d_output: *const f32,
    d_bias: *mut f32,
    n_cols: u32,
    n_rows: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let col = block_x * 256 + tid;

        if col < n_cols {
            let mut sum: f32 = 0.0;
            let mut row = 0u32;
            while row < n_rows {
                sum += *d_output.add((row * n_cols + col) as usize);
                row += 1;
            }
            *d_bias.add(col as usize) = sum;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (d_output, d_bias, n_cols, n_rows);
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
// Matrix transpose kernel — for matmul B column-major conversion
// ============================================================

/// Transpose a row-major matrix to column-major (or vice versa).
///
/// input [rows, cols] row-major → output [cols, rows] row-major
/// (equivalently: output[col * rows + row] = input[row * cols + col])
///
/// grid_dim = (ceil(total/256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrix_transpose(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = rows * cols;

        if global_id < total {
            let row = global_id / cols;
            let col = global_id % cols;
            let val = *input.add(global_id as usize);
            *output.add((col * rows + row) as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, rows, cols);
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

/// Zero-pad and copy: copy [rows, cols] into [rows_padded, cols_padded] with zeros.
///
/// grid_dim = (ceil(rows_padded * cols_padded / 256), 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrix_pad(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    rows_padded: u32,
    cols_padded: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total_padded = rows_padded * cols_padded;

        if global_id < total_padded {
            let r = global_id / cols_padded;
            let c = global_id % cols_padded;
            let val = if r < rows && c < cols {
                *input.add((r * cols + c) as usize)
            } else {
                0.0f32
            };
            *output.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, rows, cols, rows_padded, cols_padded);
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

/// Extract unpadded submatrix from a padded matrix on GPU.
///
/// Copies [rows, cols] from [rows_padded, cols_padded] (skipping padding).
/// grid_dim = (ceil(rows * cols / 256), 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrix_unpad(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    cols_padded: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = rows * cols;

        if global_id < total {
            let r = global_id / cols;
            let c = global_id % cols;
            *output.add(global_id as usize) = *input.add((r * cols_padded + c) as usize);
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, rows, cols, cols_padded);
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
// Optimizer kernels
// ============================================================

/// SGD step: param[i] -= lr * grad[i]
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sgd_step(
    param: *mut f32,
    grad: *const f32,
    lr: f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let gid = nvptx::_block_idx_x() as u32 * 256 + tid;
        if gid < n {
            let p = *param.add(gid as usize);
            let g = *grad.add(gid as usize);
            *param.add(gid as usize) = p - lr * g;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (param, grad, lr, n);
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
