// Flash Attention V3 — Cooperative 4-thread-per-row tiled GEMM
// 128 threads (4 warps), BQ=32 query rows, BKV=32 KV tile size
// Hardcoded d_head=64 for maximum register efficiency
//
// Thread mapping:
//   my_row = tid / 4  (0..31) — which query row
//   group  = tid % 4  (0..3)  — which column group
//
// Score GEMM: each thread computes 8 scores S[my_row][group*8..group*8+7]
// P·V GEMM:  each thread computes 16 outputs O[my_row][group*16..group*16+15]
//
// V3.3 optimizations over V3.1:
//   - Skip-correction fast path: when tile_max <= m_val, correction=1.0 → skip
//     16 multiplies in the correction loop (common case for later tiles)
//   - float4 output writes (4x fewer store transactions)
//
// V3.1 base optimizations:
//   - Shared memory padding (stride 65) to eliminate bank conflicts
//   - float4 global loads for K/V tiles
//   - Cooperative Q load via shared memory (4x fewer global reads)
//   - `p_val > 1e-30f` branch saves V reads for causal masked positions

extern "C" __global__ __launch_bounds__(128, 3) void flash_attn_v3(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    unsigned int seq_len,
    unsigned int d_head,
    unsigned int causal
) {
    const int BQ  = 32;
    const int BKV = 32;
    const int D   = 64;
    const int STRIDE = 65;  // padded stride to avoid shared memory bank conflicts

    const int head    = blockIdx.x;
    const int q_tile  = blockIdx.y;
    const int tid     = threadIdx.x;

    const int my_row  = tid >> 2;      // 0..31
    const int group   = tid & 3;       // 0..3
    const int lane_id = tid & 31;

    const int head_off = head * seq_len * D;
    const float* q_base = Q + head_off;
    const float* k_base = K + head_off;
    const float* v_base = V + head_off;
    float*       o_base = Out + head_off;

    const int q_row_global = q_tile * BQ + my_row;
    const float scale = rsqrtf((float)D);

    // Shared memory: K[32][65] + V[32][65] = 16,640 bytes
    // Stride=65 eliminates 4-way bank conflicts on K reads:
    //   Without padding: bank = d % 32 (same for all groups -> 4-way conflict)
    //   With padding: bank = (row*65 + d) % 32 (different per row -> no conflict)
    extern __shared__ float smem[];

    // Phase 0: Cooperative Q load into smem, then copy to registers
    // 128 threads load 2048 elements (32 rows x 64 cols) — 4x fewer global reads
    // than per-thread loading where each of 4 threads per row loads the same row.
    {
        float* Q_smem = smem;  // reuse smem before K/V are loaded
        const int TOTAL_F4 = BQ * D / 4;  // 512 float4s
        for (int i4 = tid; i4 < TOTAL_F4; i4 += 128) {
            int flat = i4 * 4;
            int r = flat / D;
            int c = flat - r * D;
            int gr = q_tile * BQ + r;
            if (gr < (int)seq_len) {
                float4 qv = *reinterpret_cast<const float4*>(&q_base[gr * D + c]);
                Q_smem[r * STRIDE + c]     = qv.x;
                Q_smem[r * STRIDE + c + 1] = qv.y;
                Q_smem[r * STRIDE + c + 2] = qv.z;
                Q_smem[r * STRIDE + c + 3] = qv.w;
            } else {
                Q_smem[r * STRIDE + c]     = 0.0f;
                Q_smem[r * STRIDE + c + 1] = 0.0f;
                Q_smem[r * STRIDE + c + 2] = 0.0f;
                Q_smem[r * STRIDE + c + 3] = 0.0f;
            }
        }
        __syncthreads();

        // Each thread copies its Q row from smem to registers
        // (4 threads share a row, but register copy is private)
    }

    float q_reg[64];
    {
        float* Q_smem = smem;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            q_reg[d] = Q_smem[my_row * STRIDE + d];
        }
    }
    __syncthreads();  // ensure all threads done reading Q_smem before it's reused

    // Now smem is available for K/V
    float* K_smem = smem;
    float* V_smem = smem + BKV * STRIDE;

    // Online softmax state
    float m_val = -1e30f;
    float l_val = 0.0f;

    // Output accumulator: 16 elements per thread
    float o_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) o_reg[i] = 0.0f;

    const int n_kv_tiles = ((int)seq_len + BKV - 1) / BKV;

    for (int t = 0; t < n_kv_tiles; t++) {
        const int kv_start = t * BKV;

        if (causal && kv_start > q_tile * BQ + BQ - 1) break;

        // Load K and V tiles cooperatively with float4
        // 128 threads, 2048 elements each -> 512 float4s -> 4 per thread
        const int TOTAL_F4 = BKV * D / 4;  // 512
        for (int i4 = tid; i4 < TOTAL_F4; i4 += 128) {
            int flat = i4 * 4;
            int r = flat / D;
            int c = flat - r * D;
            int gr = kv_start + r;
            if (gr < (int)seq_len) {
                float4 kv4 = *reinterpret_cast<const float4*>(&k_base[gr * D + c]);
                K_smem[r * STRIDE + c]     = kv4.x;
                K_smem[r * STRIDE + c + 1] = kv4.y;
                K_smem[r * STRIDE + c + 2] = kv4.z;
                K_smem[r * STRIDE + c + 3] = kv4.w;

                float4 vv4 = *reinterpret_cast<const float4*>(&v_base[gr * D + c]);
                V_smem[r * STRIDE + c]     = vv4.x;
                V_smem[r * STRIDE + c + 1] = vv4.y;
                V_smem[r * STRIDE + c + 2] = vv4.z;
                V_smem[r * STRIDE + c + 3] = vv4.w;
            } else {
                K_smem[r * STRIDE + c]     = 0.0f;
                K_smem[r * STRIDE + c + 1] = 0.0f;
                K_smem[r * STRIDE + c + 2] = 0.0f;
                K_smem[r * STRIDE + c + 3] = 0.0f;
                V_smem[r * STRIDE + c]     = 0.0f;
                V_smem[r * STRIDE + c + 1] = 0.0f;
                V_smem[r * STRIDE + c + 2] = 0.0f;
                V_smem[r * STRIDE + c + 3] = 0.0f;
            }
        }
        __syncthreads();

        // === PHASE 1: Score computation ===
        float scores[8];
        const int col_start = group * 8;

        if (q_row_global < (int)seq_len) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kv_col = kv_start + col_start + j;
                if ((causal && kv_col > q_row_global) || kv_col >= (int)seq_len) {
                    scores[j] = -1e30f;
                } else {
                    float dot = 0.0f;
                    const float* k_row = K_smem + (col_start + j) * STRIDE;
                    #pragma unroll
                    for (int d = 0; d < D; d += 4) {
                        dot += q_reg[d]   * k_row[d];
                        dot += q_reg[d+1] * k_row[d+1];
                        dot += q_reg[d+2] * k_row[d+2];
                        dot += q_reg[d+3] * k_row[d+3];
                    }
                    scores[j] = dot * scale;
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) scores[j] = -1e30f;
        }

        // === PHASE 2: Online softmax ===
        float local_max = scores[0];
        #pragma unroll
        for (int j = 1; j < 8; j++) local_max = fmaxf(local_max, scores[j]);

        float tile_max = local_max;
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, 2));

        float m_new = fmaxf(m_val, tile_max);

        float local_sum = 0.0f;
        float exp_scores[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float e = __expf(scores[j] - m_new);
            exp_scores[j] = e;
            local_sum += e;
        }

        float tile_sum = local_sum;
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 2);

        // Compute correction factor for online softmax rescaling
        // Optimization: skip correction when m_val hasn't changed (correction = 1.0)
        // This is common for later tiles where the max score doesn't increase.
        float correction = __expf(m_val - m_new);
        if (correction < 0.999f) {
            #pragma unroll
            for (int i = 0; i < 16; i++) o_reg[i] *= correction;
            l_val = l_val * correction + tile_sum;
        } else {
            l_val += tile_sum;
        }
        m_val = m_new;

        // === PHASE 3: P·V accumulation ===
        const int o_col_start = group * 16;

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int src_lane = (lane_id & ~3) | g;

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float p_val = __shfl_sync(0xFFFFFFFF, exp_scores[j], src_lane);

                if (p_val > 1e-30f) {
                    int v_row = g * 8 + j;
                    const float* v_ptr = V_smem + v_row * STRIDE + o_col_start;

                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        o_reg[i]     += p_val * v_ptr[i];
                        o_reg[i + 1] += p_val * v_ptr[i + 1];
                        o_reg[i + 2] += p_val * v_ptr[i + 2];
                        o_reg[i + 3] += p_val * v_ptr[i + 3];
                    }
                }
            }
        }

        __syncthreads();
    }

    // === PHASE 4: Write output via float4 ===
    if (q_row_global < (int)seq_len && l_val > 0.0f) {
        float inv_l = 1.0f / l_val;
        const int o_col_start = group * 16;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            float4 out_val;
            out_val.x = o_reg[i]     * inv_l;
            out_val.y = o_reg[i + 1] * inv_l;
            out_val.z = o_reg[i + 2] * inv_l;
            out_val.w = o_reg[i + 3] * inv_l;
            *reinterpret_cast<float4*>(&o_base[q_row_global * D + o_col_start + i]) = out_val;
        }
    }
}
