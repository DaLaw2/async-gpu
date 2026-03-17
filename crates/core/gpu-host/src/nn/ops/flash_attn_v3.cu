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

extern "C" __global__ void flash_attn_v3(
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

    // Shared memory: K[32][64] + V[32][64] = 16KB
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + BKV * D;

    // Load Q row into registers (all 4 threads per row load the same row)
    float q_reg[64];
    if (q_row_global < (int)seq_len) {
        #pragma unroll
        for (int d = 0; d < D; d++) {
            q_reg[d] = q_base[q_row_global * D + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < D; d++) q_reg[d] = 0.0f;
    }

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

        // Load K and V tiles cooperatively (128 threads, 2048 elements each)
        #pragma unroll
        for (int i = tid; i < BKV * D; i += 128) {
            int r = i / D;
            int c = i - r * D;
            int gr = kv_start + r;
            float kv = (gr < (int)seq_len) ? k_base[gr * D + c] : 0.0f;
            K_smem[i] = kv;
            float vv = (gr < (int)seq_len) ? v_base[gr * D + c] : 0.0f;
            V_smem[i] = vv;
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
                    const float* k_row = K_smem + (col_start + j) * D;
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

        // Reduce max across 4 threads in same row
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

        // Reduce sum across 4 threads
        float tile_sum = local_sum;
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 2);

        float correction = __expf(m_val - m_new);
        #pragma unroll
        for (int i = 0; i < 16; i++) o_reg[i] *= correction;

        l_val = l_val * correction + tile_sum;
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
                    const float* v_ptr = V_smem + v_row * D + o_col_start;

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

    // === PHASE 4: Write output ===
    if (q_row_global < (int)seq_len && l_val > 0.0f) {
        float inv_l = 1.0f / l_val;
        const int o_col_start = group * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            o_base[q_row_global * D + o_col_start + i] = o_reg[i] * inv_l;
        }
    }
}
