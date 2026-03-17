# perf-attn v3.1: High-Performance Flash Attention CUDA C Kernel Design

**Task**: Design cooperative tiled-GEMM flash attention kernel targeting ≥70% cuDNN FA2.

## Architecture Summary

- **128 threads** (4 warps) per block, processing **BQ=32** query rows
- **BKV=32** KV rows per tile
- **d_head=64** (hardcoded for maximum register/smem efficiency)
- Thread layout: **8×16 grid** (thread_row = tid/16, thread_col = tid%16)
- Score GEMM tile: each thread owns a **4×2** sub-tile of S[32,32]
- P·V GEMM tile: each thread owns a **4×4** sub-tile of O[32,64]
- Inner K-dimension tiled by **TK=16**

## Shared Memory Layout

```
Total shared memory: (32*64 + 32*64 + 32*64) * 4 = 24,576 bytes
  Q_smem[32][65]  — 32 query rows × 64+1 cols (padded to avoid bank conflicts)
  K_smem[32][65]  — 32 KV rows × 64+1 cols
  V_smem[32][65]  — 32 KV rows × 64+1 cols
```

Note: S[32,32] is kept entirely in registers (each thread holds its 4×2 sub-tile).

## Thread-to-Output Mapping

### Score GEMM: S[32,32] = Q[32,64] × K^T[64,32]

Thread (tr, tc) where tr=tid/16, tc=tid%16:
- Owns S rows: [tr*4 .. tr*4+3]  (4 rows)
- Owns S cols: [tc*2 .. tc*2+1]  (2 cols)
- Sub-tile: s_reg[4][2] in registers

Inner loop over d_head in TK=16 chunks (4 iterations for d=64):
```
for tk = 0..4:
  Load Q_frag[4] from Q_smem[tr*4+i][tk*16 + lane]  // 4 rows, broadcast across cols
  Load K_frag[2] from K_smem[tc*2+j][tk*16 + lane]   // 2 rows of K = 2 cols of K^T
  // But we need full 16-element dot product...
```

Actually — for a 32×32×64 GEMM with 128 threads, a better approach is:
- Tile K-dim by TK=16: 4 outer iterations
- Each iteration: 128 threads load 32×16 Q sub-block and 32×16 K sub-block
- Then compute rank-16 update to S[32,32]

For the rank-16 update with 128 threads computing 1024 elements of S:
- Each thread computes 8 elements of S
- Layout: 128 threads as 32×4 (32 S-rows, 4 groups of 8 S-cols)
- Each thread: 1 row × 8 cols → but that's only 32*32=1024 with 32*8=256... no.

**Final chosen layout for score GEMM**:
- 16 rows × 8 cols thread grid
- Each thread: 2 S-rows × 4 S-cols = 8 elements
- 16*2=32, 8*4=32 ✓

### P·V GEMM: O[32,64] += diag(1/l) * P[32,32] × V[32,64]

Thread (tr, tc) where tr=tid/16, tc=tid%16:
- Owns O rows: [tr*2 .. tr*2+1]  (2 rows, matching score GEMM row ownership)
- Owns O cols: [tc*4 .. tc*4+3]  (4 cols)
- Sub-tile: o_reg[2][4] in registers — wait, 16*4=64 ✓, 16*2=32... but tr goes 0..7 (128/16=8), so 8*2=16 rows, not 32.

Let me reconsider. 128 threads, S has 32*32=1024 elements, O has 32*64=2048 elements.

**Revised thread layout**:
- For score GEMM S[32,32]: 1024 elements / 128 threads = 8 per thread
- For P·V GEMM O[32,64]: 2048 elements / 128 threads = 16 per thread

Best approach: **decouple the two GEMMs' thread mappings**.

For score: 128 threads → 4 threads per Q row (32*4=128), each thread computes 8 scores (32/4=8).
- Thread: q_row = tid/4, group = tid%4
- Scores: S[q_row][group*8 .. group*8+7]
- Inner K: loop d=0..63, each thread accumulates full dot product for its 8 scores

For P·V: 128 threads → 2 threads per Q row (32*2=64)... no, that's only 64 threads.
Better: 128 threads → 4 threads per output row, each computes 16 output cols.
- Thread: o_row = tid/4, group = tid%4
- Output: O[o_row][group*16 .. group*16+15]
- Inner K: loop over 32 KV positions

This makes thread mapping consistent between score and P·V: same (row, group) assignment!

## Complete Kernel Source

```cuda
// Flash Attention V3 — Cooperative Tiled GEMM
// 128 threads (4 warps), BQ=32 query rows, BKV=32 KV tile size
// Hardcoded d_head=64 for maximum register efficiency
//
// Thread mapping:
//   q_row = tid / 4     (0..31)  — which query row this thread is responsible for
//   group = tid % 4     (0..3)   — which column group
//
// Score GEMM: S[q_row][group*8..group*8+7] — each thread computes 8 scores
// P·V GEMM:  O[q_row][group*16..group*16+15] — each thread computes 16 output elements
//
// Both phases use the SAME row assignment, so online softmax state
// (per-row max and sum) naturally lives in the thread that owns the row.
// The 4 threads sharing a row coordinate via warp shuffles for reductions.

extern "C" __global__ void flash_attn_v3(
    const float* __restrict__ Q,   // [total_rows, 64]
    const float* __restrict__ K,   // [total_rows, 64]
    const float* __restrict__ V,   // [total_rows, 64]
    float* __restrict__ Out,       // [total_rows, 64]
    unsigned int seq_len,
    unsigned int d_head,           // must be 64
    unsigned int causal
) {
    // === Constants ===
    const int BQ  = 32;
    const int BKV = 32;
    const int D   = 64;

    // === Block/thread IDs ===
    const int head    = blockIdx.x;
    const int q_tile  = blockIdx.y;
    const int tid     = threadIdx.x;   // 0..127

    // Thread's row and column group
    const int my_row  = tid >> 2;      // tid / 4, range 0..31
    const int group   = tid & 3;       // tid % 4, range 0..3

    // Warp info for shuffles
    const int warp_id    = tid >> 5;   // tid / 32
    const int lane_id    = tid & 31;
    // The 4 threads for the same row within a warp have lane offsets 0,1,2,3
    // relative to (my_row % 8) * 4. Mask for the 4 threads sharing a row:
    // Within each group of 32 consecutive threads (one warp), threads
    // 0-3 share row 0 (of the warp's 8 rows), 4-7 share row 1, etc.
    // Shuffle mask for 4 threads: 0x3 (XOR with 1, 2, 3)

    // === Pointers ===
    const int head_off = head * seq_len * D;
    const float* q_base = Q + head_off;
    const float* k_base = K + head_off;
    const float* v_base = V + head_off;
    float*       o_base = Out + head_off;

    const int q_row_global = q_tile * BQ + my_row;
    const float scale = rsqrtf((float)D);

    // === Shared memory ===
    // Layout: K_smem[32][64] + V_smem[32][64]
    // Q stays in registers (each thread loads its own row)
    // Total: 2 * 32 * 64 * 4 = 16384 bytes
    extern __shared__ float smem[];
    float* K_smem = smem;              // [BKV][D] = 2048 floats
    float* V_smem = smem + BKV * D;    // [BKV][D] = 2048 floats

    // === Load Q row into registers ===
    // Each thread loads its own Q row (the row it's responsible for in output)
    // 64 values per row, 4 threads share the same row, so each loads all 64.
    // This is redundant but avoids shared memory for Q.
    float q_reg[D];
    if (q_row_global < (int)seq_len) {
        #pragma unroll
        for (int d = 0; d < D; d++) {
            q_reg[d] = q_base[q_row_global * D + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < D; d++) {
            q_reg[d] = 0.0f;
        }
    }

    // === Online softmax state (per thread, later reduced across group) ===
    float m_val = -1e30f;   // running max of scores for this row
    float l_val = 0.0f;     // running sum of exp(score - max)

    // === Output accumulator ===
    // Each thread accumulates 16 output columns: group*16 .. group*16+15
    float o_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) o_reg[i] = 0.0f;

    // === Main loop over KV tiles ===
    const int n_kv_tiles = ((int)seq_len + BKV - 1) / BKV;

    for (int t = 0; t < n_kv_tiles; t++) {
        const int kv_start = t * BKV;

        // Causal early exit: if all KV positions in this tile are after
        // all Q positions in this block, skip
        if (causal && kv_start > q_tile * BQ + BQ - 1) break;

        // === Cooperative load of K and V tiles ===
        // 128 threads load 32 rows × 64 cols = 2048 elements each for K and V
        // Each thread loads 2048/128 = 16 elements per matrix
        // Strategy: thread tid loads elements tid, tid+128, tid+256, ...
        // up to 2048
        #pragma unroll
        for (int i = tid; i < BKV * D; i += 128) {
            int r = i / D;       // row in tile (0..31)
            int c = i - r * D;   // col (0..63), avoid modulo
            int global_r = kv_start + r;
            float kval = (global_r < (int)seq_len) ? k_base[global_r * D + c] : 0.0f;
            K_smem[i] = kval;
            float vval = (global_r < (int)seq_len) ? v_base[global_r * D + c] : 0.0f;
            V_smem[i] = vval;
        }
        __syncthreads();

        // ============================================================
        // PHASE 1: Score computation — S[my_row][group*8 .. group*8+7]
        // ============================================================
        // Each thread computes 8 dot products: Q[my_row,:] · K[j,:]
        // for j = group*8 .. group*8+7
        //
        // This is a 1×8 sub-block of S, with inner dimension D=64.

        float scores[8];
        const int col_start = group * 8;

        if (q_row_global < (int)seq_len) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot = 0.0f;
                const int kv_col = kv_start + col_start + j;

                // Causal mask: if kv position > query position, mask it
                if (causal && kv_col > q_row_global) {
                    scores[j] = -1e30f;
                } else if (kv_col >= (int)seq_len) {
                    scores[j] = -1e30f;
                } else {
                    const float* k_row = K_smem + (col_start + j) * D;
                    // Unrolled dot product over D=64
                    #pragma unroll
                    for (int d = 0; d < D; d += 4) {
                        dot += q_reg[d]     * k_row[d];
                        dot += q_reg[d + 1] * k_row[d + 1];
                        dot += q_reg[d + 2] * k_row[d + 2];
                        dot += q_reg[d + 3] * k_row[d + 3];
                    }
                    scores[j] = dot * scale;
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) scores[j] = -1e30f;
        }

        // ============================================================
        // PHASE 2: Online softmax — need row-wise max and sum
        // ============================================================
        // Each thread has 8 of 32 scores for its row.
        // The 4 threads sharing this row (group 0,1,2,3) must cooperate
        // to find the row max and row sum.

        // Step 2a: Local max across this thread's 8 scores
        float local_max = scores[0];
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            local_max = fmaxf(local_max, scores[j]);
        }

        // Step 2b: Reduce max across 4 threads in the same row
        // The 4 threads for the same row are at lanes (my_row%8)*4+0..3
        // within their warp. They differ in the low 2 bits of lane_id.
        // Use __shfl_xor_sync with masks 1 and 2 to reduce across 4 threads.
        float tile_max = local_max;
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, 2));

        // Step 2c: New running max
        float m_new = fmaxf(m_val, tile_max);

        // Step 2d: Compute exp(score - m_new) and local sum
        float local_sum = 0.0f;
        float exp_scores[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float e = __expf(scores[j] - m_new);  // fast exp
            exp_scores[j] = e;
            local_sum += e;
        }

        // Step 2e: Reduce sum across 4 threads in the same row
        float tile_sum = local_sum;
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, 2);

        // Step 2f: Correction factor for old accumulator
        float correction = __expf(m_val - m_new);

        // Rescale old output accumulator
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            o_reg[i] *= correction;
        }

        // Update running statistics
        l_val = l_val * correction + tile_sum;
        m_val = m_new;

        // ============================================================
        // PHASE 3: P·V accumulation — O[my_row][group*16..group*16+15]
        // ============================================================
        // O += P[my_row, :] × V[:, group*16..group*16+15]
        // P[my_row, k] = exp_scores for this row (distributed across 4 threads)
        //
        // Each thread needs ALL 32 P values for its row, but only has 8.
        // Use warp shuffles to broadcast P values from each group.
        //
        // Strategy: iterate over the 4 groups (g=0,1,2,3).
        // For group g, the thread at (my_row, g) has exp_scores[0..7]
        // = P[my_row, g*8..g*8+7].
        // Broadcast these 8 values to all 4 threads via __shfl_sync.

        const int o_col_start = group * 16;

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            // Source lane within the 4-thread group for row my_row
            // In the warp, the 4 threads for row (my_row%8) are at
            // lanes (my_row%8)*4 + 0,1,2,3
            // We want to read from the thread with group=g, which is at
            // lane (my_row%8)*4 + g
            int src_lane = (lane_id & ~3) | g;  // replace low 2 bits with g

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                // Get P value from the source thread
                float p_val = __shfl_sync(0xFFFFFFFF, exp_scores[j], src_lane);

                // Accumulate: O[my_row][o_col] += p_val * V[g*8+j][o_col]
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

        __syncthreads();  // Ensure smem is safe to overwrite in next tile
    }

    // ============================================================
    // PHASE 4: Write output — normalize by l_val and store
    // ============================================================
    if (q_row_global < (int)seq_len && l_val > 0.0f) {
        float inv_l = 1.0f / l_val;
        const int o_col_start = group * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            o_base[q_row_global * D + o_col_start + i] = o_reg[i] * inv_l;
        }
    }
}
```

## Correctness Walkthrough

### Test case: seq=4, d_head=64, causal=true, 1 head

**Setup**: BQ=32 (only 4 rows active), BKV=32 (1 tile), 128 threads.

Threads 0-15 handle row 0 (groups 0-3), threads 16-31 handle row 1, etc.
Only threads with q_row_global < 4 are active (threads 0-15 for the 4 rows,
each row having 4 threads).

**Score GEMM for row 0 (tid=0,1,2,3)**:
- tid=0 (group=0): computes S[0][0..7] — only S[0][0] is valid (causal: j>0 masked)
- tid=1 (group=1): computes S[0][8..15] — all masked (causal: kv_col=8..15 > q_row=0)
- tid=2 (group=2): computes S[0][16..23] — all masked
- tid=3 (group=3): computes S[0][24..31] — all masked (and kv_col ≥ seq_len=4)

✓ Correct: row 0 only attends to position 0.

**Score GEMM for row 2 (tid=8,9,10,11)**:
- tid=8 (group=0): S[2][0..7] — positions 0,1,2 valid; 3..7 either causal-masked or OOB
- tid=9 (group=1): S[2][8..15] — all OOB (kv_col ≥ seq_len=4)

✓ Correct: row 2 attends to positions 0, 1, 2.

**Online softmax for row 0**:
- Only S[0][0] has a real score; everything else is -1e30
- local_max for tid=0: score[0]; for tid=1,2,3: -1e30
- After shuffle reduce: tile_max = score[0] ✓
- exp_scores: only exp(0)=1 for the valid entry; rest ≈ 0
- tile_sum ≈ 1.0 ✓
- Output: O[0][:] = 1.0 * V[0][:] / 1.0 = V[0][:] ✓ (row 0 only sees itself)

**P·V accumulation**:
- For row 0: P[0,0]=1.0, P[0,j≠0]≈0
- g=0, j=0: p_val = exp_scores[0] from src_lane for group 0 = 1.0
  - o_reg[i] += 1.0 * V[0][o_col_start+i] ✓
- All other (g,j): p_val ≈ 0, contributes nothing ✓

**Causal mask edge case — row 2, position 2**:
- kv_col = kv_start + col_start + j = 0 + 0 + 2 = 2
- causal && kv_col > q_row_global → 2 > 2 → false → NOT masked ✓
- Position 3: kv_col = 3 > 2 → masked ✓

### Multiple KV tiles (seq=128)

n_kv_tiles = 128/32 = 4. For q_tile=0 (rows 0..31):
- Tile 0 (kv 0..31): causal check — kv_start=0 ≤ 31 → process
- Tile 1 (kv 32..63): kv_start=32 > 31 → `causal && 32 > 31` → break!

Wait, this is wrong! Row 31 needs to attend to all positions 0..31, and tile 0 covers that.
But row 31 should NOT attend to positions 32+, so breaking at tile 1 is correct. ✓

For q_tile=1 (rows 32..63):
- Tile 0 (kv 0..31): kv_start=0 ≤ 63 → process ✓
- Tile 1 (kv 32..63): kv_start=32 ≤ 63 → process ✓
- Tile 2 (kv 64..95): kv_start=64 > 63 → break ✓

### Softmax correctness across tiles

First tile: m_val=-1e30, l_val=0
- tile_max = max of valid scores, say M1
- m_new = max(-1e30, M1) = M1
- correction = exp(-1e30 - M1) ≈ 0
- o_reg *= 0 (no-op, was 0)
- l_val = 0 * 0 + tile_sum1 = tile_sum1 ✓

Second tile:
- tile_max = M2
- m_new = max(M1, M2)
- correction = exp(M1 - m_new) — rescales old accumulator ✓
- l_val = l_val_old * correction + tile_sum2 ✓
- o_reg rescaled then new P·V added ✓

This correctly implements the online softmax from FlashAttention. ✓

## Shared Memory Requirements

- K_smem: 32 × 64 × 4 = 8,192 bytes
- V_smem: 32 × 64 × 4 = 8,192 bytes
- Total: **16,384 bytes** (16 KB)

No bank conflicts because D=64 and access patterns are row-major with
consecutive threads accessing different rows.

## Register Usage Per Thread

- q_reg[64]: 64 registers
- o_reg[16]: 16 registers
- scores[8] + exp_scores[8]: 16 registers (reusable)
- Scalars (m_val, l_val, correction, etc.): ~10 registers
- Total: ~106 registers per thread

At 128 threads/block and 106 regs/thread = 13,568 registers.
SM has 65,536 registers → can run 4 blocks per SM. Good occupancy.

## Performance Estimate

For seq=128, 12 heads, d_head=64:
- Grid: (12, 4) = 48 blocks
- Per block: 4 KV tiles max (depending on causal early-exit)
- Average KV tiles (causal): ~2.5 per block
- Per tile: 8 score dot products (each 64 FMAs), softmax, 32×16 P·V FMAs
- Score FLOPs per tile: 8 × 64 × 2 × 128_threads...

Actually let's compute differently:
- Score GEMM per tile: 2 × 32 × 32 × 64 = 131,072 FLOPs
- P·V GEMM per tile: 2 × 32 × 64 × 32 = 131,072 FLOPs
- Total per tile: 262,144 FLOPs + softmax overhead
- Per block (4 tiles avg for non-causal): 1,048,576 FLOPs
- 48 blocks total: 50,331,648 FLOPs

At 50% arithmetic throughput (memory-bound): ~2000 GFLOPS → 0.025ms
Target: ≤0.069ms → should be achievable.

## Rust Integration

The kernel function signature matches the existing NVRTC integration:

```rust
let config = cudarc::driver::LaunchConfig {
    grid_dim: (n_heads as u32, seq_len.div_ceil(32) as u32, 1),
    block_dim: (128, 1, 1),
    shared_mem_bytes: 16384,
};
// Launch args: (Q, K, V, Out, seq_len, d_head, causal)
```

## Key Advantages Over Current Kernel

| Aspect | Current (V2) | This Design (V3) |
|--------|-------------|-------------------|
| Threads/block | 32 (1 warp) | 128 (4 warps) |
| Score computation | 1 thread per row, scalar dot | 4 threads per row, parallel |
| P·V computation | Sequential per KV position | Cooperative via shuffle broadcast |
| Register pressure | 64 regs (Q) + 64 regs (O) = 128 | 64+16+16 = 96 regs |
| Memory efficiency | Each thread loads full K,V rows | Cooperative smem loads |
| ILP | Low (1 dot product at a time) | High (8 scores + 4-way unroll) |
