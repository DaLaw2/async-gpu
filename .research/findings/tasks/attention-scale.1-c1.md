# attention-scale.1: Attention Algorithm Survey for seq>32
**Cycle**: 1 | **Theme**: attention-scale | **Kind**: investigation | **Status**: done

## Summary

Surveyed four approaches for scaling attention beyond seq=32: naive multi-block, multi-pass,
FlashAttention, and block-sparse. FlashAttention (specifically a simplified v1 variant) is
the recommended approach for seq=128+ in inline PTX. It avoids materializing the O(seq^2)
score matrix, requires only O(seq) extra memory, and integrates naturally with the existing
MMA and shared-memory primitives. A simpler multi-pass approach is viable as a stepping stone
for seq=128 but becomes memory-prohibitive at seq=1024.

## Findings

### Q: Memory and compute trade-offs of different approaches?

**A:** See comparison table below. The fundamental tension is between implementation simplicity
and memory efficiency. Naive/multi-pass approaches are simple but require O(N^2) global memory
for the score matrix, which becomes prohibitive at N=1024 (4MB per head in f32). FlashAttention
trades implementation complexity for O(N) memory by never materializing the full score matrix.

| seq_len | Score matrix (f32) | Fits shared mem (48KB)? | Fits shared mem (100KB)? |
|---------|-------------------|------------------------|-------------------------|
| 32      | 4 KB              | Yes                    | Yes                     |
| 64      | 16 KB             | Yes                    | Yes                     |
| 128     | 64 KB             | Barely (no room left)  | Yes                     |
| 256     | 256 KB            | No                     | No                      |
| 512     | 1 MB              | No                     | No                      |
| 1024    | 4 MB              | No                     | No                      |

**Confidence**: high

### Q: Is FlashAttention feasible in inline PTX?

**A:** Yes, but with caveats. A simplified FlashAttention v1 is feasible. The algorithm requires:

1. **Tiled GEMM (Q x K^T)** — We already have `mma.sync.aligned.m16n8k16` working. Can reuse.
2. **Online softmax** — Requires only scalar `max` and `sum` tracking per row, plus `exp` and
   multiply. We already have `gpu_exp_f32`. Warp-level reductions exist.
3. **Rescaled accumulation (P x V)** — Another tiled GEMM with a rescaling multiply before
   accumulation. Straightforward extension of existing MMA pattern.
4. **Causal mask** — Applied during score computation by setting masked positions to -inf before
   softmax. Natural fit since we process tiles sequentially.

The main complexity is the rescaling logic in the output accumulator: when a new tile produces
a larger maximum, all previously accumulated outputs must be multiplied by `exp(m_old - m_new)`.
This is ~10-15 lines of PTX per tile iteration.

Estimated implementation: ~300-500 lines of Rust+PTX (vs ~120 lines for current naive kernel).
Most complexity is bookkeeping, not new PTX primitives.

What we do NOT need: warp-shuffle for cross-warp communication (single-warp-per-row suffices
for d_head=64), cp.async (nice-to-have but not required), TMA instructions (Hopper-only).

**Confidence**: high

### Q: Minimum viable approach for seq=128?

**A:** Two viable paths:

**Path A — Multi-pass (simplest, seq<=256):**
Store the 128x128 score matrix (64KB f32) in shared memory if using extended shared memory
(100KB on SM86), or in global memory. Three-kernel approach:
1. Kernel 1: Q x K^T -> scores in global memory
2. Kernel 2: row-wise softmax on scores
3. Kernel 3: scores x V -> output

Pros: Each kernel is simple, reuses existing patterns.
Cons: 3 kernel launches per head, O(N^2) global memory, bandwidth-bound.

**Path B — Single-pass FlashAttention (recommended for seq>=128):**
Tile K and V into blocks of B_c columns (e.g., B_c=32). For each Q-row tile (B_r rows):
- Iterate over K/V tiles, computing partial scores and accumulating output with online softmax.
- Only need shared memory for: Q tile (B_r x d) + K tile (B_c x d) + V tile (B_c x d) + output accumulator (B_r x d) + running stats (B_r x 2 scalars).

For B_r=32, B_c=32, d_head=64:
- Q tile: 32 x 64 x 4 = 8 KB
- K tile: 32 x 64 x 4 = 8 KB
- V tile: 32 x 64 x 4 = 8 KB
- Output acc: 32 x 64 x 4 = 8 KB
- Stats (m, l): 32 x 2 x 4 = 256 B
- **Total: ~32.25 KB — fits comfortably in 48KB shared memory**

This works for ANY seq_len since we stream K/V tiles from global memory.

**Confidence**: high

## Algorithm Comparison

| Approach | Global Memory | Shared Memory | Kernel Launches | Causal Mask | Complexity (LoC) | Seq Limit | Recommendation |
|----------|--------------|---------------|-----------------|-------------|-----------------|-----------|----------------|
| **Current naive** | O(Nd) | O(N^2) | 1 | Easy | ~120 | 32 | Keep for small seq |
| **Naive multi-block** | O(N^2) per head | O(B^2) per block | 1 (but needs global atomics) | Awkward | ~200 | ~256 | Not recommended |
| **Multi-pass** | O(N^2) per head | O(tile) | 3 | Moderate | ~250 | ~1024 (mem limited) | Stepping stone only |
| **FlashAttention v1** | O(Nd) output only | O(B_r*d + B_c*d) | 1 | Natural | ~400 | Unlimited | **Recommended** |
| **Block-sparse** | O(Nw) (w=window) | O(B*w) | 1 | N/A (approx) | ~500 | Unlimited | Not applicable (need full attention) |

## FlashAttention Deep Dive

### Online Softmax: The Key Insight

Standard softmax requires two passes: (1) find max over all elements, (2) compute exp and sum.
This requires materializing all scores before normalization.

Online softmax maintains running statistics `(m, l)` where:
- `m` = running maximum of scores seen so far
- `l` = running sum of `exp(score - m)` (the softmax denominator)

When processing a new block of scores `s_new[0..B_c]`:

```
m_new = max(m_old, max(s_new))
l_new = l_old * exp(m_old - m_new) + sum(exp(s_new - m_new))
```

The rescaling factor `exp(m_old - m_new)` corrects the old denominator when a new maximum
is discovered. If `m_new == m_old` (common case), the correction is just `* 1.0`.

### FlashAttention v1 Algorithm (Simplified)

```
Input:  Q[N, d], K[N, d], V[N, d]  in global memory (HBM)
Output: O[N, d]                     in global memory

For each Q-tile i (rows i*B_r .. (i+1)*B_r):
    Load Q_i[B_r, d] into shared memory
    Initialize: O_i = zeros[B_r, d]     (output accumulator in shared memory/registers)
                m_i = -inf[B_r]          (running max per row)
                l_i = 0[B_r]            (running sum per row)

    For each KV-tile j (cols j*B_c .. (j+1)*B_c):
        Load K_j[B_c, d] into shared memory
        Load V_j[B_c, d] into shared memory

        // Step 1: Compute tile of scores
        S_ij = Q_i @ K_j^T              // [B_r, B_c] — tiled GEMM (MMA)
        S_ij *= scale                    // 1/sqrt(d_head)

        // Step 1b: Apply causal mask (if j*B_c + col > i*B_r + row, set to -inf)

        // Step 2: Online softmax update
        m_new = max(m_i, rowmax(S_ij))   // new running max
        P_ij  = exp(S_ij - m_new)        // exponentiated scores
        l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)

        // Step 3: Rescale and accumulate output
        O_i = O_i * (l_i * exp(m_i - m_new) / l_new)  // rescale old output
        O_i += (P_ij / l_new) @ V_j                     // add new contribution

        // Update stats
        m_i = m_new
        l_i = l_new

    // O_i is now the final output for these rows
    Write O_i to global memory O[i*B_r .. (i+1)*B_r]
```

**Note:** The division by `l_new` can be deferred to the end (accumulate unnormalized, divide
once after all KV-tiles). This is simpler and avoids repeated divisions:

```
O_i = O_i * exp(m_i - m_new) + P_ij @ V_j   // unnormalized accumulation
...
// After all KV-tiles:
O_i = O_i / l_i                              // single final normalization
```

### Memory Analysis for GPT-2 Parameters

GPT-2 small: n_heads=12, d_head=64, d_model=768.

With FlashAttention, tile sizes B_r=32, B_c=32:

| Component | Size | Location |
|-----------|------|----------|
| Q tile (B_r x d_head) | 32 x 64 x 4 = 8 KB | Shared memory |
| K tile (B_c x d_head) | 32 x 64 x 4 = 8 KB | Shared memory |
| V tile (B_c x d_head) | 32 x 64 x 4 = 8 KB | Shared memory |
| Score tile S (B_r x B_c) | 32 x 32 x 4 = 4 KB | Shared memory |
| Output acc O (B_r x d_head) | 32 x 64 x 4 = 8 KB | Shared memory |
| Stats m, l (B_r x 2) | 32 x 2 x 4 = 256 B | Registers/shared |
| **Total** | **~36.25 KB** | **< 48 KB limit** |

At seq=1024: need ceil(1024/32) = 32 KV-tile iterations per Q-tile. Each iteration does
one 32x32 GEMM (score) + one 32x64 GEMM (output accum). Total GEMM ops per head:
32 Q-tiles x 32 KV-tiles x 2 GEMMs = 2048 MMA calls. Very feasible.

Global memory per head: only Q, K, V, O = 4 x 1024 x 64 x 4 = 1 MB. No score matrix stored.

### Numerical Stability

The online softmax approach is numerically stable because:
1. We always subtract the running max before `exp()`, preventing overflow.
2. The rescaling uses `exp(m_old - m_new)` where `m_old <= m_new`, so the exponent is <= 0,
   preventing overflow in the correction factor.
3. Final results are mathematically identical to standard softmax (exact, not approximate).

### Causal Mask Integration

Causal masking (upper-triangular -inf) integrates naturally:
- When computing S_ij, if the KV-tile column index exceeds the Q-tile row index, set to -inf.
- For tiles entirely above the diagonal: skip the tile entirely (S = -inf, P = 0, no V contribution).
- For tiles on the diagonal: mask individual elements.
- For tiles below the diagonal: no masking needed.

This can actually improve performance for causal attention since ~50% of KV-tiles can be skipped.

## Recommended Approach

**FlashAttention v1 (simplified)** is the recommended approach for the following reasons:

1. **Memory efficiency**: O(N*d) instead of O(N^2). Critical for seq=1024 (saves 4MB per head).
2. **Single kernel**: No multi-kernel overhead or global memory synchronization.
3. **Reuses existing primitives**: MMA m16n8k16, shared memory, `gpu_exp_f32`, warp reductions.
4. **Scales indefinitely**: Works for seq=128, 512, 1024, or beyond.
5. **Numerically stable**: Online softmax with running max is more stable than naive softmax.
6. **Causal mask bonus**: Can skip ~50% of computation for autoregressive inference.

The main cost is implementation complexity (~400 lines vs ~120 for naive), but the algorithm
is well-understood and each component maps directly to existing project primitives.

## Implementation Sketch

```
// Kernel signature
fn flash_attention_fwd(
    q: *const f32, k: *const f32, v: *const f32, out: *mut f32,
    seq_len: u32, d_head: u32, n_heads: u32, is_causal: u32,
)

// Grid: (n_heads, ceil(seq_len/B_R), 1)
// Block: (32, 1, 1) — one warp per tile-row
// B_R = 32 (rows of Q per block), B_C = 32 (cols of K per tile)
// Shared memory: Q_tile[32][64] + K_tile[32][64] + V_tile[32][64]
//              + S_tile[32][32] + O_acc[32][64] + stats[32][2]
//              = ~36 KB

// Each block processes Q rows [block_y * B_R .. (block_y+1) * B_R]:
//
// 1. Load Q_tile from global -> shared
// 2. Init O_acc = 0, m = -inf, l = 0
// 3. For kv_tile in 0 .. ceil(seq_len / B_C):
//    a. If causal and kv_tile * B_C > (block_y+1) * B_R: break (all masked)
//    b. Load K_tile, V_tile from global -> shared
//    c. bar.sync
//    d. Compute S_tile = Q_tile @ K_tile^T using MMA (or scalar dot products)
//    e. Apply scale (1/sqrt(d_head))
//    f. Apply causal mask if needed
//    g. Compute row-max of S_tile, update m_new
//    h. Compute P_tile = exp(S_tile - m_new)
//    i. Compute row-sum of P_tile, update l_new
//    j. Rescale O_acc: O_acc *= exp(m_old - m_new)
//    k. Accumulate: O_acc += P_tile @ V_tile
//    l. Update m = m_new, l = l_new
//    m. bar.sync
// 4. Normalize: O_acc /= l
// 5. Write O_acc to global output
```

### New Primitives Needed

| Primitive | Description | Estimated LoC | Difficulty |
|-----------|-------------|---------------|------------|
| Tile load (global->shared) | Cooperative load of B x d f32 tile | ~20 | Low (existing pattern) |
| Row-max reduction | Max across B_C columns within a tile row | ~15 | Low (existing warp reduce) |
| Row-sum reduction | Sum across B_C columns within a tile row | ~15 | Low (existing warp reduce) |
| Element-wise exp | exp(S_ij - m) for tile elements | ~10 | Low (existing gpu_exp_f32) |
| Rescale accumulator | O *= exp(m_old - m_new) | ~10 | Low (scalar multiply) |
| Causal mask logic | Set S[r][c] = -inf if col > row | ~10 | Low |
| **Total new code** | | **~80 lines** | |

The remaining ~300 lines are the outer loop, tile indexing, and MMA calls (reused from GEMM).

## Impact on Downstream Tasks

1. **GPT-2 inference**: Enables full context window (seq=1024). Currently blocked at seq=32.
2. **Transformer layer pipeline**: The attention kernel is the bottleneck. Scaling it unlocks
   end-to-end transformer inference.
3. **Memory budget**: FlashAttention's O(N*d) memory means attention no longer dominates the
   memory footprint — weight matrices (d_model^2 parameters) become the limiting factor instead.
4. **Performance**: Single-kernel FlashAttention avoids the multi-launch overhead of multi-pass,
   and the tiled approach maximizes data reuse in shared memory.
