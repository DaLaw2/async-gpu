# mma-splitk.1: Study split-K GEMM implementations
**Cycle**: 351 | **Theme**: mma-splitk | **Kind**: investigation | **Status**: done

## Summary

Investigated split-K GEMM patterns from CUTLASS, Triton, and industry references.
Split-K partitions the K dimension across multiple thread blocks (grid.z), each computing
a partial result, then reduces via atomicAdd or workspace buffer. This limits per-block
accumulation error and improves SM utilization for skinny GEMM shapes.

## Findings

### Q: How does CUTLASS implement split-K?
A: Two approaches:

**Parallel split-K (two-pass)**:
1. Launch GEMM kernel with grid = (M_tiles, N_tiles, split_k_slices)
2. Each z-slice computes partial [M_tile × N_tile] for K/split_k_slices elements
3. Results stored in workspace buffer of size M × N × split_k_slices × sizeof(f32)
4. Separate reduction kernel sums across split_k dimension

**Serial split-K (single kernel)**:
1. Same grid, but z-slices use semaphores/atomic to coordinate
2. First z-slice writes its result to output
3. Subsequent z-slices atomicAdd their partial results
4. No separate workspace needed, but atomic contention can hurt performance

CUTLASS example (06_splitK_gemm) shows: m=128, n=128, k=4096, split_k=16 →
each z-slice processes k_chunk=256 elements (16 MMA tiles for K=16 instruction).
**Confidence**: high

### Q: What K_CHUNK sizes are optimal for SM 86 (RTX 3060)?
A: No fixed optimal value — depends on problem size and GPU occupancy.

General guidelines from research:
- **A100**: split_k=4 optimal for W4A16 quantized GEMM
- **H100**: split_k=8 optimal (33% more SMs than A100)
- **RTX 3060 (SM 86, 28 SMs)**: split_k=4-8 reasonable starting point
- Too many splits → atomic contention and scheduling overhead dominate
- Too few splits → error accumulation and low occupancy

For our GPT-2 dimensions:
- K=768: split_k=4 → k_chunk=192 (12 MMA tiles per chunk)
- K=2304: split_k=4 → k_chunk=576 (36 tiles), split_k=8 → k_chunk=288 (18 tiles)
- K=3072: split_k=4 → k_chunk=768 (48 tiles), split_k=8 → k_chunk=384 (24 tiles)

Recommendation: start with split_k=4, tune upward if precision still insufficient.
**Confidence**: medium (needs empirical tuning)

### Q: Atomic reduction vs two-pass workspace — trade-offs?
A:

**AtomicAdd approach (preferred for our case)**:
- Pros: Single kernel launch, no workspace allocation, simpler code
- Cons: Atomic contention on output tiles, non-deterministic ordering
- f32 atomicAdd is natively supported on SM 60+ (including SM 86)
- For our problem sizes (M,N ≤ 3072), contention is moderate
- Precision: f32 atomicAdd is IEEE-754 compliant, but addition order is non-deterministic

**Workspace approach**:
- Pros: Deterministic, no atomic contention, exact f32 accumulation
- Cons: Extra workspace (M×N×split_k×4 bytes), two kernel launches, more complex
- For 3072×3072×4 = 144MB workspace — significant memory overhead

**Recommendation**: Use atomicAdd. The non-determinism in addition order is negligible
for f32 (associativity error is ~machine epsilon). Our primary concern is per-multiply
rounding (f16×f16), not summation order. AtomicAdd eliminates the second kernel launch.
**Confidence**: high

### Q: How to handle non-power-of-2 K dimensions?
A: Standard approach: last z-slice handles the remainder.

```
k_per_split = ceil(K / split_k)
k_start = z_idx * k_per_split
k_end = min(k_start + k_per_split, K)
k_tiles_this_split = (k_end - k_start) / tile_k
```

For K=768, split_k=4: each gets exactly 192 (divides evenly).
For K=2304, split_k=4: each gets exactly 576 (divides evenly).
For K=3072, split_k=4: each gets exactly 768 (divides evenly).
All our GPT-2 dimensions divide evenly by 4 and 8.
**Confidence**: high

## Implementation Plan for mma-splitk.2

Based on investigation, the split-K f16 MMA kernel should:

1. **Grid**: (M_tiles, N_tiles, split_k) where M_tiles = M/32, N_tiles = N/16
2. **Each block**: Process k_per_split/16 MMA tiles (K_CHUNK/16 iterations)
3. **K range**: k_start = blockIdx.z * k_per_split, iterate over that range only
4. **Reduction**: atomicAdd f32 to output buffer
5. **Zero-init**: Output buffer must be zeroed before launch (cudaMemset or first z-slice writes, others atomicAdd)
6. **First z-slice optimization**: z=0 writes directly (no atomic), z>0 uses atomicAdd
   This avoids atomic contention for the first partial result.

Key differences from current `multi_block_gemm`:
- Add `split_k` and `k_per_split` parameters
- K-loop iterates over partial range, not full K
- Output uses atomicAdd instead of direct store (for z>0)
- Input: f16 weights loaded directly (no f32→f16 conversion needed if weights are f16)

## Unexpected Discoveries

1. **Split-K is primarily for occupancy, not precision**: Industry uses split-K mainly
   to improve SM utilization for skinny matrices (small M×N, large K). The precision
   benefit (limiting accumulation error per block) is a secondary advantage.

2. **Stream-K is the next evolution**: CUTLASS 3.x introduces Stream-K which
   dynamically distributes work across SMs, further reducing wave quantization.
   Not needed for our use case but worth noting.

3. **Triton's split-K uses atomic_add directly**: No workspace needed. The atomic
   contention with split_k=4-8 is acceptable for most problem sizes.

## Impact on Downstream Tasks

- **mma-splitk.2**: Clear implementation path — modify multi_block_gemm to add
  grid.z dimension, k_per_split parameter, atomicAdd reduction
- **mma-splitk.3**: f16 weight loading is independent — can be done in parallel
- **mma-splitk.4**: Inference validation needs both .2 and .3 complete
