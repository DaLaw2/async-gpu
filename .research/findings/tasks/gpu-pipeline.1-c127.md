# gpu-pipeline.1: Fragment-to-matrix index mapping for MMA (non-uniform matrices)
**Cycle**: 127 | **Theme**: gpu-pipeline | **Kind**: experiment | **Status**: done

## Summary

Empirically determined and verified the complete per-thread fragment-to-matrix element mapping for `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on SM 86. The key discovery: the D output mapping is transposed relative to naive expectations — rows come from lane_id, columns from group_id.

## Findings

### Q: What is the per-thread fragment-to-matrix element mapping for mma.sync.aligned.m16n8k16?
A: For thread tid, with group = tid/4 (0..7), lane = tid%4 (0..3):

**A fragment (16×16, row-major, f16x2 packed as u32[16][8]):**
- a0 = A_packed[group][lane]       → {A[group, lane*2], A[group, lane*2+1]}
- a1 = A_packed[group][lane+4]     → {A[group, lane*2+8], A[group, lane*2+9]}
- a2 = A_packed[group+8][lane]     → {A[group+8, lane*2], A[group+8, lane*2+1]}
- a3 = A_packed[group+8][lane+4]   → {A[group+8, lane*2+8], A[group+8, lane*2+9]}

**B fragment (16×8, f16x2 packed as u32[16][4]):**
- b0 = B_packed[group][lane]       → {B[group, lane*2], B[group, lane*2+1]}
- b1 = B_packed[group+8][lane]     → {B[group+8, lane*2], B[group+8, lane*2+1]}

**D output (16×8, f32):**
- d0 = D[lane*2, group]
- d1 = D[lane*2+1, group]
- d2 = D[lane*2+8, group]
- d3 = D[lane*2+9, group]

Key insight: The row stride between the two register pairs (a0/a1 vs a2/a3, b0 vs b1, d0/d1 vs d2/d3) is **+8**, NOT **×2+1** as initially assumed. And the D mapping is **transposed** — rows come from lane, columns from group.
**Confidence**: high (verified with A=identity, B=sequential values)

### Q: Can we implement correct MMA with non-uniform A and B matrices?
A: Yes. Verified with A = 16×16 identity matrix and B = 16×8 matrix with unique values (1..128). All 128 D output elements match expected values D[i][j] = B[i][j] exactly.
**Confidence**: high

## Unexpected Discoveries

1. **D mapping is transposed**: The output fragment layout has row=f(lane) and col=f(group), which is the opposite of the A fragment layout where row=f(group). This is counter-intuitive but consistent with how the MMA instruction distributes work across the warp.

2. **First attempt mapping was completely wrong**: Initial mapping used group*2/group*2+1 (stride 2) instead of group/group+8 (stride 8). This caused 127/128 mismatches. The empirical approach (identity matrix test) was essential for discovering the correct mapping.

## Changes Made
- **crates/gpu-kernel/src/lib.rs**: Added `test_mma_mapped` kernel with correct fragment indexing
- **crates/gpu-host/src/main.rs**: Added `run_mma_mapped_test()` with A=identity verification

## Open Questions
1. Does this mapping hold for other MMA shapes (m8n8k16, m16n16k16)?
2. Can `ldmatrix.sync.aligned` simplify fragment loading?

## Impact on Downstream Tasks
- **gpu-pipeline.2 (Multi-tile GEMM)**: UNBLOCKED — fragment mapping now known
- **gpu-pipeline.3 (End-to-end pipeline)**: Can use correct GEMM for real computations
