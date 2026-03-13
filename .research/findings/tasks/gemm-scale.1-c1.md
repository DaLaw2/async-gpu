# gemm-scale.1: Multi-warp Output Tiling
**Cycle**: 1 | **Theme**: gemm-scale | **Kind**: experiment | **Status**: done

## Summary
Implemented 4-warp (128 threads) GEMM with 2×2 tiling layout. Each warp computes a 16×8 MMA tile, producing D(32×16) = A(32×K) × B(K×16). Discovered and corrected two critical MMA fragment mapping errors that affected all previous single-warp tests (hidden by uniform data).

## Findings

### Q: Can 4 warps cooperatively compute a 32×16 GEMM using MMA?
A: Yes. 4 warps in a 2×2 layout (warp_m×warp_n) each compute a 16×8 tile using `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Cooperative shared memory loading distributes the work: 128 threads load A[32][8] (2 each) and B[16][8] (1 each) per K-tile.
**Confidence**: high

### Q: What is the correct MMA m16n8k16 fragment mapping?
A: The NVIDIA PTX ISA specifies `.col` for B, meaning the B fragment uses **column-major** packing. For thread with groupID `g = tid/4` and lane `l = tid%4`:

**A fragments (row-major, .row):**
- a0 = pack(A[g][l*2], A[g][l*2+1])
- a1 = pack(A[g][l*2+4], A[g][l*2+5])  (shifted by 4 in K)
- a2 = pack(A[g+8][l*2], A[g+8][l*2+1])
- a3 = pack(A[g+8][l*2+4], A[g+8][l*2+5])

**B fragments (column-major, .col):**
- b0 = pack(B[l*2][g], B[l*2+1][g])   ← two consecutive K-rows of column g
- b1 = pack(B[l*2+8][g], B[l*2+9][g])

**D output mapping:**
- d0 = D[g][l*2]
- d1 = D[g][l*2+1]
- d2 = D[g+8][l*2]
- d3 = D[g+8][l*2+1]

This means: `g` indexes the M-dimension (row of output), `l` indexes the N-dimension (column of output, packed pairs). The B matrix must be stored in column-major packed format where each u32 contains two consecutive K-dimension elements from the same N-column.

**Confidence**: high (empirically verified with 5 test cases including non-uniform A and B)

## Unexpected Discoveries

1. **All previous MMA tests had wrong fragment mapping comments** — the comments said `d0=D[l*2][g]` (lane=row, group=column) but the actual mapping is `d0=D[g][l*2]` (group=row, lane=column). These tests passed only because they used uniform/identity matrices that can't distinguish the two mappings.

2. **B must be column-major packed** — feeding row-major B to the MMA effectively computes A × B^T. Previous tests used uniform B (all 1s), so B = B^T and the error was invisible. Only non-uniform B exposes this bug.

3. **The MMA fragment mapping matches the NVIDIA PTX ISA specification** — the `.col` qualifier on B means column-major fragment layout. Thread (g,l) provides B elements from column g at rows determined by l.

## Open Questions

1. Should the single-warp GEMM kernels (gpu-compute.5, gpu-pipeline.1-3) be updated to use column-major B? They currently work because they only use uniform B data.

## Impact on Downstream Tasks

- **gemm-scale.2** (multi-block GEMM): Can proceed. The correct fragment mapping is now established.
- **gemm-scale.3** (768×768 validation): Must use column-major B packing.
- **All future MMA-based kernels**: Must use column-major B and correct output mapping.
- **ADR needed**: Document the correct MMA fragment mapping as an Architecture Decision Record.
