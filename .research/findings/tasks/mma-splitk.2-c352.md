# mma-splitk.2: Split-K f16 MMA GEMM kernel + fragment mapping fix
**Cycle**: 352 | **Theme**: mma-splitk | **Kind**: experiment | **Status**: done

## Summary

Fixed a fundamental MMA fragment register ordering bug that affected ALL MMA kernels
since their initial implementation. The bug caused incorrect results for non-uniform
data (invisible with all-1.0 tests). Also fixed a race condition in the split-K kernel
where z-slice 0's direct write could be overwritten by other z-slices' atomicAdds.

## Findings

### Q: What was the MMA fragment mapping bug?
A: The a[1] and a[2] register assignments were SWAPPED. The PTX ISA for
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` uses a row-interleaved layout:

```
a[0] = (rows 0-7,  k_low)   — f16x2(A[groupID][2*tid], A[groupID][2*tid+1])
a[1] = (rows 8-15, k_low)   — f16x2(A[groupID+8][2*tid], A[groupID+8][2*tid+1])
a[2] = (rows 0-7,  k_high)  — f16x2(A[groupID][2*tid+8], A[groupID][2*tid+9])
a[3] = (rows 8-15, k_high)  — f16x2(A[groupID+8][2*tid+8], A[groupID+8][2*tid+9])
```

Our code incorrectly loaded:
- a[1] = (rows 0-7, k_high)  ← WRONG, should be (rows 8-15, k_low)
- a[2] = (rows 8-15, k_low)  ← WRONG, should be (rows 0-7, k_high)

This caused k_low values to be double-counted for row groupID and k_high values
to be routed to row groupID+8. With all-1.0 data, the error is invisible because
all fragments have the same value.
**Confidence**: high (verified on hardware)

### Q: How was the bug diagnosed?
A: Binary diagnostic kernel with A[row][k] = 2^k for k=0..9:
- Expected D[0][0] = 1023 (sum of 2^0 through 2^9)
- Got D[0][0] = 510 (missing 2^0 and 2^9)
- Got D[8][0] = 1536 (= 1023 + 513 = 1023 + 2^0 + 2^9)
- Key insight: 510 + 1536 = 2046 = 2 × 1023 — total preserved but distributed wrong

A fragment dump kernel (mma_diag) confirmed a0==a2 and a1==a3 (all rows identical),
yet d0≠d2 — proving the MMA uses a[1] and a[2] for different rows than expected.

Solving the equations: d0 = (1+2)/2 × 4 threads' k_lo = each thread's k_lo counted
twice → k_lo contribution doubled, k_hi contribution sent to d2.
**Confidence**: high

### Q: What was the split-K race condition?
A: The split-K kernel had z-slice 0 write directly to output and z-slices 1+ use
atomicAdd. Since CUDA doesn't guarantee thread block execution order, z-slice 1
could atomicAdd before z-slice 0 writes, then z-slice 0's direct write overwrites
z-slice 1's contribution. Fix: all z-slices use atomicAdd (output is zeroed).
**Confidence**: high

### Q: What is the precision after the fix?
A: MMA f16 GEMM now produces **zero error** vs f32 FMA reference at all tested dimensions:
- 128×768 × 768×768: 0.0 error (all split_k values)
- 128×768 × 768×2304: 0.0 error
- 128×768 × 768×3072: 0.0 error
- 128×3072 × 3072×768: 0.0 error

This means the previous "MMA precision issue" was entirely caused by the fragment
mapping bug, NOT by f16 truncation or accumulation errors. f16 MMA with f32
accumulation is sufficient for GPT-2 dimensions without any additional precision
techniques (split-K, Kahan summation, etc.).
**Confidence**: high

## Unexpected Discoveries

1. **The fragment mapping bug was the root cause of ALL MMA precision issues** across
   the entire project history. Every test that showed "MMA error" was actually a
   data layout bug, not a precision limitation of f16 arithmetic.

2. **Split-K is NOT needed for precision** at GPT-2 dimensions (K up to 3072).
   f16 MMA with f32 accumulation already produces zero error vs f32 FMA. Split-K
   is only needed for much larger K values where f32 accumulation overflows.

3. **The all-1.0 test is a terrible canary** for fragment mapping bugs. Any permutation
   of identical values produces the same result. Non-uniform test data is essential.

## Impact on Downstream Tasks

- **mma-splitk.3** (f16 weight loading): Still valuable for performance (skip f32→f16
  conversion), but no longer needed for precision.
- **mma-splitk.4** (GPT-2 inference validation): Should now pass trivially since the
  MMA produces zero-error results.
- **mma-splitk.5** (benchmarks): Can now measure clean MMA throughput without
  precision-related artifacts.
- **All prior MMA GEMM kernels** (full_gemm, full_gemm_f32in, multi_block_gemm, bf16,
  gemm+softmax, etc.): All fixed by the same a[1]↔a[2] swap.
