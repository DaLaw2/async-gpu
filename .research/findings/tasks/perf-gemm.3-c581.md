# perf-gemm.3: 128×64 tile + 4×8 register blocking GEMM
**Cycle**: 581 | **Theme**: perf-gemm | **Kind**: experiment | **Status**: done

## Summary
Scaled `gemm_f32_v2` from 64×64+4×4 to 128×64+4×8 tile with 32 named accumulators.
Result: **998 GFLOPS at 4096³** (36.8% of cuBLAS), up from 905 GFLOPS (33%).

## Results

| Shape | v1 (old) | v2 64×64 | v2 128×64 | cuBLAS | % cuBLAS |
|-------|----------|----------|-----------|--------|----------|
| 512³ | 149 | 767 | 719 | 2,290 | 31.4% |
| 1024³ | 157 | 893 | 957 | 2,768 | **34.6%** |
| 2048³ | 157 | 888 | 960 | 2,741 | **35.0%** |
| 4096³ | 160 | 905 | 999 | 2,711 | **36.8%** |
| GPT-2 128×768² | 126 | 607 | 513 | 1,996 | 25.7% |

## Observations
1. 128×64 beats 64×64 for large matrices (4096³: 999 vs 905 = +10.4%)
2. 128×64 is worse for small matrices (512³: 719 vs 767) — too many threads per tile
3. GPT-2 shapes (M=128) are worse with BM=128 because only 1 block in M dimension
4. The FMA:load ratio is 32/12 = 2.67:1 (vs 1:1 for v1) — much better but still below 4:1

## Next Steps for Further Improvement
1. **float4 vectorized global loads**: Currently scalar loads → 4x fewer transactions
2. **Try 64×128+8×4**: Wider N-tile may help GPT-2 shapes where N is large
3. **Larger K tile (BK=16)**: More FMAs per smem load
4. **Auto-select tile size**: Use 64×64 for small M, 128×64 for large M

**Confidence**: high
