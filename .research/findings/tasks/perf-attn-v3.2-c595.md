# perf-attn-v3.2: Cooperative Score GEMM Optimizations
**Cycle**: 595 | **Theme**: perf-attn-v3 | **Kind**: experiment | **Status**: done

## Summary
Implemented three key optimizations to flash_attn_v3.cu: shared memory padding (stride 65),
float4 global loads for K/V, and cooperative Q loading. Fixed NVRTC arch target (sm_86 → sm_75).

## Findings

### Q: What bank conflicts existed in the old kernel?
A: 4-way bank conflict on K reads in score GEMM. All 4 thread groups (reading different K rows
at the same d-column) hit the same shared memory bank because stride=64 is a multiple of 32 (bank count).
**Confidence**: high (mathematical analysis)

### Q: Does padding by 1 (stride 65) fix bank conflicts?
A: Yes. With stride 65, bank = (row*65 + d) % 32. Since gcd(65,32)=1, rows 8 apart
map to banks 8 apart mod 32, which are all distinct.
**Confidence**: high

### Q: What performance does the optimized V3 kernel achieve?
A: On GTX 1660 (sm_75, 5 TFLOPS, 192 GB/s):
- seq=128: 337 GFLOPS (causal), 464 GFLOPS (bidir)
- seq=256: 460 GFLOPS (causal), 532 GFLOPS (bidir)
- seq=512: 559 GFLOPS (causal), 600 GFLOPS (bidir)
**Confidence**: high (measured, 20 iterations)

### Q: Why was removing the `p_val > 1e-30f` branch slower?
A: The branch saves V shared memory reads for masked positions (where p_val ≈ 0).
For causal attention, ~50% of positions are masked, so skipping V reads saves significant
shared memory bandwidth. The branch divergence cost is less than the saved memory traffic.
**Confidence**: high (measured: 556 with branch vs 540 without)

## Optimizations Implemented
1. **Smem padding**: K_smem/V_smem stride = 65 (not 64) → eliminates 4-way K bank conflicts
2. **float4 global loads**: K/V tile loading uses float4 (128-bit) reads
3. **Cooperative Q load**: 128 threads cooperatively load Q tile via float4,
   then each thread copies its row to registers. 4x fewer global reads vs redundant per-thread loads.
4. **NVRTC arch fix**: sm_86 → sm_75 (matches actual GPU)
5. **Launch bounds**: `__launch_bounds__(128, 3)` for register allocation guidance

## Key Metrics
| seq | Mode | Time (ms) | GFLOPS |
|-----|------|-----------|--------|
| 512 | causal | 0.721 | 559 |
| 512 | bidir | 1.342 | 600 |

Peak FP32 utilization: 600/5000 = 12% (limited by low occupancy + memory latency on sm_75).
Roofline (AI=13.4 FLOP/byte, BW=192 GB/s) predicts 2573 GFLOPS → gap due to latency/occupancy.

## Open Questions
- Would BKV=16 (smaller tiles) improve occupancy enough to offset increased overhead?
- Would a completely different thread mapping (e.g., 8 threads per row) help?
- Is tensor-core emulation via f16 accumulation worth the precision trade?

## Impact on Downstream Tasks
- perf-attn-v3.3 (P·V cooperative) may yield diminishing returns — P·V is already shuffle-based
- perf-attn-v3.4 (benchmark + iterate) should set concrete 70% target based on cuBLAS reference
