# perf-attn-v3.3 — Cooperative P·V + Online Softmax Optimization

## Status: done
## Summary
Experimented with three optimization strategies for the Flash Attention V3 kernel on GTX 1660 (SM75).
Two approaches regressed performance; the third (skip-correction fast path + float4 output writes)
yielded a modest 1-6% improvement depending on sequence length.

## Changes Made

### Approach 1: BKV=16 (halved KV tile) — REGRESSED
- Halved BKV from 32 to 16, reducing shared memory from 16640 to 8320 bytes
- Theoretical occupancy gain: 3 -> 5 blocks/SM
- **Result**: 6-18% regression across all sizes (worst at small seq)
- **Root cause**: 2x more tile iterations (more __syncthreads, more K/V loads, more softmax/shuffle
  rounds) overwhelmed the occupancy benefit. Overhead is per-tile-constant, not per-element.

### Approach 2: Batched shuffles + fused correction FMA — REGRESSED
- Pre-shuffled all 8 p_vals per group before FMA loop (vs one shuffle per p_val)
- Fused correction into first group's FMA: `o_reg = fma(o_reg, correction, p*v)`
- **Result**: 6-11% regression vs HEAD V3.1
- **Root cause**: Batched shuffles keep all 8 p_vals live in registers simultaneously, increasing
  register pressure. The simple one-at-a-time shuffle approach lets the compiler overlap shuffle
  latency with FMA execution better. fmaf may also schedule differently than separate mul+add.

### Approach 3: Skip-correction fast path + float4 output — IMPROVED
- Added a branch to skip the 16-multiply correction loop when correction >= 0.999 (m_val unchanged)
- Replaced scalar output writes with float4 (4x fewer store transactions)
- **Result**: 1-6% improvement (larger gains at smaller seq lengths)

## Performance Results

### Baseline (HEAD V3.1)
| seq | mode | time (ms) | GFLOPS |
|-----|------|-----------|--------|
| 128 | causal | 0.075 | 337 |
| 128 | bidir | 0.108 | 464 |
| 256 | causal | 0.218 | 461 |
| 256 | bidir | 0.382 | 527 |
| 512 | causal | 0.724 | 556 |
| 512 | bidir | 1.342 | 600 |

### V3.3 (skip-correction + float4 output)
| seq | mode | time (ms) | GFLOPS | delta |
|-----|------|-----------|--------|-------|
| 128 | causal | 0.070 | 358 | +6.2% |
| 128 | bidir | 0.104 | 484 | +4.3% |
| 256 | causal | 0.211 | 477 | +3.5% |
| 256 | bidir | 0.375 | 537 | +1.9% |
| 512 | causal | 0.714 | 564 | +1.4% |
| 512 | bidir | 1.329 | 606 | +1.0% |

### BKV=16 (regressed, reverted)
| seq | mode | time (ms) | GFLOPS | delta |
|-----|------|-----------|--------|-------|
| 128 | causal | 0.091 | 276 | -18% |
| 128 | bidir | 0.117 | 428 | -8% |
| 256 | causal | 0.259 | 389 | -15% |
| 256 | bidir | 0.427 | 472 | -11% |
| 512 | causal | 0.798 | 504 | -10% |
| 512 | bidir | 1.424 | 565 | -6% |

## Key Insights

1. **Occupancy is not the bottleneck** at BKV=32. The 3 blocks/SM is sufficient for SM75.
   The kernel is compute+latency bound, not occupancy-starved.

2. **Simple shuffle patterns beat batched shuffles** on SM75. The per-shuffle-per-FMA pattern
   allows the compiler to overlap shuffle latency with FMA execution naturally.

3. **Correction skip matters more at small seq** because there are fewer tiles and each tile
   is more likely to be the first (where m_val always changes). At seq=512 with 16 tiles,
   only the first few tiles need correction.

4. **Current utilization**: 606/5000 = 12.1% of peak FP32. The gap is structural:
   - Roofline predicts 2573 GFLOPS at AI=13.4 FLOP/byte
   - Only 3 blocks/SM (384 threads vs 1024 max)
   - Memory latency hiding requires more concurrent warps
   - Fundamental limit for this thread mapping on SM75 without tensor cores

5. **The simple V3.1 kernel structure is already near-optimal** for this hardware.
   Further gains require a fundamentally different approach (different thread count,
   different tile shape, or exploiting tensor cores on newer hardware).

## Files Changed
- `crates/core/gpu-host/src/nn/ops/flash_attn_v3.cu` — skip-correction fast path, float4 output writes
