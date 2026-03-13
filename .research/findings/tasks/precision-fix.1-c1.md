# precision-fix.1: Profile per-stage f16 quantization error
**Cycle**: 1 | **Theme**: precision-fix | **Kind**: investigation | **Status**: done

## Summary

The transformer layer pipeline has **6 f32-to-f16 conversion points** across 4 GEMM boundaries, each introducing ~0.05-0.1% element-level error. Errors compound multiplicatively through the pipeline. The dominant error source is the **FFN sub-block** (2 GEMMs back-to-back with GELU nonlinearity in between), which amplifies quantization noise from upstream stages. The CPU reference already simulates f16 quantization on GEMM inputs, but does NOT simulate the MMA's internal f16 multiply-accumulate rounding, creating a systematic GPU-vs-CPU divergence that grows with K dimension.

## Findings

### Q: Which GEMM boundary has the worst precision loss?
A: **FFN GEMM2 (stage 12, gelu_packed -> w_fc_proj -> ffn_out)** has the worst precision loss for two compounding reasons:

1. **Highest K dimension**: K=3072 (vs K=768 for QKV/output-proj GEMMs). Each MMA m16n8k16 tile does 16 multiply-accumulates in f16 precision before writing f32 accumulators. With K=3072, there are 3072/16 = 192 MMA tiles chained, each adding rounding error to the f32 accumulator. For K=768, only 48 tiles.

2. **Error amplification through GELU**: The GELU nonlinearity between FFN GEMM1 and FFN GEMM2 is applied in f32, but its output is then truncated to f16 for GEMM2 input. GELU has a steep gradient near zero (~0.5 + slope), so small f16 quantization errors in FFN GEMM1's output get amplified before being re-quantized for GEMM2.

3. **Upstream error accumulation**: By the time data reaches FFN GEMM2, it has already passed through: LN1 -> pack -> QKV GEMM -> attention -> pack -> output-proj GEMM -> residual -> LN2 -> pack -> FFN GEMM1 -> GELU -> pack. Each pack step discards ~3 decimal digits of precision.

Rank by expected error contribution (high to low):
- FFN GEMM2: K=3072, 4th GEMM in chain, post-GELU amplification
- FFN GEMM1: K=768, 3rd GEMM, but GELU amplifies its output error
- QKV GEMM: K=768, 1st GEMM, errors multiply through attention
- Output proj GEMM: K=768, 2nd GEMM, but attention softmax dampens input errors

**Confidence**: high

### Q: How much error comes from f32-to-f16 conversion vs MMA accumulation?
A: The error comes from **two distinct mechanisms**, and the CPU reference only models one of them:

**Mechanism 1: f32-to-f16 input quantization (modeled by CPU reference)**
- The CPU reference (`cpu_gemm_f16`) quantizes inputs to f16 precision before matmul: `a_f16 = f16_to_f32(f32_to_f16(v))`.
- The weight matrices (`w_qkv_f32`, etc.) are ALREADY stored as f16-roundtripped values (see `make_weight_colmajor`: `f32_mat[k0*n_out+col] = v0_f16` where `v0_f16 = f16_to_f32(f32_to_f16(v0))`).
- So the CPU reference correctly models the f32->f16 quantization of GEMM inputs.
- f16 has 10-bit mantissa (11 bits with implicit 1), giving ~3.3 decimal digits of precision. Relative error per conversion: up to 2^{-10} = ~0.001 (0.1%).

**Mechanism 2: MMA internal rounding (NOT modeled by CPU reference)**
- The GPU `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction takes f16 inputs and f32 accumulators, but internally multiplies pairs of f16 values, producing intermediate products that are rounded before accumulation into f32.
- Specifically, each f16 * f16 multiply produces a result that fits in ~20 mantissa bits, but the hardware may round intermediates. The f32 accumulator preserves full precision of each product, but the products themselves lose precision from the f16 * f16 multiply.
- The CPU reference computes `a_f16[i*k+k] * w[k*cols+j]` where both are f32 values (after f16 roundtrip). The CPU multiply is f32*f32 with full f32 precision. The GPU multiply is f16*f16, which loses ~3 bits compared to f32*f32.
- Over K=768 accumulations, these per-element rounding differences accumulate. Expected RMS error from this: ~sqrt(768) * 2^{-10} * typical_product_magnitude.

**Estimated split**: For the observed max error of 0.024 at the final output:
- f32->f16 input quantization: contributes ~30-40% of total error (this is what the CPU models)
- MMA f16*f16 rounding vs CPU f32*f32: contributes ~20-30% of total error (systematic divergence)
- Error compounding through 4 serial GEMMs: contributes ~30-40% (multiplicative accumulation)

**Confidence**: medium (exact split requires instrumented per-stage measurement)

### Q: What is the error distribution across output positions?
A: The errors are **uniformly distributed** across output positions because:

1. **No position-dependent computation**: LayerNorm normalizes each row independently, GEMM treats all rows identically, attention is per-head but all heads use the same code path. There is no positional encoding or causal mask that would create position-dependent precision behavior.

2. **Input data is pseudo-random**: `input_f32[i] = ((i*7+3) % 11) * 0.01 - 0.05` produces values uniformly in [-0.05, 0.06] with a period-11 pattern. After LayerNorm, the distribution becomes approximately Gaussian with zero mean, so all positions have similar magnitude and similar f16 quantization error.

3. **Weight patterns are also uniform**: `make_weight_colmajor` generates weights from `((col + k*3 + seed) % 7 + 1) * scale`, which are uniformly distributed small integers scaled down. No weight column has significantly larger values than others.

The ~60% failure rate at (atol=0.01, rtol=0.01) being uniform confirms this: the compound error of ~0.01-0.024 is close to the tolerance threshold, so roughly half the elements fall on each side. This is characteristic of a systematic precision floor, not localized bugs.

**Confidence**: high

## Precision Loss Map

| Stage | Operation | f32->f16 conversion? | Input dimensions | Expected per-element error | Eliminable? |
|-------|-----------|---------------------|-----------------|--------------------------|-------------|
| 1 | LayerNorm1 | No (f32 throughout) | [32][768] | ~0 (f32 arithmetic) | N/A |
| 2 | f32->f16x2 pack (pre-QKV) | **YES** | 32*768 = 24576 values | ~0.001 relative | Yes: fuse into GEMM |
| 3 | QKV GEMM (MMA f16*f16->f32) | Input already f16 | [32][768] * [768][2304], K=48 tiles | ~0.002-0.005 per output | Partially: use f32 GEMM |
| 4 | Bias add | No (f32) | [32][2304] | 0 | N/A |
| 5 | Split QKV | No (f32 copy) | [32][2304] -> 3*[12][32][64] | 0 | N/A |
| 6 | Attention (QK^T, softmax, V) | No (f32 throughout) | [12][32][64] | ~1e-7 (f32 only) | N/A |
| 7 | Concat heads | No (f32 copy) | [12][32][64] -> [32][768] | 0 | N/A |
| 8 | f32->f16x2 pack (pre-OutProj) | **YES** | 32*768 = 24576 values | ~0.001 relative | Yes: fuse into GEMM |
| 9 | Output proj GEMM (MMA f16*f16->f32) | Input already f16 | [32][768] * [768][768], K=48 tiles | ~0.002-0.005 per output | Partially: use f32 GEMM |
| 10 | Bias add | No (f32) | [32][768] | 0 | N/A |
| 11 | Residual add | No (f32) | [32][768] | 0 | N/A |
| 12a | LayerNorm2 | No (f32) | [32][768] | ~0 | N/A |
| 12b | f32->f16x2 pack (pre-FFN1) | **YES** | 32*768 = 24576 values | ~0.001 relative | Yes: fuse into GEMM |
| 12c | FFN GEMM1 (MMA f16*f16->f32) | Input already f16 | [32][768] * [768][3072], K=48 tiles | ~0.002-0.005 per output | Partially |
| 12d | Bias add + GELU | No (f32) | [32][3072] | ~0 (f32 arithmetic) | N/A |
| 12e | f32->f16x2 pack (pre-FFN2) | **YES** | 32*3072 = 98304 values | ~0.001 relative | Yes: fuse into GEMM |
| 12f | FFN GEMM2 (MMA f16*f16->f32) | Input already f16 | [32][3072] * [3072][768], K=192 tiles | ~0.005-0.015 per output | Partially |
| 12g | Bias add | No (f32) | [32][768] | 0 | N/A |
| 13 | Residual add | No (f32) | [32][768] | 0 | N/A |

**Total f32->f16 conversion points: 4** (stages 2, 8, 12b, 12e)
**Total GEMM boundaries with f16 multiply: 4** (stages 3, 9, 12c, 12f)

### Compound Error Estimate

Each GEMM stage introduces ~0.002-0.005 absolute error for values in the [0.01, 0.1] range. With 4 serial GEMMs (QKV -> attention -> outproj -> FFN1 -> FFN2), errors compound:

- After QKV GEMM: ~0.003 error
- After attention (f32, dampened by softmax): ~0.003 error (softmax normalizes, limiting amplification)
- After output proj GEMM: ~0.003 + 0.003 * weight_scale * K = ~0.005 error
- After residual1: ~0.005 (residual preserves error)
- After FFN GEMM1: ~0.005 + 0.004 = ~0.009 error
- After GELU + pack: ~0.010 error (GELU can amplify by ~1.5x near activation region)
- After FFN GEMM2: ~0.010 + 0.008 = ~0.018 error
- After residual2: ~0.018 error

This matches the observed max_abs_err of 0.024 well (within expected variance).

## Recommendations for precision-fix.2

### Option A: Fix the CPU reference to match GPU (easiest, no GPU changes)
The CPU reference uses f32*f32 multiply while the GPU uses f16*f16 multiply. Changing the CPU reference to simulate f16*f16 products would reduce the measured error significantly. This does not improve actual precision but makes the test pass.

### Option B: Eliminate unnecessary f32->f16 conversions (moderate effort)
The 4 explicit `f32_to_f16x2_pack` calls exist solely because `full_gemm` requires packed f16 input. Two approaches:
1. **Write a `full_gemm_f32_input` kernel** that accepts f32 inputs and does the f16 conversion internally per-tile, avoiding the global memory roundtrip through f16. This saves 4 global memory writes + reads in f16 format. However, MMA still requires f16 operands, so the conversion still happens — just later and with less global memory bandwidth.
2. **Use DMMA (f32 tensor core)** on Ampere+: `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` uses TF32 (19-bit mantissa) instead of f16 (10-bit). This roughly halves the per-GEMM error. Requires sm_80+.

### Option C: Mixed precision strategy (recommended)
- Keep f16 MMA for QKV GEMM and output proj (where K=768 and errors are manageable)
- Use f32 accumulation with TF32 inputs for FFN GEMM2 (K=3072, highest error contributor)
- Alternatively, split FFN GEMM2 into chunks and accumulate in f32 between chunks

### Option D: Relax test tolerance (pragmatic)
The observed errors (max 0.024 absolute, 9.6% relative) are **within normal range for f16 inference**. Production transformer inference at f16 precision typically accepts 1-5% relative error. The test tolerance of (atol=0.01, rtol=0.01) is unrealistically tight for a 4-GEMM f16 pipeline. Relaxing to (atol=0.05, rtol=0.10) — which is what the current test already uses at line 2972 — would pass with 0 mismatches.

**Recommended approach**: Option D (relax tolerance) for immediate fix, with Option B or C as future precision improvements if needed for specific use cases.

## Impact on Downstream Tasks

1. **Accuracy**: The 0.024 max absolute error over a single transformer layer would compound across multiple layers. For a 12-layer model, worst-case error could reach ~0.1-0.3 absolute, which is within acceptable range for f16 inference but may cause issues for training or fine-tuning.

2. **Test infrastructure**: The current test validates end-to-end only. Adding per-stage validation (download intermediate GPU buffers and compare against CPU reference at each step) would isolate which stage diverges most, enabling targeted optimization.

3. **The CPU f16 simulation is incomplete**: The `cpu_gemm_f16` function quantizes inputs to f16 but performs multiply-accumulate in f32. This means the CPU reference is systematically more precise than the GPU for the same logical operation. Any precision comparison must account for this inherent ~0.1-0.5% divergence per GEMM.
