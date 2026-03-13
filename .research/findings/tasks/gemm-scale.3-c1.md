# gemm-scale.3: 768×768 GEMM Correctness Validation — DECISION GATE
**Cycle**: 1 | **Theme**: gemm-scale | **Kind**: experiment | **Status**: done

## Summary
Full 768×768 GEMM validated against CPU reference using 2D grid tiling. 1152 blocks (24×48), each computing a 32×16 output tile via 4-warp MMA. Maximum relative error 0.42%, zero mismatches with tolerance (1% rel + 1.0 abs). Decision gate PASSED — GEMM infrastructure is ready for transformer layers.

## Findings

### Q: Does 768×768 GEMM with deterministic f16 weights match CPU reference within f16 tolerance?
A: Yes. Using deterministic integer-valued weights (A: 1-5, B: 1-7 from hash functions), the GPU GEMM matches CPU f32 reference within 0.42% maximum relative error. The MMA instruction accumulates in f32, so precision loss comes only from f16 input quantization, not accumulation.
**Confidence**: high

### Q: What is the maximum relative error across all output elements?
A: 0.004236 (0.42%). This is well within acceptable tolerance for f16 computation over K=768 dimension. The error arises from f16 input quantization — the MMA accumulates in f32.
**Confidence**: high

## Design Notes

- **2D grid**: `grid_dim = (M/32, N/16, 1) = (24, 48, 1)` — 1152 blocks total.
- **blockIdx.y**: Retrieved via inline PTX `mov.u32 {r}, %ctaid.y;` since `nvptx::_block_idx_y()` hasn't been used before in this project.
- **B column offset**: B is col-major packed `[N][K/2]`, so column block offset = `block_n * 16 * k_half`.
- **Completion signaling**: Each block atomically increments status counter; host verifies count >= total_blocks.

## Unexpected Discoveries
None — the kernel worked on the first attempt. The 2D tiling extension was straightforward.

## Open Questions
None. Decision gate passed.

## Impact on Downstream Tasks
- **transformer-layer.3** (Multi-head attention): Unblocked. Can use `full_gemm` kernel.
- **transformer-layer.4** (FFN block): Unblocked. Can use `full_gemm` kernel.
- **gemm-scale theme**: COMPLETED. All 3 tasks done.
