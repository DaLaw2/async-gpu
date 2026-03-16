# ve-run-gpt2.1: Run GPT-2 inference example
**Cycle**: 409 | **Theme**: ve-run-gpt2 | **Kind**: experiment | **Status**: partial

## Summary
Fixed flash_attention kernel launch config (missing shared memory + wrong grid dims).
GPT-2 example now runs end-to-end without crashing, but produces garbage text.

## Findings

### Q: Does gpt2-inference example run end-to-end?
A: Yes, after fixing the attention launch config. Model loads (278ms), builds (656ms),
generates 50 tokens per prompt (~790ms/token). No crashes.
**Confidence**: high

### Q: Is the output correct?
A: No. All prompts produce incoherent text (repeated tokens like "Highlands", "apolog").
The raw tests_inference.rs code produces correct output with the same model, so the
issue is in the nn API layer, not the weights or model architecture.
**Confidence**: high

## Bug Fix: flash_attention launch config
The nn ops `scaled_dot_product_attention` was launching flash_attention with:
- `grid_dim: (seq_len, 1, 1)` — WRONG (should be `(1, n_q_tiles, 1)` per head)
- `shared_mem_bytes: 0` — WRONG (needs `2 * 32 * d_head * 4`)

Fixed to match the working tests_inference.rs config. This eliminated the
CUDA_ERROR_ILLEGAL_ADDRESS crash.

## Root Cause Hypothesis for Garbage Output
The nn `matmul()` function extracts results from the GEMM output using
`d_host[r * n_pad + c]` (row-major). For some matrix dimensions, this produces
incorrect results (same bug pattern as conv2d). The Linear layer uses matmul, so
all transformer layer outputs may be corrupted.

## Open Questions
- Is the GEMM extraction correct for GPT-2's specific matrix dimensions (768×768, 768×2304)?
- Does the raw test code handle the GEMM output differently from nn ops?

## Impact on Downstream Tasks
- ve-run-gpt2.2 (cached vs non-cached) is blocked until output is correct
- Need a GEMM output layout investigation task
