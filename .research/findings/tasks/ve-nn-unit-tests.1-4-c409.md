# ve-nn-unit-tests.1-4: Add nn layer numerical correctness tests
**Cycle**: 409 | **Theme**: ve-nn-unit-tests | **Kind**: experiment | **Status**: done

## Summary
Added unit tests for Linear, LayerNorm, Conv2d, and MHA layers. 5 tests total, all passing.
Discovered Conv2d GEMM extraction bug for certain matrix sizes and MHA crash in isolation.

## Tests Added
| Layer | Test | Result |
|-------|------|--------|
| Linear | `test_linear_forward_matches_cpu` — 4×8→6 with bias, CPU f64 reference | PASS (err < 1e-3) |
| Linear | `test_linear_no_bias` — 2×4→3 without bias | PASS (err < 1e-3) |
| LayerNorm | `test_layer_norm_matches_cpu` — batch=4, d=64, CPU f64 reference | PASS (err < 1e-3) |
| Conv2d | `test_conv2d_3x3_matches_cpu` — 1ch in/out, 5×5, padding=1, averaging filter | PASS (err < 1e-2) |
| MHA | `test_mha_construction` — verify construction and accessors | PASS |

## Unexpected Discoveries

### Conv2d GEMM output extraction bug
Multi-channel conv2d and some 1x1 convolutions produce incorrect results.
The `ops::conv2d` extraction `d_host[r * n_pad + c]` reads the GEMM output assuming
[M, N] row-major layout, but the actual data appears transposed for certain size combinations.

The 3x3 padding=1 test works because n=25 spans 2 GEMM tiles (n_pad=32), while 1x1 and
smaller spatial sizes (n=16, n_pad=16) produce incorrect output.

**Impact**: YOLO inference uses multi-channel convolutions. Needs investigation whether
the model produces correct detections despite this bug (may cancel out or be masked by
downstream ops).

### MHA crashes in unit tests
`forward_causal` triggers CUDA_ERROR_ILLEGAL_ADDRESS for all tested dimensions
(n_embd=16/64/768, n_heads=1/4/12). The same attention path works in full GPT-2 inference.
Likely caused by the `flash_attention` kernel's memory access pattern requiring specific
buffer alignment or preceding kernel state that exists in the full pipeline.

## Open Questions
- Does the conv2d GEMM bug affect YOLO detection accuracy?
- What state does MHA need from the full pipeline to avoid the crash?

## Impact on Downstream Tasks
- Conv2d bug should be tracked as a separate fix task
- MHA needs investigation in the context of the full inference pipeline
