# nn-registry.1-4: KernelRegistry & Auto-Launch
**Cycle**: 377-380 | **Theme**: nn-registry | **Kind**: experiment + investigation | **Status**: done

## Summary
Implemented KernelRegistry that loads 23 ML-relevant kernel functions from PTX at construction.
Provides auto-config launch helpers (config_1d, config_gemm, config_layernorm, config_attention,
config_embedding, config_im2col, config_batchnorm). Kernel catalog audited: 53 total compute
kernels across 7 compute_*.rs files, 23 selected as ML-relevant.

## Findings
### Q: How many ML-relevant kernels exist?
A: 23 kernels across compute_cnn.rs (8), compute_transformer.rs (13), compute_gemm.rs (2).
The remaining 30 are test/demo/MMA-test kernels not needed for the nn API.
**Confidence**: high

### Q: What launch configs are needed?
A: 6 config types cover all ML kernels:
- config_1d(n): 256-thread blocks for element-wise ops
- config_gemm(m, n): 128-thread blocks, 32x16 output tiles
- config_layernorm(rows): 1 warp per row
- config_attention(seq_len): 1 warp per query position
- config_embedding(seq_len): 256-thread blocks per token
- config_batchnorm(n) = config_1d(n)
**Confidence**: high

## Impact on Downstream Tasks
- nn-ops can now use registry.get("kernel_name") + config helpers
- No user code needs to know PTX function names or grid/block sizes
