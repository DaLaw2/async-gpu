# transformer-layer.5: Weight Loading Strategy Design
**Cycle**: 1 | **Theme**: transformer-layer | **Kind**: design | **Status**: done

## Summary
Decided on pre-allocated device buffers for weight loading. Model weights are uploaded via `cudarc::htod_sync_copy()` before kernel launch, and kernel receives raw pointers as launch parameters. This is simpler, faster, and leverages existing infrastructure compared to hostcall streaming.

## Findings

### Q: Is cuMemAlloc pre-allocation or hostcall streaming more aligned with the autonomy thesis?
A: Pre-allocation. For inference, weights are deterministic static inputs — not runtime decisions. The kernel's autonomy is expressed through its compute graph (LayerNorm → MHA → FFN pipeline), not through requesting weights at runtime. Hostcall streaming adds latency and complexity for no benefit in this use case.
**Confidence**: high

### Q: How to pass weight buffer pointers to the kernel?
A: Each weight tensor is a separate `CudaSlice<T>` on the host side. For kernel launch, extract raw device pointers and pass as kernel arguments. This is already how all existing kernels work (e.g., `full_gemm` receives `a_global`, `b_global`, `d_global`).
**Confidence**: high

## Design

### Buffer Layout for GPT-2 Layer 0

| Weight | Shape | Format | Size |
|--------|-------|--------|------|
| ln_1.weight (gamma) | [768] | f32 | 3 KB |
| ln_1.bias (beta) | [768] | f32 | 3 KB |
| attn.c_attn.weight | [768, 2304] | col-major f16x2 [2304][384] u32 | 3.4 MB |
| attn.c_attn.bias | [2304] | f32 | 9 KB |
| attn.c_proj.weight | [768, 768] | col-major f16x2 [768][384] u32 | 1.1 MB |
| attn.c_proj.bias | [768] | f32 | 3 KB |
| ln_2.weight (gamma) | [768] | f32 | 3 KB |
| ln_2.bias (beta) | [768] | f32 | 3 KB |
| mlp.c_fc.weight | [768, 3072] | col-major f16x2 [3072][384] u32 | 4.5 MB |
| mlp.c_fc.bias | [3072] | f32 | 12 KB |
| mlp.c_proj.weight | [3072, 768] | col-major f16x2 [768][1536] u32 | 4.5 MB |
| mlp.c_proj.bias | [768] | f32 | 3 KB |
| **Total** | | | ~13.6 MB |

### Key Decision: ADR-012
Pre-allocated device buffers. See decisions.md.

## Impact on Downstream Tasks
- **transformer-layer.3** (MHA): Use pre-allocated weight pointers.
- **transformer-layer.4** (FFN): Use pre-allocated weight pointers.
- **transformer-layer.6** (end-to-end): All weights pre-loaded before single kernel launch.
