# Theme Synthesis: fusion-codegen — Fused Kernel Codegen

## Status: active (3/3 core tasks done)

## What exists
`FusionCodegen` in `crates/core/gpu-host/src/nn/fusion.rs`:
NVRTC codegen engine generating CUDA C from elementwise op chains (7 ops:
BiasAdd, ElemAdd, ElemMul, Gelu, Relu, Silu, Sigmoid). Float4-vectorized
main path + scalar tail; register-only intermediates (no global memory
round-trips); vectorized bias loads when `n_cols % 4 == 0`. Thread-safe
cache keyed by op-chain hash + n_cols. GPU-verified: 14 tests passing.

## Performance
Fused vs unfused (ElemMul+ElemAdd+Gelu, 1M elements, GTX 1660):
**2.05x speedup** (101.7 us vs 208.8 us). Meets epic >= 2x target.

## Integration gap
Engine is standalone. `FusionOptimizer` detects `ElementwiseChain` groups
and `FusionCodegen` can compile them, but no orchestration layer connects
detection to execution during a forward pass.

## Constraints & next steps
- cudarc 0.12: all fused kernels use `"fused_kernel"` func name (differentiated by module)
- `ElemMul` backward not yet implemented (stub only)
- Wire `FusionCodegen` into traced forward-pass replay loop
