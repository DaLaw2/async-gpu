# Theme Synthesis: fusion-codegen — Fused Kernel Codegen

## Status: active (2/2 tasks done — investigation + experiment complete)

## What exists

`FusionCodegen` in `crates/core/gpu-host/src/nn/fusion.rs`:
- NVRTC-based codegen engine generating CUDA C from elementwise op chains
- Supported ops: BiasAdd, ElemAdd, Gelu, Relu, Silu, Sigmoid
- Float4-vectorized main path + scalar tail for non-aligned sizes
- Thread-safe `Mutex<HashMap>` cache keyed by op-chain hash + n_cols params
- GPU-verified: fused output matches unfused within f32 tolerance (1e-4 to 1e-6)

## Integration gap

The codegen engine is standalone — it compiles and runs fused kernels, but is
not yet wired into the autograd tape execution path. The `FusionOptimizer`
detects `ElementwiseChain` groups, and `FusionCodegen` can compile them, but
no orchestration layer connects detection to execution during a forward pass.
This is the remaining work for the theme (Phase 3 integration in the design).

## Key constraint

cudarc 0.12 `load_ptx` requires `&'static str` for function names. All fused
kernels use the fixed name `"fused_kernel"`, differentiated by module name.

## Next steps

- Wire `FusionCodegen` into traced forward-pass replay loop + benchmark on real model
- Consider BiasAdd float4-vectorized path when `n_cols % 4 == 0`
