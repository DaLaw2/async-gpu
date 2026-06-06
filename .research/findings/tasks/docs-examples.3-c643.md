# docs-examples.3: Create examples for auto-fusion, par_iter, gpu_test

## Status: DONE

## Summary

Created three standalone examples demonstrating post-cycle-607 features:

### 1. auto-fusion (examples/std/auto-fusion/)
- Flat std-style example (Cargo.toml + src/main.rs + README.md)
- Uses gpu-host with `nn` and `cublas` features
- Five demos: GPT-2 block fusion detection, elementwise chain detection,
  fused kernel JIT compilation + GPU execution, fused vs unfused performance
  comparison, fan-out detection
- Host side compiles clean, exercises FusionOptimizer, FusionPlan, FusionCodegen

### 2. par-iter (examples/hostcall/par-iter/)
- Host/kernel split structure matching hello-gpu/vector-math pattern
- Six demos: map+collect, map+sum, enumerate+map+collect, zip+map+collect,
  filter+map+sum, chained map fusion proof
- Host loads pre-compiled kernel PTX (KERNEL_STD) containing par_iter_demo kernels
- Kernel crate is documentation-only (actual kernels in gpu-kernel-compute)
- Host side compiles clean

### 3. gpu-test (examples/std/gpu-test/)
- Flat std-style example showing #[gpu_test] macro usage
- Binary prints usage documentation and API patterns
- Cargo.toml has gpu-test-macro as dev-dependency for test usage
- Host side compiles clean

## Verification

All three host sides pass `cargo check` with zero warnings.
Kernel side not compiled (requires full pipeline) — as expected per task spec.
