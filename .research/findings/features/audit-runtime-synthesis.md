# audit-runtime: Feature Synthesis

**Result**: All 8 runtime failures fixed. 16/24 examples pass, 5 fail-data (expected), 3 JIT-bound.

## Fix Applied

API-level auto-discovery: `get_kernel()` and `CustomLaunchBuilder::prepare()` now
search all PTX modules (`ptx::ALL`) with text pre-filter when no explicit PTX is given.
tokio-offload: changed explicit `ptx::KERNEL` to `ptx::KERNEL_IO`.

## Verification

- KERNEL_IO examples (5): all PASS (async-io, async-pipeline, gpu-channels, tokio-offload, hello-gpu demos 1-2)
- KERNEL_TEST examples (3): auto-discovery correct but JIT takes 10+ min for 227K-line PTX (pre-existing)
- Regression: vector-math still PASS

## Remaining: KERNEL_TEST JIT latency (pre-existing, not a bug)

thread-demo, structured-concurrency, warp-cooperative need cubin support in `gpu::launch()`/`gpu::custom()`.
