# audit-runtime: Feature Synthesis

**Result**: 11/24 pass, 8 fail-runtime (1 root cause), 5 fail-data

## Key Finding

One bug causes all 8 runtime failures: `ptx::KERNEL` = `KERNEL_COMPUTE`,
but 8 examples need symbols from `KERNEL_IO` or `KERNEL_TEST`. APIs
default to `KERNEL_COMPUTE`; examples with own `build.rs` + `.ptx()` work.

## Affected Examples

- hello-gpu, async-io, async-pipeline, gpu-channels, tokio-offload (KERNEL_IO)
- thread-demo, structured-concurrency, warp-cooperative (KERNEL_TEST)

## Fix

Per-example: `.ptx(ptx::KERNEL_IO)` or `.ptx(ptx::KERNEL_TEST)`.
Alternative: `get_kernel()` auto-searches multiple modules.

## Data-Missing (expected)

cifar-train, mnist-cnn, mnist-train, gpt2-lora, yolo-detect — need external data.

## Passing (11)

benchmark, diff-physics, dynamic-control, gpt2-inference, gpu-rag,
graph-algorithms, monte-carlo, resnet-cifar, vector-math, parallel-search, tcp-echo
