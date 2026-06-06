# audit-runtime.1: Runtime audit — run all examples on GPU

**Status**: DONE
**Date**: 2026-06-06
**Machine**: GTX 1660 (6 GB), CUDA 13.3

## Summary

Ran all 24 compilable examples on GPU. **11 pass**, **5 fail-data** (missing datasets/models),
**8 fail-runtime** (PTX symbol mismatch — all the same root cause).

The dominant failure mode is a **single bug**: `ptx::KERNEL` aliases `KERNEL_COMPUTE`, but
8 examples reference kernel functions that live in `KERNEL_IO` or `KERNEL_TEST`. The
`gpu::run_with_output()`, `gpu::launch()`, and `gpu::custom()` APIs all default to
`ptx::KERNEL` (= `KERNEL_COMPUTE`), so any example using hostcall I/O kernels, threading
kernels, channel kernels, or cooperative compute kernels fails with
`CUDA_ERROR_NOT_FOUND "named symbol not found"`.

## Findings

### hostcall/ examples (10)

| Example | Status | Notes |
|---|---|---|
| hello-gpu | fail-runtime | Needs `hostcall_print_hello` (KERNEL_IO), `thread_spawn_test` (KERNEL_TEST); default PTX is KERNEL_COMPUTE |
| vector-math | **pass** | SAXPY, dot product, softmax all pass. Own kernel PTX via build.rs |
| parallel-search | **pass** | 32-lane warp grep, verification exact match. Own kernel PTX via build.rs |
| async-io | fail-runtime | Needs `hostcall_file_test`, `pipelined_compute` (KERNEL_IO); default PTX is KERNEL_COMPUTE |
| async-pipeline | fail-runtime | Needs `branching_pipeline`, `pipelined_compute` (KERNEL_IO); default PTX is KERNEL_COMPUTE |
| gpu-channels | fail-runtime | Needs `channel_oneshot_demo`, `channel_mpsc_demo`, `executor_demo` (KERNEL_IO); default PTX is KERNEL_COMPUTE |
| structured-concurrency | fail-runtime | Needs `sc_producer_consumer`, `sc_cooperative_parallel`, etc. (KERNEL_TEST); default PTX is KERNEL_COMPUTE |
| tcp-echo | **pass** | GPU TCP connect/send/read/close, echo verified. Own kernel PTX via build.rs |
| tokio-offload | fail-runtime | Needs `hostcall_print_hello` (KERNEL_IO); loads `ptx::KERNEL` (KERNEL_COMPUTE) |
| warp-cooperative | fail-runtime | Needs `cooperative_compute_test`, `cooperative_map_test`, etc. (KERNEL_TEST); default PTX is KERNEL_COMPUTE |

### std/ examples (14)

| Example | Status | Notes |
|---|---|---|
| benchmark | **pass** | SGEMM ~90% cuBLAS, flash attention, conv2d, GPT-2 profiling. Full output verified |
| cifar-train | fail-data | Missing `models/cifar10/data_batch_1.bin`. Needs `scripts/download-cifar10.sh` |
| diff-physics | **pass** | 2D spring-mass, 50 optimization steps, loss 7.07->1.55, PASSED |
| dynamic-control | **pass** | Variable-length GPT-2 generation, stochastic sampling, temperature sweep, early-exit inference. All demos complete (~4 min total) |
| gpt2-inference | **pass** | GPT-2 Small 124M params loaded, 3 prompts generated, cached/non-cached outputs match |
| gpt2-lora | fail-data | Missing `models/wikitext2/train.txt`. Model loads fine but training data absent |
| gpu-rag | **pass** | RAG pipeline: embed + search + generate. 1030 chunks, GPT-2 generation, PASSED |
| graph-algorithms | **pass** | BFS (2.49x GPU speedup), PageRank (4.19x GPU speedup), GPU vs CPU verification PASS |
| mnist-cnn | fail-data | Missing `models/mnist/` directory |
| mnist-train | fail-data | Missing `models/mnist/` directory |
| monte-carlo | **pass** | Pi estimation (42.9x throughput speedup), Black-Scholes (491.4x speedup), all tests passed |
| resnet-cifar | **pass** | ResNet-18 forward pass valid, no NaN, PASSED (10% accuracy = random weights, expected) |
| thread-demo | fail-runtime | Needs `thread_spawn_test`, `thread_reuse_test` (KERNEL_TEST); default PTX is KERNEL_COMPUTE |
| yolo-detect | fail-data | Missing `models/yolov8n.safetensors` |

## Root Cause Analysis

### PTX symbol mismatch (8 examples)

All 8 fail-runtime examples share the same root cause:

```
ptx::KERNEL = ptx::KERNEL_COMPUTE  (line 163 of gpu-host/src/lib.rs)
```

The `gpu::run()`, `gpu::run_with_output()`, `gpu::launch()`, and `gpu::custom()` APIs
all default to loading `ptx::KERNEL` when no `.ptx()` override is provided. But:

- **KERNEL_IO** (`kernel_io.ptx`) contains: `hostcall_print_hello`, `hostcall_file_test`,
  `branching_pipeline`, `pipelined_compute`, `channel_oneshot_demo`, `channel_mpsc_demo`,
  `executor_demo`
- **KERNEL_TEST** (`kernel_test.ptx`) contains: `thread_spawn_test`, `thread_reuse_test`,
  `cooperative_compute_test`, `cooperative_map_test`, `cooperative_reduce_test`,
  `cooperative_map_ext_test`, `cooperative_matmul_test`, `sc_producer_consumer`,
  `sc_cooperative_parallel`, `sc_nested_scopes`, `sc_combined_demo`, `sc_grid_reduce`
- **KERNEL_COMPUTE** (`kernel_compute.ptx`) contains: compute kernels (GEMM, conv, etc.)

The 3 working hostcall examples (vector-math, parallel-search, tcp-echo) have their own
`build.rs` that compiles a dedicated kernel crate to PTX and pass it via
`include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))` + `.ptx(KERNEL_PTX)`.

### Fix options (not implemented — investigation only)

1. **Per-example fix**: Each failing example should specify `.ptx(ptx::KERNEL_IO)` or
   `.ptx(ptx::KERNEL_TEST)` instead of relying on the default.
2. **API-level fix**: The `get_kernel()` function in `gpu.rs` could try multiple PTX
   modules (KERNEL_COMPUTE, KERNEL_IO, KERNEL_TEST) until the symbol is found.
3. **Unified PTX**: Merge all PTX modules into a single file (but JIT time would be huge).

### Missing data files (5 examples)

These are expected failures — the examples need datasets/models that must be downloaded:

- `cifar-train`: `models/cifar10/` (via `scripts/download-cifar10.sh`)
- `mnist-cnn`, `mnist-train`: `models/mnist/` (via `scripts/download-mnist.sh`)
- `gpt2-lora`: `models/wikitext2/train.txt`
- `yolo-detect`: `models/yolov8n.safetensors` (via `scripts/export_yolo.py`)

## Open Questions

1. Should `ptx::KERNEL` be changed to point to a module that includes more kernels,
   or should each example explicitly select its PTX module?
2. The `dynamic-control` example takes ~4 minutes to run — is this acceptable for a demo?
3. `resnet-cifar` shows 10% accuracy with random weights — should it load pretrained weights?
4. `gpt2-inference` cached path is 4x slower than non-cached (126ms/tok vs 32ms/tok) —
   is the KV cache implementation correct or is this a performance regression?
