# showcase-examples.1 — Feature Example Coverage Audit

## Completed Features vs Example Status

| # | Feature | Example(s) | Status | Notes |
|---|---------|-----------|--------|-------|
| 1 | **Kernel launch** (`gpu::run`, `gpu::launch`, `gpu::custom`) | `hello-gpu`, `vector-math`, `parallel-search`, `tcp-echo`, `thread-demo` | **Covered** | All three API tiers demonstrated across multiple examples |
| 2 | **File I/O on GPU** (hostcall-based) | `hello-gpu` (Demo 2), `async-io` | **Covered** | OPEN+WRITE+READ+CLOSE sequence demonstrated |
| 3 | **Network I/O on GPU** (TCP client) | `tcp-echo` | **Covered** | GPU-initiated TCP connect/send/recv/close with host echo server |
| 4 | **std::thread on GPU** (`thread::spawn`, `JoinHandle`) | `thread-demo`, `hello-gpu` (Demo 3) | **Covered** | Standalone `thread-demo` + embedded in `hello-gpu` |
| 5 | **Cooperative compute** (`cooperative_map`, `cooperative_reduce`) | _(none)_ | **Missing** | Functions exist in `gpu-runtime/src/thread.rs` but no example uses them directly. `structured-concurrency` uses `spawn_all` (the scope-based alternative) but not the raw cooperative APIs |
| 6 | **Structured concurrency** (`BlockScope`, `GridScope`, channels) | `structured-concurrency` | **Covered** | 5 demos: producer-consumer, spawn_all, nested scopes, combined, GridScope reduce |
| 7 | **GPU async/await** (WarpFuture, async fn on GPU) | `async-pipeline`, `warp-cooperative` | **Covered** | `async-pipeline` shows real I/O pipeline with await points; `warp-cooperative` tests multi-await state machines |
| 8 | **GPU async executor** (`GpuExecutor`, task scheduling) | `gpu-channels` (Demo 3) | **Partial** | Executor is demonstrated as part of the channels example (Demo 3: 8 tasks polled to completion), but no standalone executor-only example showing pure task scheduling without channels |
| 9 | **GPU channels** (oneshot, MPSC) | `gpu-channels` | **Covered** | Oneshot (4 pairs) + MPSC (3 producers, 1 consumer) with full verification |
| 10 | **Par iter** (`GpuParallelIterator`) | _(none)_ | **Missing** | 1230-line `par_iter.rs` module with `GpuParIter`, `map`, `filter`, `fold`, `collect`, `zip`, `GpuParallelIterator` trait — zero examples anywhere. This is a kernel-side API (runs inside GPU code, not from host) |
| 11 | **Tokio integration** (`AsyncGpuRuntime`, `GpuTask`) | `tokio-offload` | **Covered** | Non-blocking kernel launch + event streaming via `next_event().await` |
| 12 | **NN inference: GPT-2** | `gpt2-inference` | **Covered** | Full GPT-2 Small (124M params) text generation with tokenizer |
| 13 | **NN inference: YOLO** | `yolo-detect` | **Covered** | YOLOv8-nano object detection with COCO classes |
| 14 | **NN inference: ResNet** | `resnet-cifar` | **Covered** | ResNet-18 inference + mini-ResNet training on CIFAR-10 |
| 15 | **GPU-RAG pipeline** | `gpu-rag` | **Covered** | Embed + cosine similarity + GPT-2 generation. Also has `--bench-fused` and `--bench-int8` sub-modes |
| 16 | **Autograd / training** | `mnist-train`, `mnist-cnn`, `cifar-train`, `gpt2-lora` | **Covered** | Multiple training examples with autograd tape + backward pass |
| 17 | **Kernel fusion** (fused GEMM+bias+GELU) | `gpu-rag --bench-fused` | **Partial** | Only accessible as a benchmark sub-mode of `gpu-rag`, not a standalone example demonstrating the feature |
| 18 | **INT8 quantized inference** | `gpu-rag --bench-int8` | **Partial** | Only accessible as a benchmark sub-mode of `gpu-rag`, not a standalone example |
| 19 | **ONNX inference** | `diff-physics --test-onnx`, `resnet-cifar --onnx` | **Partial** | ONNX is only accessible as a CLI sub-mode inside other examples, not standalone |
| 20 | **Differentiable physics** | `diff-physics` | **Covered** | Spring-mass simulation with analytical backward + gradient optimization |
| 21 | **Dynamic control flow** | `dynamic-control` | **Covered** | Variable-length generation, top-k sampling, per-sample early stopping |
| 22 | **Monte Carlo simulations** | `monte-carlo` | **Covered** | xoshiro256++ PRNG, Pi estimation, Black-Scholes pricing |
| 23 | **Graph algorithms** | `graph-algorithms` | **Covered** | CSR representation, BFS, PageRank with GPU acceleration |
| 24 | **GPU benchmarks** (GFLOPS/GB/s) | `benchmark` | **Covered** | SGEMM vs cuBLAS, LayerNorm, Softmax throughput |
| 25 | **Pure compute** (SAXPY, dot product, softmax) | `vector-math` | **Covered** | Three compute patterns with CPU-GPU cooperation |
| 26 | **Warp-parallel search** | `parallel-search` | **Covered** | 32-lane grep with bulk I/O + parallel pattern matching |

## Summary

### Fully Covered (20 features)
Kernel launch, File I/O, Network I/O, std::thread, Structured concurrency, GPU async/await, GPU channels, Tokio integration, GPT-2, YOLO, ResNet, GPU-RAG, Autograd/training, Diff-physics, Dynamic control flow, Monte Carlo, Graph algorithms, Benchmarks, Pure compute, Warp-parallel search.

### Partially Covered (3 features)
- **GPU executor**: Embedded in `gpu-channels` Demo 3, not standalone
- **Kernel fusion**: Hidden behind `gpu-rag --bench-fused` flag
- **INT8 quantized inference**: Hidden behind `gpu-rag --bench-int8` flag
- **ONNX inference**: Hidden behind `--test-onnx` / `--onnx` flags in other examples

### Missing (2 features)
- **Cooperative compute** (`cooperative_map`, `cooperative_reduce`): No example at all. The API exists in `gpu-runtime/src/thread.rs` but is only referenced in docs/comments of the scope module (which recommends `spawn_all` instead). May be intentional — `spawn_all` inside `BlockScope` is the preferred API.
- **Par iter** (`GpuParallelIterator`): Zero examples despite a substantial 1230-line implementation. This is a kernel-side API, so examples would need to be GPU kernel code that uses `data.par_iter().map(...).collect()`.

### Key Observations

1. **Hostcall examples** use the split host/kernel architecture (separate `host/` and `kernel/` directories with their own Cargo.toml). The std examples are single-crate.

2. **Facade gap**: The `async-gpu` facade crate does NOT re-export structured concurrency types (BlockScope, GridScope), channels (OneshotSlot, MpscChannel), or executor (GpuExecutor). These are only accessible through internal crates (`gpu-runtime`). The `structured-concurrency` and `gpu-channels` examples use `async_gpu::gpu` for launching but don't import the kernel-side types through the facade.

3. **ONNX, kernel fusion, and INT8** are buried as CLI sub-modes rather than being discoverable standalone examples. A developer browsing the examples directory would not discover these features.

4. **Par iter** is the biggest gap — it's a complete, sophisticated API (map, filter, fold, collect, zip, min, max, sum, product, count, any, all, find, position) with zero usage examples anywhere in the repository.
