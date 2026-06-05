# async_gpu — Rust Async/Await on NVIDIA GPUs

[![CI](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

**What if the GPU could drive its own computation?** Open files, read data, branch on results, loop until convergence, write output — all from GPU code, with zero CPU orchestration between steps.

async_gpu makes this real: **Rust async/await running natively on NVIDIA GPUs**, with a custom rustc MIR pass that turns standard `async fn` into warp-cooperative state machines — and GPU compute kernels powerful enough to run **GPT-2 inference in 25ms** (8.8x optimized), **YOLOv8-nano object detection**, **graph algorithms** (BFS, PageRank), and **Monte Carlo simulations** (129x throughput). Custom SGEMM at **90% of cuBLAS**, Flash Attention V3 at **47-60% of cuDNN FA2**.

```rust
// GPU kernel — looks like normal Rust, runs on GPU
#[no_mangle]
pub unsafe extern "gpu-kernel" fn matmul_pipeline(buf: *mut u8, result: *mut u32) {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::thread;

    // Read matrices from files — real std::fs on GPU
    let a = read_matrix(File::open("a.bin").unwrap());  // M×K
    let b = read_matrix(File::open("b.bin").unwrap());  // K×N

    // Matrix multiply — all warps cooperate in parallel
    let mut c = vec![0.0f32; a.rows * b.cols];
    thread::cooperative(|| {
        let wid = thread::current_id() as usize;
        let n_warps = thread::available_parallelism() + 1;
        for row in (wid..a.rows).step_by(n_warps) {
            for col in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..a.cols { sum += a[(row, k)] * b[(k, col)]; }
                c[row * b.cols + col] = sum;
            }
        }
    });

    // Write result — same std::fs, back to a file
    File::create("c.bin").unwrap().write_all(as_bytes(&c)).unwrap();
    println!("[GPU] {}×{} matmul complete", a.rows, b.cols);
}
```

```rust
// Host side — one line launches the entire pipeline
fn main() -> async_gpu::Result<()> {
    async_gpu::gpu::run("matmul_pipeline")
}
```

Kernel entry uses `extern "gpu-kernel"` — no custom attribute macros needed. A **custom rustc MIR pass** auto-applies to all `async fn` on the `nvptx64` target, inserting `bar.warp.sync` + `shfl.sync` at every `.await` point for warp convergence. Standard Rust syntax, standard `Future` trait.

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) with nightly toolchain: `rustup toolchain install nightly-2026-06-03`
- nvptx64 target: `rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-06-03`
- Rust nightly src (for `-Zbuild-std`): `rustup component add rust-src --toolchain nightly-2026-06-03`
- NVIDIA GPU (SM 70+) with CUDA driver (runtime driver sufficient; CUDA toolkit optional)

### Run an Example

Each example is self-contained with automated PTX compilation via `build.rs`:

```bash
git clone https://github.com/DaLaw2/async-gpu.git
cd async-gpu

# Hello GPU — GPU print, file I/O, thread::spawn (gpu:: one-liner API)
cargo run --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml

# Thread Demo — std::thread::spawn on GPU, join results
cargo run --manifest-path examples/std/thread-demo/Cargo.toml

# Vector Math — SAXPY, dot product, softmax (gpu::custom() builder API)
cargo run --manifest-path examples/hostcall/vector-math/host/Cargo.toml

# Structured Concurrency — block_scope, scoped spawn, oneshot channels
cargo run --release --manifest-path examples/hostcall/structured-concurrency/Cargo.toml

# GPU Channels — MPSC channels + GpuExecutor multi-task scheduling
cargo run --release --manifest-path examples/hostcall/gpu-channels/Cargo.toml

# Warp Cooperative — cooperative compute, all warps process in parallel
cargo run --release --manifest-path examples/hostcall/warp-cooperative/Cargo.toml
```

<details>
<summary>All examples</summary>

| Example | Description | Toolchain |
|---------|-------------|-----------|
| **Hostcall examples** (`examples/hostcall/`) | | |
| `hello-gpu` | GPU print, file I/O, thread::spawn (`gpu::run_with_output` API) | Stock nightly |
| `async-pipeline` | Warp-cooperative async pipelines (`gpu::run_with_output` API) | Patched rustc |
| `async-io` | Multi-file write pipeline + read-transform-write | Stock nightly |
| `parallel-search` | 32-lane GPU grep with `shfl.sync` warp reduction (`gpu::custom` API) | Stock nightly |
| `vector-math` | SAXPY, dot product, softmax (`gpu::custom` builder API) | Stock nightly |
| `tcp-echo` | GPU-initiated TCP networking (`gpu::custom` + hostcall) | Stock nightly |
| `tokio-offload` | Async kernel launch from tokio runtime | Stock nightly |
| `structured-concurrency` | Block-scoped spawn, oneshot channels, shared memory (`gpu::custom` API) | Stock nightly |
| `gpu-channels` | MPSC channels + `GpuExecutor` multi-task scheduling (`gpu::custom` API) | Stock nightly |
| `warp-cooperative` | Cooperative compute showcase + MIR pass verification (`gpu::custom` API) | Patched rustc |
| **Std / NN API examples** (`examples/std/`) | | |
| `thread-demo` | `std::thread::spawn` on GPU — spawn, join, warp reuse (`gpu::launch` API) | Stock nightly |
| `gpt2-inference` | GPT-2 Small text generation using `nn` module | Stock nightly |
| `yolo-detect` | YOLOv8-nano object detection using `nn` module | Stock nightly |
| `mnist-train` | MNIST MLP training (91.2% accuracy in 5 epochs) | Stock nightly |
| `cifar-train` | CIFAR-10 tiny CNN training with loss convergence | Stock nightly |
| `gpt2-lora` | GPT-2 LoRA fine-tuning on WikiText-2 (ppl 128→16, rank=8) | Stock nightly |
| `mnist-cnn` | MNIST CNN training (96.4% accuracy, 2.62x GPU speedup) | Stock nightly |
| `resnet-cifar` | ResNet-18 pretrained inference (91.3% CIFAR-10) + ONNX inference (91.2%) + full conv training | Stock nightly |
| `gpu-rag` | GPU-Autonomous RAG: 1030-chunk vector search + GPT-2 generation | Stock nightly |
| `diff-physics` | Differentiable 2D spring-mass / N-body gravity (47.1x GPU speedup) | Stock nightly |
| `dynamic-control` | Data-dependent GPU control flow: variable-length gen, early exit, sampling | Stock nightly |
| `graph-algorithms` | GPU BFS + PageRank on RMAT graphs (CSR, 1M+ vertices, 4.3x speedup) | Stock nightly |
| `monte-carlo` | GPU Monte Carlo: Black-Scholes pricing (129x), Pi estimation (12x) | Stock nightly |
| `benchmark` | SGEMM/Conv2D/Attention vs cuBLAS, memory bandwidth, GPT-2 profiling | Stock nightly |

</details>

### Patched Toolchain (for async warp convergence)

The MIR pass auto-applies to all `async fn` on the `nvptx64` target — no `#[warp_cooperative]` attribute needed. Without the patched toolchain, examples using stock nightly still work (hello-gpu, async-io, vector-math, thread-demo), but `async-pipeline` and `warp-cooperative` need the MIR pass.

```bash
# Linux
bash scripts/build-toolchain.sh

# Windows (cmd)
.\scripts\build-toolchain.bat
```

This clones the latest rustc, applies patches from `rustc-patches/` and `std-patches/`, and builds a stage1 compiler at `patched-rustc/build/host/stage1/`. The build requires ~30GB disk, cmake, ninja, and clang/gcc. The `async-pipeline` example's `build.rs` automatically detects and uses it.

## Progressive Examples

Three snippets, increasing complexity. Each is extracted from a runnable example.

**1. Hello GPU** -- spawn threads on GPU, join results. Looks like normal Rust. ([hello-gpu](examples/hostcall/hello-gpu/), [thread-demo](examples/std/thread-demo/))

```rust
// GPU kernel — spawn two warps as threads, join results
#[no_mangle]
pub unsafe extern "gpu-kernel" fn thread_spawn_test(result: *mut u32) {
    thread::gpu_main(|| {
        let h1 = thread::spawn(|| 42u32);
        let h2 = thread::spawn(|| 99u32);
        let (r1, r2) = (h1.join(), h2.join());
    });
}

// Host — one line launches the kernel and downloads results
let result: Vec<u32> = gpu::launch("thread_spawn_test", 4, 128)?;
assert_eq!(result[0], 42);
```

**2. Cooperative Compute** -- all warps process data in parallel, then return to sequential. ([warp-cooperative](examples/hostcall/warp-cooperative/))

```rust
// All warps cooperate: each handles rows where row % n_warps == warp_id
thread::cooperative(|| {
    let wid = thread::current_id() as usize;
    let n_warps = thread::available_parallelism() + 1;
    for row in (wid..M).step_by(n_warps) {
        for col in 0..N {
            let mut sum = 0.0;
            for k in 0..K { sum += a[row * K + k] * b[k * N + col]; }
            c[row * N + col] = sum;
        }
    }
});
```

**3. Structured Concurrency Pipeline** -- scoped spawn, oneshot channels, lifetime-bounded shared memory. ([structured-concurrency](examples/hostcall/structured-concurrency/))

```rust
// Block-scoped producer-consumer on GPU — memory freed when scope exits
block_scope(|scope| {
    let data: &mut [u32] = scope.alloc::<u32>(64);   // shared memory
    let (tx, rx) = block_oneshot(scope.alloc_slot()); // oneshot channel

    scope.spawn(move || {                 // producer warp
        for i in 0..64 { data[i] = i; }
        tx.send(1);                       // signal completion
    });
    scope.spawn(move || -> u32 {          // consumer warp
        let _signal = rx.recv_spin();     // wait for data
        data.iter().sum()                 // sum = 2016
    });
});
```

## Feature Matrix

### Language & Runtime

| Feature | Description | Example |
|---------|-------------|---------|
| `gpu::run` / `gpu::launch` / `gpu::custom` | One-liner, pure-compute, and builder kernel launch APIs | [hello-gpu](examples/hostcall/hello-gpu/), [vector-math](examples/hostcall/vector-math/) |
| `extern "gpu-kernel"` | Native GPU entry point — no proc macros needed | [hello-gpu](examples/hostcall/hello-gpu/) |
| Real `std` on GPU | `Vec`, `String`, `HashMap`, `Mutex`, `println!`, `File`, `stdin` via patched std | [hello-gpu](examples/hostcall/hello-gpu/) |
| `std::thread::spawn` on GPU | Warp-as-thread model with `JoinHandle::join()` and warp reuse | [thread-demo](examples/std/thread-demo/) |
| GPU async/await | `async fn` with warp-cooperative state machines (custom MIR pass) | [async-pipeline](examples/hostcall/async-pipeline/) |
| GPU async executor | `GpuExecutor` — multi-task scheduling on GPU | [gpu-channels](examples/hostcall/gpu-channels/) |
| Cooperative compute | `cooperative()` closure — all warps process in parallel | [hello-gpu](examples/hostcall/hello-gpu/) |
| Structured concurrency | `BlockScope`, `GridScope`, `spawn_all`, nested scopes | [structured-concurrency](examples/hostcall/structured-concurrency/) |
| GPU channels | Oneshot and MPSC channels for inter-warp communication | [gpu-channels](examples/hostcall/gpu-channels/) |
| Parallel iterator | `GpuParallelIterator` — `map`, `filter`, `fold`, `collect`, `zip` on GPU | — |
| Tokio integration | `AsyncGpuRuntime`, `GpuTask`, non-blocking kernel launch | [tokio-offload](examples/hostcall/tokio-offload/) |

### I/O & Networking

| Feature | Description | Example |
|---------|-------------|---------|
| File I/O on GPU | `std::fs::File` open/read/write/close via hostcall | [hello-gpu](examples/hostcall/hello-gpu/), [async-io](examples/hostcall/async-io/) |
| GPU TCP networking | GPU-initiated TCP connect/send/recv/close | [tcp-echo](examples/hostcall/tcp-echo/) |
| Lock-free hostcall protocol | ROCm-inspired GPU-host RPC (TLA+ verified, 367M states) | [formal/](formal/) |

### Compute Patterns

| Feature | Description | Example |
|---------|-------------|---------|
| SAXPY / dot product / softmax | Pure compute kernels with `gpu::custom` builder | [vector-math](examples/hostcall/vector-math/) |
| Warp-parallel search | 32-lane GPU grep with `shfl.sync` warp reduction | [parallel-search](examples/hostcall/parallel-search/) |
| Monte Carlo simulation | xoshiro256++ PRNG, Pi estimation, Black-Scholes (129x CPU) | [monte-carlo](examples/std/monte-carlo/) |
| Graph algorithms | BFS + PageRank on RMAT CSR graphs (1M+ vertices, 4.3x CPU) | [graph-algorithms](examples/std/graph-algorithms/) |
| Dynamic control flow | Variable-length generation, top-k sampling, early exit | [dynamic-control](examples/std/dynamic-control/) |
| Differentiable physics | Spring-mass / N-body with analytical backward (47.1x CPU) | [diff-physics](examples/std/diff-physics/) |
| Kernel fusion | Fused GEMM+bias+GELU in a single kernel launch | [gpu-rag](examples/std/gpu-rag/) `--bench-fused` |
| GPU benchmarks | SGEMM/Conv2D/Attention vs cuBLAS, memory bandwidth profiling | [benchmark](examples/std/benchmark/) |

### ML / AI

| Feature | Description | Example |
|---------|-------------|---------|
| GPT-2 inference (124M) | Full transformer, BPE tokenizer, KV-cached generation | [gpt2-inference](examples/std/gpt2-inference/) |
| YOLOv8-nano detection | 23-layer backbone, DFL decode, NMS — all pure Rust PTX | [yolo-detect](examples/std/yolo-detect/) |
| ResNet-18 inference | Pretrained CIFAR-10 (91.3%), full conv training | [resnet-cifar](examples/std/resnet-cifar/) |
| GPU-RAG pipeline | 1030-chunk vector search + GPT-2 generation | [gpu-rag](examples/std/gpu-rag/) |
| Autograd (tape-based AD) | Reverse-mode AD, SGD/Adam, cross-entropy/MSE | [mnist-train](examples/std/mnist-train/), [mnist-cnn](examples/std/mnist-cnn/) |
| LoRA fine-tuning | GPT-2 LoRA on WikiText-2 (ppl 128 to 16, rank=8) | [gpt2-lora](examples/std/gpt2-lora/) |
| ONNX runtime | 43 operators, graph fusion, prost parser (no protoc) | [resnet-cifar](examples/std/resnet-cifar/) `--onnx` |
| INT8/INT4 quantization | dp4a GEMM, W4A16 quantized inference (7.5x mem reduction) | [gpu-rag](examples/std/gpu-rag/) `--bench-int8` |

### GPU Compute Kernels (pure Rust inline PTX)

SGEMM (f32 FMA + f16 Tensor Core + INT8 dp4a), FlashAttention (tiled online softmax, causal, KV cache), Conv2D (im2col + GEMM), BatchNorm+SiLU (fused), LayerNorm, GELU, Softmax, MaxPool2D, Upsample, Embedding — 130+ kernels total.

## Real Rust `std` on GPU

GPU kernels can use **actual Rust standard library** types and traits — not custom wrappers:

```rust
// This runs on the GPU, using real std
println!("[GPU] Hello from Rust std on GPU!");

let mut data = Vec::new();
for i in 0..10 {
    data.push(format!("item-{}", i));
}

let file = std::fs::File::create("gpu_output.txt")?;
std::io::Write::write_all(&mut &file, b"Written from GPU")?;

let line = std::io::stdin().lock().lines().next().unwrap()?;
println!("[GPU] Read from stdin: {}", line);
```

This works via a **patched std** (`-Zbuild-std=std`) with a CUDA platform adaptation layer (PAL) that routes `sys` calls through the hostcall protocol.

**What works** (multi-thread safe): `println!`, `format!`, `Vec`, `String`, `Box`, `HashMap`, `Mutex`, `std::fs::File` (create/read/write), `std::io::stdin().read_line()`, `std::thread::spawn` + `JoinHandle::join()`, `Result<T, E>` with `?` operator and `std::io::Error`.

## GPT-2 Inference (124M Parameters)

End-to-end transformer inference — real HuggingFace weights, custom BPE tokenizer, 12 transformer layers, KV-cached autoregressive generation. Available via both the raw kernel API and the composable **nn module** (`Linear`, `LayerNorm`, `MultiHeadAttention`, `Gpt2Model`). All compute kernels in pure Rust with inline PTX, no CUDA C++ or cuBLAS.

```
--- Greedy autoregressive generation (with KV cache) ---
  [1/3] Prompt: "The capital of France is" -> 5 tokens, generating 50
  Generated: " the capital of the French Republic, and the capital of
  the French Republic is the capital of the French Republic..."
  Time: 3400ms total, 68ms/token  (2.07x faster with KV cache)
  PASSED (50 tokens, no NaN)
```

GPU compute kernels: GEMM (f32 FMA + f16 Tensor Core MMA with split-K + INT8 dp4a), FlashAttention (tiled online softmax, causal masking, KV cache), LayerNorm, GELU, Softmax, Embedding, fused GEMM+bias+activation — all in Rust inline PTX.

Standalone example: `cargo run --manifest-path examples/std/gpt2-inference/Cargo.toml --release` (requires `models/model.safetensors` — run `bash scripts/download-models.sh`).

<details>
<summary>More demos</summary>

### GPU-Autonomous File Transform

Single kernel launch, 8-step I/O pipeline + compute — zero CPU intervention:

```
--- File Transform Pipeline ---
  16-state WarpFuture: open->read->transform->open->write->close->close->print
  1024 bytes: ASCII case toggled correctly, Elapsed: 4.183ms
```

### GPU-Autonomous Vector Search

20-state WarpFuture: open database, read vectors, cosine similarity across 32 lanes, merge top-K via warp shuffle, write results — one kernel launch:

```
--- Vector Similarity Search ---
  rank 1: id=42 score=1.0000, rank 2: id=82 score=0.2103, rank 3: id=18 score=0.0913
  Elapsed: 6.434ms
```

### Compute Pipeline (1.91x vs multi-launch)

Newton-Raphson sqrt with warp-cooperative convergence — single-launch async (24.1 us) vs multi-launch CUDA-style (46.1 us, 3 separate kernels).

</details>

## YOLOv8-nano Object Detection

End-to-end real-time object detection — SafeTensors weights, 23-layer backbone/neck, decoupled detect head with DFL decode + NMS. All compute kernels in pure Rust inline PTX, no cuDNN or cuBLAS.

```
--- YOLOv8-nano end-to-end inference ---
  Image: 810x1080 → letterbox 640x640
  7 detections found:
  [ 0] person          conf=0.931  box=(672, 391, 810, 877)
  [ 1] person          conf=0.925  box=(222, 409, 344, 856)
  [ 2] person          conf=0.878  box=(53, 400, 243, 905)
  [ 3] bus             conf=0.865  box=(32, 237, 797, 747)
  [ 4] person          conf=0.508  box=(1, 548, 59, 877)
  [ 5] car             conf=0.469  box=(686, 505, 778, 680)
  [ 6] tie             conf=0.298  box=(135, 477, 152, 518)
```

GPU compute kernels: Conv2D (im2col + GEMM), BatchNorm+SiLU (fused elementwise), MaxPool2D, Upsample (nearest-neighbor), C2f blocks, SPPF, Sigmoid — all in Rust inline PTX.

Standalone example: `cargo run --manifest-path examples/std/yolo-detect/Cargo.toml --release` (requires `models/yolov8n.safetensors` — run `uv run --with ultralytics --with safetensors scripts/export_yolo.py`).

## How It Works

### Host SDK

**One-liner API** (`async_gpu::gpu`):

| Function | Purpose |
|----------|---------|
| `gpu::run("kernel")` | Hostcall-enabled kernel (supports `println!`, file I/O) |
| `gpu::run_with_output("kernel", n)` | Hostcall + output buffer, returns `Vec<T>` |
| `gpu::launch("kernel", n, threads)` | Pure compute with output buffer, no hostcall |
| `gpu::custom("kernel")` | Builder API for multi-argument kernels (`.ptx()`, `.threads()`, `.hostcall()`, `.prepare()`) |

**Core types**:

| Type | Purpose |
|------|---------|
| `GpuRuntime` | Device init, PTX loading, kernel launch, multi-GPU support |
| `HostcallBuffer` | GPU-host RPC communication (print, file I/O, stdin) |
| `MappedBuffer<T>` | RAII pinned device-mapped memory (auto-freed on drop) |
| `GpuStream` | CUDA stream wrapper for overlapping compute and I/O |
| `GpuContext` | Prepared launch context from `gpu::custom()` — upload, alloc, launch |
| `GpuResult` | Post-launch handle for downloading device buffers |

### Lock-Free Hostcall Protocol

GPU-host communication uses a ROCm-inspired two-stack design over CUDA mapped memory:

- **Free stack**: Available packets for GPU to claim (one CAS per warp)
- **Ready stack**: Filled packets for host to process
- **Per-block sharding**: Reduces CAS contention at scale
- **Sideband buffer**: Separate mapped memory for bulk data beyond the 56-byte packet payload

Formally verified with TLA+ (367M safety states, 337K liveness states, 0 violations). See [`formal/`](formal/).

### Async on GPU

The custom MIR pass auto-applies to **all** `async fn` on the `nvptx64` target — no attributes needed. It inserts `bar.warp.sync` at each `.await` point so all 32 SIMT lanes always agree on the current state.

Standard `async fn` + `.await` is the only path needed — no proc macros required.

### GPU Threading Model

`std::thread::spawn` works on GPU — each warp (32 SIMT lanes) acts as a single thread:

| API | GPU Behavior |
|-----|-------------|
| `thread::spawn(closure)` | Wakes a sleeping warp, assigns closure, returns `JoinHandle` |
| `handle.join()` | Blocks parent warp until child completes, returns result |
| `thread::available_parallelism()` | Returns number of free warps |
| `thread::current()` / `thread::yield_now()` | Thread identity and cooperative yield |

Warp 0 runs `main()`, other warps sleep until `thread::spawn()` wakes them. Warps return to the idle pool after their closure completes, enabling reuse.

## Neural Network Module (`async_gpu::nn`)

PyTorch-style composable layers and autograd, running on GPU via the kernel registry:

```rust
use async_gpu::nn::{GpuTensor, KernelRegistry, Module};
use async_gpu::nn::layers::{Linear, LayerNorm, GELU};
use async_gpu::nn::models::gpt2::Gpt2Model;

// Build model from safetensors weights — no raw kernel launches needed
let model = Gpt2Model::from_weights(&weights, config, &registry)?;
let tokens = model.generate(&prompt_tokens, 50)?;
```

**Layers**: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm2d`, `Embedding`, `MultiHeadAttention`, `GELU`, `SiLU`, `Sigmoid`, `ReLU`, `MaxPool2d`, `Sequential`, `Int4Linear`.

**ONNX Runtime** (`async_gpu::onnx`):
- Load any `.onnx` file via prost protobuf parser (no protoc needed)
- 43 ONNX operators: Conv (incl. grouped/depthwise), MatMul, Gemm, Relu, BatchNorm, LayerNorm, Softmax, Add, Mul, Sub, Reshape, Transpose, Gather, Split, Where, Concat, Identity, GlobalAveragePool, ReduceMean, and more
- `OnnxSession`: initializer caching + weight prepadding for repeated inference
- Graph fusion pass: MatMul+Add+Activation pattern matching
- GPT-2 ONNX text generation verified (150ms/forward, 1107 nodes)
- ResNet-18 ONNX: 91.2% CIFAR-10 accuracy (matches ORT exactly)
- MobileNetV2 ONNX: 209 nodes, 1000-class output, end-to-end verified

**Autograd** (tape-based reverse-mode AD):
- Forward ops automatically record on a thread-local tape when `requires_grad = true`
- `backward()` traverses tape in reverse with chain rule dispatch
- Backward kernels: GELU, SiLU, sigmoid, ReLU, matmul, LayerNorm, BatchNorm (GPU), Conv2d (im2col), MaxPool2d (gradient routing), UpsampleNearest (4-to-1), bias_add, elementwise_add
- Optimizers: SGD (with momentum), Adam
- Losses: cross-entropy, MSE
- Verified via numerical gradient checks (finite differences)

## Crate Map

```
crates/
  core/
    gpu-host/          Host-side SDK: gpu:: API, GpuRuntime, HostcallBuffer, MappedBuffer, GpuStream
      nn/              Neural network module: GpuTensor, KernelRegistry, ops, layers, models
        autograd/      Tape-based reverse-mode AD: backward, optimizers, losses
        models/        GPT-2, YOLOv8-nano, and ResNet-18 model implementations
        ops/quantize/  INT8/INT4 quantization pack/unpack utilities
        test_utils/    Numerical comparison harness, CPU f64 references, golden files
      onnx_rt/         ONNX Runtime: protobuf parser (prost), graph executor (43 ops), fusion pass
    gpu-protocol/      Shared constants: packet layout, service IDs, error codes
    gpu-runtime/       GPU-side runtime: index, math, warp, block, thread, nn, executor, channels
    gpu-atomics/       System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
    gpu-libc/          Minimal libc shim for GPU: routes sys calls to hostcall
  kernel/
    gpu-kernel-std/    Unified GPU kernel crate (130+ kernels: compute, hostcall, pipeline, fused, physics, persistent, thread, std)
  test/
    async-hostcall-test/   Async hostcall integration tests
    async-pipeline-test/   Async pipeline integration tests
    embassy-test/          Embassy async executor tests
    gpu-critical-section/  GPU critical section tests
    gpu-std-test/          Patched std integration tests
    multi-warp-test/       Multi-warp coordination tests
    std-build-test/        Patched std build verification

rustc-patches/       Custom MIR pass patches for rustc (auto-applies to async fn on nvptx64)
scripts/             Build/CI automation, model download (download-models.sh, export_yolo.py)
examples/
  hostcall/          10 hostcall examples using gpu:: API (hello-gpu, async-pipeline, vector-math, structured-concurrency, gpu-channels, etc.)
  std/               14 std/nn examples (thread-demo, gpt2-inference, yolo-detect, mnist-train, mnist-cnn, cifar-train, gpt2-lora, resnet-cifar, gpu-rag, diff-physics, dynamic-control, graph-algorithms, monte-carlo, benchmark)
formal/              TLA+ specification and model-checking config
```

## Performance

**Inference** (RTX 3060, SM 86):

| Metric | Value |
|--------|-------|
| GPT-2 per-token f32 FMA (KV cache) | ~68ms/token |
| GPT-2 per-token f16 MMA (Tensor Core) | ~26ms/token (2.18x over f32 FMA) |
| YOLOv8-nano inference | 374ms, 34 detections on 640x640 |
| ResNet-18 pretrained (CIFAR-10) | 91.3% accuracy, 16.0ms/image |
| Compute pipeline speedup | 1.91x vs multi-launch |
| N-body gravity (4096 particles) | 47.1x GPU vs CPU |
| **ONNX Runtime** (ResNet-18, 48 nodes) | 42ms/inference, 91.2% CIFAR-10 (matches ORT) |
| **ONNX Runtime** (GPT-2, 1107 nodes) | 150ms/forward pass, text generation works |
| **ONNX Runtime** (MobileNetV2, 209 nodes) | 409ms/inference, 1000-class output verified |
| **INT4 GPT-2** (W4A16 quantized) | 43ms/token, 7.5x memory reduction (45MB vs 340MB) |
| **GPU PageRank** (1M vertices, 16M edges) | 4.3x speedup over CPU (scale=22) |
| **GPU Monte Carlo** (Black-Scholes, f32) | 129x throughput speedup, 0.004% error |

**Kernel Performance vs cuBLAS / cuDNN** (NVIDIA A2 SM 86 unless noted):

| Kernel | async-gpu | cuBLAS/cuDNN | % of Reference | Improvement |
|--------|-----------|-------------|----------------|-------------|
| **GPT-2 forward** (seq=128) | **25.1ms** | ~20ms est. | — | **8.8x** over baseline |
| **GPT-2 forward** (seq=128)¹ | **39.4ms** | — | — | **5.6x** over baseline |
| **SGEMM** (4096³) | 2,691 GFLOPS | 2,987 GFLOPS | **90%** | 17.1x over v1 |
| **Flash Attention V3** (seq=512, causal)¹ | 559 GFLOPS | ~1,000-1,200 est. | **47-60%** | V3 rewrite |
| **Flash Attention** (seq=64) | 0.056ms | 0.030ms (FA2) | **54%** | 8.2x over v1 |
| **Flash Attention** (seq=128) | 0.134ms | 0.048ms (FA2) | **36%** | 9.3x over v1 |
| **Conv2D** (128→128, 28²) | 425 GFLOPS | 522 GFLOPS | **81%** | 3.9x over v1 |
| **Conv2D** (256→256, 14²) | 556 GFLOPS | 243 GFLOPS | **229%** | 4.9x over v1 |
| **LayerNorm** (128×768)¹ | 199 GB/s eff. | 200 GB/s peak | **~100%** | 6.6x over v1 |
| **Fused LN+residual**¹ | 154 GB/s eff. | — | — | 2.01x speedup |
| **elementwise_add** (in-place)¹ | 160 GB/s | 192 GB/s peak | **83%** | 1.5x over PyTorch |

¹ Measured on GTX 1660 (SM 75, 192 GB/s). FA V3 % is vs estimated cuDNN FA2 on SM 75 (no tensor cores).

**Training** (GPU matmul + autograd tape):

| Example | CPU | GPU | Speedup | Accuracy |
|---------|-----|-----|---------|----------|
| MNIST MLP (60K, 5 epochs) | 44.0s (8.8s/ep) | 7.8s (1.6s/ep) | **5.6x** | 91.2% |
| MNIST CNN (60K, 5 epochs) | 541.3s (107.5s/ep) | 206.7s (41.3s/ep) | **2.62x** | 96.4% |
| CIFAR-10 CNN (2K, 10 epochs) | 6.5s (0.7s/ep) | 7.2s (0.7s/ep) | 0.90x | 27.2%/21.0% |
| Mini-ResNet (2K, 20 ep, full conv bwd) | — | 468.9s | — | 32.1% |

MNIST MLP shows clear GPU advantage for matmul-heavy workloads (batch=64, 784×128 GPU GEMM). MNIST CNN uses full GPU conv2d backward (im2col + matmul + col2im) — 2.62x over CPU. CIFAR-10 GPU produces **identical** loss/accuracy curves to CPU. All use `--cpu` for comparison.

**Hostcall**:

| Metric | Value |
|--------|-------|
| Round-trip (1 thread) | ~42-101 us, 10-15K calls/s |
| Round-trip (32 threads) | ~1.1 ms, 20-23K calls/s |

## Limitations

- **Nightly Rust**: Requires `asm_experimental_arch`, `abi_gpu_kernel`, `-Zbuild-std`. Async warp convergence MIR pass needs patched rustc
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip, not suitable for per-element I/O in hot loops
- **Partial std**: `HashMap`, `Mutex`, File I/O work; `OsRng`/`getrandom` not available
- **f32 + f16 MMA**: f32 FMA and f16 Tensor Core MMA (split-K accumulation) both supported; BF16/TF32 not yet implemented

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
