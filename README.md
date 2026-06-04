# async_gpu — Rust Async/Await on NVIDIA GPUs

[![CI](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

**What if the GPU could drive its own computation?** Open files, read data, branch on results, loop until convergence, write output — all from GPU code, with zero CPU orchestration between steps.

async_gpu makes this real: **Rust async/await running natively on NVIDIA GPUs**, with a custom rustc MIR pass that turns standard `async fn` into warp-cooperative state machines — and GPU compute kernels powerful enough to run **GPT-2 inference in 25ms** (8.8x optimized), **YOLOv8-nano object detection**, **graph algorithms** (BFS, PageRank), and **Monte Carlo simulations** (129x throughput). Custom SGEMM at **63% of cuBLAS**, Flash Attention at **54% of cuDNN FA2**.

```rust
#[warp_cooperative]
pub async fn data_pipeline(buf: *mut u8) -> u32 {
    // Open input file — yields warp during I/O wait
    let fd = GpuOpenFuture::new(buf, b"input.txt", FILE_OPEN_READ).await?;

    // Read data (each .await inserts bar.warp.sync for warp convergence)
    let mut data = [0u8; 48];
    let n = GpuReadFuture::new(buf, fd, &mut data).await?;
    GpuCloseFuture::new(buf, fd).await?;

    // Transform on GPU
    let mut out = [0u8; 48];
    for i in 0..n { out[i] = data[i].to_ascii_uppercase(); }

    // Write output
    let out_fd = GpuOpenFuture::new(buf, b"output.txt", FILE_OPEN_WRITE_CREATE).await?;
    let written = GpuWriteFuture::new(buf, out_fd, &out[..n]).await?;
    GpuCloseFuture::new(buf, out_fd).await?;

    Ok(written as u32)
}

// Entry point: drive async pipeline with spin-polling executor
let result = block_on(data_pipeline(buf)).unwrap_or(0xDEAD);
```

The `#[warp_cooperative]` attribute is a **custom rustc MIR pass** that inserts `bar.warp.sync` + `shfl.sync` at every `.await` point, ensuring all 32 GPU lanes yield and resume together. Standard Rust `async fn` syntax, standard `Future` trait — no macros, no custom runtime.

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

# Hello GPU — vector add, GPU print, file I/O, bulk transfer
cargo run --manifest-path examples/hello-gpu/host/Cargo.toml

# Async Pipeline — #[warp_cooperative] async fn with real I/O (requires patched rustc)
cargo run --manifest-path examples/async-pipeline/host/Cargo.toml

# Vector Math — SAXPY, dot product, softmax (pure GPU compute)
cargo run --manifest-path examples/vector-math/host/Cargo.toml
```

<details>
<summary>All examples</summary>

| Example | Description | Toolchain |
|---------|-------------|-----------|
| **Hostcall examples** (`examples/hostcall/`) | | |
| `hello-gpu` | Vector add, GPU print, file I/O, bulk sideband | Stock nightly |
| `async-pipeline` | `#[warp_cooperative] async fn` with hostcall Futures | Patched rustc |
| `async-io` | Multi-file write pipeline + read-transform-write | Stock nightly |
| `parallel-search` | 32-lane GPU grep with `shfl.sync` warp reduction | Stock nightly |
| `vector-math` | SAXPY, dot product, softmax (pure compute) | Stock nightly |
| `tcp-echo` | GPU-initiated TCP networking via hostcall | Stock nightly |
| `tokio-offload` | Async kernel launch from tokio runtime | Stock nightly |
| `warp-cooperative` | MIR pass verification tests | Patched rustc |
| **NN API examples** (`examples/std/`) | | |
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

### Patched Toolchain (for `#[warp_cooperative]`)

The `#[warp_cooperative]` MIR pass requires a patched rustc. Without it, examples using stock nightly still work (hello-gpu, async-io, vector-math), but `async-pipeline` needs the MIR pass.

```bash
# Linux
bash scripts/build-toolchain.sh

# Windows (cmd)
.\scripts\build-toolchain.bat
```

This clones the latest rustc, applies patches from `rustc-patches/` and `std-patches/`, and builds a stage1 compiler at `patched-rustc/build/host/stage1/`. The build requires ~30GB disk, cmake, ninja, and clang/gcc. The `async-pipeline` example's `build.rs` automatically detects and uses it.

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

**What works** (multi-thread safe): `println!`, `format!`, `Vec`, `String`, `Box`, `HashMap`, `Mutex`, `std::fs::File` (create/read/write), `std::io::stdin().read_line()`, `Result<T, E>` with `?` operator and `std::io::Error`.

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

| Type | Purpose |
|------|---------|
| `GpuRuntime` | Device init, PTX loading, kernel launch, multi-GPU support |
| `HostcallBuffer` | GPU-host RPC communication (print, file I/O, stdin) |
| `MappedBuffer<T>` | RAII pinned device-mapped memory (auto-freed on drop) |
| `GpuStream` | CUDA stream wrapper for overlapping compute and I/O |

### Lock-Free Hostcall Protocol

GPU-host communication uses a ROCm-inspired two-stack design over CUDA mapped memory:

- **Free stack**: Available packets for GPU to claim (one CAS per warp)
- **Ready stack**: Filled packets for host to process
- **Per-block sharding**: Reduces CAS contention at scale
- **Sideband buffer**: Separate mapped memory for bulk data beyond the 56-byte packet payload

Formally verified with TLA+ (367M safety states, 337K liveness states, 0 violations). See [`formal/`](formal/).

### Warp-Cooperative GPU Async

| Feature | `#[warp_cooperative]` (recommended) | `#[warp_async]` |
|---------|--------------------------------------|-----------------|
| Syntax | Standard `async fn` + `.await` | `warp_*!()` macros |
| Toolchain | Patched rustc | Stock nightly |
| Warp convergence | `bar.warp.sync` at `.await` | State machine by construction |

All 32 lanes always agree on the current state — warp convergence is maintained by construction.

## Neural Network Module (`gpu_host::nn`)

PyTorch-style composable layers and autograd, running on GPU via the kernel registry:

```rust
use gpu_host::nn::{GpuTensor, KernelRegistry, Module};
use gpu_host::nn::layers::{Linear, LayerNorm, GELU};
use gpu_host::nn::models::gpt2::Gpt2Model;

// Build model from safetensors weights — no raw kernel launches needed
let model = Gpt2Model::from_weights(&weights, config, &registry)?;
let tokens = model.generate(&prompt_tokens, 50)?;
```

**Layers**: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm2d`, `Embedding`, `MultiHeadAttention`, `GELU`, `SiLU`, `Sigmoid`, `ReLU`, `MaxPool2d`, `Sequential`, `Int4Linear`.

**ONNX Runtime** (`gpu_host::onnx_rt`):
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
    gpu-host/          Host-side SDK: GpuRuntime, HostcallBuffer, MappedBuffer, GpuStream
      nn/              Neural network module: GpuTensor, KernelRegistry, ops, layers, models
        autograd/      Tape-based reverse-mode AD: backward, optimizers, losses
        models/        GPT-2, YOLOv8-nano, and ResNet-18 model implementations
        ops/quantize/  INT8/INT4 quantization pack/unpack utilities
        test_utils/    Numerical comparison harness, CPU f64 references, golden files
      onnx_rt/         ONNX Runtime: protobuf parser (prost), graph executor (43 ops), fusion pass
    gpu-protocol/      Shared constants: packet layout, service IDs, error codes
    gpu-runtime/       GPU-side runtime: index, math, warp, block, nn, executor, channels
    gpu-atomics/       System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
    gpu-libc/          Minimal libc shim for GPU: routes sys calls to hostcall
  kernel/
    gpu-kernel/        Main GPU kernel crate (130+ kernels: compute, hostcall, pipeline, backward, fused, physics, persistent, elementwise)
    gpu-kernel-std/    GPU kernels using patched Rust std (println!, Vec, File, stdin)
  macro/
    warp-macro/        #[warp_async] proc macro (generates WarpFuture state machines)

rustc-patches/       Custom MIR pass patches for rustc
scripts/             Build/CI automation, model download (download-models.sh, export_yolo.py)
examples/
  hostcall/          8 raw-API examples (hello-gpu, async-pipeline, vector-math, etc.)
  std/               13 nn-API examples (gpt2-inference, yolo-detect, mnist-train, mnist-cnn, cifar-train, gpt2-lora, resnet-cifar, gpu-rag, diff-physics, dynamic-control, graph-algorithms, monte-carlo, benchmark)
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

**Kernel Performance vs cuBLAS / cuDNN** (NVIDIA A2, SM 86):

| Kernel | async-gpu | cuBLAS/cuDNN | % of Reference | Improvement |
|--------|-----------|-------------|----------------|-------------|
| **GPT-2 forward** (seq=128) | **25.1ms** | ~20ms est. | — | **8.8x** over baseline |
| **SGEMM** (4096³) | 1,760 GFLOPS | 2,800 GFLOPS | **63%** | 11.2x over v1 |
| **Flash Attention** (seq=64) | 0.056ms | 0.030ms (FA2) | **54%** | 8.2x over v1 |
| **Flash Attention** (seq=128) | 0.134ms | 0.048ms (FA2) | **36%** | 9.3x over v1 |
| **Conv2D** (128→128, 28²) | 425 GFLOPS | 522 GFLOPS | **81%** | 3.9x over v1 |
| **Conv2D** (256→256, 14²) | 556 GFLOPS | 243 GFLOPS | **229%** | 4.9x over v1 |
| **LayerNorm** (128×768) | 199 GB/s eff. | 200 GB/s peak | **~100%** | 6.6x over v1 |
| **elementwise_add** | 152 GB/s | 200 GB/s peak | **76%** | 1.5x over PyTorch |

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

- **Nightly Rust**: Requires `asm_experimental_arch`, `-Zbuild-std`. `#[warp_cooperative]` needs patched rustc
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip, not suitable for per-element I/O in hot loops
- **Partial std**: `HashMap`, `Mutex`, File I/O work; `OsRng`/`getrandom` not available
- **f32 + f16 MMA**: f32 FMA and f16 Tensor Core MMA (split-K accumulation) both supported; BF16/TF32 not yet implemented

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
