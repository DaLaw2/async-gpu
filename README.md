# async_gpu — Rust Async/Await on NVIDIA GPUs

[![CI](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

**What if the GPU could drive its own computation?** Open files, read data, branch on results, loop until convergence, write output — all from GPU code, with zero CPU orchestration between steps.

async_gpu makes this real: **Rust async/await running natively on NVIDIA GPUs**, with a custom rustc MIR pass that turns standard `async fn` into warp-cooperative state machines — and GPU compute kernels powerful enough to run **end-to-end GPT-2 inference** entirely from Rust inline PTX.

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

- [Rust](https://rustup.rs/) with nightly toolchain: `rustup toolchain install nightly-2026-03-11`
- nvptx64 target: `rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-03-11`
- Rust nightly src (for `-Zbuild-std`): `rustup component add rust-src --toolchain nightly-2026-03-11`
- NVIDIA GPU (SM 70+) with CUDA 12.x driver

### Run an Example

Each example is self-contained with automated PTX compilation via `build.rs`:

```bash
git clone https://github.com/DaLaw2/async-gpu.git
cd async-gpu

# Hello GPU — vector add, GPU print, file I/O, bulk transfer
cargo run --manifest-path examples/hello-gpu/host/Cargo.toml

# Async Pipeline — #[warp_cooperative] async fn with real I/O
cargo run --manifest-path examples/async-pipeline/host/Cargo.toml

# Async I/O — multi-file write pipeline + read-transform-write
cargo run --manifest-path examples/async-io/host/Cargo.toml

# Parallel Search — 32-lane GPU grep with warp reduction
cargo run --manifest-path examples/parallel-search/host/Cargo.toml

# Vector Math — SAXPY, dot product, softmax (pure GPU compute)
cargo run --manifest-path examples/vector-math/host/Cargo.toml

# TCP Echo — GPU-initiated TCP networking via hostcall
cargo run --manifest-path examples/tcp-echo/host/Cargo.toml
```

<details>
<summary>Run the full test suite (includes GPT-2 inference)</summary>

```bash
# Build all GPU kernels (requires nightly toolchain)
bash scripts/ci-lint.sh

# Run the full test suite (downloads GPT-2 weights on first run)
cd crates/gpu-host
cargo run --release
```
</details>

### Patched Toolchain (for `#[warp_cooperative]`)

The `#[warp_cooperative]` MIR pass requires a patched rustc. Without it, examples using stock nightly still work (hello-gpu, async-io, vector-math), but `async-pipeline` needs the MIR pass.

```bash
# Linux
bash scripts/build-toolchain.sh

# Windows (cmd)
.\scripts\build-toolchain.bat
```

This clones rustc, applies patches from `rustc-patches/`, and builds a stage1 compiler at `patched-rustc/build/`. The `async-pipeline` example's `build.rs` automatically detects and uses it.

## Examples

All examples use the **gpu-host SDK** — three core types for GPU programming:

| Type | Purpose |
|------|---------|
| `GpuRuntime` | Device init, PTX loading, kernel launch, data transfer |
| `HostcallBuffer` | GPU-host RPC communication (print, file I/O, stdin) |
| `MappedBuffer<T>` | RAII pinned device-mapped memory (auto-freed on drop) |

### hello-gpu — Hostcall Basics

Four demos: vector addition (pure compute), GPU-to-host print, file write from GPU, and bulk sideband read.

### async-pipeline — Warp-Cooperative Async I/O

`#[warp_cooperative] async fn` with real hostcall Futures: read file → transform on GPU → write output. Two demos: small I/O (48-byte packet payload) and bulk I/O (sideband, up to 1MB). PTX has `bar.warp.sync` at every `.await` point.

### parallel-search — 32-Lane GPU Grep

ALL 32 warp lanes active: thread 0 reads 4KB file via sideband bulk I/O, then each lane searches 1/32 of the data for a byte pattern. Results gathered via `shfl.sync.idx` warp reduction. Exact match with CPU reference count.

### async-io — Multi-Step File I/O

Write pipeline: GPU creates 3 files in sequence. Transform pipeline: GPU reads a file, uppercases on-device, writes result — all from one kernel launch.

### vector-math — Pure GPU Compute

SAXPY, dot product (GPU multiply + CPU reduce), and softmax (multi-pass GPU-CPU cooperation with numerically stable exp via PTX `ex2.approx.ftz.f32`).

### tcp-echo — GPU-Initiated TCP Networking

GPU kernel connects to a local TCP echo server, sends "Hello from GPU!", receives the echo response, and verifies it. Demonstrates the TCP hostcall services (connect, write, read, close) with async Futures and `block_on()`.

### warp-cooperative — MIR Pass Verification (requires patched rustc)

Tests the `#[warp_cooperative]` MIR pass directly: `simple_add` (no `.await`, bar.warp.sync only), `multi_await` (2 `.await` points, `shfl.sync.idx` broadcast), and `async_pipeline` (6 `.await` points simulating I/O). Verifies all 32 lanes produce correct results.

## Real Rust `std` on GPU

GPU kernels can use **actual Rust standard library** types and traits — not custom wrappers:

```rust
// This runs on the GPU, using real std (multi-thread safe!)
use std::io::BufRead;

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

This works via a **patched std** (`-Zbuild-std=std`) with a CUDA platform adaptation layer (PAL) that routes `sys` calls through the hostcall protocol. The `gpu-libc` crate provides the libc shim.

**What works** (multi-thread safe): `println!`, `format!`, `Vec`, `String`, `Box`, `std::fs::File` (create/read/write), `std::io::stdin().read_line()`, `?` operator with `std::io::Error`.

> **Multi-thread support**: Thread-local storage uses per-thread arrays indexed by hardware thread ID (`gpu_threads.rs`), and the bump allocator uses atomic CAS. Verified with 32-thread Vec allocation and 4-thread concurrent `println!`. For compute-heavy multi-thread kernels, `no_std` kernels with `gpu-runtime` are also available — see the 65+ kernels in `gpu-kernel` (including full GPT-2 inference).

## GPU Error Handling

GPU kernels can use `?` and return `Result<T, E>` to the host:

- Hostcall errors propagate as `Result`, not panic
- Host receives structured error info (error type + message)
- `std::io` and `std::fs` errors produce proper `std::io::Error`

## Demos

### GPU-Autonomous File Transform

A single kernel launch where the GPU self-coordinates an 8-step I/O pipeline + per-thread compute — no CPU intervention between steps:

```
--- File Transform Pipeline ---
  [HOST] FILE OPEN: "gpu_input.txt" flags=0 -> fd=1
  [HOST] BULK READ: fd=1 1024 bytes read
  [HOST] FILE OPEN: "gpu_output.txt" flags=1 -> fd=2
  [HOST] BULK WRITE: fd=2 1024 bytes written
  [HOST] FILE CLOSE: fd=1 closed
  [HOST] FILE CLOSE: fd=2 closed
  [HOST] GPU says: "pipeline: done"
  Elapsed: 4.183ms
    16-state WarpFuture: open->read->transform->open->write->close->close->print
    1024 bytes: ASCII case toggled correctly
    Zero CPU intervention between steps
```

### GPU-Autonomous Vector Search

20-state WarpFuture: open database, read vectors, open query, compute cosine similarity across all 32 lanes, merge top-K via warp shuffle, write results — one kernel launch:

```
--- Vector Similarity Search ---
  [HOST] BULK READ: fd=1 51208 bytes read   (100 vectors x 128 dims)
  [HOST] BULK READ: fd=2 516 bytes read     (query vector)
  [HOST] BULK WRITE: fd=3 84 bytes written  (top-K results)
  Elapsed: 6.434ms
    rank 1: id=42 score=1.0000   <- exact match
    rank 2: id=82 score=0.2103
    rank 3: id=18 score=0.0913
```

### GPU-Autonomous Compute Pipeline

Multi-stage iterative compute — Newton-Raphson sqrt with warp-cooperative convergence:

```
--- Compute Pipeline Demo ---
  GPU pipeline: 5 iterations, 3.07 us GPU time
  Single-launch async: 24.1 us host time
  Multi-launch CUDA-style: 46.1 us (3 separate kernel launches)
  Speedup: 1.91x (eliminates kernel launch overhead)

  With 5 iterations: estimated 9.56x overhead for CUDA-style multi-launch
```

Each iteration runs on GPU without any host roundtrip — convergence is checked
entirely by warp-cooperative max-error reduction across all 32 lanes.

### GPT-2 Inference (124M Parameters)

End-to-end transformer inference — real HuggingFace weights, custom BPE tokenizer, 12 transformer layers, KV-cached autoregressive generation. All compute kernels in pure Rust with inline PTX.

```
--- Greedy autoregressive generation (with KV cache) ---
  [1/3] Prompt: "The capital of France is" -> 5 tokens, generating 50
  Generated: " the capital of the French Republic, and the capital of
  the French Republic is the capital of the French Republic..."
  Time: 3400ms total, 68ms/token  (2.07x faster with KV cache)
  PASSED (50 tokens, no NaN)
```

Pipeline: token embedding -> 12x (LayerNorm -> Multi-Head FlashAttention with KV cache -> FFN with GELU -> residual connections) -> final LayerNorm -> LM head (CPU). KV cache eliminates redundant recomputation — only the new token passes through the model per generation step.

GPU compute kernels — all in Rust inline PTX, no CUDA C++ or cuBLAS:

| Kernel | Implementation |
|--------|---------------|
| **GEMM** | Pure f32 FMA with shared memory tiling, column-major weights. Handles arbitrary dimensions (768x768, 768x2304, 768x3072). |
| **FlashAttention** | Tiled attention with online softmax (numerically stable), causal masking, multi-head support, KV cache. |
| **LayerNorm** | Shared-memory parallel reduction for mean + variance, per-element affine transform. |
| **GELU** | Numerically stable tanh approximation with overflow clamping via `ex2.approx.f32` PTX. |
| **Softmax** | Shared-memory max reduction + exp-sum reduction, numerically stable. |
| **Embedding** | Token + positional embedding lookup with element-wise addition. |
| **KV Cache** | Append-only cache for key/value tensors, avoids recomputing past tokens. |

## How It Works

```
+------------------------------------------------------+
|  GPU Kernel (nvptx64, Rust nightly)                  |
|                                                      |
|  use std::{fs, io, vec};  // real Rust std on GPU    |
|  #[warp_cooperative] async fn  (MIR pass)            |
|  #[warp_async]  -->  WarpFuture (proc macro)         |
|         |                                            |
|  gpu-runtime: hostcall helpers, WarpFuture trait      |
|  gpu-atomics: inline PTX (CAS, shfl, activemask)     |
|  gpu-protocol: packet layout, service IDs             |
|  gpu-libc: libc shim (routes sys calls to hostcall)   |
|         |                                            |
|  ---------- CUDA mapped memory ----------------      |
|         |                                            |
|  Host CPU (gpu-host SDK)                             |
|  GpuRuntime: device init, PTX loading, kernel launch  |
|  HostcallBuffer: poll buffer, serve I/O requests      |
|  MappedBuffer<T>: RAII pinned device-mapped memory    |
+------------------------------------------------------+
```

### Lock-Free Hostcall Protocol

GPU-host communication uses a ROCm-inspired two-stack design over CUDA mapped memory:

- **Free stack**: Available packets for GPU to claim (one CAS per warp)
- **Ready stack**: Filled packets for host to process
- **Per-block sharding**: Reduces CAS contention at scale
- **Warp-granular packets**: 32 lanes share one packet (not 32 separate ones)
- **Sideband buffer**: Separate mapped memory for bulk data beyond the 56-byte packet payload

### Warp-Cooperative GPU Async — Two Approaches

**`#[warp_cooperative]` MIR Pass** (recommended) — Standard Rust `async fn` with standard `Future` trait. A custom rustc MIR pass inserts `bar.warp.sync` + `shfl.sync` at every `.await` suspension point. Requires patched toolchain.

**`#[warp_async]` Proc Macro** — Sequential Rust code with `warp_*!()` macro calls, compiled into a `WarpFuture` state machine. Works on stock nightly.

| Feature | `#[warp_cooperative]` | `#[warp_async]` |
|---------|----------------------|-----------------|
| Syntax | Standard `async fn` + `.await` | `warp_*!()` macros |
| Toolchain | Patched rustc | Stock nightly |
| Control flow | `if`/`match`/`loop`/`?` | `if`/`match`/`loop`/`?` |
| Warp convergence | `bar.warp.sync` at `.await` | State machine by construction |
| Future types | `GpuOpenFuture`, `GpuReadFuture`, etc. | `warp_open!()`, `warp_read!()`, etc. |

All 32 lanes always agree on the current state — warp convergence is maintained by construction.

## Formal Verification (TLA+)

The lock-free CAS hostcall protocol has been formally verified using [TLA+](https://lamport.azurewebsites.net/tla/tla.html). The spec lives in the [`formal/`](formal/) directory.

**Safety** — 367 million states explored with no violations:
- No double-ownership of packets (two actors never hold the same slot)
- No lost packets (every packet is always reachable from a stack or in use)
- No ABA corruption (epoch tags prevent stale CAS from corrupting the free stack)

**Liveness** — 337K states under fairness constraints:
- Every GPU request eventually receives a host response
- Every packet eventually returns to the free pool for reuse

The verification confirms that the ABA prevention mechanism via epoch tags is essential — removing it produces counterexamples within seconds. See `formal/` for the full TLA+ specification and model-checking configuration.

## Crate Map

```
crates/
  core/
    gpu-host/          Host-side SDK: GpuRuntime, HostcallBuffer, MappedBuffer
    gpu-protocol/      Shared constants: packet layout, service IDs, error codes
    gpu-runtime/       GPU-side runtime: index, math, warp, block, nn, executor, channels
    gpu-atomics/       System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
    gpu-libc/          Minimal libc shim for GPU: routes sys calls to hostcall
    gpu-critical-section/  No-op critical-section impl for GPU Embassy executor
  kernel/
    gpu-kernel/        Main GPU kernel crate (94+ kernels: compute, hostcall, pipeline)
    gpu-kernel-std/    GPU kernels using patched Rust std (println!, Vec, File, stdin)
  macro/
    warp-macro/        #[warp_async] proc macro (generates WarpFuture state machines)

rustc-patches/       Custom MIR pass patches for rustc: inserts bar.warp.sync at async yield points
scripts/             build-toolchain.sh/.bat: build patched rustc, ci-lint.sh: local CI checks

examples/
  hello-gpu/         4 demos: vector_add, print, file I/O, bulk transfer
  async-pipeline/    #[warp_cooperative] async fn: small I/O + bulk sideband I/O
  parallel-search/   32-lane GPU grep with shfl.sync warp reduction
  async-io/          Multi-file write pipeline + read-transform-write
  vector-math/       SAXPY, dot product, softmax (pure compute, no hostcall)
  tcp-echo/          GPU-initiated TCP networking: connect, send, receive, close
  warp-cooperative/  MIR pass verification: simple, multi-await, 6-stage pipeline (requires patched rustc)
```

## Capabilities

| Category | What works |
|----------|-----------|
| **I/O from GPU** | `println!()`, `std::fs::File`, `std::io::stdin()`, bulk sideband transfer (sync + async Futures), TCP networking (connect/send/recv) |
| **Std library** | `Vec`, `String`, `Box`, `format!()`, `?` operator — real Rust std via patched PAL (multi-thread safe) |
| **Error handling** | `Result<T, E>` propagation from GPU to host, `std::io::Error` |
| **Async runtime** | `block_on()` executor, Embassy on GPU, `futures::join!()`, per-thread and per-warp executors |
| **GPU Async Runtime** | GpuExecutor with dynamic task spawning, oneshot channels, lock-free MPMC work queue |
| **Warp-cooperative** | `#[warp_cooperative]` MIR pass (async fn + .await), `#[warp_async]` proc macro (warp_*! macros) |
| **Compute** | GEMM (f32 FMA), FlashAttention, LayerNorm, GELU, softmax — all in Rust inline PTX |
| **Inference** | GPT-2 small (124M params) with KV cache: 68ms/token (2.07x speedup) |
| **Scaling** | Multi-block with per-block sharding, 512+ concurrent threads |
| **Buffered I/O** | `println!()` auto-buffers via sideband, flushed as single hostcall (O(1) vs O(N)) |
| **Compute Utils** | math (sin/cos/exp/log/tanh/sigmoid), warp reductions (sum/max/min), block reductions, nn (GELU/ReLU/softmax/layer_norm) |
| **Autonomous GPU** | Data-dependent iteration (Newton's method convergence loop), multi-command kernels, cross-launch pipelines, compute pipeline benchmark (1.91x vs multi-launch) |
| **Testing** | `cargo test` integration harness for GPU kernels, 3 proof-of-concept tests |
| **Safety** | GPU panic handler with visible `[GPU PANIC]` messages via hostcall |

## Performance

Hostcall round-trip latency (RTX 3060, SM 86):

| Threads | p50 Latency | Throughput |
|---------|-------------|------------|
| 1 thread | ~42-101 us | 10-15K calls/s |
| 32 (1 warp) | ~1.1 ms | 20-23K calls/s |
| 128 (4 warps) | ~5-6 ms | ~14K calls/s |

GPT-2 inference (RTX 3060, SM 86):

| Metric | f32 FMA |
|--------|---------|
| 12-layer forward pass (seq=5) | ~35ms |
| Per-token generation (with KV cache) | ~68ms/token |
| 50-token generation (3 prompts) | ~3.4s each |

Compute pipeline (single-launch vs multi-launch):

| Approach | Host median | GPU time |
|----------|-------------|----------|
| Single-launch (async pipeline) | 24.1 us | 1.02 us |
| Multi-launch (3 kernels x 1 iter) | 46.1 us | N/A |
| Speedup | 1.91x | -- |

<details>
<summary>Reproduce</summary>

```bash
cd crates/core/gpu-host
cargo run --release 2>&1 | grep -A 10 "Hostcall Latency Benchmark"
```

Numbers vary by ~30% between runs depending on GPU load.
</details>

## Limitations

- **Nightly Rust**: Requires `asm_experimental_arch`, `-Zbuild-std`, PTX target support. `#[warp_cooperative]` needs patched rustc (see build instructions)
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip, not suitable for per-element I/O in hot loops
- **Uniform I/O**: `#[warp_async]` requires all 32 lanes to execute the same I/O sequence
- **Hostcall-limited concurrency**: `println!` and file I/O are multi-thread safe but constrained by the 16-packet hostcall pool — 4 concurrent I/O threads recommended, 32+ threads for pure compute (Vec, String, allocator)
- **Partial std**: Networking and threading primitives are stubbed; `HashMap` and `Mutex` both work on GPU, but `OsRng`/`getrandom` are not available
- **f32 only**: Tensor Core MMA has precision issues with reduced formats; using f32 FMA

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
