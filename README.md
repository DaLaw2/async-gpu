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

# Async Pipeline — #[warp_cooperative] async fn with real I/O (requires patched rustc)
cargo run --manifest-path examples/async-pipeline/host/Cargo.toml

# Vector Math — SAXPY, dot product, softmax (pure GPU compute)
cargo run --manifest-path examples/vector-math/host/Cargo.toml
```

<details>
<summary>All examples</summary>

| Example | Description | Toolchain |
|---------|-------------|-----------|
| `hello-gpu` | Vector add, GPU print, file I/O, bulk sideband | Stock nightly |
| `async-pipeline` | `#[warp_cooperative] async fn` with hostcall Futures | Patched rustc |
| `async-io` | Multi-file write pipeline + read-transform-write | Stock nightly |
| `parallel-search` | 32-lane GPU grep with `shfl.sync` warp reduction | Stock nightly |
| `vector-math` | SAXPY, dot product, softmax (pure compute) | Stock nightly |
| `tcp-echo` | GPU-initiated TCP networking via hostcall | Stock nightly |
| `tokio-offload` | Async kernel launch from tokio runtime | Stock nightly |
| `warp-cooperative` | MIR pass verification tests | Patched rustc |

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

End-to-end transformer inference — real HuggingFace weights, custom BPE tokenizer, 12 transformer layers, KV-cached autoregressive generation. All compute kernels in pure Rust with inline PTX, no CUDA C++ or cuBLAS.

```
--- Greedy autoregressive generation (with KV cache) ---
  [1/3] Prompt: "The capital of France is" -> 5 tokens, generating 50
  Generated: " the capital of the French Republic, and the capital of
  the French Republic is the capital of the French Republic..."
  Time: 3400ms total, 68ms/token  (2.07x faster with KV cache)
  PASSED (50 tokens, no NaN)
```

GPU compute kernels: GEMM (f32 FMA, shared memory tiling), FlashAttention (tiled online softmax, causal masking, KV cache), LayerNorm, GELU, Softmax, Embedding — all in Rust inline PTX.

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

## Crate Map

```
crates/
  core/
    gpu-host/          Host-side SDK: GpuRuntime, HostcallBuffer, MappedBuffer, GpuStream
    gpu-protocol/      Shared constants: packet layout, service IDs, error codes
    gpu-runtime/       GPU-side runtime: index, math, warp, block, nn, executor, channels
    gpu-atomics/       System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
    gpu-libc/          Minimal libc shim for GPU: routes sys calls to hostcall
  kernel/
    gpu-kernel/        Main GPU kernel crate (94+ kernels: compute, hostcall, pipeline)
    gpu-kernel-std/    GPU kernels using patched Rust std (println!, Vec, File, stdin)
  macro/
    warp-macro/        #[warp_async] proc macro (generates WarpFuture state machines)

rustc-patches/       Custom MIR pass patches for rustc
scripts/             Build/CI automation (build-toolchain, ci-lint, pre-push, install-toolchain)
examples/            8 self-contained examples (see Quick Start)
formal/              TLA+ specification and model-checking config
```

## Performance

RTX 3060, SM 86:

| Metric | Value |
|--------|-------|
| Hostcall round-trip (1 thread) | ~42-101 us, 10-15K calls/s |
| Hostcall round-trip (32 threads) | ~1.1 ms, 20-23K calls/s |
| GPT-2 per-token (with KV cache) | ~68ms/token (2.07x faster) |
| Compute pipeline speedup | 1.91x vs multi-launch |

## Limitations

- **Nightly Rust**: Requires `asm_experimental_arch`, `-Zbuild-std`. `#[warp_cooperative]` needs patched rustc
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip, not suitable for per-element I/O in hot loops
- **Partial std**: `HashMap`, `Mutex`, File I/O work; `OsRng`/`getrandom` not available
- **f32 only**: Tensor Core MMA has precision issues with reduced formats; using f32 FMA

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
