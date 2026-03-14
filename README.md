# async_gpu — Rust Async/Await on NVIDIA GPUs

**What if the GPU could drive its own computation?** Open files, read data, branch on results, loop until convergence, write output — all from GPU code, with zero CPU orchestration between steps.

async_gpu makes this real: **Rust async/await running natively on NVIDIA GPUs**, with a proc macro that turns sequential GPU code into warp-cooperative state machines — and GPU compute kernels powerful enough to run **end-to-end GPT-2 inference** entirely from Rust inline PTX.

```rust
#[warp_async]
unsafe fn autonomous_pipeline(buf: *mut u8, mode: u64) -> bool {
    warp_print!(buf, b"auto: start");
    match mode {
        0 => {
            let fd = warp_open!(buf, b"data.txt", FILE_OPEN_WRITE_CREATE);
            warp_write!(buf, fd, b"GPU-autonomous-output", 21);
            warp_close!(buf, fd);
            warp_print!(buf, b"auto: file-written");
        }
        1 => {
            let fd = warp_open!(buf, b"data.txt", FILE_OPEN_READ);
            let n = warp_read!(buf, fd, 56);
            warp_close!(buf, fd);
            if n > 10 {
                warp_print!(buf, b"auto: large-payload");
            } else {
                warp_print!(buf, b"auto: small-payload");
            }
        }
        _ => { warp_print!(buf, b"auto: unknown-mode"); }
    }
    warp_print!(buf, b"auto: done");
}
```

The `#[warp_async]` proc macro compiles this into a warp-cooperative state machine where all 32 GPU lanes share one state, decisions are broadcast from lane 0 via `shfl.sync`, and hostcall I/O happens with a single CAS per warp. The code above replaces 150+ lines of hand-written state machine.

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

# Async I/O — multi-file write pipeline + read-transform-write
cargo run --manifest-path examples/async-io/host/Cargo.toml

# Vector Math — SAXPY, dot product, softmax (pure GPU compute)
cargo run --manifest-path examples/vector-math/host/Cargo.toml
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

## Examples

All examples use the **gpu-host SDK** — three core types for GPU programming:

| Type | Purpose |
|------|---------|
| `GpuRuntime` | Device init, PTX loading, kernel launch, data transfer |
| `HostcallBuffer` | GPU-host RPC communication (print, file I/O, stdin) |
| `MappedBuffer<T>` | RAII pinned device-mapped memory (auto-freed on drop) |

### hello-gpu — Hostcall Basics

Four demos: vector addition (pure compute), GPU-to-host print, file write from GPU, and bulk sideband read.

### async-io — Multi-Step File I/O

Write pipeline: GPU creates 3 files in sequence. Transform pipeline: GPU reads a file, uppercases on-device, writes result — all from one kernel launch.

### vector-math — Pure GPU Compute

SAXPY, dot product (GPU multiply + CPU reduce), and softmax (multi-pass GPU-CPU cooperation with numerically stable exp via PTX `ex2.approx.ftz.f32`).

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
|  #[warp_async]  -->  WarpFuture state machine        |
|    match/if/loop       (auto-generated)              |
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

### `#[warp_async]` Proc Macro

Transforms sequential Rust code with `warp_*!()` calls into a `WarpFuture` state machine:

| Feature | How it works |
|---------|-------------|
| Sequential calls | Each `warp_*!()` becomes an INIT + WAIT state pair |
| `.await` | Standard `impl Future` polled warp-cooperatively (lane 0 polls, broadcast result) |
| `?` operator | Error discriminant broadcast via `shfl.sync`, short-circuit on `Err` |
| `if`/`else` | Lane 0 evaluates condition, broadcasts decision via `shfl.sync` |
| `match` | Lane 0 evaluates scrutinee, maps to arm index, broadcasts |
| `loop` + `break` | DECISION state: break condition checked by lane 0, broadcast |
| Nesting | Full support (if inside match, match inside loop, etc.) |
| Variable capture | `let fd = warp_open!(...)` stores result as struct field, available in later states |

All 32 lanes always agree on the current state — warp convergence is maintained by construction.

## Crate Map

```
crates/
  gpu-host/          Host-side SDK: GpuRuntime, HostcallBuffer, MappedBuffer
                     Also contains GPT-2 inference binary (feature-gated)
  gpu-protocol/      Shared constants: packet layout, service IDs, error codes
  gpu-runtime/       GPU-side facade: hostcall helpers, WarpFuture trait, sideband I/O
  gpu-atomics/       System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
  gpu-libc/          Minimal libc shim for GPU: routes sys calls to hostcall
  gpu-critical-section/  No-op critical-section impl for GPU Embassy executor
  warp-macro/        #[warp_async] proc macro (generates WarpFuture state machines)
  gpu-kernel/        Main GPU kernel crate (94 kernels: compute, hostcall, pipeline)
  gpu-kernel-std/    GPU kernels using patched Rust std (println!, Vec, File, stdin)

rustc-patches/       Skeleton MIR pass for warp-cooperative async/await (future rustc integration)

examples/
  hello-gpu/         4 demos: vector_add, print, file I/O, bulk transfer
  async-io/          Multi-file write pipeline + read-transform-write
  vector-math/       SAXPY, dot product, softmax (pure compute, no hostcall)
```

## Capabilities

| Category | What works |
|----------|-----------|
| **I/O from GPU** | `println!()`, `std::fs::File`, `std::io::stdin()`, bulk sideband transfer |
| **Std library** | `Vec`, `String`, `Box`, `format!()`, `?` operator — real Rust std via patched PAL (multi-thread safe) |
| **Error handling** | `Result<T, E>` propagation from GPU to host, `std::io::Error` |
| **Async runtime** | Embassy executor on GPU, `futures::join!()`, per-thread and per-warp executors |
| **Warp-cooperative** | `#[warp_async]` with if/else, loop/break, match, `.await`, `?` operator |
| **Compute** | GEMM (f32 FMA), FlashAttention, LayerNorm, GELU, softmax — all in Rust inline PTX |
| **Inference** | GPT-2 small (124M params) with KV cache: 68ms/token (2.07x speedup) |
| **Scaling** | Multi-block with per-block sharding, 512+ concurrent threads |
| **Buffered I/O** | `println!()` auto-buffers via sideband, flushed as single hostcall (O(1) vs O(N)) |
| **Autonomous GPU** | Data-dependent iteration (Newton's method convergence loop), multi-command kernels, cross-launch pipelines |
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

<details>
<summary>Reproduce</summary>

```bash
cd crates/gpu-host
cargo run --release 2>&1 | grep -A 10 "Hostcall Latency Benchmark"
```

Numbers vary by ~30% between runs depending on GPU load.
</details>

## Limitations

- **Nightly Rust**: Requires `asm_experimental_arch`, `-Zbuild-std`, PTX target support
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip, not suitable for per-element I/O in hot loops
- **Uniform I/O**: `#[warp_async]` requires all 32 lanes to execute the same I/O sequence
- **Hostcall-limited concurrency**: `println!` and file I/O are multi-thread safe but constrained by the 16-packet hostcall pool — 4 concurrent I/O threads recommended, 32+ threads for pure compute (Vec, String, allocator)
- **Partial std**: Networking and threading primitives are stubbed; `HashMap` works (address-based seed) but `OsRng`/`getrandom` are not available
- **f32 only**: Tensor Core MMA has precision issues with reduced formats; using f32 FMA

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
