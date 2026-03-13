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

- [Rust](https://rustup.rs/) (nightly toolchain auto-installed via `rust-toolchain.toml`)
- NVIDIA GPU (SM 70+) with CUDA driver

### Run

```bash
git clone https://github.com/DaLaw2/async-gpu.git
cd async-gpu
./run-hello-gpu.sh        # Linux/macOS
run-hello-gpu.bat          # Windows
```

One command. The build script compiles the GPU kernel to PTX, builds the host binary, and runs four demos: vector addition, GPU-to-host print, file I/O from GPU, and bulk sideband transfer.

<details>
<summary>Manual build (or run the full test suite)</summary>

```bash
# Build all GPU kernels
cd crates/gpu-kernel
cargo build --release

# Copy PTX to host crate
cp target/nvptx64-nvidia-cuda/release/gpu_kernel.ptx ../gpu-host/kernel.ptx

# Run the full test suite
cd ../gpu-host
cargo run --release
```
</details>

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
  [HOST] BULK READ: fd=1 51208 bytes read   (100 vectors × 128 dims)
  [HOST] BULK READ: fd=2 516 bytes read     (query vector)
  [HOST] BULK WRITE: fd=3 84 bytes written  (top-K results)
  Elapsed: 6.434ms
    rank 1: id=42 score=1.0000   ← exact match
    rank 2: id=82 score=0.2103
    rank 3: id=18 score=0.0913
```

### GPT-2 Inference (124M Parameters)

End-to-end transformer inference running entirely on GPU — token embedding, 12 transformer layers (LayerNorm → Multi-Head Attention → FFN → residual connections), LM head projection, and greedy autoregressive generation:

```
--- GPT-2 Forward Pass (12 layers) ---
  Prompt: "The capital of France is"
  Token IDs: [464, 3139, 286, 4881, 318]
  Forward pass: 12 layers × (LayerNorm → MHA → FFN → residual)
  Elapsed: ~30ms (f16 Tensor Core) / ~34ms (f32 FMA)

--- Greedy Generation (20 tokens) ---
  "The capital of France is - and the French government has been ..."
  59ms/token (full recompute, no KV cache)
```

All compute kernels are written in pure Rust with inline PTX assembly — Tensor Core MMA (`mma.sync.aligned.m16n8k16`), FlashAttention with causal masking, tiled GEMM with shared memory, LayerNorm, GELU, softmax. Real GPT-2 weights loaded from HuggingFace safetensors format with a custom BPE tokenizer validated against HuggingFace.

### GPU-Driven Multi-Step Pipeline with Branching

The GPU autonomously chooses processing paths based on parameters and hostcall results:

```
--- Autonomous Pipeline (Mode 1: read + classify) ---
    [GPU] auto: start
  [HOST] FILE OPEN: "gpu_autonomous.txt" flags=0 -> fd=1
  [HOST] FILE READ: fd=1 21 bytes read
  [HOST] FILE CLOSE: fd=1 closed
    [GPU] auto: large-payload          ← GPU decided: 21 bytes > 10
    [GPU] auto: done
```

Three modes: file write pipeline (3 hostcall steps), read + classify (4 steps + GPU-decided branch), roundtrip verification (6 steps + GPU-decided verify). All generated from `#[warp_async]` with `match` + `if/else`.

## How It Works

```
+------------------------------------------------------+
|  GPU Kernel (nvptx64, Rust nightly)                  |
|                                                      |
|  #[warp_async]  ──►  WarpFuture state machine        |
|    match/if/loop       (auto-generated)              |
|         │                                            |
|  gpu-runtime: hostcall helpers, WarpFuture trait      |
|  gpu-atomics: inline PTX (CAS, shfl, activemask)     |
|  gpu-protocol: packet layout, service IDs             |
|         │                                            |
|  ───────┼──── CUDA mapped memory ────────────────    |
|         │                                            |
|  Host CPU                                            |
|  hostcall listener: poll buffer, serve I/O            |
|  Services: print, file, bulk read/write, panic, ...   |
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
| `if`/`else` | Lane 0 evaluates condition, broadcasts decision via `shfl.sync` |
| `match` | Lane 0 evaluates scrutinee, maps to arm index, broadcasts |
| `loop` + `break` | DECISION state: break condition checked by lane 0, broadcast |
| Nesting | Full support (if inside match, match inside loop, etc.) |
| Variable capture | `let fd = warp_open!(...)` stores result as struct field, available in later states |

All 32 lanes always agree on the current state — warp convergence is maintained by construction.

### WarpFuture: Warp-Cooperative Async

All 32 GPU lanes in a warp share a single state machine:

- **I/O phases**: Cooperative hostcall — one packet per warp, not 32
- **Compute phases**: Each lane works independently on its data slice
- **Reconvergence**: `syncwarp()` ensures all lanes rejoin before the next I/O phase

### GPU Compute Kernels

All compute is written in Rust with inline PTX assembly — no CUDA C++, no cuBLAS:

| Kernel | Implementation |
|--------|---------------|
| **GEMM** | Tensor Core `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, multi-block tiling, f32 accumulation. Handles arbitrary dimensions up to 768x768+. Also: pure f32 FMA variant for precision validation. |
| **FlashAttention** | Tiled attention with online softmax, causal masking, multi-head support. Scales to seq=256+. |
| **LayerNorm** | Shared-memory parallel reduction for mean + variance, per-element affine transform. |
| **GELU** | Approximation via `ex2.approx.f32` PTX instruction. |
| **Softmax** | Shared-memory max reduction + exp-sum reduction, numerically stable. |
| **Embedding** | Token + positional embedding lookup with element-wise addition. |

## Capabilities

| Category | What works |
|----------|-----------|
| **I/O from GPU** | `println!()`, file open/read/write/close, bulk sideband transfer |
| **Async runtime** | Embassy executor on GPU, `futures::join!()`, per-thread and per-warp executors |
| **Std library** | `Vec`, `String`, `format!()` via patched std with hostcall-backed libc shim |
| **Warp-cooperative** | `#[warp_async]` with if/else, loop/break, match, nested control flow |
| **Compute** | Tensor Core GEMM, FlashAttention, LayerNorm, GELU, softmax — all in Rust inline PTX |
| **Inference** | End-to-end GPT-2 small (124M params): tokenize → embed → 12 transformer layers → generate |
| **Scaling** | Multi-block with per-block sharding, 512+ concurrent threads |
| **Safety** | GPU panic handler with visible `[GPU PANIC]` messages via hostcall |

## Project Structure

```
async_gpu/
├── crates/
│   ├── gpu-protocol/       Shared packet layout, service IDs (#![no_std])
│   ├── gpu-atomics/        System-scope atomics + warp intrinsics via inline PTX
│   ├── gpu-runtime/        Hostcall helpers, WarpFuture trait, sideband I/O
│   ├── gpu-libc/           Minimal libc shim routing syscalls through hostcall
│   ├── gpu-kernel/         GPU kernels compiled to PTX (nvptx64)
│   ├── gpu-host/           Host-side CUDA harness, hostcall listener, test runner
│   ├── warp-macro/         #[warp_async] proc macro for WarpFuture generation
│   └── (6 test crates)     Async pipeline, Embassy, multi-warp, std tests
├── examples/hello-gpu/     One-command demo (host + kernel)
└── .research/              163 cycles of autonomous research findings
```

## Performance

Hostcall round-trip latency (RTX 3060, SM 86):

| Threads | p50 Latency | Throughput |
|---------|-------------|------------|
| 1 thread | ~42-101 us | 10-15K calls/s |
| 32 (1 warp) | ~1.1 ms | 20-23K calls/s |
| 128 (4 warps) | ~5-6 ms | ~14K calls/s |

Per-block sharding reduces CAS contention at higher thread counts. Latency is dominated by host I/O processing, not the protocol itself.

GPT-2 inference (RTX 3060, SM 86, seq=5 prompt):

| Metric | f16 Tensor Core | f32 FMA |
|--------|----------------|---------|
| 12-layer forward pass | ~30ms | ~34ms |
| Per-token generation | ~59ms | — |
| GEMM overhead (f32 vs f16) | baseline | +13% |

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
- **Limited std**: File I/O, print, Vec, String work; networking and threading are stubbed
- **No KV cache**: GPT-2 generation recomputes all layers per token (O(n^2) total)

## Research Background

This project was built through an autonomous Think/Do/Check research loop: 163 cycles, 35+ themes, 162 verified experiments, 10 architecture decision records. Full research state and findings are in `.research/`.

Key research milestones:
- **Cycles 1-47**: Foundation — toolchain, hostcall protocol, GPU atomics, libc shim, Embassy async executor, std on GPU
- **Cycles 48-89**: Scaling — multi-block, per-block sharding, bulk data transfer, GPU panic handler, WarpFuture trait, `#[warp_async]` proc macro, hybrid executor
- **Cycles 90-108**: Product — CI pipeline, API documentation, hello-gpu example, control flow in proc macro (if/else, loop, match)
- **Cycles 109-163**: GPU inference — Tensor Core MMA, tiled GEMM, FlashAttention, LayerNorm, GELU, safetensors loading, BPE tokenizer, 12-layer GPT-2 forward pass, autoregressive generation

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
