# Changelog

## v0.1.0 — 2026-03-15

First public release. Rust async/await running natively on NVIDIA GPUs.

### Core Features

- **Lock-free hostcall protocol**: GPU-host RPC over CUDA mapped memory with per-block sharding and ABA-tagged lock-free stacks
- **Warp-cooperative async/await**: Custom rustc MIR pass (`#[warp_cooperative]`) inserts `bar.warp.sync` at every `.await` yield point for automatic warp convergence
- **`#[warp_async]` proc macro**: Alternative warp-cooperative coding style using `warp_*!()` macros (works on stock nightly)
- **Real Rust `std` on GPU**: `println!`, `Vec`, `String`, `Box`, `std::fs::File`, `std::io::stdin()` via patched std + gpu-libc shim
- **GPU error handling**: `Result<T, E>` propagation from GPU to host with `std::io::Error` support
- **GPU debugging**: `gpu_trace!()` structured tracing, `gpu_assert!()`, flight recorder with conditional compilation
- **`block_on()` executor**: Spin-poll executor for driving `Future`s on GPU with nanosleep yield

### I/O

- File I/O: open, read, write, close via hostcall (up to 48 bytes inline)
- Bulk sideband I/O: up to 1MB transfers via `SERVICE_BULK_READ` / `SERVICE_BULK_WRITE`
- Async I/O Futures: `GpuOpenFuture`, `GpuReadFuture`, `GpuWriteFuture`, `GpuCloseFuture`, `GpuBulkReadFuture`, `GpuBulkWriteFuture`
- Buffered print: `println!()` auto-buffers via sideband, flushed as single hostcall
- Stdin: `std::io::stdin().read_line()` proxied through hostcall

### GPU Compute

- GPT-2 inference (124M params): embedding, 12-layer transformer, KV-cached autoregressive generation
- GEMM: f32 FMA with shared memory tiling
- FlashAttention: tiled attention with online softmax, causal masking, KV cache
- LayerNorm, GELU, softmax, embedding — all in Rust inline PTX

### SDK

- `gpu-host` library crate: `GpuRuntime`, `HostcallBuffer`, `MappedBuffer<T>`
- 4 examples: `hello-gpu`, `async-pipeline`, `async-io`, `vector-math`
- Automated PTX compilation via `build.rs` in each example

### Infrastructure

- Patched toolchain build scripts (Linux `.sh` + Windows `.bat`)
- CI lint script with PTX validation
- `ARCHITECTURE.md` documenting system design
- 290 research cycles, 292 completed tasks

### Requirements

- Rust nightly (2026-03-11)
- NVIDIA GPU (SM 70+) with CUDA 12.x driver
- Patched rustc for `#[warp_cooperative]` (optional — examples work on stock nightly)
