# Changelog

## v0.2.0 — 2026-03-15

TCP networking from GPU kernels and developer experience improvements.

### Networking

- **TCP hostcall services**: 8 new services (connect, write, read, close, bind, accept, bulk_write, bulk_read) enabling GPU kernels to make network connections
- **GPU-side TCP Futures**: `GpuTcpConnectFuture`, `GpuTcpWriteFuture`, `GpuTcpReadFuture`, `GpuTcpCloseFuture`, `GpuTcpBulkWriteFuture`, `GpuTcpBulkReadFuture`
- **Unified fd namespace**: `FdResource` enum (`File | TcpStream | TcpListener`) — sockets and files share the same fd table
- **TCP bulk I/O**: Up to 1MB TCP transfers via sideband buffer (`SERVICE_TCP_BULK_WRITE` / `SERVICE_TCP_BULK_READ`)
- **TCP server support**: `SERVICE_TCP_BIND` + `SERVICE_TCP_ACCEPT` for GPU-side server patterns

### Examples

- **tcp-echo**: GPU kernel connects to a local TCP echo server, sends a message, reads the echoed response
- **parallel-search**: 32-lane warp-parallel byte-pattern search over bulk-read file data

### Infrastructure

- CI coverage expanded: all examples (kernel PTX + host check) in `ci-lint.sh`
- Per-example `README.md` with architecture, running instructions, expected output
- `CONTRIBUTING.md` developer guide
- CI badge and license badge in README
- Convenience `run.sh` / `run.bat` scripts for examples
- Toolchain build scripts split into `.sh` (Linux) + `.bat` (Windows)
- `VALIDATION.md` first-run hardware validation checklist
- 309 research cycles, 329 completed tasks

### Formal Verification

- **TLA+ specification** of the CAS hostcall protocol (750 lines)
- **Safety verified**: 367M states, no double-ownership, no lost packets, no ABA
- **Liveness verified**: 337K states, all packets complete full lifecycle

### Code Quality

- ~83 `// SAFETY` comments added to all high-risk unsafe blocks
- Module-level `//!` docs for `gpu-host` lib.rs and hostcall.rs

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
