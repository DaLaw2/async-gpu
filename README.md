# async_gpu — Rust Async/Await on GPU

An experimental reproduction of [VectorWare](https://www.vectorware.com/)'s technology for running Rust's standard library and async/await on NVIDIA GPUs via CUDA, as described in their blog posts:

- [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/)
- [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/)

## What This Is

This project explores whether Rust's `std` library and async/await can run on GPU hardware using the `nvptx64-nvidia-cuda` target. Rather than writing raw CUDA kernels, the idea is to let GPU threads use familiar Rust abstractions — `File::open()`, `println!()`, `async/await` — with I/O routed to the host CPU through a shared-memory hostcall protocol.

The project is structured as an autonomous research loop with 12 completed research themes and 47 verified experiments.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  GPU Kernel (nvptx64)                                │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Rust std     │  │  Embassy    │  │  gpu-kernel  │ │
│  │  (patched)    │  │  executor   │  │  (PTX asm)  │ │
│  │  File, println│  │  async/await│  │  atomics    │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                 │                │         │
│  ┌──────▼─────────────────▼────────────────▼──────┐  │
│  │  gpu-libc shim → gpu-protocol → hostcall buf   │  │
│  └────────────────────────┬───────────────────────┘  │
│                           │ shared memory (CUDA)     │
├───────────────────────────┼──────────────────────────┤
│  Host CPU                 │                          │
│  ┌────────────────────────▼───────────────────────┐  │
│  │  gpu-host: hostcall listener + CUDA runtime    │  │
│  │  (file I/O, memory alloc, stdin, time, etc.)   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Crates

| Crate | Purpose |
|-------|---------|
| `gpu-protocol` | Shared hostcall packet layout, service IDs, error encoding (`#![no_std]`) |
| `gpu-atomics` | System-scope GPU atomics via inline PTX (`atom.*.sys`, `membar.sys`) |
| `gpu-critical-section` | No-op critical-section impl for per-thread Embassy executors |
| `gpu-libc` | Minimal libc shim routing `write`/`read`/`open`/`close` through hostcall |
| `gpu-kernel` | GPU kernel crate (`cdylib` → PTX), integration tests |
| `gpu-host` | Host-side CUDA harness using `cudarc`, hostcall listener, test runner |

### Test Crates

| Crate | What It Tests |
|-------|---------------|
| `embassy-test` | Embassy executor compiles and runs on nvptx64 |
| `async-hostcall-test` | Async hostcall I/O with futures combinators |
| `async-pipeline-test` | Multi-step async I/O pipeline (read → process → write) |
| `multi-warp-test` | Multi-warp scaling (32+ threads, concurrent hostcall) |
| `gpu-std-test` | `std::fs::File` I/O on GPU |
| `std-build-test` | Vendored std with `-Zbuild-std=std`, `println!()` |

## Key Technical Decisions

1. **Hostcall Protocol (ADR-3)**: ROCm-style lock-free two-stack protocol with warp-granular packets (32 lanes × 8 u64 slots), tagged pointers for ABA prevention, doorbell counter, and adaptive-timeout host polling via `cudaMemHostAlloc`.

2. **Embassy Executor (ADR-2/4)**: Each GPU thread runs its own Embassy executor — no cross-thread synchronization needed. Fat LTO links the executor across crate boundaries. Critical section is a no-op (safe because single-thread-per-executor).

3. **Patched std (ADR-1)**: Minimal patches to vendored Rust std source — `cfg_select!` gates for nvptx64 covering the allocator (slab+bitmap, 8 size classes), stdio (routed through hostcall), thread-local storage (disabled), and OnceLock (bypassed for `println!()`).

4. **System-scope Atomics**: LLVM's `core::sync::atomic` lacks `.sys` scope qualifiers needed for GPU-CPU shared memory. All cross-device atomics use inline PTX through `gpu-atomics`.

## Current Status

All 12 active research themes are **completed** (47/47 tasks):

- Toolchain, hostcall, gpu-std, async-runtime, integration
- Atomics, std-pal, product, allocator, multiblock
- OnceLock bypass, error handling

## Strengths

- **Familiar Rust API on GPU**: GPU code can use `File::open()`, `println!()`, `async/await` — no raw CUDA C needed.
- **Lock-free hostcall**: The two-stack protocol enables GPU↔host I/O without mutex contention, scaling to 512+ threads across multiple blocks.
- **Async/await works**: Embassy's poll-based executor runs on GPU with minimal overhead. Multiple futures can be composed with standard combinators (`join`, `select`).
- **Cross-platform host**: Error propagation uses `io::ErrorKind` mapping (not raw errno), so the host side works on both Linux and Windows.
- **Minimal std patching**: Only ~4 patch files touch the vendored std source, keeping upgrade friction low.
- **No custom rustc fork**: Everything builds on stock nightly rustc with `-Zbuild-std`.

## Limitations and Shortcomings

- **Nightly-only**: Requires Rust nightly for `#![feature(abi_ptx)]`, `-Zbuild-std`, `core::arch::asm!` with PTX, and other unstable features. Breakage on toolchain updates is expected.
- **NVIDIA-only**: Targets `nvptx64-nvidia-cuda` exclusively. No AMD/Intel GPU support. Requires CUDA toolkit and an SM70+ GPU.
- **No warp-cooperative execution**: Each thread runs independently. There is no warp-level collective communication or cooperative kernel design — threads cannot share async work.
- **Static thread limits**: Per-thread Embassy executor storage is allocated as fixed-size static arrays sized at compile time. Dynamic scaling requires the slab allocator, which adds complexity.
- **Performance not characterized**: The `benchmark` theme is parked. There are no latency/throughput measurements for hostcall overhead, async state machine cost, or comparison against native CUDA.
- **Hostcall is synchronous on host side**: The host listener polls in a busy loop. There is no interrupt-driven notification from GPU to host — this wastes CPU cycles.
- **Limited std coverage**: Only `std::fs` (File), `std::io` (print/stdin), and basic allocation work. Networking, threading, and most of std are stubbed out.
- **Fat LTO required**: All GPU crates must link with Fat LTO, increasing compile times significantly.
- **Single-kernel execution model**: No support for CUDA graphs, streams, or overlapping kernel execution on the host side.
- **No error recovery on GPU**: If a hostcall times out, the GPU thread returns an error code but has no way to retry or recover gracefully.

## Building

### Prerequisites

- Rust nightly toolchain with `nvptx64-nvidia-cuda` target
- CUDA toolkit 12.0+
- NVIDIA GPU (SM70+, e.g., Volta/Turing/Ampere/Ada/Hopper)

### Steps

```bash
# 1. Prepare patched std source
cd std-patches && bash apply.sh && cd ..

# 2. Build a GPU test crate (e.g., gpu-kernel)
cd crates/gpu-kernel
cargo +nightly build -Zbuild-std=core,alloc --target nvptx64-nvidia-cuda --release

# 3. Build and run the host harness
cd crates/gpu-host
cargo run --release
```

> Note: The exact build process varies by crate. Some test crates require `__CARGO_TESTS_ONLY_SRC_ROOT` set to the patched-std directory for `-Zbuild-std=std` support.

## Research Structure

This project uses an autonomous research loop (Think → Do → Check) managed by a state machine in `.research/state.toml`. Findings, brainstorms, and reviews are stored in `.research/findings/`. See `CLAUDE.md` for the full workflow specification.

## License

This is an experimental research project. No license has been specified yet.

## Acknowledgments

This work is a reproduction and exploration of techniques described by [VectorWare](https://www.vectorware.com/). All credit for the original ideas goes to them.
