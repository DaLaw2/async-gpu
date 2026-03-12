# async_gpu — Rust Async/Await on GPU

An experimental reproduction of [VectorWare](https://www.vectorware.com/)'s technology for running Rust's standard library and async/await on NVIDIA GPUs via CUDA, as described in their blog posts:

- [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/)
- [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/)

## What This Is

This project explores whether Rust's `std` library and async/await can run on GPU hardware using the `nvptx64-nvidia-cuda` target. Rather than writing raw CUDA kernels, the idea is to let GPU threads use familiar Rust abstractions — `File::open()`, `println!()`, `async/await` — with I/O routed to the host CPU through a shared-memory hostcall protocol.

The project is structured as an autonomous research loop with 17+ completed research themes and 68 verified experiments.

## Architecture

```
+------------------------------------------------------+
|  GPU Kernel (nvptx64)                                |
|  +--------------+  +-------------+  +-------------+ |
|  |  Rust std     |  |  Embassy    |  |  gpu-runtime | |
|  |  (patched)    |  |  executor   |  |  (facade)   | |
|  |  File, println|  |  async/await|  |  prelude    | |
|  +------+-------+  +------+------+  +------+------+ |
|         |                 |                |         |
|  +------v-----------------v----------------v------+  |
|  |  gpu-libc shim -> gpu-protocol -> hostcall buf |  |
|  +------------------------+----------------------+   |
|                           | shared memory (CUDA)     |
+---------------------------+--------------------------+
|  Host CPU                 |                          |
|  +------------------------v-----------------------+  |
|  |  gpu-host: hostcall listener + CUDA runtime    |  |
|  |  adaptive polling + dedicated I/O thread       |  |
|  +------------------------------------------------+  |
+------------------------------------------------------+
```

### Crates

| Crate | Purpose |
|-------|---------|
| `gpu-protocol` | Shared hostcall packet layout, service IDs, error encoding (`#![no_std]`) |
| `gpu-atomics` | System-scope GPU atomics via inline PTX (`atom.*.sys`, `membar.sys`) |
| `gpu-critical-section` | No-op critical-section impl for per-thread Embassy executors |
| `gpu-runtime` | Facade crate: re-exports gpu-protocol + gpu-atomics + hostcall helpers via `prelude` |
| `gpu-libc` | Minimal libc shim routing `write`/`read`/`open`/`close` through hostcall |
| `gpu-kernel` | GPU kernel crate (`cdylib` -> PTX), integration tests + benchmark kernels |
| `gpu-host` | Host-side CUDA harness using `cudarc`, hostcall listener, test runner |

### Test Crates

| Crate | What It Tests |
|-------|---------------|
| `embassy-test` | Embassy executor compiles and runs on nvptx64 |
| `async-hostcall-test` | Async hostcall I/O with futures combinators |
| `async-pipeline-test` | Multi-step async I/O pipeline (read -> process -> write) |
| `multi-warp-test` | Multi-warp scaling (32+ threads, concurrent hostcall) |
| `gpu-std-test` | `std::fs::File` I/O on GPU |
| `std-build-test` | Vendored std with `-Zbuild-std=std`, `println!()` |

## Performance

Measured on RTX 3060 (SM_86) with the NOP hostcall latency benchmark (benchmark.2). Each thread performs 10 sequential NOP hostcalls with `%globaltimer` timing.

### Hostcall Round-Trip Latency

| Threads | Packets | p50 | p95 | p99 | Mean | CAS retries/call | Throughput |
|---------|---------|-----|-----|-----|------|-------------------|------------|
| 1 | 4 | 19-25 us | 19-25 us | 19-25 us | 19-25 us | 0.0 | 26-41K calls/s |
| 32 | 64 | 1.0 ms | 1.3 ms | 1.3 ms | 1.0 ms | 3-7 | 20-24K calls/s |
| 128 | 64 | 5.8 ms | 7.3 ms | 45 ms | 6.2 ms | 28-43 | 14K calls/s |
| 512 | 64 | 14 ms | 38 ms | 115 ms | 22 ms | 42-52 | 8K calls/s |

**Key observations:**
- Single-thread latency (~20 us) is competitive with CUDA C++ cooperative polling protocols (typically 5-15 us)
- CAS retries dropped ~6x at 32 threads after nightly upgrade (1.91 -> 1.96, improved LLVM codegen)
- Throughput **does not scale** with thread count — CAS contention on the lock-free free-stack is the bottleneck
- At 512 threads with 64 packets, **~70% of threads are starved** (only 1256-1651/5120 calls completed)
- Host listener uses I/O thread separation (ADR-6): fast services (NOP/PRINT/TIME/PANIC) inline, blocking FILE I/O offloaded

### PTX Register Pressure (Virtual Registers)

| Kernel | Est. Regs | Stack Spill? | Theoretical Occupancy (SM_86) |
|--------|-----------|--------------|-------------------------------|
| vector_add | 31 | No | 100% |
| hostcall_print_hello (sync) | 92 | No | 46% |
| async_hostcall_single | 57 | Yes | 73% |
| async_hostcall_two | 82 | Yes | 50% |
| pipeline_kernel (async) | 57 | Yes | 73% |

All async kernels use stack spilling (local memory) for Embassy executor state. Virtual register counts are moderate — actual hardware counts (via cuobjdump) would be lower.

### When to Use This

- **Debugging GPU kernels**: `println!()` from GPU at ~20 us per call — useful for development
- **Coarse-grained I/O**: File setup/teardown, configuration reads — 20 us overhead is negligible
- **Async I/O pipelines**: Embassy drives ~77K poll cycles/s per thread, adequate for I/O-bound tasks
- **NOT for**: Per-element I/O, high-throughput data transfer, or latency-critical hot loops

## Key Technical Decisions

1. **Hostcall Protocol (ADR-3)**: ROCm-style lock-free two-stack protocol with warp-granular packets (32 lanes x 8 u64 slots), tagged pointers for ABA prevention, doorbell counter, and CONTROL_FILLED state to prevent duplicate processing. Host uses adaptive polling (spin 10 us, then sleep 100 us).

2. **Embassy Executor (ADR-2/4)**: Each GPU thread runs its own Embassy executor — no cross-thread synchronization needed. Fat LTO links the executor across crate boundaries. Critical section is a no-op (safe because single-thread-per-executor).

3. **Patched std (ADR-1)**: Minimal patches to vendored Rust std source — `cfg_select!` gates for nvptx64 covering the allocator (slab+bitmap, 8 size classes), stdio (routed through hostcall), thread-local storage (disabled), and OnceLock (bypassed for `println!()`).

4. **System-scope Atomics**: LLVM's `core::sync::atomic` lacks `.sys` scope qualifiers needed for GPU-CPU shared memory. All cross-device atomics use inline PTX through `gpu-atomics`.

5. **GPU Panic Handler (ADR-5)**: Panic handler sends message (56 bytes) + thread/block metadata via `SERVICE_PANIC` hostcall, then executes `trap;` for clean termination. Replaces the previous `loop {}` infinite hang.

6. **Host Listener I/O Thread (ADR-6)**: Fast services (NOP, PRINT, TIME, PANIC) handled inline on the listener thread. Blocking FILE I/O and STDIN offloaded to a dedicated I/O thread via `mpsc` channel. Unified via `StdinSource` trait.

## Current Status

17 research themes completed, 2 active, 3 parked (68/73 tasks done):

- **Core**: toolchain, hostcall, gpu-std, async-runtime, integration
- **Infrastructure**: atomics, std-pal, allocator, error-handling, oncelock
- **Scaling**: multiblock, product, benchmark
- **Phase 2**: host-listener, ci (GitHub Actions), api (gpu-runtime facade + example)
- **Phase 3**: gpu-panic (panic handler), host-scaling (I/O thread), nightly-compat (1.91 -> 1.96)
- **Active**: large-payload (bulk data transfer design)
- **Parked**: warp-coop (incompatible with ADR-4), networking, upstream

## Strengths

- **Familiar Rust API on GPU**: GPU code can use `File::open()`, `println!()`, `async/await` — no raw CUDA C needed
- **Lock-free hostcall**: The two-stack protocol enables GPU-host I/O without mutex contention, scaling to 512+ threads across multiple blocks
- **Async/await works**: Embassy's poll-based executor runs on GPU. Multiple futures can be composed with standard combinators (`join`, `select`)
- **Cross-platform host**: Error propagation uses `io::ErrorKind` mapping (not raw errno), so the host side works on both Linux and Windows
- **Minimal std patching**: Only ~4 patch files touch the vendored std source, keeping upgrade friction low
- **No custom rustc fork**: Everything builds on stock nightly rustc with `-Zbuild-std`
- **~20 us single-thread latency**: Competitive with CUDA C++ polling protocols for hostcall round-trip
- **GPU panic handler**: Panics produce visible `[GPU PANIC] block=N thread=M: message` output instead of silent hangs
- **I/O thread separation**: Blocking FILE I/O won't stall fast services (PRINT, PANIC)

## Limitations and Known Issues

- **Nightly-only**: Requires Rust nightly (pinned to `nightly-2026-03-11`, rustc 1.96.0) for `#![feature(abi_ptx)]`, `-Zbuild-std`, `core::arch::asm!` with PTX, and other unstable features. Breakage on toolchain updates is expected.
- **NVIDIA-only**: Targets `nvptx64-nvidia-cuda` exclusively. No AMD/Intel GPU support. Requires CUDA runtime (loaded via `cudarc`) and an SM70+ GPU.
- **No warp-cooperative execution**: Each thread runs independently (ADR-4). Warp-cooperative allocation is architecturally incompatible with per-lane async execution.
- **Throughput does not scale**: CAS contention on the lock-free free-stack is the primary bottleneck. Throughput peaks at ~26-41K calls/s for 1 thread and *decreases* with more threads.
- **Packet pool starvation**: At 512 threads with 64 packets, ~70% of threads starve. Size the packet pool to at least 2x your active thread count.
- **56-byte payload limit**: Each hostcall can transfer at most 56 bytes of data. Bulk data transfer (large-payload theme) is under development using a sideband mapped buffer approach.
- **Limited std coverage**: Only `std::fs` (File), `std::io` (print/stdin), and basic allocation work. Networking, threading, and most of std are stubbed out.
- **Fat LTO required**: All GPU crates must link with `lto = "fat"` in Cargo.toml release profile. Required since nightly 1.96.0 for cross-crate function resolution.
- **All async kernels spill to local memory**: Embassy executor state is stored on the stack, causing register spilling for all async kernels. This adds latency per access but does not prevent execution.
- **CudaDevice Drop panics after GPU trap**: After a GPU panic (which executes `trap;`), the CUDA context enters a sticky error state. `std::process::exit(0)` is used to avoid cudarc's Drop impl panicking.

## Building

### Prerequisites

- Rust nightly toolchain (pinned: `nightly-2026-03-11`) with `nvptx64-nvidia-cuda` target
- `llvm-bitcode-linker` component
- NVIDIA GPU (SM70+, e.g., Volta/Turing/Ampere/Ada/Hopper) with CUDA driver

### Quick Start (Example)

```bash
# Build and run the hello-gpu example (auto-compiles kernel PTX via build.rs)
cd examples/hello-gpu/host
cargo run --release
```

### Full Build

```bash
# 1. Build a GPU kernel crate (e.g., gpu-kernel)
cd crates/gpu-kernel
cargo build --release
# .cargo/config.toml auto-sets target=nvptx64-nvidia-cuda and build-std

# 2. Copy PTX to gpu-host
cp target/nvptx64-nvidia-cuda/release/gpu_kernel.ptx ../gpu-host/kernel.ptx

# 3. Build and run the host test harness
cd ../gpu-host
cargo run --release
```

> **Note**: Some test crates require `__CARGO_TESTS_ONLY_SRC_ROOT` set to the patched-std directory for `-Zbuild-std=std` support.

## Research Structure

This project uses an autonomous research loop (Think -> Do -> Check) managed by a state machine in `.research/state.toml`. Findings, brainstorms, and reviews are stored in `.research/findings/`. See `CLAUDE.md` for the full workflow specification.

## License

MIT OR Apache-2.0 (dual-licensed)

## Acknowledgments

This work is a reproduction and exploration of techniques described by [VectorWare](https://www.vectorware.com/). All credit for the original ideas goes to them.
