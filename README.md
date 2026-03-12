# async_gpu — GPU as Autonomous Compute Environment

**What if the GPU could drive its own I/O?** Open files, read data, process it, write results — all from GPU code, with zero CPU intervention between steps.

This project makes it real: **Rust async/await running natively on NVIDIA GPUs**, turning the GPU from a passive accelerator into a self-coordinating compute environment.

```
                     One kernel launch. Zero CPU intervention.

    GPU Kernel ──► open("input.txt")
                   read(1024 bytes via sideband)
                   transform(32 lanes toggle ASCII case)     ← per-thread compute
                   open("output.txt")
                   write(transformed data)
                   close(input)
                   close(output)
                   print("pipeline: done")                   ← all done, GPU returns
```

## The Demo

A single kernel launch where the GPU self-coordinates an 8-step I/O pipeline + per-thread compute:

```
--- File Transform Pipeline (async-pipeline) ---
  Created gpu_input.txt (1024 bytes)
  Launching file_transform_pipeline kernel...
  [HOST] FILE OPEN: "gpu_input.txt" flags=0 -> fd=1
  [HOST] BULK READ: fd=1 1024 bytes read
  [HOST] FILE OPEN: "gpu_output.txt" flags=1 -> fd=2
  [HOST] BULK WRITE: fd=2 1024 bytes written
  [HOST] FILE CLOSE: fd=1 closed
  [HOST] FILE CLOSE: fd=2 closed
  [HOST] GPU says: "pipeline: done"
  Status: 1 (1=success)
  Elapsed: 4.183ms
  File Transform Pipeline: PASSED!
    16-state WarpFuture: open->read->transform->open->write->close->close->print
    GPU self-coordinated 8 I/O steps + 1 compute step
    1024 bytes: ASCII case toggled correctly
    Zero CPU intervention between steps
```

The GPU decides what to read, how to process it, and where to write — all expressed as a Rust state machine on the GPU side. The CPU only provides I/O services when asked.

## How It Works

GPU threads communicate with the host through a **lock-free hostcall protocol** over CUDA shared memory:

```
+------------------------------------------------------+
|  GPU Kernel (nvptx64)                                |
|  +--------------+  +-------------+  +-------------+ |
|  |  Rust std     |  |  WarpFuture |  |  gpu-runtime | |
|  |  (patched)    |  |  state      |  |  hostcall    | |
|  |  File, println|  |  machine    |  |  helpers     | |
|  +------+-------+  +------+------+  +------+------+ |
|         |                 |                |         |
|  +------v-----------------v----------------v------+  |
|  |     gpu-protocol  ->  hostcall buffer (mapped)  | |
|  +------------------------+----------------------+   |
|                           | CUDA mapped memory       |
+---------------------------+--------------------------+
|  Host CPU                 |                          |
|  +------------------------v-----------------------+  |
|  |  hostcall listener: poll buffer, serve I/O     |  |
|  |  13 services: print, file, bulk, panic, ...    |  |
|  +------------------------------------------------+  |
+------------------------------------------------------+
```

**Key insight**: Async is not for replacing SIMT computation — it's for coordinating control flow between computation steps. Homogeneous I/O uses WarpFuture (warp-cooperative, 1 CAS per warp). Heterogeneous computation uses per-thread blocks (divergent, each lane independent).

### WarpFuture: Warp-Cooperative Async

All 32 GPU lanes in a warp share a single state machine:

- **I/O phases**: All lanes participate in cooperative hostcall (one packet per warp, not 32)
- **Compute phases**: Each lane works independently on its data slice
- **Reconvergence**: `syncwarp()` ensures all lanes rejoin before the next I/O phase

The file transform demo is a 16-state `WarpFuture`:

```
OPEN_IN → WAIT → BULK_READ → WAIT → COMPUTE → OPEN_OUT → WAIT → BULK_WRITE → WAIT
  → CLOSE_IN → WAIT → CLOSE_OUT → WAIT → PRINT → WAIT → DONE
```

## Quick Start

### Prerequisites

- Rust nightly (`nightly-2026-03-11`) with `nvptx64-nvidia-cuda` target
- `llvm-bitcode-linker` component
- NVIDIA GPU (SM70+) with CUDA driver

### Run the Demo

```bash
# Build the GPU kernel
cd crates/gpu-kernel
cargo build --release
# .cargo/config.toml auto-sets target and build-std

# Copy PTX to host crate
cp target/nvptx64-nvidia-cuda/release/gpu_kernel.ptx ../gpu-host/kernel.ptx

# Build and run (includes the file transform pipeline demo)
cd ../gpu-host
cargo run --release
```

Or use the standalone hello-gpu example:

```bash
cd examples/hello-gpu/host
cargo run --release
```

## What Works

| Capability | Status | Example |
|------------|--------|---------|
| `println!()` from GPU | Working | `hostcall_print_hello` kernel |
| `File::open/read/write/close` | Working | `hostcall_file_test` kernel |
| Async/await (Embassy executor) | Working | `async_hostcall_single` kernel |
| `futures::join!()` on GPU | Working | `futures_join_kernel` |
| `Vec`, `String`, `format!()` | Working | `std_hello_kernel` (vendored std) |
| WarpFuture (warp-cooperative async) | Working | `warp_future_print_test` |
| `#[warp_async]` proc macro | Working | `warp_macro_print_test` |
| Hybrid executor (I/O + compute) | Working | `hybrid_executor_test` |
| Bulk data transfer (sideband) | Working | `bulk_io_test` (4KB+) |
| GPU-autonomous pipeline | Working | `file_transform_pipeline` |
| GPU panic handler | Working | Visible `[GPU PANIC]` messages |
| Multi-block scaling | Working | 16 blocks, per-block sharding |

## Performance

Measured on RTX 3060 (SM_86). Hostcall round-trip via NOP benchmark:

| Threads | p50 Latency | Throughput | CAS/call |
|---------|-------------|------------|----------|
| 1 | ~20 us | 26-41K/s | 0 |
| 32 | ~1 ms | 20-24K/s | 3-7 |
| 128 | ~6 ms | 14K/s | 28-43 |

Per-block sharding reduces CAS contention by **99%** (from ~53 retries/call to ~0.5).

## Project Structure

### Core Crates

| Crate | Purpose |
|-------|---------|
| `gpu-protocol` | Shared packet layout, service IDs, error encoding (`#![no_std]`) |
| `gpu-atomics` | System-scope GPU atomics via inline PTX |
| `gpu-runtime` | Hostcall helpers, WarpFuture trait, sideband bulk I/O |
| `gpu-libc` | Minimal libc shim routing syscalls through hostcall |
| `gpu-kernel` | GPU kernels compiled to PTX |
| `gpu-host` | Host-side CUDA harness, hostcall listener, test runner |
| `warp-macro` | `#[warp_async]` proc macro for WarpFuture generation |

### Key Design Decisions

1. **Lock-free hostcall protocol**: ROCm-style two-stack design with tagged pointers, warp-granular packets (32 lanes x 8 slots), per-block sharding
2. **WarpFuture**: Warp-cooperative async where 32 lanes share one state machine — reduces CAS from 32/warp to 1/warp
3. **Hybrid executor**: WarpFuture for I/O + per-thread compute blocks with `syncwarp()` reconvergence
4. **Sideband buffer**: Separate mapped memory for bulk data transfer beyond the 56-byte packet payload limit
5. **No custom rustc fork**: Stock nightly with `-Zbuild-std`, inline PTX for system-scope atomics

## Limitations

- **Nightly-only**: Requires unstable Rust features (`abi_ptx`, `-Zbuild-std`, PTX asm)
- **NVIDIA-only**: `nvptx64-nvidia-cuda` target, SM70+ GPU required
- **WarpFuture requires uniform I/O**: All lanes must execute the same I/O sequence; divergent I/O falls back to per-thread futures
- **~20 us hostcall latency**: Not suitable for per-element I/O or latency-critical hot loops
- **Limited std coverage**: `File`, `println!`, `Vec`, `String` work; networking/threading stubbed out

## Research

This project was built through 26 research themes and 90 verified experiments using an autonomous Think/Do/Check loop. Research state and findings are in `.research/`.

Inspired by [VectorWare](https://www.vectorware.com/)'s blog posts on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0 (dual-licensed)
