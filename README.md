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

### Vector Similarity Search

A GPU-autonomous vector search pipeline: open database, read vectors, open query, read query, compute cosine similarity across all 32 warp lanes, merge results via `shfl.sync`, write top-K — all in one kernel launch:

```
--- Vector Similarity Search (ml-workload) ---
  Created vecdb.bin (100 vectors × 128 dims = 51208 bytes)
  Created query.bin (query = db[42], expect top-1 match at id ~42)
  CPU reference top-3: ["id=42 score=1.0000", "id=82 score=0.2103", "id=18 score=0.0913"]
  Launching vector_search_pipeline kernel...
  [HOST] FILE OPEN: "vecdb.bin" flags=0 -> fd=1
  [HOST] BULK READ: fd=1 51208 bytes read
  [HOST] FILE CLOSE: fd=1 closed
  [HOST] FILE OPEN: "query.bin" flags=0 -> fd=2
  [HOST] BULK READ: fd=2 516 bytes read
  [HOST] FILE CLOSE: fd=2 closed
  [HOST] FILE OPEN: "results.bin" flags=1 -> fd=3
  [HOST] BULK WRITE: fd=3 84 bytes written
  [HOST] FILE CLOSE: fd=3 closed
  Status: 1 (1=success)
  Elapsed: 6.434ms
  Results: K=10
    rank 1: id=42 score=1.0000   ← exact match found
    rank 2: id=82 score=0.2103
    rank 3: id=18 score=0.0913
```

20-state `VecSearchFuture` — the GPU reads a database, reads a query, computes cosine similarity with all 32 lanes processing different vectors in parallel, merges per-lane top-K via warp shuffle, and writes results. 9 hostcall round-trips, 6.4ms end-to-end.

### Batch Vector Search

Five queries processed in a single kernel launch — I/O cost amortized, 1.6ms/query:

```
--- Batch Vector Search ---
  Launching batch_search_pipeline kernel (5 queries)...
  Elapsed: 7.997ms
  Query 0: [id=10 s=1.0000] [id=73 s=0.3801] [id=50 s=0.0853]
  Query 1: [id=42 s=1.0000] [id=82 s=0.2103] [id=18 s=0.0913]
  Query 2: [id=77 s=1.0000] [id=14 s=0.3891] [id=47 s=0.2168]
  Query 3: [id=3 s=1.0000] [id=66 s=0.3961] [id=57 s=0.0927]
  Query 4: [id=95 s=1.0000] [id=32 s=0.4102] [id=29 s=0.2164]
    5 queries, amortized 1.6ms/query (vs 6.4ms single-query)
```

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
| Vector similarity search | Working | `vector_search_pipeline` (20-state, full warp merge) |
| Batch vector search | Working | `batch_search_pipeline` (5 queries, 1.6ms/query amortized) |
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

This project was built through 26 research themes and 95+ verified experiments using an autonomous Think/Do/Check loop. Research state and findings are in `.research/`.

Inspired by [VectorWare](https://www.vectorware.com/)'s blog posts on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0 (dual-licensed)
