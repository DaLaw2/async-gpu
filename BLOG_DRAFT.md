# Rust Async/Await on NVIDIA GPUs: A Deep Dive

*How we brought standard Rust `async fn` to GPU kernels, complete with file I/O, warp-cooperative execution, and real `std` library support.*

---

## The Problem

GPU programming has a fundamental limitation: kernels can't do I/O. Need to read a file? Return to the CPU, read it, copy the data, re-launch the kernel. Need to make a decision based on results? Same round-trip. This isn't just inconvenient — it's a performance cliff for workloads that interleave compute and I/O.

What if a GPU kernel could autonomously open files, read data, transform it, write results, and make decisions — all without returning to the host?

## The Solution: async_gpu

We built a system that enables **standard Rust async/await on NVIDIA GPUs**. Here's what it looks like:

```rust
#[warp_cooperative]
pub async fn data_pipeline(buf: *mut u8) -> u32 {
    let fd = GpuOpenFuture::new(buf, b"input.txt", FILE_OPEN_READ).await?;
    let mut data = [0u8; 48];
    let n = GpuReadFuture::new(buf, fd, &mut data).await?;
    GpuCloseFuture::new(buf, fd).await?;

    // Transform on GPU (all 32 lanes in parallel)
    for i in 0..n { data[i] = data[i].to_ascii_uppercase(); }

    let out_fd = GpuOpenFuture::new(buf, b"output.txt", FILE_OPEN_WRITE_CREATE).await?;
    GpuWriteFuture::new(buf, out_fd, &data[..n]).await?;
    GpuCloseFuture::new(buf, out_fd).await?;
    n as u32
}
```

This is a real GPU kernel. The `#[warp_cooperative]` attribute is a custom rustc MIR pass that makes `async fn` work correctly with NVIDIA's SIMT execution model. Each `.await` is a yield point where all 32 lanes in a warp synchronize. Between awaits, lanes execute in lockstep.

The system has three key innovations.

## Innovation 1: Lock-Free Hostcall Protocol

GPU kernels communicate with the host via a lock-free two-stack protocol over CUDA mapped memory:

```
GPU                              Host
 │ Pop free packet (CAS)          │
 │ Fill payload + service ID      │
 │ Push to ready stack            │
 │ Ring doorbell ─────────────────┤
 │                                ├─ Detect doorbell
 │                                ├─ Pop ready packet
 │                                ├─ Execute service (file I/O, print, stdin)
 │                                ├─ Write response
 │                                ├─ Set CONTROL_READY ────────────┤
 │ Poll control word ◄────────────────────────────────────────────┤
 │ Read response                  │
 │ Push to free stack             │
```

The protocol uses ABA-tagged compare-and-swap for both stacks, per-block sharding to reduce contention, and a sideband buffer for bulk transfers up to 1MB. Round-trip latency is ~42-100μs for a single thread, scaling to ~20K calls/second with a full warp.

## Innovation 2: Warp-Cooperative MIR Pass

NVIDIA GPUs execute 32 threads (lanes) in lockstep as a "warp." When Rust's async state machine suspends at an `.await` point, all 32 lanes must agree on where to resume — otherwise the warp diverges and lanes can deadlock.

Our solution is a custom rustc compiler pass that runs after the standard `StateTransform` (which converts `async fn` into a coroutine state machine). The pass modifies the generated MIR to:

1. **Broadcast the state discriminant** from lane 0 to all lanes via `shfl.sync.idx`. This ensures all lanes resume at the same suspension point.

2. **Insert `bar.warp.sync` barriers** before each return (yield). This ensures all lanes complete each poll cycle before any lane can advance to the next state.

The result: standard Rust `async fn`, standard `Future` trait, standard `.await` — and the compiler handles warp convergence automatically. No macros, no custom runtime, no manual state machines.

## Innovation 3: Real Rust `std` on GPU

GPU kernels can use actual Rust standard library types:

```rust
use std::fs::File;
use std::io::Write;

println!("[GPU] Hello from Rust std on GPU!");
let mut data = Vec::new();
for i in 0..10 { data.push(format!("item-{}", i)); }
File::create("output.txt")?.write_all(b"Written from GPU")?;
```

This works via a patched `std` with a CUDA platform adaptation layer (PAL) that routes system calls through the hostcall protocol. The `gpu-libc` crate provides a minimal libc shim: `open`/`read`/`write`/`close` become hostcall requests, `malloc`/`free` use an atomic bump allocator, and unsupported operations return `ENOSYS`.

Multi-thread safety is verified: 32 threads can concurrently allocate `Vec`s and call `println!`.

## Showcase: 32-Lane Parallel File Search

To demonstrate genuine GPU parallelism, we built a "GPU grep" — all 32 warp lanes search different chunks of a file for a byte pattern:

```
[host] Created search_input.txt (4096 bytes)
[host] CPU count of "GPU": 168
[host] Launching parallel_search kernel (32 threads)...
  [HOST] BULK READ: fd=1 4096 bytes read
[GPU] parallel search done
[host] GPU result: 168
[host] Verification: PASSED (exact match)
```

Thread 0 reads the file via sideband bulk I/O. All 32 threads search their 128-byte chunk. Lane 0 gathers results via `shfl.sync.idx` warp reduction. The GPU count matches the CPU reference exactly.

## What's Next

The project is [open source](https://github.com/DaLaw2/async-gpu) under MIT/Apache-2.0. It includes:

- 5 working examples (hello-gpu, async-pipeline, parallel-search, async-io, vector-math)
- A complete host SDK (`GpuRuntime`, `HostcallBuffer`, `MappedBuffer`)
- GPT-2 124M inference running entirely on GPU (68ms/token)
- Comprehensive [ARCHITECTURE.md](ARCHITECTURE.md) for contributors

The main limitation is the patched rustc requirement — the MIR pass hasn't been upstreamed. Stock nightly works for everything except `#[warp_cooperative]`.

We'd love feedback and contributions. If you're interested in GPU programming, Rust compiler internals, or async runtimes, there's something here for you.
