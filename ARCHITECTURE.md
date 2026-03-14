# Architecture

This document explains how async_gpu works — from the lock-free GPU-host communication protocol to the custom rustc MIR pass that makes `async fn` warp-cooperative.

## Overview

async_gpu enables Rust `async`/`await` on NVIDIA GPUs. A GPU kernel can open files, read data, transform it, write results, and make decisions — all without returning to the host between steps. The system has three key innovations:

1. **Lock-free hostcall protocol**: GPU threads request host services (file I/O, print, stdin) via a two-stack packet system over CUDA mapped memory
2. **Warp-cooperative MIR pass**: A custom rustc compiler pass inserts `bar.warp.sync` at every `.await` yield point, ensuring all 32 GPU lanes stay synchronized
3. **Platform adaptation layer**: A patched Rust `std` that routes system calls through hostcall, enabling `std::fs::File`, `println!`, `Vec`, etc. on GPU

## Hostcall Protocol

GPU-host communication uses a ROCm-inspired lock-free design over CUDA mapped memory (visible to both GPU and CPU simultaneously).

### Buffer Layout

```
┌──────────────────────────────────────────────┐
│ Header (64 bytes)                            │
│   [0..8]   Free stack ptr (tagged, ABA)      │
│   [8..16]  Ready stack ptr (tagged, ABA)     │
│   [16..24] Doorbell counter (u64)            │
│   [24..28] Shutdown flag (u32)               │
│   [28..32] Packet count (u32)                │
│   [32..64] Shard metadata                    │
├──────────────────────────────────────────────┤
│ Shard Array (16 bytes per shard)             │
│   Each shard: free_stack + ready_stack       │
├──────────────────────────────────────────────┤
│ Packets (2112 bytes each, 64-byte aligned)   │
│   [0..8]   Next pointer (tagged, stack link) │
│   [8..12]  Active mask (u32)                 │
│   [12..16] Service ID (u32)                  │
│   [16..20] Control flags (u32)               │
│   [64..2112] Payload (32 × 8 × 8 bytes)     │
├──────────────────────────────────────────────┤
│ Sideband Buffer (1MB default)                │
│   [0..64]  Header (alloc offset, capacity)   │
│   [64..]   Data region (bump-allocated)      │
└──────────────────────────────────────────────┘
```

### Request Lifecycle

```
GPU                                     Host
 │                                       │
 ├─ 1. Pop free packet (CAS)             │
 ├─ 2. Fill service ID + payload         │
 ├─ 3. Set CONTROL_FILLED                │
 ├─ 4. Push to ready stack               │
 ├─ 5. Increment doorbell ───────────────┤
 │                                       ├─ 6. Detect doorbell change
 │                                       ├─ 7. Pop ready packet
 │                                       ├─ 8. Dispatch to service handler
 │                                       ├─ 9. Write response to payload
 │                                       ├─ 10. Set CONTROL_READY ──────┤
 ├─ 11. Poll control word ◄─────────────────────────────────────────────┤
 ├─ 12. Read response                    │
 └─ 13. Push to free stack               │
```

### Lock-Free Stack (ABA-Tagged)

Both free and ready stacks use tagged CAS to prevent the ABA problem:
- **Bits 63-32**: Monotonically increasing tag
- **Bits 15-0**: Packet index (0xFFFF = null)

Push and pop are single-CAS operations. No mutexes (GPU streaming multiprocessors cannot use spinlocks safely).

### Per-Block Sharding

Each GPU block maps to a shard via `blockIdx.x % num_shards`. Each shard has its own free/ready stack pair, reducing CAS contention when many blocks are active.

### Services

| ID | Name | Direction | Description |
|----|------|-----------|-------------|
| 1 | PRINT | GPU→Host | Print message (up to 56 bytes inline) |
| 2 | WRITE | GPU→Host | Write to file descriptor (up to 48 bytes inline) |
| 3 | READ | GPU→Host | Read from file descriptor (up to 56 bytes inline) |
| 4 | OPEN | GPU→Host | Open file, returns fd |
| 5 | CLOSE | GPU→Host | Close file descriptor |
| 8 | STDIN | GPU→Host | Read line from host stdin |
| 9 | TIME | GPU→Host | Get host timestamp |
| 10 | PANIC | GPU→Host | Report GPU panic with message |
| 11 | BULK_WRITE | GPU→Host | Write via sideband (up to 1MB) |
| 12 | BULK_READ | GPU→Host | Read via sideband (up to 1MB) |
| 15 | BULK_PRINT | GPU→Host | Print via sideband buffer |
| 16 | TCP_CONNECT | GPU→Host | Connect to TCP addr:port, returns socket fd |
| 17 | TCP_WRITE | GPU→Host | Write to TCP socket (up to 48 bytes inline) |
| 18 | TCP_READ | GPU→Host | Read from TCP socket (up to 56 bytes inline) |
| 19 | TCP_CLOSE | GPU→Host | Close TCP socket |
| 20 | TCP_BIND | GPU→Host | Bind+listen on TCP addr:port, returns listener fd |
| 21 | TCP_ACCEPT | GPU→Host | Accept connection, returns stream fd |
| 22 | TCP_BULK_WRITE | GPU→Host | Write to TCP socket via sideband (up to 1MB) |
| 23 | TCP_BULK_READ | GPU→Host | Read from TCP socket via sideband (up to 1MB) |

TCP services share the fd namespace with file I/O. The host fd table uses an `FdResource` enum (`File | TcpStream | TcpListener`) to distinguish resource types.

### Sideband Bulk I/O

For data larger than the 48-byte inline payload, the sideband buffer provides a separate mapped memory region with a GPU-side bump allocator:

1. GPU atomically allocates space in sideband (`fetch_add`)
2. GPU copies data to sideband (for writes) or just allocates space (for reads)
3. Hostcall payload carries `(fd, sideband_offset, length)`
4. Host reads/writes data directly from/to sideband
5. Response indicates bytes transferred

## Warp-Cooperative Async

### Problem

On a GPU, 32 threads (lanes) execute in lockstep as a "warp." If one lane suspends at an `.await` point while others continue, the warp diverges — some lanes are masked off, wasting compute. Worse, if the `.await` changes control flow, lanes can deadlock.

### Solution: `#[warp_cooperative]` MIR Pass

A custom rustc MIR pass runs **after** the standard `StateTransform` pass (which converts `async fn` into a coroutine state machine). It modifies the generated MIR to insert warp synchronization:

**Phase 1: Discriminant Broadcast**

The state machine's entry point reads a discriminant to decide which suspension point to resume. The MIR pass rewrites this to broadcast from lane 0:

```
Before:                          After:
  read discriminant                read discriminant
  switch on discriminant  →        activemask.b32        (read active lanes)
                                   shfl.sync.idx.b32     (broadcast from lane 0)
                                   switch on broadcast discriminant
```

All 32 lanes now see the **same** discriminant, preventing divergence.

**Phase 2: Barrier at Returns**

Each `Return` terminator (= yield point) gets a warp barrier:

```
Before:           After:
  return    →       activemask.b32
                    bar.warp.sync %mask
                    return
```

This ensures all lanes complete each poll cycle before any lane can advance.

### Result

Standard Rust `async fn` with standard `Future` trait — no macros, no custom runtime. The compiler handles warp convergence automatically:

```rust
#[warp_cooperative]
pub async fn pipeline(buf: *mut u8) -> u32 {
    let fd = GpuOpenFuture::new(buf, b"file.txt", READ).await?;  // bar.warp.sync here
    let n = GpuReadFuture::new(buf, fd, &mut data).await?;       // bar.warp.sync here
    GpuCloseFuture::new(buf, fd).await?;                          // bar.warp.sync here
    n as u32
}
```

### Alternative: `#[warp_async]` Proc Macro

For use without the patched toolchain, the `warp-macro` crate provides `#[warp_async]`, which transforms code with `warp_*!()` macro calls into a `WarpFuture` state machine at the proc-macro level. This approach works on stock nightly but requires a different coding style.

## Platform Adaptation Layer (PAL)

### Architecture

```
GPU Kernel Code
    │
    ├── use std::fs::File;    (patched std)
    ├── println!("hello");    (patched std)
    └── Vec::new();           (patched std)
          │
          ▼
   Patched Rust std (-Zbuild-std=std)
   └── sys::pal::cuda::*     (CUDA PAL module)
          │
          ▼
   gpu-libc (libc shim)
   ├── open()  → hostcall SERVICE_OPEN
   ├── read()  → hostcall SERVICE_READ
   ├── write() → hostcall SERVICE_WRITE
   ├── close() → hostcall SERVICE_CLOSE
   ├── malloc() → atomic bump allocator
   └── stubs  → ENOSYS for unsupported ops
          │
          ▼
   gpu-runtime (hostcall protocol)
```

### Patched std

The `patched-std/` directory contains patches applied to Rust's standard library:
- Adds `cfg(target_arch = "nvptx64")` paths in `sys/pal/`
- Routes file operations through `gpu_libc::open/read/write/close`
- Routes `println!` through `gpu_libc::write` to stdout fd
- Enables `Vec`, `String`, `Box` via `#[global_allocator]` backed by bump allocator

### Memory Allocator

GPU-side allocation uses an atomic bump allocator:
- `malloc(size)`: CAS loop on global bump pointer, returns aligned address
- `free(ptr)`: No-op (bump allocator, memory reclaimed on kernel exit)
- Default alignment: 64 bytes (cache-line)
- Thread-safe: all 1024 GPU threads can allocate concurrently

### Async I/O Futures

Standard `impl Future` types for non-blocking hostcall I/O:

| Future | Service | Output |
|--------|---------|--------|
| `GpuOpenFuture` | OPEN | `Result<i32, i32>` (fd or errno) |
| `GpuReadFuture` | READ | `Result<usize, i32>` (bytes read) |
| `GpuWriteFuture` | WRITE | `Result<usize, i32>` (bytes written) |
| `GpuCloseFuture` | CLOSE | `Result<(), i32>` |
| `GpuBulkReadFuture` | BULK_READ | `Result<usize, i32>` (via sideband) |
| `GpuBulkWriteFuture` | BULK_WRITE | `Result<usize, i32>` (via sideband) |

Each Future follows a three-state machine: `Init` → `Waiting { pkt_idx }` → `Done`. The `Init` state submits the hostcall; `Waiting` polls for the response; `Done` returns the result.

## Crate Map

```
┌─────────────── Host Side ──────────────────┐
│                                            │
│  gpu-host (SDK)                            │
│  ├── GpuRuntime     (CUDA device wrapper)  │
│  ├── HostcallBuffer (listener + dispatch)  │
│  ├── MappedBuffer   (RAII mapped memory)   │
│  └── depends on: cudarc                    │
│                                            │
│  examples/                                 │
│  ├── hello-gpu/host                        │
│  ├── async-pipeline/host                   │
│  ├── async-io/host                         │
│  └── vector-math/host                      │
│                                            │
└────────────────────────────────────────────┘
         │ (build.rs compiles kernel to PTX,
         │  host includes PTX as string)
         ▼
┌─────────────── GPU Side ───────────────────┐
│                                            │
│  gpu-runtime (façade)                      │
│  ├── hostcall request/release              │
│  ├── async Futures (Open/Read/Write/...)   │
│  ├── block_on() executor                   │
│  ├── WarpFuture + WarpExecutor             │
│  └── sideband bulk I/O                     │
│                                            │
│  gpu-protocol (shared constants)           │
│  ├── packet layout, service IDs            │
│  ├── error codes, control flags            │
│  └── used by both GPU and host crates      │
│                                            │
│  gpu-atomics (inline PTX)                  │
│  ├── CAS, fetch_add, load/store            │
│  ├── shfl.sync, activemask                 │
│  └── nanosleep, syncwarp                   │
│                                            │
│  gpu-libc (libc shim)                      │
│  ├── open/read/write/close → hostcall      │
│  ├── malloc/free → bump allocator          │
│  └── stubs for unsupported ops             │
│                                            │
│  warp-macro (proc macro)                   │
│  └── #[warp_async] → WarpFuture codegen    │
│                                            │
└────────────────────────────────────────────┘
```

**Key constraint**: GPU crates are `#![no_std]` and target `nvptx64-nvidia-cuda`. They cannot share a Cargo workspace with host crates due to different targets. Each example's `build.rs` compiles the kernel separately.

## Build Pipeline

### Quick Start (Stock Nightly)

```bash
rustup toolchain install nightly-2026-03-11
rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-03-11
rustup component add rust-src --toolchain nightly-2026-03-11
cargo run --manifest-path examples/hello-gpu/host/Cargo.toml
```

Works for: `hello-gpu`, `async-io`, `vector-math`. Does NOT work for `async-pipeline` (requires patched rustc).

### Patched Toolchain (for `#[warp_cooperative]`)

```bash
# Linux
bash scripts/build-toolchain.sh

# Windows
.\scripts\build-toolchain.bat
```

This:
1. Clones Rust compiler source to `rustc-src/`
2. Copies to `patched-rustc/`
3. Applies patches from `rustc-patches/` (MIR pass + feature gates)
4. Builds a stage1 compiler at `patched-rustc/build/`
5. Example `build.rs` scripts auto-detect and use the patched compiler

### PTX Compilation Flow

```
Kernel Rust source (.rs)
    │
    ▼ (cargo build --target nvptx64-nvidia-cuda --release)
LLVM IR
    │
    ▼ (LLVM nvptx backend)
Raw PTX (.ptx)
    │
    ▼ (sed post-processing: remove .ptr .align, stub panic_const)
Clean PTX
    │
    ▼ (build.rs: patch .target sm_30 → sm_86)
Final PTX
    │
    ▼ (include_str! in host binary)
Runtime PTX loading via cuModuleLoadData
```

### Key Build Flags

- **Kernel**: `--target nvptx64-nvidia-cuda`, `-Zbuild-std=core,alloc`, `+ptx78` target feature
- **Patched std kernel**: `-Zbuild-std=std,panic_abort`
- **Host**: Standard `x86_64` Rust, links against CUDA driver API via `cudarc`

## Key Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| Warp size | 32 lanes | NVIDIA hardware constant |
| Packet size | 2112 bytes | 64-byte header + 32×8×8 payload, cache-line aligned |
| Max inline write | 48 bytes | 6 payload slots (slot 0 = fd + length) |
| Max inline read | 56 bytes | 7 payload slots (slot 0 = bytes_read) |
| Sideband default | 1 MB | Sufficient for most bulk transfers |
| Spin timeout | 10M polls | ~640ms at 64ns nanosleep, prevents infinite hangs |
| Bump alignment | 64 bytes | Cache-line optimization |
| Packet alignment | 64 bytes | Cache-line boundary |
