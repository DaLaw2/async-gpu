# Architecture

> **North star**: Write plain Rust, the compiler decides GPU.

async\_gpu compiles standard Rust — `async fn`, `Vec`, `println!`, iterators —
to NVIDIA GPU code. A patched compiler, a lock-free GPU-host RPC protocol,
and a layered crate architecture make this possible without macros, custom
runtimes, or GPU-specific syntax.

This document covers the compilation pipeline, crate layout, key subsystems,
and data flow.

---

## Compilation Pipeline

```
User Rust source (.rs)
    |
    v  rustc (patched nightly, -Zbuild-std=std)
MIR passes
    |- StateTransform: async fn -> coroutine state machine
    |- WarpCooperative: insert bar.warp.sync at yield points
    |- Iterator fusion, cost model analysis
    |
    v  LLVM nvptx64 backend
Raw PTX (.ptx)
    |
    v  ptxas (optional, --prod mode)
cubin (native GPU binary)
    |
    v  include_str! / include_bytes! in host crate
Host binary (loads via cuModuleLoadData at runtime)
```

The patched compiler adds a CUDA platform adaptation layer (PAL) to Rust's
`std`, routing `println!`, `File`, `Vec`, and `TcpStream` through hostcall
RPC. GPU kernels use standard Rust types — no `#![no_std]` restriction for
user code that builds with patched std.

## Crate Map

15 crates organized in four layers:

```
┌─── Facade ──────────────────────────────────────────────────┐
│                                                             │
│  async-gpu          User-facing crate. Re-exports gpu::*,  │
│                     GpuRuntime, GpuArray, Scheduler, nn.   │
│                     Features: nn, async (tokio).            │
│                                                             │
├─── Core (5 crates) ────────────────────────────────────────┤
│                                                             │
│  gpu-host           Host-side SDK: CUDA init, PTX loading, │
│                     hostcall listener, GpuArray, scheduler, │
│                     auto-tuner, streams, nn, onnx_rt.       │
│                                                             │
│  gpu-runtime        GPU-side facade: re-exports atomics +   │
│                     protocol. Modules: index, math, warp,   │
│                     block, scope, tiered_mem, par_iter,      │
│                     channel, executor, panic, nn, safety.    │
│                                                             │
│  gpu-protocol       Shared constants: packet layout, service│
│                     IDs, control flags, error codes. Used by │
│                     both host and GPU crates.                │
│                                                             │
│  gpu-atomics        Inline PTX intrinsics: CAS, fetch_add,  │
│                     shfl.sync, activemask, nanosleep,        │
│                     syncwarp, load/store with memory order.  │
│                                                             │
│  gpu-libc           libc shim for nvptx64: open/read/write/ │
│                     close -> hostcall, malloc -> bump alloc, │
│                     stubs (ENOSYS) for unsupported ops.      │
│                                                             │
├─── Kernel (4 crates) ──────────────────────────────────────┤
│                                                             │
│  gpu-kernel-core    Shared helpers, basic ops, math utils.   │
│                     rlib + cdylib. Other kernel crates       │
│                     depend on this.                          │
│                                                             │
│  gpu-kernel-compute ML/HPC kernels: GEMM, transformer,      │
│                     CNN (Conv2D, Winograd), physics,         │
│                     persistent kernels, fused ops, MMA.      │
│                                                             │
│  gpu-kernel-io      Hostcall-enabled kernels: I/O pipelines, │
│                     hybrid warp print.                       │
│                                                             │
│  gpu-kernel-test    Test/demo kernels: std demos, warp tests,│
│                     par_iter demos, structured concurrency.  │
│                                                             │
├─── Test (9 crates) ────────────────────────────────────────┤
│                                                             │
│  gpu-test-harness   Integration test binary (all GPU tests)  │
│  gpu-test-macro     #[gpu_test] proc macro                   │
│  gpu-critical-section  critical-section impl for Embassy     │
│  async-hostcall-test   Async hostcall protocol tests         │
│  async-pipeline-test   Multi-stage pipeline tests            │
│  embassy-test          Embassy executor tests                │
│  gpu-std-test          Patched std functionality tests        │
│  multi-warp-test       Multi-warp scaling tests              │
│  std-build-test        -Zbuild-std=std compilation tests     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Build System

### Two Workspaces

The project uses two separate Cargo workspaces because host and GPU crates
target different architectures:

| Workspace | Target | Location | Contents |
|-----------|--------|----------|----------|
| Host | `x86_64-unknown-linux-gnu` | `Cargo.toml` (root) | async-gpu, gpu-host, gpu-protocol, gpu-test-harness, gpu-test-macro |
| GPU | `nvptx64-nvidia-cuda` | Each kernel crate has its own `[workspace]` | gpu-kernel-{core,compute,io,test}, gpu-runtime, gpu-atomics, gpu-libc |

GPU crates are `#![no_std]` and compile with `-Zbuild-std=std,panic_abort`
(patched std) or `-Zbuild-std=core,alloc` (no-std kernels).

### Kernel Build Flow

`scripts/build-kernels.sh` orchestrates the kernel build:

1. **PTX generation** — Builds each kernel crate sequentially (shared deps
   compile once) with `cargo +nightly build --target nvptx64-nvidia-cuda`.
2. **PTX copy** — Copies `.ptx` files to `crates/core/gpu-host/` where they
   are embedded via `include_str!` in the `ptx` module.
3. **Cubin compilation** (--prod only) — Runs `ptxas` in parallel across all
   kernel crates to produce native GPU binaries.

Two build profiles:
- **Dev** (`--release`): PTX only, opt-level 1, no LTO. Fast iteration (~30s).
- **Prod** (`--prod`): PTX + cubin, opt-level 3, fat LTO. Maximum optimization.

The host loader (`gpu.rs`) tries cubin first (sub-second load), falls back to
PTX JIT if cubin is absent.

### PTX Auto-Discovery

The `ptx` module provides a catalog of all PTX modules:

```rust
pub const ALL: &[PtxModule] = &[
    PtxModule { name: "core",    ptx: KERNEL_CORE },
    PtxModule { name: "compute", ptx: KERNEL_COMPUTE },
    PtxModule { name: "io",      ptx: KERNEL_IO },
    PtxModule { name: "test",    ptx: KERNEL_TEST },
];
```

The `gpu::run()` one-liner API iterates `ptx::ALL` to find which module
contains a given kernel function, so users never specify PTX paths manually.

## Hostcall Protocol

GPU-host communication uses a lock-free two-stack design over CUDA mapped
memory (visible to both GPU and CPU simultaneously).

### Buffer Layout

```
+----------------------------------------------+
| Header (64 bytes)                            |
|   [0..8]   Free stack ptr (tagged, ABA)      |
|   [8..16]  Ready stack ptr (tagged, ABA)     |
|   [16..24] Doorbell counter (u64)            |
|   [24..28] Shutdown flag (u32)               |
|   [28..32] Packet count (u32)                |
|   [32..64] Shard metadata                    |
+----------------------------------------------+
| Packets (2112 bytes each, 64-byte aligned)   |
|   [0..8]   Next pointer (tagged, stack link) |
|   [8..12]  Active mask (u32)                 |
|   [12..16] Service ID (u32)                  |
|   [16..20] Control flags (u32)               |
|   [64..2112] Payload (32 x 8 x 8 bytes)     |
+----------------------------------------------+
| Sideband Buffer (1MB default)                |
|   Bump-allocated scratch for bulk I/O        |
+----------------------------------------------+
```

### Request Lifecycle

1. GPU pops a free packet (CAS on tagged stack)
2. GPU fills service ID + payload, sets CONTROL_FILLED
3. GPU pushes to ready stack, increments doorbell
4. Host detects doorbell change, pops ready packet
5. Host dispatches to service handler (file I/O, TCP, print, etc.)
6. Host writes response, sets CONTROL_READY
7. GPU polls control word, reads response, pushes to free stack

### Services

Services cover file I/O (open/read/write/close), TCP networking
(connect/bind/accept/read/write), stdin, timestamps, panic reporting,
tracing, and bulk I/O via the sideband buffer. All share a unified fd
namespace backed by an `FdResource` enum (`File | TcpStream | TcpListener`).

### Persistent Sessions

`HostcallSession` keeps the listener thread and fd table alive across
multiple kernel launches. `Pipeline` adds automatic packet reinitialization
between pipeline stages.

## Warp-Cooperative Async

A GPU warp is 32 threads in lockstep. If one suspends at `.await` while
others continue, the warp diverges and may deadlock.

A custom rustc MIR pass runs after `StateTransform` (which converts
`async fn` to a coroutine state machine) and inserts warp synchronization:

1. **Discriminant broadcast** — The state machine's resume discriminant is
   broadcast from lane 0 via `shfl.sync.idx.b32`, ensuring all 32 lanes
   enter the same state.

2. **Barrier at yields** — Each return terminator (yield point) gets
   `activemask.b32` + `bar.warp.sync`, ensuring all lanes complete each
   poll cycle before advancing.

The result: standard `async fn` with standard `Future` trait works on GPU
with no macros, no custom runtime, and no annotation needed.

## Structured Concurrency

Two scope primitives enforce lifetime-bounded GPU work:

### BlockScope (intra-block, shared memory)

```rust
block_scope(|scope| {
    let buf = scope.alloc::<f32>(256);       // shared memory
    scope.spawn_all(|wid, n_warps| { ... }); // all warps participate
    // buf freed, all spawned work joined
});
```

Uses a watermark/bump allocator over shared memory. The `'scope` lifetime
prevents references from escaping. Warp-trapped detection via
`set_warp_trapped()` prevents infinite join spins on panic.

### GridScope (cross-block, global memory)

Coordinates work across independently-scheduled blocks via system-scope
atomics and a pre-allocated global memory pool. Uses work-dispatch slots
(`grid_work` module) for coordinator-worker patterns on SM75 without
cooperative launch.

## Memory Hierarchy

### GPU-Side: Tiered Memory Types

`GpuRef<'scope, T, Tier>` encodes address space at the type level:

- `SharedRef<T>` — shared memory (~2-cycle latency). Emits `ld.shared`/`st.shared`.
- `GlobalRef<T>` — global memory (~100-cycle latency). Emits `ld.global`/`st.global`.

No `Deref` impl — forces explicit `.read(i)` / `.write(i, val)` to prevent
silent fallback to generic address space loads. `SharedRef` is `!Send`
(shared memory is per-block).

### Host-Side: GpuArray\<T\>

Transparent host-device data type with automatic residency management:

- **4-state residency**: `HostOnly -> Synced -> DeviceOnly -> HostDirty`
- **Size threshold**: Arrays < 64 KiB use pinned mapped memory (zero-copy
  over PCIe); larger arrays use device VRAM with explicit copies.
- **Deref transparency**: Host code reads `&[T]` through `Deref`. Kernel
  launch triggers automatic upload via `AsDevicePtr::ensure_device()`.

## Key Subsystems

### AutoScheduler

Work-routing abstraction with three variants:

- `CpuScheduler` — CPU-only; GPU ops return `NoGpu`.
- `GpuScheduler` — GPU-capable; delegates to `gpu::launch()`.
- `AutoScheduler` — Routes by data size: small data runs on CPU, large on
  GPU. Provides `par_map()` that hides the CPU/GPU decision.

### AutoTuner

Warmup-based parameter search for kernel launch configuration:

- Generates candidate block sizes, benchmarks each, selects fastest.
- Cache key: `(kernel_name, problem_size_bucket, device_ordinal)`.
- Thread-safe `TuningCache` for cross-thread result sharing.
- Integrates with `KernelResources` for occupancy-aware filtering.

### FlightRecorder

Mapped-memory ring buffer for post-mortem GPU tracing. GPU writes trace
events directly to mapped memory (no hostcall round-trip). On kernel crash,
the host dumps the last N events for debugging.

### GPU Panic Handler

Patched std panic handler: formats message with block/warp/lane metadata,
sends via hostcall, signals `set_warp_trapped()` for BlockScope detection,
then executes `trap`.

### CUDA Streams

Two-tier model: `GpuStream` for overlapping pure compute; hostcall kernels
use the default stream (device sync required before packet reset).

### Parallel Iterators

Rayon-like `par_iter()` API for GPU kernels. Lazy adapter chains (`map`,
`enumerate`, `zip`, `filter`) fused at compile time via monomorphization.
Terminal methods: `for_each`, `collect_into`, `fold`, `sum`.

### GPU Channels

Three tiers of inter-task communication:
- `channel` — global memory, system-scope atomics (~100 cycles)
- `block_channel` — shared memory, CTA-scope atomics (~2-6 cycles, 20-50x faster)
- `unified_channel` — auto-selects transport based on scope

### GPU Executor

Work-stealing async executor on GPU. Warps dequeue tasks from a lock-free
MPMC queue, poll type-erased futures, recycle slots. Lane 0 dequeues,
broadcasts task ID via `shfl.sync` to all 32 lanes.

## Neural Network Stack

Feature-gated (`nn`). Built on the kernel crate infrastructure.

```
GpuTensor (N-dim f32)
    |
    +-- ops::*           Stateless GPU kernel wrappers
    |   gemm, conv, attention, norm, activation,
    |   pool, reshape, quantize, upsample, winograd
    |
    +-- layers::*        Module trait implementations
    |   Linear, Conv2d, Attention, LayerNorm, Embedding,
    |   LoRA, Int4Linear, Activation, Pool, Sequential
    |
    +-- autograd::*      Tape-based automatic differentiation
    |   Tape, backward(), OpKind, TensorPool
    |   Optimizers: SGD, Adam
    |   Loss: MSE, CrossEntropy
    |   Gradient checkpointing
    |
    +-- fusion::*        Tape-level fusion detection
    |   Greedy longest-match: Matmul+Bias+Gelu, ElemAdd+LayerNorm
    |
    +-- models::*        Pre-built architectures (demo feature)
    |   GPT-2, ResNet, YOLOv8
    |
    +-- KernelRegistry   Auto-discovers kernels across ptx::ALL
```

### ONNX Runtime

Feature-gated (`onnx`). Parses ONNX protobuf, executes the graph on GPU:
- `proto` — Protobuf message types + parser
- `executor` — Graph executor dispatching ONNX nodes to nn ops
- `fusion` — Graph-level operator fusion pass

## Data Flow

```
  COMPILE TIME                        RUN TIME

  Kernel Rust (.rs)                   gpu::run("my_kernel")
       |                                   |
       v                                   v
  rustc (patched nightly)             PTX auto-discovery (ptx::ALL)
  |- MIR: StateTransform                   |
  |- MIR: WarpCooperative                  v
  |- LLVM: nvptx64                    cuModuleLoadData (cubin or PTX JIT)
       |                                   |
       v                                   v
  PTX --[ptxas]--> cubin              HostcallSession::start()
       |              |               |- mapped packet buffer
       v              v               |- listener thread
  include_str  include_bytes               |
       \            /                      v
        v          v                  cuLaunchKernel
  Host binary (ptx module)            +--------+--------+
                                      | GPU    | Host   |
                                      | kernel | serves |
                                      | runs   | RPCs   |
                                      +--------+--------+
                                           |
                                           v
                                      cuCtxSynchronize
                                      session.shutdown()
```
