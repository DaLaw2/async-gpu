# Changelog

All notable changes to async-gpu are documented in this file. Organized by
milestone with approximate dates from the git history. The project uses an
internal "cycle" counter for development tracking; cycle numbers are included
where relevant.

---

## Unreleased (post-v0.2.0 — cycles 309–642+)

**642+ development cycles, 56 epics completed**

### Codebase Health & Documentation (cycle 642)
- **ARCHITECTURE.md rewrite** — 396-line document covering all 15 crates, kernel
  split, unified runtime, tiered memory, GpuArray, AutoTuner, auto-fusion,
  nn/ONNX stack, structured concurrency, par_iter, channels, executor, build
  system, and compilation pipeline
- **Workflow migration** — 3-level (Epic→Theme→Task) to 4-level
  (Epic→Story→Feature→Task) Agile hierarchy
- **Code quality** — zero clippy warnings, `pub(crate)` cleanup, dead code removal,
  error handling audit

### Conv2D Optimization (cycle 642)
- **Winograd transforms** — F(2x2,3x3) and F(4x4,3x3) for 3x3 convolutions
- **Winograd batched GEMM** — 3-5x speedup over direct convolution, 807 GFLOPS peak
- **54.8% of peak** throughput via cuBLAS caching + Winograd F(4x4) combination
- Direct convolution paths for 1x1, 5x5, 7x7 kernels with 2.1-2.6x speedup

### Ownership & Memory Model (cycle 642)
- **SharedRef<T> / GlobalRef<T>** — tiered memory types with inline PTX asm for
  shared and global address spaces
- **GpuRef<'scope, T, Tier>** — generic reference type parameterized over memory tier
- **Borrow safety** — compile-time enforcement of shared memory lifetimes
- **Register promotion analysis** — analysis framework for promoting memory accesses

### Transparent Data (cycle 642)
- **GpuArray<T>** — host-device data type with 4-state residency tracking
  (HostOnly, DeviceOnly, Both, Modified)
- **AsDevicePtr trait** — zero-cost access to underlying device pointer
- Lazy host-to-device transfer on first kernel launch
- Integrated with kernel launch API — 27 tests pass

### Auto-Tuning (cycle 642)
- **AutoTuner** — warmup-based parameter search framework
- **TuningCache** — persistent cache for tuning results
- 1.4x measured speedup on compute-bound kernels via block size optimization

### Dynamic Dispatch on GPU (cycle 642)
- **Box<dyn Trait>** — trait objects compile to PTX, vtable dispatch works
- **&dyn Fn / closures** — higher-order functions and closures on GPU
- **hashbrown** — compiles on GPU unmodified, demonstrates ecosystem compatibility
- Vec<Box<dyn Animal>> polymorphism verified on hardware

### Compile-Time Cost Model (cycles 638–642)
- **KernelResources** — hybrid ptxas -v + MIR analysis pipeline for register/smem usage
- **Occupancy warnings** — compile-time lint catches low-occupancy configurations
- **Bank conflict detection** — static analysis for shared memory access patterns
- Caught real performance issue in existing kernels

### GPU Panic Handler (cycles 639–641)
- **Patched std panic handler** — GPU panic reports include block ID, warp ID,
  and lane ID in the panic message
- `gpu_assert!` removed in favor of standard `assert!` with GPU metadata
- PTX auto-discovery fixes resolved 8 runtime failures

### API Encapsulation (cycle 641)
- **async-gpu facade crate** — curated re-exports (`gpu::run()`, `gpu::custom()`,
  `GpuRuntime`, `GpuArray`, `AutoScheduler`)
- Runtime audit and deprecation design for internal APIs
- Feature flags: `nn` for neural networks, `async` for tokio integration

### GPU Coroutines / Generators (cycles 635–638)
- **GpuGenerator trait** — yield-based cooperative multitasking on GPU warps
- **Streaming pipeline** — FibGenerator + 3 test kernels demonstrate lazy evaluation
- Generator/coroutine semantics mapped to SIMT execution model
- 429-line GpuGenerator implementation

### Unified Runtime (cycles 630–634)
- **AutoScheduler** — routes work to CPU or GPU based on type heuristics
  (I/O → CPU, compute → GPU)
- **GpuVec<T>** — zero-copy buffer for CPU/GPU data sharing
- **North Star demo** — `read → compute → write` pipeline with zero GPU concepts
  visible in user code
- Zero-overhead benchmark confirms no abstraction cost

### GPU Generics (cycles 626–629)
- **Full Rust generics on GPU** — `fn kernel<T: Add + Copy>()` compiles to
  type-specific PTX via monomorphization
- User-defined traits with `where` bounds and custom types
- Zero-overhead polymorphism — generic `parallel_reduce<T>` works for f32, i32,
  custom Vec2f

### Type-Level Safety (cycles 624–625)
- **DisjointSlice<T>** — proves exclusive access to non-overlapping regions at
  compile time
- **ThreadIndex<'kernel>** — thread index type scoped to kernel lifetime
- **WarpIndex / WarpHandle** — typed warp-level primitives
- Zero-unsafe cooperative kernel demonstrated

### GPU Test Framework (cycle 623)
- **#[gpu_test] attribute macro** — write GPU tests like `#[test]`
- **GPU-native assert!** — assertion failures report warp ID and thread ID
- `cargo test` integration — GPU tests run alongside host tests
- Multi-block par_iter 2.4-3.8x faster than Rayon

### Kernel Split (cycles 611–622)
- **4 kernel crates** — `gpu-kernel-core`, `gpu-kernel-compute`, `gpu-kernel-io`,
  `gpu-kernel-test` for parallel compilation
- Multi-module host loader with parallel build script
- PTX JIT dev path — skip ptxas for <1 min rebuild; cubin is `--prod` only
- Per-crate PTX constants + backward-compatible aliases

### GPU Iterators (cycles 608–610)
- **par_iter().map().filter().fold()** — compiles to warp-parallel kernels via
  monomorphization
- **GpuParallelIterator trait** — Rayon-style API on GPU
- Fusion codegen via NVRTC — chained operations fuse into single kernel launch
- GpuFilter support

### Structured Concurrency (2026-06-05)
- **BlockScope / GridScope** — scoped parallelism with lifetime safety
- **SharedMemAllocator** — typed shared memory allocation within scopes
- **Block channels** — inter-thread communication within a block
- **GpuChannel** — unified channel abstraction + cancellation chain-walk
- `spawn_all` + `join_all` for structured task management

### Kernel Performance Optimization (2026-06-04 – 2026-06-05)
- **SGEMM** — 1759 GFLOPS via NVRTC (63% cuBLAS), 1172 GFLOPS via inline PTX (42%)
- **Flash Attention V3** — cooperative 4-thread-per-row, 5.7x speedup over V1
- **LayerNorm V2** — single-pass Welford, 80 GB/s (2.67x improvement)
- **GPT-2 inference** — 25.1ms forward pass (from 221ms baseline, 8.8x speedup)
- cuBLAS fallback for small GEMM (M≤256) — GPT-2 32.6ms → 25.1ms

### Developer Experience (2026-06-04)
- **extern "gpu-kernel"** — stable ABI for GPU entry points (replaces `extern "ptx-kernel"`)
- **gpu::run("kernel")** — one-liner kernel launch, hides all CUDA boilerplate
- **gpu::custom("kernel")** — builder API: `.ptx()`, `.threads()`, `.elements()`,
  `.hostcall()`, `.prepare()`
- **std::thread::spawn on GPU** — warp-as-thread model, SIMT lane-0 guard
- Thread-demo example: GPU threading looks like CPU Rust
- Removed legacy `#[warp_cooperative]` attributes — MIR pass handles all async fn

### Neural Network Module (2026-03-15 – 2026-03-17)
- **GpuTensor** — N-dimensional tensor with autograd support
- **nn layers** — Linear, Conv2d, BatchNorm2d, LayerNorm, activations (SiLU, GELU,
  Sigmoid, Softmax, ReLU)
- **Autograd v1-v4** — tape-based reverse-mode AD with GPU-native backward passes
  (matmul, conv2d, attention, batch_norm, max_pool, upsample)
- **Optimizers** — SGD, Adam with GPU-side step kernels
- **Loss functions** — CrossEntropyLoss, MSELoss
- **LoRA** — rank-8 adapter fine-tuning, GPT-2 ppl 128 → 16.3

### ONNX Runtime (2026-03-17)
- **Protobuf parser** — prost-based, parses ResNet-18 / GPT-2 / MobileNetV2
- **Graph executor** — 41+ operator dispatchers for GPU inference
- **Graph compiler** — fusion pass for MatMul+Add+Activation patterns (≥20% speedup)
- **Verified models** — ResNet-18 91.2%, GPT-2 coherent text, MobileNetV2 end-to-end

### Quantization (2026-03-16 – 2026-03-17)
- **INT8** — dp4a kernel, 4×INT8 dot product per instruction, 1.3% error
- **INT4 (W4A16)** — dequantize-on-the-fly with per-group scales, 4.7x compression
- GPT-2 INT4 quantization: 522MB → 111MB

### Pre-Built Models & Demos (2026-03-15 – 2026-03-17)
- **GPT-2 inference** — 124M params, 12-layer transformer, KV-cached generation
- **YOLOv8-nano** — object detection: Conv2D, BatchNorm, SiLU, detection head, NMS
- **GPU-Autonomous RAG** — vector store + cosine similarity + GPT-2 generation
  (search 0.8ms, gen 237ms/tok)
- **ResNet-18** — pretrained CIFAR-10 inference (91.3% accuracy)
- **Training examples** — MNIST MLP (91.2%), MNIST CNN (96.4%), CIFAR-10 CNN,
  GPT-2 LoRA fine-tuning
- **Differentiable physics** — N-body spring-mass simulation (47.1x GPU speedup)
- **Graph algorithms** — BFS, PageRank with CSR representation
- **Monte Carlo** — GPU PRNG (xoshiro256++), Black-Scholes pricing

### Advanced Compute (2026-03-15 – 2026-03-17)
- **Fused GEMM+bias+activation** — single-launch matmul_fused() kernels
- **Persistent kernels** — long-running GPU compute with mapped memory work queue
- **Dynamic control flow** — variable-length generation, early-exit inference
- **Tensor core MMA** — split-K f16 MMA with 2.18x throughput, 26ms GPT-2 forward

### Infrastructure (2026-03-14 – 2026-06-05)
- **Warp-cooperative MIR pass** — custom rustc compiler pass inserts `bar.warp.sync`
  at every `.await` yield point for automatic warp convergence
- **Patched std** — platform abstraction layer routes system calls through hostcall
- **Toolchain automation** — `build-toolchain.sh` / `.bat`, `apply-std-patches.sh`,
  `apply-rustc-patches.sh`
- **Flight recorder** — `gpu_trace!()` structured tracing with conditional compilation
- **CUDA streams** — `GpuStream` wrapper for stream-based kernel launch
- **Tokio integration** — `AsyncGpuRuntime`, `GpuTask` for async kernel execution

---

## v0.2.0 — 2026-03-15

TCP networking from GPU kernels and developer experience improvements.

### Networking
- **TCP hostcall services** — 8 new services (connect, write, read, close, bind,
  accept, bulk_write, bulk_read) enabling GPU kernels to make network connections
- **GPU-side TCP Futures** — `GpuTcpConnectFuture`, `GpuTcpWriteFuture`,
  `GpuTcpReadFuture`, `GpuTcpCloseFuture`, `GpuTcpBulkWriteFuture`,
  `GpuTcpBulkReadFuture`
- **Unified fd namespace** — `FdResource` enum (`File | TcpStream | TcpListener`)
  shares the same fd table
- **TCP bulk I/O** — up to 1MB TCP transfers via sideband buffer
- **TCP server support** — `SERVICE_TCP_BIND` + `SERVICE_TCP_ACCEPT` for GPU-side
  server patterns

### Examples
- **tcp-echo** — GPU kernel connects to a local TCP echo server, sends a message,
  reads the echoed response
- **parallel-search** — 32-lane warp-parallel byte-pattern search over bulk-read
  file data

### Formal Verification
- **TLA+ specification** of the CAS hostcall protocol (750 lines)
- **Safety verified** — 367M states, no double-ownership, no lost packets, no ABA
- **Liveness verified** — 337K states, all packets complete full lifecycle

### Infrastructure
- CI coverage expanded: all examples in `ci-lint.sh`
- Per-example README with architecture, running instructions, expected output
- `CONTRIBUTING.md` developer guide
- CI badge and license badge in README
- Convenience `run.sh` / `run.bat` scripts for examples
- Toolchain build scripts split into `.sh` (Linux) + `.bat` (Windows)
- `VALIDATION.md` first-run hardware validation checklist
- ~83 `// SAFETY` comments added to all high-risk unsafe blocks

---

## v0.1.0 — 2026-03-15

First release. Rust async/await running natively on NVIDIA GPUs.

### Core
- **Lock-free hostcall protocol** — GPU-host RPC over CUDA mapped memory with
  per-block sharding and ABA-tagged lock-free stacks
- **Warp-cooperative async/await** — custom rustc MIR pass inserts `bar.warp.sync`
  at every `.await` yield point for automatic warp convergence
- **Real Rust std on GPU** — `println!`, `Vec`, `String`, `Box`, `std::fs::File`,
  `std::io::stdin()` via patched std + gpu-libc shim
- **GPU error handling** — `Result<T, E>` propagation from GPU to host
- **block_on() executor** — spin-poll executor for driving Futures on GPU

### I/O
- File I/O: open, read, write, close via hostcall (up to 48 bytes inline)
- Bulk sideband I/O: up to 1MB transfers via `SERVICE_BULK_READ` /
  `SERVICE_BULK_WRITE`
- Buffered print: `println!()` auto-buffers via sideband, flushed as single hostcall
- Stdin: `std::io::stdin().read_line()` proxied through hostcall

### GPU Compute
- GPT-2 inference (124M params): embedding, 12-layer transformer, KV-cached
  autoregressive generation
- GEMM: f32 FMA with shared memory tiling
- FlashAttention: tiled attention with online softmax, causal masking, KV cache

### SDK
- `gpu-host` library crate: `GpuRuntime`, `HostcallBuffer`, `MappedBuffer<T>`
- 4 examples: `hello-gpu`, `async-pipeline`, `async-io`, `vector-math`
- Automated PTX compilation via `build.rs` in each example

### Infrastructure
- Patched toolchain build scripts (Linux `.sh` + Windows `.bat`)
- CI lint script with PTX validation
- `ARCHITECTURE.md` documenting system design

### Requirements
- Rust nightly (patched, based on nightly-2026-03-11)
- NVIDIA GPU (SM 70+) with CUDA 12.x driver
