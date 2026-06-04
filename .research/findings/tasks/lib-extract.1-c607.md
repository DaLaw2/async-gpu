# lib-extract.1: Test Harness Dependency Inventory

**Status**: done  
**Date**: 2025-06-04

## 1. Test Files Inventory

| File | Lines | Description |
|------|------:|-------------|
| `main.rs` | 1,201 | Test dispatcher + inline tests (thread_spawn, gpu_api, std_thread, kernel_std_smoke, fusion_benchmark) |
| `tests_basic.rs` | 583 | Kernel launches, atomics, warp intrinsics, mapped memory polling |
| `tests_hostcall.rs` | 2,044 | Hostcall print, embassy async, file I/O, sessions, pipelines, flight recorder, warp-cooperative futures |
| `tests_gemm.rs` | 2,933 | Softmax, tiled/multi-tile/multi-warp/multi-block/full GEMM, BF16, TF32, split-K |
| `tests_inference.rs` | 6,181 | Full GPT-2 forward pass, generation, f32/bf16 forward, CPU f64 reference, KV-cached generation |
| `tests_pipeline.rs` | 1,176 | File transform, panic handler, bulk I/O, sharding, parallel grep, branching, pipelined compute, autonomous pipeline, buffered print, Newton sqrt |
| `tests_scaling.rs` | 1,613 | Multi-warp/block scaling, slab allocator, executor demo, channel oneshot/MPSC, compute pipeline, tokio bridge |
| `tests_search.rs` | 689 | f32 math, vector search, batch search, MMA, shared memory |
| `tests_std.rs` | 1,226 | -Zbuild-std tests, PAL stdout/stdin, std::fs, multithread malloc, buffered println |
| `tests_tokenizer.rs` | 257 | GPT-2 BPE tokenizer validation |
| `tests_transformer.rs` | 2,232 | LayerNorm, GELU, attention, flash attention, embedding, FFN, transformer layer |
| `tests_warp.rs` | 1,225 | Warp intrinsics, WarpFuture, proc macro, control flow, hybrid executor, rustc async baseline |
| `tests_benchmark.rs` | 954 | Hostcall latency, warp divergence, sharding, throughput, scalability, file I/O benchmarks |
| `tests_cnn.rs` | 1,212 | BatchNorm+SiLU, CNN ops, Conv2D, YOLO backbone, detect head, end-to-end |
| `bench_harness.rs` | 207 | BenchmarkResult, LatencyStats, statistical helpers, JSON output |
| **Total** | **23,733** | **79% of 30,146 total lines in src/** |

## 2. Per-File Dependency Analysis

### 2a. Imports from the binary crate's own modules (`crate::*`)

Every test file uses:
- `crate::error::{GpuHostError, Result}` — error types
- `crate::mapped_mem::*` — pinned memory allocation helpers
- `crate::hostcall` — HostcallBuffer, HostcallSession, Pipeline, CommandBuffer, FlightRecorder
- `crate::KERNEL_PTX` (and similar PTX constants) — re-exported from `gpu_host::ptx::*`

The `crate::` prefix here refers to the **binary** crate (`main.rs`), which includes `error`, `hostcall`, and `mapped_mem` as `mod` statements. These modules are defined in the **library** crate's `src/` directory — the binary crate re-includes them via `mod` declarations.

### 2b. Imports from the library crate (`gpu_host::*`)

| Import path | Used by | Feature gate |
|-------------|---------|-------------|
| `gpu_host::ptx::KERNEL` | main.rs | — |
| `gpu_host::ptx::EMBASSY_TEST` | main.rs | — |
| `gpu_host::ptx::ASYNC_HOSTCALL_TEST` | main.rs | — |
| `gpu_host::ptx::STD_BUILD_TEST` | main.rs | — |
| `gpu_host::ptx::ASYNC_PIPELINE_TEST` | main.rs | — |
| `gpu_host::ptx::MULTI_WARP_TEST` | main.rs | — |
| `gpu_host::ptx::KERNEL_STD` | main.rs | — |
| `gpu_host::model_dir()` | main.rs, tests_inference, tests_cnn | `gpt2` |
| `gpu_host::model::load_gpt2_weights()` | main.rs, tests_inference | `gpt2` |
| `gpu_host::tokenizer::Gpt2Tokenizer` | tests_inference, tests_tokenizer | `gpt2` |
| `gpu_host::tokenizer::{ENDOFTEXT_TOKEN_ID, GPT2_VOCAB_SIZE}` | tests_tokenizer | `gpt2` |
| `gpu_host::model_yolo::*` | tests_cnn | `gpt2` |
| `gpu_host::yolo_backbone::*` | tests_cnn | `gpt2` |
| `gpu_host::gpu::launch()` | main.rs | — |
| `gpu_host::gpu` | main.rs | — |
| `gpu_host::nn::ops::norm::*` | main.rs | `nn`, `cublas` |
| `gpu_host::nn::ops::reshape::elementwise_add_out` | main.rs | `nn`, `cublas` |
| `gpu_host::nn::ops::elementwise_add` | main.rs | `nn`, `cublas` |
| `gpu_host::nn::registry::KernelRegistry` | main.rs | `nn`, `cublas` |
| `gpu_host::nn::tensor::GpuTensor` | main.rs, tests_transformer | `nn`, `cublas` |
| `gpu_host::nn::ops::attention::multi_head_flash_attention_v3` | tests_transformer | `nn`, `cublas` |
| `gpu_host::async_rt::{AsyncGpuRuntime, GpuTask, HostcallEvent}` | tests_scaling | `async` |
| `gpu_host::memory::MappedBuffer` | tests_scaling | `async` |
| `gpu_host::runtime::GpuRuntime` | tests_scaling | `async` |
| `gpu_host::Result` (re-export) | tests_cnn | — |
| `gpu_host::error::GpuHostError` | tests_cnn | — |

### 2c. External crate imports

| Crate | Used by |
|-------|---------|
| `cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig}` | ALL test files |
| `cudarc::driver::sys::{self, lib as cuda_lib}` | tests_hostcall, tests_gemm, tests_search, tests_pipeline, main.rs |
| `cudarc::nvrtc::Ptx` | ALL test files (via `from_src()`) |
| `tokio` | tests_scaling (feature = "async") |

## 3. Dependency Classification

### Must stay pub — Real library API items

These are items any user of gpu-host would use:

| Item | Module | Current visibility |
|------|--------|--------------------|
| `GpuRuntime` | `runtime` | `pub` (re-exported at crate root) |
| `GpuHostError`, `Result` | `error` | `pub` (re-exported) |
| `HostcallBuffer` | `hostcall` | `pub` (re-exported) |
| `HostcallSession` | `hostcall` | `pub` (re-exported) |
| `Pipeline` | `hostcall` | `pub` (re-exported) |
| `MappedBuffer` | `memory` | `pub` (re-exported) |
| `gpu::launch()` | `gpu` | `pub` |
| `model_dir()` | lib root | `pub` |
| `model::load_gpt2_weights()` | `model` | `pub` (gpt2 feature) |
| `tokenizer::Gpt2Tokenizer` | `tokenizer` | `pub` (gpt2 feature) |
| `nn::*` | `nn` | `pub` (nn feature) |
| `async_rt::*` | `async_rt` | `pub` (async feature) |
| `streams::*` | `streams` | `pub` |

### Must become pub (currently accessed via binary mod inclusion)

These items are currently accessed by the test harness via `crate::` (binary crate mod inclusion), not via `gpu_host::`. They are `pub` in the library but the binary re-includes them as local modules:

| Item | Module | Current visibility | Action needed |
|------|--------|--------------------|---------------|
| `alloc_mapped_u32()` | `mapped_mem` | `pub` | Already pub; harness needs to import via `gpu_host::mapped_mem::*` |
| `alloc_mapped_result_array()` | `mapped_mem` | `pub` | Same |
| `alloc_mapped_u64_array()` | `mapped_mem` | `pub` | Same |
| `alloc_mapped_bytes()` | `mapped_mem` | `pub` | Same |
| `free_mapped_mem()` | `mapped_mem` | `pub` | Same |
| `free_mapped_bytes()` | `mapped_mem` | `pub` | Same |
| `free_mapped_u64_array()` | `mapped_mem` | `pub` | Same |
| `HostcallBuffer::new()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::new_sharded()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::new_with_sideband()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::listen()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::listen_with_stdin()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::signal_shutdown()` | `hostcall` | `pub` | Already pub |
| `HostcallBuffer::dev_ptr` | `hostcall` | `pub` field | Already pub |
| `HostcallBuffer::sideband_dev_ptr` | `hostcall` | `pub` field | Already pub |
| `HostcallSession::start()` | `hostcall` | `pub` | Already pub |
| `HostcallSession::start_with_print()` | `hostcall` | `pub` | Already pub |
| `HostcallSession::dev_ptr()` | `hostcall` | `pub` | Already pub |
| `HostcallSession::reinit_packets()` | `hostcall` | `pub` | Already pub |
| `HostcallSession::shutdown()` | `hostcall` | `pub` | Already pub |
| `CommandBuffer` | `hostcall` | `pub` | Already pub |
| `Command` | `hostcall` | `pub` | Already pub |
| `FlightRecorder` | `hostcall` | `pub` | Already pub |
| `GpuKernelErrorInfo` | `error` | `pub` | Already pub |
| `check_kernel_result()` | `error` | `pub` | Already pub |
| `BenchmarkResult` | `bench_harness` | `pub` | NOT exposed via library — lives in binary only |
| `LatencyStats` | `bench_harness` | `pub` | NOT exposed via library — lives in binary only |
| `compute_stats()` | `bench_harness` | `pub` | NOT exposed via library — lives in binary only |
| `write_results_json()` | `bench_harness` | `pub` | NOT exposed via library — lives in binary only |

### PTX constants — should become pub(crate) after extraction

| Constant | Current visibility | After extraction |
|----------|-------------------|-----------------|
| `gpu_host::ptx::KERNEL` | `pub` | `pub(crate)` — only test harness uses them |
| `gpu_host::ptx::EMBASSY_TEST` | `pub` | `pub(crate)` |
| `gpu_host::ptx::ASYNC_HOSTCALL_TEST` | `pub` | `pub(crate)` |
| `gpu_host::ptx::STD_BUILD_TEST` | `pub` | `pub(crate)` |
| `gpu_host::ptx::ASYNC_PIPELINE_TEST` | `pub` | `pub(crate)` |
| `gpu_host::ptx::MULTI_WARP_TEST` | `pub` | `pub(crate)` |
| `gpu_host::ptx::KERNEL_STD` | `pub` | `pub(crate)` |

**WAIT** — `gpu_host::ptx::KERNEL` IS used in the lib.rs doc example (`use gpu_host::ptx;` then `rt.load_ptx(ptx::KERNEL, ...)`). If we make it `pub(crate)`, the doc example breaks. Decision: keep `ptx::KERNEL` pub (it's part of the user API), make the test-only ones pub(crate) or move them to the harness.

Actually, **ALL** PTX constants are test kernels (embassy_test, async_hostcall_test, etc.) except `ptx::KERNEL` which is the main kernel. `ptx::KERNEL` is used in the doc example at lib.rs line 30. The others are purely test infrastructure. So:
- `ptx::KERNEL` — **must stay pub** (used in doc example + real user API)
- `ptx::KERNEL_STD` — could be useful for users building with -Zbuild-std
- All others — **can become pub(crate)** or move to harness

## 4. Feature Flags Required by Test Harness

| Feature | Required by | Tests affected |
|---------|-------------|----------------|
| `gpt2` | tests_inference, tests_tokenizer, tests_cnn, main.rs (model loading, weight test) | Generation, forward pass, tokenizer, YOLO |
| `nn` | main.rs (fusion benchmark), tests_transformer (flash attention bench) | LayerNorm/GELU ops, GpuTensor |
| `cublas` | main.rs (fusion benchmark), tests_transformer (flash_attention_v3_bench) | cuBLAS-based attention benchmark |
| `async` | tests_scaling (tokio_bridge_demo_test) | Tokio bridge demo |
| `onnx` | NOT used by any test file directly | — |

The binary's `Cargo.toml` `[[bin]]` section requires `gpt2` feature. The harness will need: `gpt2 + nn + cublas + async`.

## 5. Hardest Parts to Extract

### 5a. Binary re-includes library modules via `mod`
The binary `main.rs` has `mod error; mod hostcall; mod mapped_mem;` which re-includes the **library's** source files. The test files then use `crate::error`, `crate::hostcall`, `crate::mapped_mem` referring to the binary crate, NOT the library. After extraction, these must change to `gpu_host::error`, `gpu_host::hostcall`, `gpu_host::mapped_mem`.

**Impact**: Every `use crate::*` in every test file must be rewritten to `use gpu_host::*`. This is ~14 files × ~3 imports each = ~42 import changes. Mechanical but tedious.

### 5b. bench_harness.rs lives only in the binary
`bench_harness.rs` is included via `mod bench_harness` in `main.rs`. It has no counterpart in the library. It must either:
1. Move into the test harness crate as a module, or
2. Become a new utility module in the library (if benchmarks are useful to users)

Option 1 is simpler. tests_benchmark.rs is the only consumer.

### 5c. main.rs has inline test functions
`main.rs` contains 6 substantial inline test functions (not in any `tests_*.rs`):
- `run_fusion_benchmark()` — 166 lines, uses `gpu_host::nn::*`, gated on `cublas`
- `run_std_thread_spawn_demo()` — 53 lines
- `run_real_std_thread_spawn()` — 60 lines
- `run_std_thread_spawn_minimal()` — 47 lines
- `run_kernel_std_smoke()` — 114 lines
- `run_gpu_api_test()` — 32 lines
- `run_thread_spawn_test()` — 76 lines

These must be moved to test modules during extraction.

### 5d. The main() dispatcher is large
`main.rs` `main()` is 638 lines — a giant match/dispatch with ~80 ONLY_TEST cases and ~100 sequential test calls. Must be reconstructed in the harness.

### 5e. cudarc::driver::sys direct usage
Several test files use `cudarc::driver::sys::{self, lib as cuda_lib}` for raw CUDA API calls (cuMemHostAlloc, cuMemFreeHost, cuCtxSynchronize). This is through the `mapped_mem` module which is already pub in the library. The direct `sys` usage in tests_hostcall (time test) and tests_pipeline (panic test) will need cudarc as a direct dependency of the harness.

### 5f. PTX files are embedded via include_str!
The library embeds PTX via `include_str!("../kernel.ptx")` etc. The harness will need to either:
1. Import PTX from the library (`gpu_host::ptx::*`), or
2. Embed its own copies (bad — duplication)

Option 1 is correct. The PTX constants must remain accessible from outside the library.

## 6. Summary

The test harness is cleanly separable. The core pattern across all 14 test files is:
1. Import `error` types, `mapped_mem` helpers, and `hostcall` types from the binary's own modules
2. Import PTX constants from `gpu_host::ptx::*`
3. Import specific library APIs (`gpu_host::model::*`, `gpu_host::nn::*`, etc.) for higher-level tests
4. Use `cudarc` directly for kernel launches

**No test file accesses private (`pub(crate)`) library internals.** All imports are through `pub` API surfaces. The `crate::` imports are an artifact of the binary re-including library modules, not a sign of private access.

The extraction is mechanical: change `crate::*` to `gpu_host::*`, move `bench_harness.rs` and inline main.rs tests into the harness, and reconstruct the dispatcher.
