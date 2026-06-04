# lib-cleanup.1: Crate Placement Audit

Investigation: audit all files/functions — are they in the right crate?

## 1. `crates/core/gpu-host/` (should be: pure host-side library)

### src/ files

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `lib.rs` | LIBRARY | gpu-host | gpu-host | STAYS — root module, re-exports, `model_dir()` utility |
| `error.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `GpuHostError`, `Result` type |
| `runtime.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `GpuRuntime` wrapper around CudaDevice |
| `hostcall.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `HostcallSession`, `Pipeline`, listener |
| `memory.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `MappedBuffer<T>` RAII mapped memory |
| `mapped_mem.rs` | LIBRARY | gpu-host | gpu-host | STAYS — low-level mapped memory helpers |
| `gpu.rs` | LIBRARY | gpu-host | gpu-host | STAYS — one-liner GPU launch API (`gpu::run`, `gpu::launch`, `gpu::custom`) |
| `streams.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `GpuStream` for overlapping kernels |
| `async_rt.rs` | LIBRARY | gpu-host | gpu-host | STAYS — tokio integration (feature-gated `async`) |
| **`model.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — GPT-2 weight loader from safetensors; demo/app code, not core library |
| **`model_generic.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — Generic SafeTensors loader; supports model.rs |
| **`model_yolo.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — YOLOv8 weight loader + PPM image I/O; clearly demo code |
| **`yolo_backbone.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — YOLOv8 backbone inference; application code, not library |
| **`tokenizer.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — GPT-2 BPE tokenizer wrapper; application-level |

### nn/ module (feature = "nn")

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `nn/mod.rs` | LIBRARY | gpu-host | gpu-host | STAYS — NN module root |
| `nn/tensor.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `GpuTensor` |
| `nn/registry.rs` | LIBRARY | gpu-host | gpu-host | STAYS — `KernelRegistry` |
| `nn/error.rs` | LIBRARY | gpu-host | gpu-host | STAYS — NN error types |
| `nn/layers/*.rs` | LIBRARY | gpu-host | gpu-host | STAYS — reusable layer abstractions (Linear, Conv, Norm, etc.) |
| `nn/ops/*.rs` | LIBRARY | gpu-host | gpu-host | STAYS — stateless GPU op wrappers |
| `nn/autograd/*.rs` | LIBRARY | gpu-host | gpu-host | STAYS — tape-based autograd system |
| **`nn/test_utils.rs`** | **TEST** | gpu-host | **gpu-test-harness** | **MOVE** — `Tolerance`, `assert_close`, `GoldenEntry` — test-only utilities |
| **`nn/cpu_ref.rs`** | **TEST** | gpu-host | **gpu-test-harness** | **MOVE** — CPU f64 reference implementations used only for test verification |
| **`nn/models/gpt2.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — Full GPT-2 model; demo/application code (1727 lines) |
| **`nn/models/resnet.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — ResNet18 model; demo/application code (589 lines) |
| **`nn/models/yolov8.rs`** | **DEMO** | gpu-host | **separate crate or examples** | **MOVE** — YOLOv8 model; demo/application code (834 lines) |

### onnx_rt/ module (feature = "onnx")

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `onnx_rt/mod.rs` | LIBRARY | gpu-host | gpu-host | STAYS — ONNX runtime root |
| `onnx_rt/proto.rs` | LIBRARY | gpu-host | gpu-host | STAYS — ONNX protobuf parsing |
| `onnx_rt/executor.rs` | LIBRARY | gpu-host | gpu-host | STAYS — ONNX graph executor |
| `onnx_rt/fusion.rs` | LIBRARY | gpu-host | gpu-host | STAYS — graph fusion pass |

### ptx module (embedded PTX)

| Item | Classification | Should Be | Action Needed |
|------|---------------|-----------|---------------|
| `ptx::KERNEL` | LIBRARY | gpu-host | STAYS — main kernel PTX, needed by `gpu::run`/`gpu::launch` |
| `ptx::EMBASSY_TEST` | **TEST** | gpu-test-harness | **MOVE** — only used by test harness |
| `ptx::ASYNC_HOSTCALL_TEST` | **TEST** | gpu-test-harness | **MOVE** — only used by test harness |
| `ptx::STD_BUILD_TEST` | **TEST** | gpu-test-harness | **MOVE** — only used by test harness |
| `ptx::ASYNC_PIPELINE_TEST` | **TEST** | gpu-test-harness | **MOVE** — only used by test harness |
| `ptx::MULTI_WARP_TEST` | **TEST** | gpu-test-harness | **MOVE** — only used by test harness |
| `ptx::KERNEL_STD` | **MIXED** | Depends | REVIEW — used by both test harness and potentially examples; may need to stay or be re-exported differently |

### tests/ directory

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `tests/gpu_integration.rs` | TEST | gpu-host | gpu-host | STAYS — proper `#[test]` integration tests for the library |

### Visibility audit — pub items that should be pub(crate)

- `model_dir()` in `lib.rs` — arguably DEMO support; could be moved with model code
- `mapped_mem` module is `pub` — contains low-level helpers (`alloc_mapped_u32`, `alloc_mapped_result_array`, `free_mapped_mem`). These are used by test harness directly. Consider: keep `pub` since `GpuContext::mapped_buffer()` wraps them, but the raw functions leak internal details. **REVIEW**: make `pub(crate)` and expose through `MappedBuffer` API instead.

---

## 2. `crates/core/gpu-runtime/` (should be: GPU-side runtime library)

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `lib.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `prelude.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS — but see note below |
| `index.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `math.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `warp.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `block.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `nn.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `hostcall.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `sideband.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `print_buffer.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `panic.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `cmd.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `flight_recorder.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `executor.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `channel.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `sync.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `collections.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `thread.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `warp_future.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `std_future.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `warp_cooperative.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `warp_sequential.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `warp_result.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |
| `nvptx_shim.rs` | LIBRARY | gpu-runtime | gpu-runtime | STAYS |

### prelude.rs audit

The prelude exports ~60 items including many internal protocol constants (`PKT_OFF_CONTROL`, `PKT_OFF_PAYLOAD`, `SERVICE_PRINT`, etc.). These are packet-level implementation details that most kernel authors should never touch.

**Finding**: The prelude is **too broad**. It exports raw protocol constants that are only needed by hand-written hostcall code. The majority of kernel authors should use higher-level APIs (`gpu_hostcall_print`, `thread::spawn`, etc.).

**Recommendation**: Split into two tiers:
- `prelude::*` — high-level API only (print, panic_init, block_on, thread::*, sync::*, collections::*)
- Import protocol constants explicitly via `gpu_protocol::*` or `gpu_runtime::hostcall::*` when needed

### No test/demo code found in gpu-runtime

gpu-runtime is clean. All files are legitimate GPU-side runtime library code.

---

## 3. `crates/kernel/gpu-kernel/` (should be: demo/test kernel implementations)

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `lib.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `basic.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS — basic test kernels |
| `compute_cnn.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_demo.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_fused.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_gemm.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_math.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_mma.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_persistent.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_physics.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_search.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `compute_transformer.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `helpers.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `hostcall_kernels.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `hybrid.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `pipeline.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `thread_test.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |
| `warp.rs` | TEST/DEMO | gpu-kernel | gpu-kernel | STAYS |

**Finding**: All files correctly classified as demo/test kernel code. No library code hiding here.

---

## 4. `crates/kernel/gpu-kernel-std/` (should be: demo kernels using patched std)

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `lib.rs` | TEST/DEMO | gpu-kernel-std | gpu-kernel-std | STAYS |

**Finding**: Contains test kernels demonstrating std on GPU (println, Vec, File I/O, HashMap, thread::spawn, matmul). All legitimately demo/test code.

**Note**: `gpu_stdout_write()` and `gpu_stdin_read()` are `#[no_mangle]` extern functions called by the patched std PAL layer. These are **runtime infrastructure** that any std-using kernel needs, not just test code. If more std-kernel crates are created, these would need to be extracted into a shared crate (e.g., `gpu-std-bridge`). Currently fine since only one std-kernel crate exists.

---

## 5. `crates/macro/warp-macro/` (legacy — should it be removed?)

### Dependency analysis

**Who depends on it:**
- `crates/kernel/gpu-kernel/Cargo.toml` → `warp-macro = { path = "../../macro/warp-macro" }`
- Used via `#[warp_macro::warp_async]` in `gpu-kernel/src/warp.rs` (10 annotated functions)
- These kernels are tested by `gpu-test-harness/src/tests_warp.rs`

**What would break if removed:**
- 10 test kernel functions in `gpu-kernel/src/warp.rs` that use `#[warp_async]`
- All `warp_*` tests in `tests_warp.rs` that test the proc-macro-generated state machines
- The workspace `Cargo.toml` lists it as a member

**Verdict**: warp-macro is **NOT dead** — it is actively used by gpu-kernel for test kernels. It is also a legitimate library component (proc macro for kernel authors). However, the question is whether the rustc MIR pass has superseded its purpose. The MIR pass automatically makes all `async fn` on nvptx64 warp-cooperative, which was the macro's original job.

**Recommendation**: Keep for now. The macro serves a different use case (WarpFuture state machines from sequential warp_*!() calls) vs the MIR pass (standard Future cooperation). They are complementary, not redundant. Mark for future review after the MIR pass is fully mature.

---

## 6. `crates/async-gpu/` (should be: thin facade)

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `lib.rs` | LIBRARY | async-gpu | async-gpu | STAYS |

### Re-export audit

Currently re-exports:
- `gpu` module (one-liner API)
- `GpuHostError`, `GpuKernelErrorInfo`, `Result`
- `GpuRuntime`, `HostcallSession`, `MappedBuffer`, `Pipeline`
- `GpuStream`
- `model_dir`
- `nn` (feature-gated)
- `async_rt` (feature-gated)

**Missing re-exports:**
- `ptx` module — examples that use custom PTX sources via `gpu::custom().ptx(...)` need this
- `mapped_mem` — if kept public, should be re-exported
- `hostcall::CommandBuffer` — useful for advanced users
- `hostcall::FlightRecorder` — useful for debugging

**Problematic re-exports:**
- `model_dir` — this is demo/model support code. Should move out with model code.

**Finding**: The facade is mostly correct but needs to propagate the `gpt2` and `onnx` feature flags. Currently `async-gpu/Cargo.toml` only has `nn` and `async` features. Examples that need `gpt2` or `onnx` features must bypass async-gpu and depend on gpu-host directly.

---

## 7. `crates/test/gpu-test-harness/` (should be: all test/bench code)

| File | Classification | Current Location | Should Be | Action Needed |
|------|---------------|-----------------|-----------|---------------|
| `main.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `bench_harness.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_basic.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_benchmark.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_cnn.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_gemm.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_hostcall.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_inference.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_pipeline.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_scaling.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_search.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_std.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_tokenizer.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_transformer.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |
| `tests_warp.rs` | TEST | gpu-test-harness | gpu-test-harness | STAYS |

**Finding**: The extraction from gpu-host was done correctly. The test harness depends on `gpu-host` (not `async-gpu`), has its own feature flags that pass through to gpu-host, and contains all test/benchmark code.

**Missing from extraction:**
- `nn/test_utils.rs` and `nn/cpu_ref.rs` still live in gpu-host. They should be moved to gpu-test-harness or a shared test utilities crate.
- The 6 test PTX constants (`EMBASSY_TEST`, `ASYNC_HOSTCALL_TEST`, etc.) are still embedded in gpu-host's `ptx` module and re-read by gpu-test-harness. They should be embedded directly in gpu-test-harness (via its own `include_str!` from its own build script or the PTX files).

---

## 8. `examples/` (should be: clean user-facing examples)

### Dependency audit

| Example | Depends On | Should Depend On | Action Needed |
|---------|-----------|-----------------|---------------|
| `hostcall/hello-gpu/host` | **async-gpu** | async-gpu | CORRECT |
| `hostcall/async-io/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `hostcall/async-pipeline/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `hostcall/parallel-search/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `hostcall/tcp-echo/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `hostcall/tokio-offload` | gpu-host (async feature) | **async-gpu** (async feature) | **MIGRATE** |
| `hostcall/vector-math/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `hostcall/warp-cooperative/host` | gpu-host | **async-gpu** | **MIGRATE** |
| `std/benchmark` | gpu-host (nn, gpt2, cublas) | async-gpu (needs gpt2, cublas features first) | **BLOCKED** on async-gpu feature flags |
| `std/cifar-train` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/diff-physics` | gpu-host (nn, onnx, gpt2) | async-gpu (needs onnx, gpt2) | **BLOCKED** |
| `std/dynamic-control` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/gpt2-inference` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/gpt2-lora` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/gpu-rag` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/graph-algorithms` | cudarc only | cudarc only | OK — pure CUDA example |
| `std/mnist-cnn` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/mnist-train` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |
| `std/monte-carlo` | cudarc only | cudarc only | OK — pure CUDA example |
| `std/resnet-cifar` | gpu-host (nn, onnx) | async-gpu (needs onnx) | **BLOCKED** |
| `std/thread-demo` | gpu-host | **async-gpu** | **MIGRATE** |
| `std/yolo-detect` | gpu-host (nn, gpt2) | async-gpu (needs gpt2) | **BLOCKED** |

**Finding**: Only 1 of 24 host-side examples uses async-gpu (hello-gpu). All others depend on gpu-host directly. This defeats the purpose of the facade crate.

**Root cause**: async-gpu only forwards `nn` and `async` features. It does not forward `gpt2`, `onnx`, or `cublas` features. Most examples need `gpt2` (for model loading), so they cannot use async-gpu.

### Tests disguised as examples

None found. The examples are genuinely user-facing demonstrations, not tests in disguise.

---

## Summary of All Misplaced Items

### HIGH PRIORITY — Move out of gpu-host

1. **Demo/model code** (5 files, ~4000+ lines):
   - `model.rs`, `model_generic.rs`, `model_yolo.rs`, `yolo_backbone.rs`, `tokenizer.rs`
   - These are application-level code (GPT-2 + YOLO weight loading, tokenization, backbone inference)
   - **Target**: New crate `crates/models/gpu-models/` or move into examples

2. **Demo model architectures** (3 files, ~3150 lines):
   - `nn/models/gpt2.rs`, `nn/models/resnet.rs`, `nn/models/yolov8.rs`
   - Full model implementations — application code, not reusable library
   - **Target**: Same `gpu-models` crate or examples

3. **Test utilities** (2 files, ~200 lines):
   - `nn/test_utils.rs`, `nn/cpu_ref.rs`
   - Only used by test harness for numerical verification
   - **Target**: gpu-test-harness or shared test crate

4. **Test PTX constants** (6 of 7 `ptx::*` constants):
   - `EMBASSY_TEST`, `ASYNC_HOSTCALL_TEST`, `STD_BUILD_TEST`, `ASYNC_PIPELINE_TEST`, `MULTI_WARP_TEST`, `KERNEL_STD`
   - Only consumed by gpu-test-harness
   - **Target**: Embed directly in gpu-test-harness build

### MEDIUM PRIORITY — Fix async-gpu facade

5. **async-gpu missing feature flags**: Add `gpt2`, `onnx`, `cublas` feature pass-through
6. **Migrate examples**: Update 8 hostcall examples + 2 std examples to use async-gpu instead of gpu-host

### LOW PRIORITY — Cleanup

7. **Prelude too broad**: gpu-runtime prelude exports ~30 protocol constants that most users never need
8. **`mapped_mem` module visibility**: Consider `pub(crate)` for raw allocation helpers
9. **`model_dir()` in lib.rs**: Should move with model code or be behind `gpt2` feature
10. **warp-macro**: Not dead, actively used. Keep but document its relationship to the MIR pass

### Items that are correctly placed

- All core gpu-host modules (error, runtime, hostcall, memory, gpu, streams, async_rt)
- All nn library code (layers, ops, autograd, tensor, registry)
- All gpu-runtime modules
- All gpu-kernel / gpu-kernel-std test kernels
- All gpu-test-harness test modules
- gpu-protocol (shared protocol constants)
- gpu-atomics (GPU atomic primitives)
- gpu-libc (libc shim for GPU)
- gpu-critical-section (no-op critical section for Embassy)
