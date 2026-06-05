# test-framework.1: Design #[gpu_test] proc macro

## Summary

This investigation designs the `#[gpu_test]` proc macro that transforms a test function
into a GPU kernel + host runner pair. The recommended approach uses a **build-script
compilation model** — kernel code is compiled to PTX via the existing build.rs pipeline,
and the proc macro generates only the host-side `#[test]` runner that loads and launches
the kernel. This approach works within the existing PTX compilation infrastructure, avoids
dual-compilation complexity, and integrates naturally with `cargo test`.

## Design Options

### Option A: Proc macro generates both kernel + host runner in one expansion

**How it works:** `#[gpu_test]` expands a single function into two items: an
`extern "gpu-kernel"` function (compiled for nvptx64) and a `#[test]` function
(compiled for the host). The proc macro would need to emit code that compiles
differently depending on `cfg(target_arch)`.

**Pros:**
- Single source file for both kernel and host
- No build script changes needed (in theory)

**Cons:**
- **Fundamental problem:** Rust proc macros expand at compile time for ONE target.
  The kernel crate compiles for nvptx64, the test binary compiles for x86_64.
  A proc macro cannot emit code for two targets in a single expansion.
- Would require the test crate to be compiled twice (once for each target), which
  cargo does not support natively
- The `extern "gpu-kernel"` ABI is only available on nightly with nvptx64 target

**Verdict:** Not viable with current Rust tooling.

### Option B: Build script compiles kernel code, proc macro generates host runner (RECOMMENDED)

**How it works:** GPU test functions live in a kernel-side crate (or module within
`gpu-kernel-std`), compiled to PTX by build.rs. The proc macro `#[gpu_test]` is applied
to a _host-side_ stub that specifies the kernel name + expected behavior. The macro
generates a `#[test]` function that loads the PTX, launches the kernel, and checks results.

**Architecture:**

```
Kernel side (gpu-kernel-std):              Host side (test crate):
┌──────────────────────────────┐           ┌──────────────────────────────┐
│ #[no_mangle]                 │           │ #[gpu_test]                  │
│ pub unsafe extern "gpu-kernel"│    PTX   │ fn test_vector_sum() {       │
│ fn test_vector_sum(result: *mut u32) {│──>│     // test body runs on GPU│
│     gpu_main_poll(|| {       │   build   │     assert_eq!(sum, 42);     │
│         let sum = ...;       │   .rs     │ }                            │
│         assert_eq!(sum, 42); │           │                              │
│     });                      │           │ // Expands to:               │
│ }                            │           │ #[test]                      │
└──────────────────────────────┘           │ fn test_vector_sum() {       │
                                           │     gpu::run_zero_param(     │
                                           │         PTX, "test_vector_sum│
                                           │     ).unwrap();              │
                                           │ }                            │
                                           └──────────────────────────────┘
```

**Pros:**
- Works with the existing build.rs PTX pipeline (no changes needed)
- Clean separation of kernel and host code (existing pattern)
- `cargo test` discovers tests via standard `#[test]` attribute
- Minimal proc macro complexity — just generates boilerplate host code
- assert! already works on GPU via the patched std panic handler + hostcall

**Cons:**
- Kernel and test are in separate files/crates (two places to look)
- Kernel name must match between kernel-side and host-side

**Verdict:** Best fit. Aligns with existing patterns (`GpuStdModule`, `run_zero_param`).

### Option C: Custom test harness (`#![test_runner(gpu_test_runner)]`)

**How it works:** Replace libtest entirely with a custom GPU test runner that discovers
`#[gpu_test]` functions and runs them on GPU.

**Pros:**
- Full control over test discovery, execution, and reporting
- Could batch multiple GPU tests into a single kernel launch

**Cons:**
- Loses `cargo test` integration (no `--test-threads`, `--filter`, etc.)
- Cannot mix `#[test]` and `#[gpu_test]` in the same crate
- Significant implementation effort for the test runner
- Custom harness feature is unstable (`#![feature(custom_test_frameworks)]`)

**Verdict:** Too disruptive. Violates the "cargo test just works" North Star.

### Option D: Hybrid — proc macro on kernel side + auto-generated host tests

**How it works:** `#[gpu_test]` lives on the kernel-side function. A build script
scans the kernel PTX for `test_*` symbols and auto-generates `#[test]` functions
in the host test crate.

**Pros:**
- Single annotation point (on the kernel function)
- Auto-discovery of test kernels

**Cons:**
- PTX symbol scanning is fragile
- Build script dependency ordering is complex
- Test configuration (threads, grid, expected results) needs to be encoded somewhere

**Verdict:** Interesting but fragile. Could be a Phase 2 evolution of Option B.

## Recommended Approach: Option B — Split Kernel/Host with Proc Macro

### Phase 1: Minimal viable design

#### Kernel side (in `gpu-kernel-std`)

Test kernels are ordinary `extern "gpu-kernel"` functions following the zero-param
entry pattern. They use `gpu_main_poll` and the standard `assert!` macro:

```rust
// In gpu-kernel-std/src/test_kernels.rs

#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_vector_sum() {
    gpu_runtime::entry::auto_init();
    gpu_runtime::thread::gpu_main_poll(|| {
        let sum = 1 + 2 + 3;
        assert_eq!(sum, 6, "vector sum should be 6");
    });
}

#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_thread_spawn() {
    gpu_runtime::entry::auto_init();
    gpu_runtime::thread::gpu_main_poll(|| {
        let h = gpu_runtime::thread::spawn(|| 42u32);
        let result = h.join();
        assert_eq!(result, 42);
    });
}
```

#### Host side proc macro (`#[gpu_test]`)

The macro generates a `#[test]` function that:
1. Loads the unified PTX
2. Creates a `GpuStdModule` (hostcall-enabled)
3. Launches the kernel with zero params
4. Captures GPU stdout/stderr
5. Checks for kernel success (no trap/panic)

```rust
// Usage:
#[gpu_test]
fn test_vector_sum() {}

// Expands to:
#[test]
fn test_vector_sum() {
    gpu_host::gpu::run_zero_param(
        gpu_host::ptx::KERNEL_STD,
        "test_vector_sum"
    ).expect("GPU test 'test_vector_sum' failed");
}
```

With configuration attributes:

```rust
// Custom thread count
#[gpu_test(threads = 256)]
fn test_multithread() {}

// Custom grid dimensions
#[gpu_test(threads = 128, grid = (2, 1, 1))]
fn test_multiblock() {}

// Expands to:
#[test]
fn test_multiblock() {
    gpu_host::gpu::run_zero_param_with_config(
        gpu_host::ptx::KERNEL_STD,
        "test_multiblock",
        128,
        (2, 1, 1),
    ).expect("GPU test 'test_multiblock' failed");
}
```

### Phase 2: Enhanced features (future tasks)

- `#[gpu_test(output = 4)]` — kernel writes to output buffer, host checks values
- `#[should_panic]` equivalent for GPU — expect a trap
- Test timeout configuration
- GPU stdout capture and assertion
- Batch kernel loading (load PTX once, run multiple tests)

## GPU Assert Design

### Current state: assert! already works

The patched `std` provides a `#[panic_handler]` that routes panic messages through
the hostcall protocol. When a GPU kernel calls `assert!()` or `assert_eq!()`:

1. The standard `assert!` macro fires → `panic!()` → `#[panic_handler]`
2. The panic handler in `patched-std/src/panicking.rs` formats the message
3. On nvptx64, the panic handler sends the message via `gpu_runtime::panic::send_panic_hostcall()`
4. The host-side listener receives the panic message with thread/block coordinates
5. The GPU thread then executes `trap;` (asm instruction), which causes CUDA to abort

**This means standard `assert!`, `assert_eq!`, `assert_ne!` already work on GPU.**
The panic message includes the assertion text and file/line info from Rust's standard
panic infrastructure.

### Warp/thread ID in failure messages

The existing `send_panic_hostcall` already encodes `thread_idx_x` and `block_idx_x`
in the panic metadata (via `encode_panic_metadata`). The host-side handler decodes
these and can print:

```
thread 'test_vector_sum' panicked at 'assertion failed: `(left == right)`
  left: `5`,
  right: `6`', gpu-kernel-std/src/test_kernels.rs:4:9
  [GPU block 0, thread 17]
```

### Enhancement: warp-aware assert messages

The current panic handler sends `thread_idx_x` and `block_idx_x`. To add warp/lane
info, we need a thin wrapper:

```rust
// In gpu_runtime — proposed enhancement
pub fn warp_id() -> u32 { thread_idx_x() / 32 }
pub fn lane_id() -> u32 { thread_idx_x() % 32 }
```

The panic handler can be extended to include warp_id and lane_id in the metadata.
**However, this is not strictly necessary for Phase 1** — the raw `thread_idx_x`
already uniquely identifies the failing lane (warp = thread_idx / 32, lane = thread_idx % 32).

## Test Result Propagation

### How kernel pass/fail reaches the host

**Pass case (no assertion failure):**
- Kernel completes normally
- `gpu::run_zero_param()` calls `cuSynchronize()` → returns `Ok(())`
- Host `#[test]` function succeeds

**Fail case (assertion failure):**
- `assert!` fires → panic → hostcall sends panic message → `trap;`
- `trap;` causes CUDA error (CUDA_ERROR_ILLEGAL_INSTRUCTION or similar)
- `cuSynchronize()` returns an error
- `gpu::run_zero_param()` returns `Err(GpuHostError::Verification { ... })`
- Host `#[test]` function fails with `.expect()` / `.unwrap()`

**Message capture:**
- The hostcall listener thread receives panic messages before the kernel traps
- `GpuStdModule::load_with_print()` accepts a print callback that can capture output
- For richer failure messages, the `#[gpu_test]` expansion can use `load_with_print`
  to capture GPU output and include it in the test failure message

### Enhanced version with captured output

```rust
// #[gpu_test] expanded form (enhanced)
#[test]
fn test_vector_sum() {
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        gpu_host::ptx::KERNEL_STD,
        "test_vector_sum",
        128,
        (1, 1, 1),
        Some(Box::new(move |msg| {
            captured_clone.lock().unwrap().push(
                String::from_utf8_lossy(msg).to_string()
            );
        })),
    ).expect("failed to load GPU test module");

    let result = unsafe { module.launch_raw(&[]) };
    module.finish();

    if let Err(e) = result {
        let msgs = captured.lock().unwrap();
        let gpu_output = msgs.join("\n");
        panic!(
            "GPU test 'test_vector_sum' failed: {e}\nGPU output:\n{gpu_output}"
        );
    }
}
```

## cargo test Integration

### Discovery and execution

`#[gpu_test]` expands to `#[test]`, so `cargo test` discovers GPU tests the same way
as regular tests. They appear in test output alongside CPU tests:

```
running 15 tests
test cpu::test_parser ... ok
test cpu::test_format ... ok
test gpu::test_vector_sum ... ok          ← GPU test
test gpu::test_thread_spawn ... ok        ← GPU test
test gpu::test_file_io ... ok             ← GPU test
test cpu::test_config ... ok
```

### Crate structure

The recommended crate structure for a project using `#[gpu_test]`:

```
my-gpu-project/
├── Cargo.toml
├── src/
│   └── lib.rs           # Library code
├── kernel/
│   ├── Cargo.toml       # nvptx64 target, depends on gpu-runtime
│   ├── .cargo/config.toml  # target = "nvptx64-nvidia-cuda"
│   └── src/
│       └── lib.rs       # GPU kernels + GPU test kernels
├── build.rs             # Compiles kernel/ to PTX
└── tests/
    └── gpu_tests.rs     # #[gpu_test] host stubs
```

For the async-gpu project specifically, GPU test kernels go in `gpu-kernel-std`,
and host test stubs go in `gpu-test-harness` or a new `gpu-test` integration test crate.

### Parallel test execution

GPU tests cannot run in parallel (single GPU device contention). The `#[gpu_test]`
expansion should include `#[serial]` (from `serial_test` crate) or use a global
mutex to serialize GPU test execution:

```rust
lazy_static! {
    static ref GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

#[test]
fn test_vector_sum() {
    let _guard = GPU_LOCK.lock().unwrap();
    // ... launch kernel ...
}
```

Alternatively, run with `cargo test -- --test-threads=1` for GPU tests.

### CI considerations

- CI does NOT have a GPU → GPU tests must be gated behind `#[cfg(feature = "gpu")]`
  or use the existing `ONLY_TEST` env var pattern
- The proc macro itself compiles on any platform (it only generates host code)
- PTX compilation happens in build.rs (already handles missing toolchain gracefully)

## Proc Macro Crate Design

### Crate: `gpu-test-macro`

```toml
[package]
name = "gpu-test-macro"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full"] }
quote = "1"
proc-macro2 = "1"
```

### Macro implementation sketch

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitInt, Meta, Expr};

#[proc_macro_attribute]
pub fn gpu_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let kernel_name = fn_name.to_string();

    // Parse attributes: threads, grid
    let config = parse_gpu_test_config(attr);

    let threads = config.threads;  // default 128
    let grid = config.grid;        // default (1,1,1)

    let expanded = quote! {
        #[test]
        fn #fn_name() {
            gpu_host::gpu::run_zero_param_with_config(
                gpu_host::ptx::KERNEL_STD,
                #kernel_name,
                #threads,
                #grid,
            ).expect(concat!("GPU test '", #kernel_name, "' failed"));
        }
    };

    expanded.into()
}
```

## Open Questions

1. **PTX source selection:** Should `#[gpu_test]` always use `KERNEL_STD` PTX, or
   should it support custom PTX sources? For Phase 1, `KERNEL_STD` is sufficient
   since all test kernels live in `gpu-kernel-std`.

2. **Test kernel naming convention:** Should test kernel names be prefixed with
   `test_` (Rust convention) or use a different convention? Recommendation: use
   `test_` prefix to make PTX symbol scanning possible in Phase 2.

3. **Output buffer tests:** Some tests need to return values to the host for
   verification. Phase 1 relies on assert! inside the kernel. Phase 2 could add
   `#[gpu_test(output = N)]` to allocate a result buffer.

4. **Batched PTX loading:** Loading PTX is expensive (~seconds for JIT, subsecond
   for cubin). Should multiple GPU tests share a single module load? The current
   `run_zero_param` loads a fresh module each time. A test fixture could load once.

5. **GPU test timeout:** Should GPU tests have a default timeout? A hung kernel
   (infinite loop) would block `cargo test` forever. Recommendation: 30-second
   default timeout via `std::thread::spawn` + `recv_timeout`.

6. **Feature gating:** Should GPU tests be behind a cargo feature (`gpu-test`)?
   This would allow `cargo test` to work on machines without a GPU by simply
   not enabling the feature. Recommendation: yes, gate behind `gpu` feature.

## Implementation Plan (subsequent tasks)

- **test-framework.2:** Implement `gpu-test-macro` proc macro crate (the `#[gpu_test]` attribute)
- **test-framework.3:** Write 10+ GPU test kernels in `gpu-kernel-std` covering existing features
- **test-framework.4:** Create host-side test crate with `#[gpu_test]` stubs
- **test-framework.5:** Add GPU assert enhancement — warp/lane ID in panic messages
- **test-framework.6:** PTX module caching + parallel test serialization
