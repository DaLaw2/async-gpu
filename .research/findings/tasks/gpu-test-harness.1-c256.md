# gpu-test-harness.1: Design GPU test harness
**Cycle**: 256 | **Theme**: gpu-test-harness | **Kind**: investigation | **Status**: done

## Summary
Investigated the current monolithic test architecture in `crates/gpu-host/src/main.rs` and analyzed constraints for migrating to `#[test]`-based execution via `cargo test`. The crate has **110 distinct test functions** across 11 test modules, all run sequentially in `fn main()`. CudaDevice is `Send + Sync` (via unsafe impls in cudarc), which enables sharing across test threads. The recommended approach is integration tests in `tests/` with a shared `OnceLock<Arc<CudaDevice>>` and `--test-threads=1` serialization.

## Findings

### Current Test Architecture

**Structure**: `crates/gpu-host` is a binary crate (`[[bin]]`) with a library target. The binary's `main.rs`:
- Creates a single `CudaDevice::new(0)` → `Arc<CudaDevice>`
- Embeds 7 PTX files via `include_str!()` as module-level constants
- Calls ~90 test functions sequentially (some called twice: in ONLY_TEST and in the main flow)
- Has an `ONLY_TEST` env var filter for running individual tests
- One test (`run_panic_test`) uses `process::exit(0)` and is marked "MUST BE LAST"

**Test modules** (11 modules, 110 `run_*` functions):
| Module | Count | PTX sources used |
|--------|-------|------------------|
| tests_basic | 6 | KERNEL_PTX |
| tests_benchmark | 3 | KERNEL_PTX |
| tests_gemm | 11 | KERNEL_PTX |
| tests_hostcall | 25 | KERNEL_PTX, EMBASSY_PTX, ASYNC_HOSTCALL_PTX |
| tests_inference | 7 | KERNEL_PTX (1 test is CPU-only) |
| tests_pipeline | 10 | KERNEL_PTX, ASYNC_HOSTCALL_PTX |
| tests_scaling | 10 | MULTI_WARP_PTX, ASYNC_PIPELINE_PTX, STD_BUILD_TEST_PTX, ASYNC_HOSTCALL_PTX, KERNEL_PTX |
| tests_search | 6 | KERNEL_PTX |
| tests_std | 9 | STD_BUILD_TEST_PTX, KERNEL_STD_PTX, KERNEL_PTX |
| tests_tokenizer | 2 | None (CPU-only) |
| tests_transformer | 9 | KERNEL_PTX |
| tests_warp | 14 | KERNEL_PTX |

**PTX files** (7 distinct, all in `crates/gpu-host/`):
1. `kernel.ptx` — main kernel, used by most tests
2. `embassy_test.ptx` — Embassy async/await tests
3. `async_hostcall_test.ptx` — async hostcall tests
4. `std_build_test.ptx` — `-Zbuild-std=std` tests
5. `async_pipeline_test.ptx` — async pipeline tests
6. `multi_warp_test.ptx` — multi-warp scaling tests
7. `kernel_std.ptx` — kernel-std tests

**Shared state pattern**: Every test takes `Arc<CudaDevice>` and independently:
1. Calls `Ptx::from_src()` + `dev.load_ptx()` to load needed PTX (idempotent — cudarc deduplicates by module name)
2. Creates its own `HostcallBuffer` (allocated/freed per test)
3. Allocates mapped memory for results (allocated/freed per test)

### Constraint Analysis

**CudaDevice is Send + Sync**: Confirmed in cudarc 0.12.1 source (`src/driver/safe/core.rs:48-49`):
```rust
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}
```
This means a shared `Arc<CudaDevice>` can be used across test threads.

**Multiple CudaDevice::new(0) calls**: cudarc uses `cuCtxPrimaryRetain` which returns the **same** primary context for the same device ordinal. Multiple `CudaDevice::new(0)` calls create separate Rust objects but share the underlying CUDA context. However, each `CudaDevice` object has its own module registry, stream, and event — and `Drop` unloads modules and releases the context. This means:
- **Safe**: Multiple `Arc<CudaDevice>` from separate `new(0)` calls can coexist.
- **Risky**: If one `CudaDevice` drops while another is still using the context, the context ref-count decreases but remains valid. However, module unloading in `Drop` could invalidate functions held by other device objects.
- **Conclusion**: Sharing a single `Arc<CudaDevice>` via `OnceLock` is the safest approach.

**PTX loading**: `dev.load_ptx()` is idempotent per module name — calling it multiple times with the same name is safe. The PTX strings are compile-time constants, so they can be `include_str!()` from integration tests too (the files are at known relative paths).

**HostcallBuffer**: Each test creates and destroys its own buffer. No cross-test sharing. This is test-local and safe for parallel execution **if** CUDA operations don't conflict.

**Test parallelism**: CUDA kernel launches on the default stream are serialized by the driver. However, host-side listener threads + mapped memory operations could race. **Recommendation: run with `--test-threads=1`** for safety, at least initially.

**Panic test**: `run_panic_test` calls `process::exit(0)` — this would kill the entire test process. Must be either excluded from the harness or run as a separate binary.

### Approach Analysis

#### Option A: Integration tests in `tests/` directory
**Structure**: `crates/gpu-host/tests/gpu_tests.rs` (or split into multiple files)

**Pros**:
- Standard `cargo test` workflow
- Each `#[test]` function gets proper pass/fail reporting
- Can use `#[ignore]` for slow tests, `--test-threads=1` for serialization
- Filters work: `cargo test test_name`

**Cons**:
- Integration tests can only access public API of the library crate
- Current test modules are `pub(crate)` — need to either make them `pub` or duplicate logic
- PTX constants are in main.rs (binary), not in lib.rs — need to expose or re-declare them
- The binary's `kernel.ptx` path is relative to crate root, so `include_str!("../kernel.ptx")` works from `tests/` too

**Migration path**:
1. Move PTX constants to lib.rs (or a `pub mod ptx` in lib.rs)
2. Make test functions `pub` instead of `pub(crate)`
3. Create `tests/gpu_tests.rs` that uses a shared device via `OnceLock`
4. Each `#[test]` calls the existing `run_*` function

#### Option B: `#[cfg(test)] mod tests` in lib.rs
**Pros**: Direct access to `pub(crate)` items

**Cons**:
- PTX constants are in main.rs, not lib.rs — same problem
- Unit tests in lib.rs feel wrong for GPU integration tests
- Harder to organize 110 tests

#### Option C: `libtest_mimic` test binary
**Pros**:
- Keep the binary structure
- Get structured test output (pass/fail per test, filtering, etc.)
- No API visibility changes needed
- Minimal migration effort

**Cons**:
- Adds a dependency (`libtest_mimic`)
- Not standard `cargo test` — requires `cargo run` or `cargo test --test ...`
- Actually, with `[[test]]` in Cargo.toml and `harness = false`, it integrates with `cargo test`

### Recommended Approach

**Option C (libtest_mimic) for initial migration, evolve to Option A over time.**

Rationale:
1. **Minimal disruption**: No API visibility changes. The existing `run_*` functions stay `pub(crate)`. PTX constants stay in main.rs.
2. **Immediate value**: Structured test output, filtering by name, `--test-threads` control.
3. **One-file change**: Replace `fn main()` with libtest_mimic `Trial` list.
4. **Panic test handling**: `libtest_mimic` supports `Trial::test(...).with_ignored_flag(true)` or can mark tests as "should panic".

**Proof of concept** (3 representative tests):
1. `run_write_thread_idx` — simplest basic test, KERNEL_PTX only
2. `run_hostcall_print_hello` — hostcall test with listener thread
3. `run_std_build_test` — different PTX (STD_BUILD_TEST_PTX)

**Implementation sketch**:
```rust
// In Cargo.toml, add:
// [dev-dependencies]
// libtest-mimic = "0.8"
//
// [[test]]
// name = "gpu_tests"
// harness = false
// path = "src/main.rs"  # or a new file

use libtest_mimic::{Arguments, Trial};

fn main() {
    let args = Arguments::from_args();
    let dev = CudaDevice::new(0).expect("CUDA init");

    let tests = vec![
        Trial::test("basic::write_thread_idx", {
            let d = Arc::clone(&dev);
            move || tests_basic::run_write_thread_idx(d)
                .map_err(|e| e.to_string().into())
        }),
        // ... 109 more
    ];

    libtest_mimic::run(&args, tests).exit();
}
```

**Alternative without new dependency**: Simply convert `fn main()` to return structured output by wrapping each test call with timing and pass/fail tracking, printing a summary at the end. This is essentially what the current code does, but less structured.

**Longer-term (Option A)**: Once the library API stabilizes:
1. Move PTX constants to `lib.rs` as `pub const`
2. Make test functions `pub`
3. Create proper integration tests in `tests/`
4. Use `std::sync::OnceLock<Arc<CudaDevice>>` for shared device

## Open Questions

1. **Is `libtest_mimic` acceptable as a dev-dependency?** It's lightweight (~500 lines, no transitive deps). Alternative: hand-roll a minimal test runner.

2. **Panic test isolation**: `run_panic_test` calls `process::exit(0)`. Options: (a) run it as a separate process via `std::process::Command`, (b) mark as `#[ignore]`, (c) skip in harness.

3. **PTX file availability**: Integration tests assume PTX files exist at compile time. If PTX is not built, tests will fail to compile (due to `include_str!`). Should there be a feature gate or build script check?

4. **Benchmark tests**: `run_hostcall_latency_benchmark`, `run_warp_divergence_measurement`, `run_sharding_benchmark` are benchmarks, not correctness tests. Should they be separated into `#[bench]` or a separate binary?

5. **Model-dependent tests**: The GPT-2 weight loading test checks for `models/model.safetensors` at runtime. In a test harness, this should be `#[ignore]` unless the model file exists.
