# test-framework.2: #[gpu_test] proc macro + GPU assert

## Summary

Implemented the `#[gpu_test]` proc macro in `crates/test/gpu-test-macro/` and verified
it end-to-end with 3 GPU test kernels that use standard `assert!`/`assert_eq!`/`assert_ne!`.
All tests pass via `cargo test` in 2.74 seconds (with cubin fast-loading).

## What was built

### 1. `gpu-test-macro` proc macro crate (`crates/test/gpu-test-macro/`)
- `#[gpu_test]` attribute macro that expands a stub function into a `#[test]`
- Parses optional `threads` and `grid` configuration attributes
- Generates code that loads cubin (sub-second) with PTX JIT fallback
- Uses `gpu_host::gpu::run_zero_param_with_cubin` for launch

### 2. GPU assert — already works (confirmed)
- Standard `assert!`, `assert_eq!`, `assert_ne!` work on GPU via patched std
- Panic handler sends thread/block coordinates via hostcall
- Failed assertions cause `trap;` which propagates as CUDA error to host
- No custom GPU assert needed for Phase 1

### 3. `run_zero_param_with_cubin` added to gpu-host
- New public function in `gpu::` that accepts a cubin `&[u8]` for fast loading
- `run_zero_param` and `run_zero_param_with_config` remain unchanged (backward compat)
- Cubin loads in sub-second vs 15-30 minutes for PTX JIT on 10MB PTX

### 4. Three GPU test kernels in `gpu-kernel-std/src/lib.rs`
- `test_gpu_assert_basic`: arithmetic assertions (assert_eq!, assert!, assert_ne!)
- `test_gpu_vec_operations`: Vec allocation, push, indexing, sum with assertions
- `test_gpu_thread_spawn`: thread spawn/join with result assertions

### 5. Integration test (`gpu-test-harness/tests/gpu_tests.rs`)
- Uses `#[gpu_test]` macro — 3 test stubs that expand to `#[test]` functions
- Runs via standard `cargo test -p gpu-test-harness --test gpu_tests`
- All 3 tests pass (ok. 3 passed; finished in 2.74s)

## Key findings

1. **PTX JIT is prohibitively slow for testing.** The unified PTX is ~10MB and takes
   15-30 minutes to JIT compile. Pre-compiled cubin is essential for usable test cycles.
   The proc macro generates code that loads cubin from a well-known path at runtime.

2. **assert! on GPU works perfectly.** The patched std panic handler routes panic
   messages through hostcall, which causes a CUDA error on the host. No custom
   GPU assert macro is needed.

3. **Warp/thread ID is already in panic messages.** The existing `send_panic_hostcall`
   encodes `thread_idx_x` and `block_idx_x` in the panic metadata. Warp ID can be
   derived as `thread_idx_x / 32`.

4. **Pre-existing kernel build error fixed.** A closure capture issue in
   `par_iter_demo.rs:325` (missing `move` on filter closure) was fixed as a
   drive-by to unblock PTX compilation.

## Expansion example

```rust
#[gpu_test]
fn test_gpu_assert_basic() {}

// Expands to:
#[test]
fn test_gpu_assert_basic() {
    let cubin = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let cubin_path = std::path::Path::new(manifest)
            .join("../../core/gpu-host/kernel_std.cubin");
        std::fs::read(&cubin_path).unwrap_or_default()
    };
    gpu_host::gpu::run_zero_param_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "test_gpu_assert_basic",
        128,
        (1, 1, 1),
    ).expect("GPU test 'test_gpu_assert_basic' failed");
}
```
