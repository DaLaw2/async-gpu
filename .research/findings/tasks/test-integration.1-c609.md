# test-integration.1: cargo test integration — GPU tests alongside CPU tests

## Summary

Verified that `#[gpu_test]` tests coexist with regular `#[test]` functions in the same
file and are discovered/run by standard `cargo test`. Added 2 CPU tests and 1 failure
propagation test (behind feature flag). All pass.

## Evidence

### Mixed GPU + CPU test run (`cargo test -p gpu-test-harness --test gpu_tests`)
```
running 5 tests
test test_cpu_sanity_check ... ok         (CPU)
test test_gpu_test_macro_is_available ... ok  (CPU)
test test_gpu_thread_spawn ... ok         (GPU)
test test_gpu_assert_basic ... ok         (GPU)
test test_gpu_vec_operations ... ok       (GPU)
test result: ok. 5 passed; finished in 1.48s
```

### Failure propagation (`--features test-failure-propagation`)
```
running 6 tests
test test_gpu_failure_propagation ... ok  (verifies KernelNotFound error path)
test result: ok. 6 passed; finished in 1.81s
```

### Sequential execution (`--test-threads=1`)
```
running 5 tests — all pass in sequence, no GPU resource conflicts
test result: ok. 5 passed; finished in 2.75s
```

### Test name filtering
- `cargo test ... test_gpu` — runs 4 tests (3 GPU + 1 macro-check)
- `cargo test ... test_cpu` — runs 1 test (CPU only)

## Findings

1. **Cargo test integration works out of the box.** `#[gpu_test]` expands to `#[test]`,
   so cargo's test discovery finds them alongside regular tests with zero configuration.

2. **Failure propagation works.** `run_zero_param_with_cubin` returns `Err(KernelNotFound)`
   for nonexistent kernels; the `.expect()` in the macro expansion converts this to a
   test panic with a clear message. GPU-side assert failures also propagate (trap → CUDA
   error → `Err` → panic).

3. **Multiple GPU tests run without conflicts.** Both parallel (default) and sequential
   (`--test-threads=1`) execution work. Each test creates a fresh CUDA module via
   `MODULE_SEQ` atomic counter, avoiding stale state.

4. **Naming convention is clear.** GPU tests are named `test_gpu_*`, CPU tests `test_cpu_*`.
   `cargo test test_gpu` filters to GPU-only tests.

5. **Workspace-level `cargo test` is blocked by pre-existing issues.** The `gpu-host`
   crate's `gemm_bench` test requires the `nn` feature, which isn't enabled by default.
   This is unrelated to the test framework.

## Changes made

- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — added 2 CPU tests + 1 failure
  propagation test (behind `test-failure-propagation` feature flag)
- `crates/test/gpu-test-harness/Cargo.toml` — added `test-failure-propagation` feature
