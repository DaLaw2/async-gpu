# Theme: test-integration — cargo test integration + coverage

## Status: in-progress (pending cubin compilation)

## What works
- `#[gpu_test]` expands to `#[test]` — cargo discovers GPU tests automatically
- GPU and CPU tests coexist in the same file, run side by side
- Failure propagation: nonexistent kernels and GPU asserts both propagate as test failures
- 14 GPU tests + 2 CPU tests = 16 total tests in gpu_tests.rs
- Feature coverage: Box, String, HashMap, thread spawn, thread reuse, cooperative, cooperative_map, cooperative_reduce, math intrinsics, atomics, iterator chains
- `gpu_main_poll()` + `assert_eq!()` + `println!()` pattern works for all test categories

## Architecture
- Kernel functions: `crates/kernel/gpu-kernel-std/src/lib.rs` (zero-param extern "gpu-kernel")
- Host tests: `crates/test/gpu-test-harness/tests/gpu_tests.rs` (#[gpu_test] stubs)
- Macro: `crates/test/gpu-test-macro/src/lib.rs` (expands to #[test] + run_zero_param_with_cubin)

## Key constraints discovered
- GPU atomics must live in global memory (static), not stack — ptxas rejects `.local` space atomics
- Cooperative APIs work without shared memory (use global statics for data passing)
- ptxas compilation time scales super-linearly with PTX size (~30min for 11.4MB)

## Open items
- cubin compilation in progress — tests will run once ptxas completes
- Workspace-level `cargo test` blocked by pre-existing `nn` feature gate (unrelated)
