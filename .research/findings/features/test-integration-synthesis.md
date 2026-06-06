# Theme: test-integration — cargo test integration + coverage

## Status: tests running (PTX JIT in progress, all code merged)

## What works
- `#[gpu_test]` expands to `#[test]` — cargo discovers GPU tests automatically
- GPU and CPU tests coexist in the same file, run side by side
- 14 GPU tests + 2 CPU tests = 16 total tests in gpu_tests.rs
- Feature coverage: assert, Vec, thread spawn, Box, String, HashMap, thread data passing, thread reuse, cooperative, cooperative_map, cooperative_reduce, math intrinsics, atomics, iterator chains
- `stdio_auto_init()` + `gpu_main_poll()` + `assert_eq!()` + `println!()` pattern works for all categories
- Stash merged cleanly into post-split crate structure (gpu-kernel-test)

## Architecture
- Kernel functions: `crates/kernel/gpu-kernel-test/src/lib.rs` (zero-param extern "gpu-kernel")
- Host tests: `crates/test/gpu-test-harness/tests/gpu_tests.rs` (#[gpu_test] stubs)
- Macro: `crates/test/gpu-test-macro/src/lib.rs` (expands to #[test] + run_zero_param_with_cubin)

## Key constraints
- GPU atomics must live in global memory (static), not stack — ptxas rejects `.local` space atomics
- Cooperative APIs work without shared memory (use global statics for data passing)
- PTX JIT takes ~15-20 min for 7MB PTX; cubin build eliminates this per-test cost
- Tests must run `--test-threads=1` without cubin (GPU contention with parallel JIT)

## Open items
- PTX JIT compile in progress — tests completing once JIT finishes
- Building cubin via ptxas would eliminate per-run JIT overhead
