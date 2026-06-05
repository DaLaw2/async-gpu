# Theme: test-integration — cargo test integration + coverage

## Status: in-progress

## What works
- `#[gpu_test]` expands to `#[test]` — cargo discovers GPU tests automatically
- GPU and CPU tests coexist in the same file, run side by side
- Failure propagation: nonexistent kernels → `Err` → test failure; GPU asserts → trap → CUDA error → test failure
- Multiple GPU tests run without resource conflicts (parallel and sequential)
- Test name filtering (`cargo test test_gpu`) works for selective execution
- All 5 default tests pass in 1.5s; 6 tests (with failure feature) pass in 1.8s

## Open items
- Workspace-level `cargo test` blocked by pre-existing `nn` feature gate on `gemm_bench` (not test-framework issue)
- Coverage tooling (test-integration.2+) not yet explored
- No GPU-side assert failure test yet (would require a deliberately-failing kernel + cubin rebuild)

## Key insight
The proc macro approach means zero special configuration — `#[gpu_test]` is invisible to cargo's test harness. Standard `cargo test` flags (filtering, threading, output capture) all work unchanged.
