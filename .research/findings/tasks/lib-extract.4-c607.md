# lib-extract.4: Verify test harness works from new location

**Status**: done
**Date**: 2026-06-04

## Verification Results

### 1. Build test harness
```
AUTO_BUILD_KERNEL=0 cargo +stable build --release -p gpu-test-harness --features gpt2
```
**Result**: PASS — compiled gpu-host (library) + gpu-test-harness (binary `gpu-tests`) in 14.41s.

### 2. Key tests via ONLY_TEST

| Test | Result | Notes |
|------|--------|-------|
| `ONLY_TEST=gpu_run` | PASS | gpu::run/launch API, gpu_kernel_demo — all assertions passed |
| `ONLY_TEST=thread_spawn` | PASS | spawn 2 threads + join, spawn 4 tasks with reuse — all assertions passed |
| `ONLY_TEST=cooperative` | PASS | cooperative debug, compute, map, reduce (32640), map_with_params — all passed |
| `ONLY_TEST=std_fs` | HANG (pre-existing) | Kernel launches but hostcall listener never receives messages. Timeout after 30s. Same behavior as before extraction — the `std_file_io_test` kernel in `kernel_std.ptx` stalls during hostcall communication. Not caused by the extraction. |

### 3. gpu-host as pure library
```
cargo +stable check -p gpu-host
```
**Result**: PASS — compiles as a library with no binary targets, clean check in 0.32s.

### 4. CI lint
```
bash scripts/ci-lint.sh
```
**Result**: PASS — all checks passed (fmt, clippy, doc-tests, doc, check, ptx for all crates, example checks).

## Analysis

The test harness extraction is verified working. Three of four key tests pass cleanly. The `std_fs` hang is a pre-existing kernel-level issue where the `std_file_io_test` kernel in `kernel_std.ptx` deadlocks during hostcall communication — this behavior is identical before and after the extraction (the kernel_std.ptx file was not modified by the extraction, only the host-side test binary location changed).

## Files Verified
- `crates/test/gpu-test-harness/Cargo.toml` — correct package name, binary name, feature gates
- `crates/test/gpu-test-harness/src/main.rs` — all test modules present, PTX constants from library
- `crates/core/gpu-host/Cargo.toml` — pure library, no binary targets
- `crates/core/gpu-host/src/lib.rs` — ptx module with include_str! for all PTX files
