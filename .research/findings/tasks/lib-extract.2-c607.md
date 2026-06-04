# lib-extract.2: Create gpu-test-harness crate

**STATUS**: done
**SUMMARY**: Created `crates/test/gpu-test-harness/` and moved all 15 test/binary files
(main.rs, bench_harness.rs, 13 tests_*.rs) out of gpu-host. Rewrote ~42 `crate::error`,
`crate::hostcall`, and `crate::mapped_mem` imports to `gpu_host::*`. Removed `[[bin]]`
section and changed `default = ["gpt2"]` to `default = []` in gpu-host/Cargo.toml.

## Changes Made

### New crate: `crates/test/gpu-test-harness/`
- `Cargo.toml` — `[[bin]] name = "gpu-tests"`, depends on gpu-host + cudarc, features: gpt2, nn, cublas, async
- `src/main.rs` — moved from gpu-host, removed `mod error/hostcall/mapped_mem`, updated imports
- `src/bench_harness.rs` — moved as-is (standalone, no crate:: refs to rewrite)
- `src/tests_*.rs` (13 files) — moved, `crate::error` → `gpu_host::error`, `crate::hostcall` → `gpu_host::hostcall`, `crate::mapped_mem` → `gpu_host::mapped_mem`

### Modified: `crates/core/gpu-host/Cargo.toml`
- Removed `[[bin]]` section
- Changed `default = ["gpt2"]` to `default = []`

### Modified: `Cargo.toml` (workspace root)
- Added `"crates/test/gpu-test-harness"` to workspace members

### Modified: `scripts/ci-lint.sh`
- Added `test/gpu-test-harness` to CRATES_FMT list
- Added `check gpu-test-harness` step with `--features gpt2`

### Deleted from `crates/core/gpu-host/src/`
- main.rs, bench_harness.rs, tests_basic.rs, tests_benchmark.rs, tests_cnn.rs,
  tests_gemm.rs, tests_hostcall.rs, tests_inference.rs, tests_pipeline.rs,
  tests_scaling.rs, tests_search.rs, tests_std.rs, tests_tokenizer.rs,
  tests_transformer.rs, tests_warp.rs

## Verification
- `cargo +stable check -p gpu-host` — PASS (pure library, no bin target)
- `cargo +stable check -p gpu-test-harness --features gpt2` — PASS
- `cargo +stable check -p gpu-test-harness --features gpt2,cublas,nn,async` — PASS
- `cargo +stable fmt --check` — PASS (both crates)
- `cargo +stable clippy -p gpu-host -- -D warnings` — PASS
- `bash scripts/ci-lint.sh` — ALL PASS

## Import rewrite summary
- `use crate::error::*` → `use gpu_host::error::*` (13 files)
- `use crate::hostcall` → `use gpu_host::hostcall` (8 files + inline refs in tests_warp.rs)
- `use crate::mapped_mem::*` → `use gpu_host::mapped_mem::*` (10 files + inline refs)
- `crate::error::GpuHostError::*` → `gpu_host::error::GpuHostError::*` (inline in tests_cnn.rs)
- PTX constants (`crate::KERNEL_PTX` etc.) left as `crate::` — defined in main.rs
- `crate::bench_harness` left as `crate::` — bench_harness.rs moved with harness

## Bailout counter
- Syntax/typo: 0/5
- Missing API: 0/2
- Linker: 0/2
