# panic-deprecate-gpu-assert.2 — Experiment: Deprecate gpu_assert! and migrate callers

**Status**: PASS
**Date**: 2026-06-06

## Baseline

- Workspace builds clean (`cargo build --workspace`)
- CI lint passes (`scripts/ci-lint.sh`)
- Single call site: `gpu_runtime::gpu_assert!(buf, 1 + 1 == 2, "math works")` in hostcall_kernels.rs

## Changes Made (3 phases)

### Phase 1: Replace call site
- `crates/kernel/gpu-kernel-io/src/hostcall_kernels.rs:1279`
- `gpu_runtime::gpu_assert!(buf, 1 + 1 == 2, "math works")` → `assert!(1 + 1 == 2, "math works")`

### Phase 2: Remove macro
- Deleted `gpu_assert!` macro (both `cfg(feature = "gpu-trace")` and non-trace variants) from `crates/core/gpu-runtime/src/lib.rs` (~55 lines removed)

### Phase 3: Remove SERVICE_ASSERT plumbing
- `crates/core/gpu-runtime/src/hostcall.rs` — deleted `gpu_hostcall_assert()` function (~78 lines)
- `crates/core/gpu-host/src/hostcall.rs` — deleted `handle_assert()` method and its dispatch arm
- `crates/core/gpu-protocol/src/lib.rs` — deleted `SERVICE_ASSERT` constant (14), `ASSERT_MAX_MSG_LEN` constant, and ASSERT payload layout comments
- `crates/core/gpu-runtime/src/prelude.rs` — removed `gpu_hostcall_assert` and `SERVICE_ASSERT` re-exports

## Verification

- `cargo build --workspace` — PASS
- `cargo +stable fmt --check` — PASS (after reformatting)
- `cargo +stable clippy -- -D warnings` — pre-existing warnings only (unrelated to changes)
- `scripts/ci-lint.sh` — all checks PASS (fmt, clippy, doc-tests, PTX compilation, workspace check, examples)
- Test kernel `test_gpu_assert_basic` uses standard `assert!`/`assert_eq!` — unaffected

## Notes

- Standard `assert!` on GPU now goes through the panic handler (SERVICE_PANIC), which provides better diagnostics (file/line/column via `#[panic_handler]`) than the old gpu_assert! macro
- No `#[allow(dead_code)]` added — all removed code was truly dead after migration
- SERVICE_ASSERT ID 14 is now unused; no renumbering needed since IDs are wire constants
