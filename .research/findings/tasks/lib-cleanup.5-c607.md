# lib-cleanup.5 — pub(crate) visibility for demo modules

## Summary

Simplified the `demo` feature gate pattern in `gpu-host` by removing the
dual `#[cfg(feature = "demo")] pub` / `#[cfg(not(feature = "demo"))] pub(crate)`
declarations. All affected modules are now unconditionally `pub(crate)`.

## Changes Made

### `crates/core/gpu-host/src/lib.rs`

Replaced 20 lines of dual-cfg declarations with 10 lines:

- `model` — always `pub(crate)`, gated on `gpt2`
- `model_generic` — always `pub(crate)`, gated on `gpt2`
- `model_yolo` — always `pub(crate)`, gated on `gpt2`
- `tokenizer` — always `pub(crate)`, gated on `gpt2`
- `yolo_backbone` — always `pub(crate)`, gated on `gpt2`

Updated module-level doc comment to list these as "Internal modules" rather
than "Optional modules".

### `crates/core/gpu-host/src/nn/mod.rs`

Removed dual-cfg pattern for three submodules:

- `cpu_ref` — always `pub(crate)`
- `models` — always `pub(crate)`
- `test_utils` — always `pub(crate)`

Updated doc comments to say "not part of the public API" instead of
"enable the `demo` feature to access".

## Modules That Must Stay `pub`

### `ptx` module — MUST stay `pub`

Used extensively outside `gpu-host`:

- `crates/test/gpu-test-harness/src/main.rs` (7 PTX constants)
- `crates/core/gpu-host/tests/gpu_integration.rs`
- `examples/hostcall/tokio-offload/src/main.rs`
- Referenced in doc examples in `lib.rs`, `gpu.rs`, `nn/registry.rs`

### `cubin` module — MUST stay `pub`

Same rationale as `ptx` — external crates load cubin for fast module init.

## `demo` Feature Status

**Cannot be fully removed from `Cargo.toml`.**

The `demo` feature is still referenced by:

1. **`crates/test/gpu-test-harness/Cargo.toml`** — `demo = ["gpu-host/demo"]`
   Removing `demo` from gpu-host would cause a cargo error here.

2. **`crates/core/gpu-host/tests/gpu_integration.rs`** — `#![cfg(feature = "demo")]`
   This file uses `demo` as a compile gate for integration tests. Without it,
   the tests would never compile when running `cargo test -p gpu-host`.

The `demo` feature remains as `demo = []` (no-op) in `Cargo.toml`. It no
longer controls any visibility — it only serves as:
- A downstream dependency target for `gpu-test-harness`
- A compile gate for `gpu_integration.rs`

## Downstream Impact

### `gpu-test-harness` — BROKEN (expected, needs follow-up)

`cargo check -p gpu-test-harness --features demo,gpt2` fails with 36 errors.
The test harness accesses `gpu_host::model`, `gpu_host::tokenizer`,
`gpu_host::model_yolo`, `gpu_host::yolo_backbone` which are now `pub(crate)`.

**Fix needed** (outside scope of this task — constraint: only modify `crates/core/gpu-host/`):
The test harness should either:
1. Use public re-exports via the `nn` API (e.g., `gpu_host::nn::models::gpt2`)
2. Get its own copies of weight-loading code
3. Access these through a `#[cfg(test)]`-style mechanism

### Examples — Already broken (pre-existing)

Multiple examples (`gpt2-inference`, `gpu-rag`, `yolo-detect`, `dynamic-control`,
etc.) reference `gpu_host::model`, `gpu_host::tokenizer`, `gpu_host::nn::models`
without enabling the `demo` feature. Under the PREVIOUS code, these were already
`pub(crate)` without `demo`, so the examples were already non-compilable.
This change does not make their situation worse.

## Verification

- `cargo clippy -p gpu-host -- -D warnings` — PASS
- `cargo clippy -p gpu-host --features demo -- -D warnings` — PASS
- `cargo doc --no-deps -p gpu-host` — PASS (clean docs, no demo modules visible)
- `cargo check -p gpu-test-harness --no-default-features` — PASS
- `cargo clippy -p gpu-host --features demo,gpt2,nn -- -D warnings` — pre-existing
  failures in `nn` ops (manual_div_ceil lint), unrelated to this change
