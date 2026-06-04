# lib-cleanup.2: Move model/yolo/tokenizer demo code out of gpu-host

**Status**: blocked
**Date**: 2025-06-04

## Summary

Moving model/tokenizer/yolo files to gpu-test-harness is blocked by structural
constraints: the test harness is a binary (not a library), 50+ call sites
reference these modules via `gpu_host::` paths, and examples also depend on
them. The nn test utilities (`test_utils.rs`, `cpu_ref.rs`) are consumed by
`#[cfg(test)]` blocks *inside* gpu-host itself. All target modules are already
properly feature-gated behind `gpt2`.

## Experiment Details

### Goal 1: Move model.rs, model_generic.rs, model_yolo.rs, yolo_backbone.rs, tokenizer.rs -> gpu-test-harness/src/

**Result**: Blocked.

**Blockers**:
1. `gpu-test-harness` is a `[[bin]]` target, not a library crate. Moving modules
   there would make them inaccessible to anything else.
2. 50+ import sites in `tests_inference.rs`, `tests_cnn.rs`, `tests_tokenizer.rs`,
   and `main.rs` reference these via `gpu_host::model::*`, `gpu_host::tokenizer::*`,
   `gpu_host::model_yolo::*`, `gpu_host::yolo_backbone::*`.
3. `examples/std/` (gpt2-inference, gpu-rag, dynamic-control, yolo-detect) also
   import these modules from `gpu_host`.
4. `model_generic.rs` is a dependency of `nn/models/gpt2.rs`, `nn/models/resnet.rs`,
   and `nn/models/yolov8.rs`.

**Current state**: Already correctly feature-gated behind `gpt2` in `lib.rs`:
```rust
#[cfg(feature = "gpt2")]
pub mod model;
#[cfg(feature = "gpt2")]
pub mod model_generic;
#[cfg(feature = "gpt2")]
pub mod model_yolo;
#[cfg(feature = "gpt2")]
pub mod tokenizer;
#[cfg(feature = "gpt2")]
pub mod yolo_backbone;
```

### Goal 2: Move nn/test_utils.rs, nn/cpu_ref.rs -> gpu-test-harness/src/

**Result**: Blocked.

**Blockers**:
1. `nn/models/gpt2.rs` uses `crate::nn::test_utils::{GoldenEntry, Tolerance}` and
   `crate::nn::test_utils::{golden_dir, assert_close}` in `#[cfg(test)]` blocks.
2. `nn/models/yolov8.rs` similarly uses `crate::nn::test_utils` in tests.
3. `test_utils.rs` itself has extensive `#[cfg(test)]` GPU validation tests that
   use `crate::nn::ops::*`, `crate::nn::registry::*`, `crate::nn::autograd::*`,
   and other crate-internal APIs (~900 lines of GPU gradient checks).
4. `cpu_ref.rs` is re-exported via `test_utils.rs` (`pub use super::cpu_ref::*`).

No external crate references `gpu_host::nn::cpu_ref` or `gpu_host::nn::test_utils`.

### Goal 3: Feature-gate nn/models/ behind gpt2

**Result**: Already done.

- `nn/models/gpt2.rs` — `#[cfg(feature = "gpt2")]` in `models/mod.rs`
- `nn/models/yolov8.rs` — `#[cfg(feature = "gpt2")]` in `models/mod.rs`
- `nn/models/resnet.rs` — Not feature-gated at module level, but its
  `from_safetensors()` method is inline-gated with `#[cfg(feature = "gpt2")]`.
  The rest of ResNet (struct, forward, from_weights) compiles without `gpt2`.

`cargo +stable check -p gpu-host --features nn` passes (no gpt2) confirming
resnet compiles fine standalone.

## Recommended Next Steps

The correct factoring would be a **new library crate** (e.g., `crates/models/gpu-models`)
that depends on `gpu-host` and provides the model loading/inference code. This would:
- Give the code a proper home (library, not binary)
- Allow test-harness and examples to depend on it
- Keep gpu-host focused on runtime/hostcall/nn infrastructure

This is a larger refactor (new crate, update 50+ imports, update Cargo.toml
dependencies in test-harness + all examples). Scope for a separate task.

## Files Changed

None — experiment concluded that the move is blocked without a new crate.

## Baseline Verification

- `cargo +stable check -p gpu-host` — PASS
- `cargo +stable check -p gpu-host --features nn` — PASS (1 dead_code warning)
- `cargo +stable check -p gpu-test-harness --features gpt2` — PASS
