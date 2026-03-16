# ve-model-paths.2+3: Create model_dir() helper and update all references
**Cycle**: 408 | **Theme**: ve-model-paths | **Kind**: experiment | **Status**: done

## Summary
Added `gpu_host::model_dir(start)` function that resolves to workspace-root `models/`.
Updated all 19+ hardcoded relative paths across 5 files to use the new helper.

## Changes
- `crates/core/gpu-host/src/lib.rs`: Added `pub fn model_dir(start: Option<&str>) -> PathBuf`
  - Resolution: `ASYNC_GPU_MODELS` env var → walk-up-to-workspace-root → fallback to `models/`
- `crates/core/gpu-host/src/main.rs`: 1 reference updated
- `crates/core/gpu-host/src/tests_inference.rs`: 7 references updated (replace_all)
- `crates/core/gpu-host/src/tests_cnn.rs`: 3 references updated (2 patterns)
- `examples/std/gpt2-inference/src/main.rs`: 1 reference updated, removed unused `use std::path::Path`
- `examples/std/yolo-detect/src/main.rs`: 2 references updated (model + image)

## Verification
- `cargo +stable check -p gpu-host --features nn,gpt2` — passes
- `cargo +stable check` in gpt2-inference — passes (0 warnings after removing unused import)
- `cargo +stable check` in yolo-detect — passes
- `cargo +stable fmt` — clean on all files
- `grep` for `../../../models/` or `../../models/` — 0 results

## Impact on Downstream Tasks
- ve-run-gpt2.1 and ve-run-yolo.1 now unblocked (model paths are centralized)
