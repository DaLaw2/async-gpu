# ve-model-paths.1: Audit all model path references across codebase
**Cycle**: 408 | **Theme**: ve-model-paths | **Kind**: investigation | **Status**: done

## Summary
19+ model path references across 9 files, all using hardcoded relative paths with no centralized helper.
`models/` directory doesn't exist in repo (gitignored) — users download models separately.

## Findings

### Q: Where are model files referenced?
A: All references point to repo-root `models/` via relative paths:

| File | Count | Pattern | Depth |
|------|-------|---------|-------|
| `crates/core/gpu-host/src/main.rs:427` | 1 | `../../models/model.safetensors` | 2 levels |
| `examples/std/gpt2-inference/src/main.rs:35` | 1 | `CARGO_MANIFEST_DIR + ../../../models/model.safetensors` | 3 levels |
| `examples/std/yolo-detect/src/main.rs:56,79` | 2 | `CARGO_MANIFEST_DIR + ../../../models/{yolov8n.safetensors,bus.ppm}` | 3 levels |
| `crates/core/gpu-host/src/tests_inference.rs` | 7 | `CARGO_MANIFEST_DIR + ../../../models/model.safetensors` | 3 levels |
| `crates/core/gpu-host/src/tests_cnn.rs:681,1113,1114` | 3 | Mixed: `../../../models/` and `base.join("models/")` | 3 levels / root |

**Library loaders** (`model.rs`, `model_yolo.rs`) correctly take `&Path` — no hardcoded paths there.

**Confidence**: high

### Q: Is there a centralized path helper?
A: No. Every call site constructs its own relative path.
**Confidence**: high

### Q: Is the models/ directory present?
A: No. Gitignore covers `*.safetensors`, `*.pt`, `/models/`. Users must download models separately.
**Confidence**: high

## Unexpected Discoveries
- `main.rs` uses 2-level depth (`../../models/`) while everything else uses 3-level (`../../../models/`). This is likely a bug since main.rs is at `crates/core/gpu-host/src/main.rs` (4 levels deep from root).
- `tests_cnn.rs` has TWO different patterns: one uses `CARGO_MANIFEST_DIR` + relative, the other resolves a `base` path to join `models/`.

## Open Questions
- Should the helper use `CARGO_MANIFEST_DIR` or runtime workspace root detection?
- Should we also handle env var override (e.g., `ASYNC_GPU_MODELS_DIR`)?

## Impact on Downstream Tasks
- ve-model-paths.2: Need a `model_dir()` helper. Best approach: find repo root from CARGO_MANIFEST_DIR or use env var fallback.
- ve-model-paths.3: 19+ call sites to update. Straightforward search-and-replace once helper exists.
