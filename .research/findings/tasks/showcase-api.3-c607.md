# showcase-api.3 — cargo doc verification

## Status: PASS (with 3 minor fixes applied)

## What was done

Ran `cargo doc --no-deps -p async-gpu` and verified the output.

### Checks performed

1. **`cargo doc --no-deps -p async-gpu`** — clean, no warnings
2. **`RUSTDOCFLAGS="-W missing_docs" cargo doc`** — clean, no missing docs
3. **`RUSTDOCFLAGS="-W rustdoc::broken_intra_doc_links" cargo doc`** — clean, no broken links
4. **`cargo check -p async-gpu`** — compiles without errors
5. **Visual inspection of generated HTML** — all types properly documented

### Issues found and fixed

Three re-exports had `///` doc comments that were **concatenated** with the source
crate's own doc comment, producing duplicate/redundant descriptions in the generated
docs:

| Item | Before (in docs) | After |
|------|-------------------|-------|
| `FlightRecorder` | "Post-mortem GPU trace ring buffer for crash investigation. A mapped-memory ring buffer that stores the last N trace events." | "A mapped-memory ring buffer that stores the last N trace events." |
| `GpuStream` | "CUDA stream overlap support. A CUDA stream for overlapping kernel execution." | "A CUDA stream for overlapping kernel execution." |
| `Result` | "Convenience type alias: `Result<T, GpuHostError>`. Convenience type alias for `Result<T, GpuHostError>`." | "Convenience type alias for `Result<T, GpuHostError>`." |

**Fix**: Removed the redundant `///` doc comments from the `pub use` re-exports in
`crates/async-gpu/src/lib.rs`. The source crate's docs carry through automatically.

### Doc structure overview

The generated docs at `target/doc/async_gpu/index.html` show:

- **Crate-level docs**: Quick Start example with `gpu::run()`, `gpu::launch()`,
  `gpu::custom()` builder pattern. Advanced Usage section. Feature Flags section.
- **Modules**: `gpu` (one-liner API with `run`, `launch`, `custom`, `run_zero_param`,
  `GpuStdModule`, `CustomLaunchBuilder`, `GpuContext`, `GpuResult`)
- **Structs**: `FlightRecorder`, `GpuKernelErrorInfo`, `GpuRuntime`, `GpuStream`,
  `HostcallBuffer`, `HostcallSession`, `MappedBuffer`, `Pipeline`
- **Enums**: `GpuHostError`, `HostcallError`
- **Functions**: `error_category_name`
- **Type Aliases**: `Result`
- **Hidden**: `ptx`, `model_dir` (internal use only)

All types have proper rustdoc descriptions. The docs are well-organized for a new user:
simple entry points (`gpu` module) are prominent, while advanced types are available at
the crate root.

### CI note

`scripts/ci-lint.sh` fails due to a pre-existing `GpuFilter` reference error in
`crates/core/gpu-runtime/src/par_iter.rs` — this is unrelated to the doc changes.
The async-gpu crate itself compiles and documents cleanly.

## Files modified

- `crates/async-gpu/src/lib.rs` — removed 3 redundant `///` doc comments on re-exports
