# docs-refresh.3: Rewrite CHANGELOG.md + getting-started.md for Current API Surface

## Summary

Rewrote both docs/CHANGELOG.md and docs/getting-started.md to reflect the full
project history and current API surface.

**CHANGELOG.md** (was 129 lines, now ~250 lines): Reorganized from 3 sections
(Unreleased + v0.2.0 + v0.1.0) into a comprehensive chronological record.
The Unreleased section now covers all major features from cycles 309-642+
organized by subsystem: Conv2D optimization, ownership/memory model,
transparent data (GpuArray<T>), auto-tuning, dynamic dispatch, compile-time
cost model, GPU panic, API encapsulation, coroutines/generators, unified
runtime, generics, type safety, test framework, kernel split, iterators,
structured concurrency, kernel performance, developer experience, neural
networks, ONNX runtime, quantization, pre-built models, and infrastructure.
v0.2.0 and v0.1.0 sections preserved and lightly cleaned.

**getting-started.md** (was 457 lines, now ~370 lines): Updated with current
API surface while preserving the proven tutorial flow. Added new sections for
GpuArray<T> transparent data, AutoTuner usage, GPU test framework (#[gpu_test]),
workspace structure diagram, and async/tokio feature flag. Updated toolchain
version references to nightly-2026-06-03. Expanded "Next Steps" with nn module
and async integration paths.

## Findings

1. CHANGELOG now covers 56 completed epics across the full project timeline
2. getting-started.md now documents 3 launch APIs: gpu::run(), gpu::launch(),
   gpu::custom() builder
3. Added GpuArray<T> and AutoTuner sections — these were completely missing
4. Workspace structure diagram added showing all 15 crates and 24 examples
5. Troubleshooting section preserved and updated with current toolchain date

## Open Questions

1. docs/ is still gitignored — these files are not tracked in version control.
   Decision pending from docs-refresh.1 audit.
