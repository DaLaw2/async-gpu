# audit-api — Feature Synthesis
## Status: active (1/N tasks complete)

## Task 1 Complete: API Surface Review (audit-api.1)
Systematic review of public API across all 7 core crates. API is clean overall:
zero anyhow, zero dead_code allows, consistent naming, good docs, well-designed facade.

## Key Issues Found (prioritized)
1. **GpuHostError::Verification overload**: Catch-all for ~32 error modes in gpu.rs, blocks programmatic handling.
2. **HostcallBuffer pub fields**: Raw pointer fields `pub` instead of `pub(crate)` + accessors.
3. **GpuVec not in facade**: User-facing type missing from async-gpu re-exports.
4. **onnx_rt missing docs**: `#[allow(missing_docs)]` on public module.
5. **No thiserror**: All error types hand-rolled — consistent but boilerplate.
6. **Pipeline hardcoded sleep**: 100ms sleep in Pipeline::run() looks like workaround.

## Clean Areas
- Naming consistent (snake_case fns, CamelCase types, SCREAMING_SNAKE consts)
- No anyhow, no #[allow(dead_code)], no stale deprecations
- #[allow(unused_mut)] in warp.rs justified (nvptx64 compat); error From impls clean
- Facade re-exports cover primary user API surface

## Suggested Next Tasks
- audit-api.2: Fix HostcallBuffer pub fields → pub(crate) + accessors
- audit-api.3: Split GpuHostError::Verification into typed variants
- audit-api.4: Add GpuVec + scheduler re-exports to async-gpu facade
- audit-api.5: Add doc comments to onnx_rt public API
