# api-docs.1: Add rustdoc to public crates
**Cycle**: 109 | **Theme**: api-docs | **Kind**: experiment | **Status**: done

## Summary

Audited rustdoc coverage for the 3 public-facing crates. gpu-protocol is fully documented (0 missing). warp-macro had 1 missing doc (the main `warp_async` proc macro function) — fixed with a comprehensive doc comment including usage example. gpu-runtime CANNOT build docs on x86_64 due to inline PTX asm in its dependency chain (gpu-atomics) — needs `#[cfg(doc)]` stubs as follow-up.

## Findings

### Q: Which public items currently lack documentation?
A:
- **gpu-protocol**: 0 missing docs. Fully documented. `cargo doc` builds clean with `-D missing_docs`.
- **warp-macro**: 1 missing doc (`pub fn warp_async`) — FIXED. Now passes `-D missing_docs`.
- **gpu-runtime**: Cannot generate docs on x86_64. `cargo doc` fails with 38 errors from `gpu-atomics` crate (inline PTX asm, `asm_experimental_arch` feature, nvptx intrinsics). Needs `#[cfg(doc)]` stubs in gpu-atomics and gpu-runtime to allow doc generation on the host target.

**Confidence**: high

### Q: What usage examples are most valuable for users?
A:
- warp-macro: Added example showing `#[warp_async]` transforming a sequential function into a WarpFuture. This is the primary user-facing API.
- gpu-protocol: Already has doc-tests (verified in CI).
- gpu-runtime: Prelude re-exports are the main user touchpoint (`use gpu_runtime::prelude::*`). Examples would be valuable but require the cfg(doc) stubs first.

**Confidence**: high

## Changes Made
- Added comprehensive doc comment to `warp_async` proc macro function (crates/warp-macro/src/lib.rs:468)
- Verified gpu-protocol and warp-macro pass `RUSTDOCFLAGS="-D missing_docs"` cargo doc

## Unexpected Discoveries
- gpu-protocol is surprisingly well-documented already — all public constants, structs, and functions have doc comments.
- gpu-runtime's doc-build failure is a real blocker for the api-docs success criteria. It's the crate that kernel authors interact with most (via prelude), but it's the one that can't generate docs.

## Open Questions
- Should we add `#[cfg(doc)]` stubs to gpu-atomics and gpu-runtime? This would allow `cargo doc` on x86_64 but requires careful stubbing of all inline PTX functions.
- Alternative: generate docs using `cargo doc --target nvptx64-nvidia-cuda`? This may not work with standard cargo doc tooling.

## Impact on Downstream Tasks
- Need a follow-up task (api-docs.2) for gpu-runtime cfg(doc) stubs
- CI should add `cargo doc --no-deps -D missing_docs` for gpu-protocol and warp-macro
