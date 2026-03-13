# docs.2: Add doc-comments to undocumented public APIs
**Cycle**: 197 | **Theme**: docs | **Kind**: experiment | **Status**: done

## Summary
Added doc-comments to all undocumented public items in gpu-host (11 items across
error.rs, hostcall.rs, lib.rs). Enforced `-D missing_docs` on gpu-host in CI.

## Findings
### Q: Which crates have missing public API docs?
A: gpu-host had 11 undocumented items: error variant fields (4), CannedStdin::new (1),
HostcallBuffer fields (6). Fixed all. gpu-protocol and warp-macro already enforced.
**Confidence**: high

### Q: Should -D missing_docs be enforced on more crates?
A: Now enforced on gpu-host, gpu-protocol, warp-macro. gpu-atomics and gpu-runtime
are GPU-side crates with internal APIs — not enforcing yet. gpu-libc is a shim layer.
**Confidence**: high

## Open Questions
None.
