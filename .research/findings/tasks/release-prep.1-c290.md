# release-prep.1: Catalog key milestones for CHANGELOG
**Cycle**: 290 | **Theme**: release-prep | **Kind**: investigation | **Status**: done

## Summary
Reviewed 220 git commits to identify major milestones. CHANGELOG.md written with 6 sections covering all project capabilities. Version numbering: v0.1.0 (research prototype, pre-1.0).

## Findings

### Q: What are the major milestones?
A: Key milestones in chronological order:
1. First GPU kernel execution (RTX 3060)
2. Lock-free hostcall protocol (two-stack CAS)
3. GPU println via hostcall
4. Async/await + File I/O on GPU (first MILESTONE)
5. Sideband bulk I/O (4KB+)
6. GPU-autonomous file transform pipeline
7. gpu-host SDK extraction
8. Multi-thread std on GPU
9. WarpFuture trait + #[warp_async] proc macro
10. GPT-2 124M inference on GPU
11. #[warp_cooperative] MIR pass in patched rustc
12. Real std (File, Vec, String, println!) on GPU via patched sysroot
13. Async bulk I/O Futures
14. ARCHITECTURE.md + comprehensive README

### Q: What version numbering scheme?
A: v0.1.0 — research prototype. Pre-1.0 because:
- Requires patched rustc (not upstreamable yet)
- API may change
- Single-GPU, single-user tested

**Confidence**: high
