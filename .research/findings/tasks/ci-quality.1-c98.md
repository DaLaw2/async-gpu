# ci-quality.1+2+3: Clippy, fmt, and CI workflow
**Cycle**: 98-100 | **Theme**: ci-quality | **Kind**: experiment+design | **Status**: done

## Summary
All host-side crates cleaned up: 98 clippy warnings fixed (mostly unnecessary_cast, manual_clamp),
needless_range_loop suppressed for volatile I/O byte loops, cargo fmt applied. GitHub Actions
workflow updated with 3 parallel jobs: lint (clippy+fmt), build-ptx (kernel), build-host (check).

## Findings

### Q: What clippy warnings exist in the codebase?
A: 98 warnings in gpu-host, zero in gpu-protocol and warp-macro. Breakdown:
- ~89 `unnecessary_cast` (u64 as u64) — auto-fixed by `cargo clippy --fix`
- 6 `needless_range_loop` — volatile I/O byte loops, suppressed with `#![allow]`
- 1 `manual_clamp` — `.min(64).max(4)` → `.clamp(4, 64)`
- 1 `empty_line_after_doc_comments` — in build.rs
- 2 pointer casts in Drop impl needed restoration (were `*mut u8` → `*mut c_void`, not same-type)
**Confidence**: high

### Q: Does the codebase pass cargo fmt --check?
A: After formatting, yes. Main changes: function argument wrapping and expression line breaks.
**Confidence**: high

### Q: What CI workflow setup works for this project?
A: Three parallel jobs:
1. **lint**: stable Rust, clippy -D warnings + fmt --check on gpu-host, gpu-protocol, warp-macro
2. **build-ptx**: nightly-2026-03-11, nvptx64 target, builds gpu-kernel to PTX
3. **build-host**: stable Rust, cargo check on gpu-host
GPU-side crates (gpu-atomics, gpu-runtime, gpu-libc) cannot run clippy (nvptx64 only).
**Confidence**: high

## Impact on Downstream Tasks
- ci-quality theme complete — all 3 tasks done
- CI enforces quality on every push/PR
