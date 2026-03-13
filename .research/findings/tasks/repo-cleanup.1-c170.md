# repo-cleanup.1: Gitignore PTX + remove duplicate model file
**Cycle**: 170 | **Theme**: repo-cleanup | **Kind**: experiment | **Status**: done

## Summary
Removed 9 tracked PTX build artifacts from git (3.4MB total), added `*.ptx` to
.gitignore, and deleted the duplicate 523MB model file at crates/gpu-host/models/.

## Findings

### Q: Are any PTX files referenced by path in code that would break after git rm --cached?
A: No. PTX files are loaded via `include_str!("../kernel.ptx")` which reads from
the filesystem at compile time. The files still exist on disk (just untracked).
CI would need to build PTX first, but there is no CI currently.
**Confidence**: high

### Q: Which model path does inference code reference — root or crate-local?
A: All inference code uses `"../../models/model.safetensors"` (root `models/` dir).
The crate-local `crates/gpu-host/models/model.safetensors` was never referenced in
code — it was a leftover copy. Safely deleted (saves 523MB disk space).
**Confidence**: high

## Changes Made
1. Added `*.ptx` to `.gitignore`
2. `git rm --cached` 9 PTX files (3.4MB removed from tracking)
3. Deleted `crates/gpu-host/models/` directory (523MB duplicate)

## Impact on Downstream Tasks
- repo-cleanup.2 and .3 can proceed (no blockers from cleanup)
- Build instructions remain the same (PTX build step was already required)
