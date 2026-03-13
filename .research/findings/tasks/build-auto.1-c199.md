# build-auto.1: Auto-generate PTX stub list from crate scan
**Cycle**: 199 | **Theme**: build-automation | **Kind**: experiment | **Status**: done

## Summary
Replaced the hardcoded PTX stub list in ci-lint.sh with grep-based auto-discovery.
The script now scans `crates/gpu-host/src/` for `include_str!("../*.ptx")` patterns
and creates stubs for any missing PTX files.

## Findings
### Q: Can ci-lint.sh discover PTX filenames from gpu-host include_str! calls?
A: Yes. `grep -roh 'include_str!("\.\.\/[^"]*\.ptx")' | sed` extracts 7 filenames.
Adding a new kernel only requires adding the `include_str!` call — no manual stub
list update needed.
**Confidence**: high

### Q: Can CI workflow share the same logic?
A: The CI workflow (.github/workflows/build.yml) runs ci-lint.sh directly, so it
inherits the auto-discovery automatically. The PTX kernel build list (PTX_KERNELS)
remains manual because some crates need special flags (e.g., -Zbuild-std=std for
gpu-kernel-std).
**Confidence**: high

## Open Questions
None.
