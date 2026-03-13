# docs.1: Rewrite README.md to reflect current project state
**Cycle**: 197 | **Theme**: docs | **Kind**: experiment | **Status**: done

## Summary
Rewrote README.md with updated Quick Start (toolchain setup + 3 example commands),
new sections for real std on GPU and GPU error handling, updated GPT-2 section with
KV cache performance (68ms/tok), added crate map, and fixed outdated limitations.

## Findings
### Q: What sections are outdated or missing?
A: Quick Start referenced non-existent scripts. GPT-2 section showed no-KV-cache
numbers. Capabilities table was missing std and error handling. No crate map.
Performance table was stale.
**Confidence**: high

### Q: Should README include crate dependency diagram?
A: Added an ASCII crate map listing all crates/ and examples/ with one-line
descriptions. A full dependency diagram would be too noisy for a README.
**Confidence**: high

### Q: What setup instructions are needed for new contributors?
A: Added: nightly toolchain install, nvptx64 target, rust-src component, CUDA
driver requirement. Each example is self-contained with automated build.rs.
**Confidence**: high

## Open Questions
None.
