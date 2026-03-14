# std-hardening.1: Update README with single-thread std limitation
**Cycle**: 205 | **Theme**: std-hardening | **Kind**: experiment | **Status**: done

## Summary
Updated README.md to accurately document that real Rust std on GPU requires
single-thread kernel launch (block_dim: (1,1,1)). Added notes in three places:
code example comment, "What works" line, Capabilities table, and Limitations section.

## Findings
### Q: What README sections needed updating?
A: Four locations: (1) code example comment, (2) "What works" qualifier,
(3) Capabilities table "Std library" row, (4) Limitations section — split old
"Partial std" into "Single-thread std" + "Partial std" (HashMap panic).
**Confidence**: high
