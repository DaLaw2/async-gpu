# readme-fix.1: Fix README Limitations — single-thread std claim is outdated
**Cycle**: 247 | **Theme**: readme-fix | **Kind**: experiment | **Status**: done

## Summary
Fixed two outdated claims in README Limitations section that contradicted the body text.

## Findings
### Q: What was wrong?
A: Two issues:
1. "Single-thread std" bullet claimed "thread-local storage, errno, and allocator are not yet multi-thread safe" — but all three were fixed by std-multithread theme (gpu_threads.rs, atomic CAS allocator, per-thread errno).
2. "HashMap panics (no random seed source)" — but hashmap-fix theme proved HashMap works via address-based seed (no fill_bytes() call).

**Confidence**: high

### Q: What was changed?
A: Replaced the two bullets:
- "Single-thread std" → "Hostcall-limited concurrency" — documents the real limitation (16-packet pool constrains I/O concurrency, not thread safety)
- Updated HashMap status to "works (address-based seed)" with note that OsRng/getrandom are not available

## Impact on Downstream Tasks
None — README is now consistent. Body (lines 103, 122, 124) and Limitations (line 318) agree on multi-thread safety.
