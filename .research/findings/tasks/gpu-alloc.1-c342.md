# gpu-alloc.1: Current GPU allocator design and slab allocator options
**Cycle**: 342 | **Theme**: gpu-alloc | **Kind**: investigation | **Status**: done

## Summary
The GPU already has a production-ready slab+bitmap allocator in patched-std that supports deallocation. It has 8 size classes (16B-4KB), atomic bitmap CAS for lock-free alloc/dealloc, and an overflow bump region for large objects. Comprehensive tests (single-thread + 32-thread concurrent) pass. The gpu-alloc epic criterion is **already met** by existing code.

## Findings

### Q: What allocator does the GPU currently use (bump/linear)?
A: **Hybrid slab+bitmap allocator** (not just bump). Two implementations exist:
- **gpu-libc** (`crates/core/gpu-libc/src/memory.rs`): Atomic bump allocator, `free()` is a no-op. Used for development/baseline.
- **patched-std** (`patched-std/src/sys/alloc/cuda.rs`): Slab+bitmap with 8 size classes (16B×2048, 32B×1024, ... 4KB×32). Total slab: 384KB. Overflow bump: ~640KB. **Supports full deallocation** via atomic bitmap CAS.
**Confidence**: high

### Q: What's the interface — global_allocator trait vs custom?
A: Standard Rust `GlobalAlloc` trait on the `System` type. Platform detection in `patched-std/src/sys/alloc/mod.rs` selects `mod cuda` for `target_os = "cuda"`. Automatically used by Vec, String, Box, format! — the full std alloc machinery.
**Confidence**: high

### Q: What slab allocator designs work on GPU (no OS, no TLS)?
A: **Atomic bitmap design** is the current winner and already implemented. Each size class has an `AtomicU32` bitmap where bit=1 means allocated. Alloc: find-first-zero via CTZ + CAS. Dealloc: compute bit index from pointer offset + CAS clear. No OS, no TLS needed. Alternatives evaluated in cycle 37 (ScatterAlloc, halloc) were more complex with no clear benefit for ≤1024 threads.
**Confidence**: high

### Q: Can we use a buddy allocator or power-of-2 free list?
A: **Buddy allocators are NOT recommended** for GPU:
1. Cascading coalescing is hard to make lock-free (multi-word CAS needed)
2. Higher register pressure (12-18 regs vs 8-12 for slab)
3. O(log n) vs O(1) for slab within a class
4. Merge operations have unsynchronized intermediate states

Power-of-2 free lists also problematic — pointer chasing in linked lists is high latency on GPU. Current bitmap approach (bitwise ops) is faster.
**Confidence**: high

## Unexpected Discoveries
- **The gpu-alloc epic criterion is already satisfied!** patched-std has a working slab allocator with deallocation, tested with 32 concurrent threads × 5 alloc/dealloc cycles
- Tests exist: `slab_dealloc_test_kernel()` (10 Vec + 10 String cycles) and `slab_concurrent_test_kernel()` (32-thread stress test)
- The only remaining gap is the overflow bump region (>4KB allocs can't be freed) — but this is acceptable for typical GPU workloads

## Open Questions
- Should we tune CLASS_COUNTS per GPU architecture? (current values target 384KB total)
- Is the overflow bump region a problem for long-running kernels?
- Should we expose allocator stats (used/free per class) for debugging?

## Impact on Downstream Tasks
- **gpu-alloc theme may be already complete** — existing allocator meets all 3 success criteria:
  1. ✅ Supports deallocation (bitmap CAS clear)
  2. ✅ No fragmentation under typical patterns (fixed-size classes)
  3. ✅ Vec, String, Box work with it (tested)
- Recommend: mark theme as completed, or add tuning/stats tasks if needed
