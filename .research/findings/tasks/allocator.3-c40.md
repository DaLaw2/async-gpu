# allocator.3: Concurrent allocator stress test (32 threads)
**Cycle**: 40 | **Theme**: allocator | **Kind**: experiment | **Status**: done

## Summary
32 GPU threads concurrently performed 5 alloc/dealloc cycles each (160 total operations)
using the slab+bitmap allocator. All 160 cycles completed successfully in 442.8µs with
zero corruption, zero OOM, and zero data integrity errors. The atomic bitmap CAS correctly
handles 32-way contention on shared bitmap words.

## Findings

### Q: Can 32 threads allocate and free concurrently without corruption?
A: **Yes.** All 32 threads completed all 5 cycles with correct data in their Vecs.
Each thread allocated a Vec with 3-10 elements (varying by thread ID), verified the sum,
and dropped the Vec. No thread saw corrupted data or allocated the same block as another
thread.

**Confidence**: high (160/160 cycles passed, data integrity verified)

### Q: What is the throughput under contention?
A: **~362K alloc/dealloc pairs per second** (160 operations in 442.8µs). This is for
small allocations (12-40 bytes per Vec buffer) in the 32B and 64B slab classes. The
bitmap CAS contention across 32 threads is modest because:
- Threads target different bitmap words based on allocation order
- CAS retry rate is low for sparse bitmaps (most words have many free bits)
- Dealloc CAS is essentially uncontested (each thread frees its own blocks)

**Confidence**: medium (single measurement, not averaged)

### Q: Does fragmentation remain bounded?
A: **Yes, by design.** Slab allocators have zero external fragmentation within a size
class. Internal fragmentation is bounded by the size class ratio (max 2x for the worst
case of allocating 17 bytes in a 32-byte slab). After all 160 alloc/dealloc cycles, all
memory is returned to the free pools (bitmaps reset to 0).

**Confidence**: high (design property)

## Test Results
| Test | Config | Expected | Result |
|------|--------|----------|--------|
| slab_concurrent_test_kernel | 32 threads × 5 cycles | 160/160 success | **PASSED** (442.8µs) |

## Files Modified
- `crates/std-build-test/src/lib.rs` — MODIFIED: added slab_concurrent_test_kernel
- `crates/gpu-host/std_build_test.ptx` — UPDATED
- `crates/gpu-host/src/main.rs` — MODIFIED: added run_slab_concurrent_test

## Impact
The allocator theme is now complete:
- allocator.1: Design survey → slab+bitmap recommended
- allocator.2: Implementation → 20 single-thread cycles pass
- allocator.3: Concurrent stress test → 32 threads × 5 cycles pass
