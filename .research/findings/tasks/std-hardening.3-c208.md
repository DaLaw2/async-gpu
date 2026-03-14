# std-hardening.3: Multi-thread allocation test
**Cycle**: 208 | **Theme**: std-hardening | **Kind**: experiment | **Status**: done

## Summary
Added `test_multithread_malloc` kernel to gpu-kernel/src/basic.rs that
launches 32 threads, each calling gpu_libc::malloc(64). Added host-side
test in tests_std.rs that verifies all 32 pointers are non-null and
non-overlapping. Also added gpu-libc as dependency of gpu-kernel.

## Findings
### Q: Can the test verify thread-safety without running on GPU?
A: The code compiles and the kernel builds. Verification requires running
on actual GPU hardware (ONLY_TEST=mt_malloc). The atomic CAS allocator
logic is straightforward — each thread atomically advances the bump pointer,
so concurrent allocations cannot overlap.
**Confidence**: high (code correctness), medium (awaiting GPU verification)
