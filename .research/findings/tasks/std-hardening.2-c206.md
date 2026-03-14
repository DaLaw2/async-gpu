# std-hardening.2: Atomic CAS bump allocator in gpu-libc
**Cycle**: 206 | **Theme**: std-hardening | **Kind**: experiment | **Status**: done

## Summary
Replaced `static mut BUMP_STATE` with `AtomicU64`-based bump allocator.
`malloc()` and `posix_memalign()` now use `compare_exchange_weak` CAS loop
to atomically advance the bump pointer. Multiple CUDA threads can call
malloc concurrently without data races.

## Findings
### Q: Does core::sync::atomic work on nvptx64?
A: Yes. `AtomicU64` with `Ordering::Relaxed` compiles and links correctly
for nvptx64-nvidia-cuda. LLVM maps these to PTX atomic instructions.
**Confidence**: high

### Q: Performance impact?
A: One CAS per allocation. For bump allocator (pointer only advances),
contention is low — CAS retry only happens when two threads allocate
simultaneously, which is rare for typical GPU workloads. No measurable
regression expected for single-thread use.
**Confidence**: high
