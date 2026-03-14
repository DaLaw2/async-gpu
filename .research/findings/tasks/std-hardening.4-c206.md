# std-hardening.4: Per-thread errno in gpu-libc
**Cycle**: 206 | **Theme**: std-hardening | **Kind**: experiment | **Status**: done

## Summary
Replaced single `static mut GPU_ERRNO: c_int` with a thread-ID indexed
array `ERRNO_ARRAY: [c_int; 1024]`. Each CUDA thread gets its own errno
slot via `thread_id_in_block()` (inline PTX `%tid.x/y/z` + `%ntid.x/y`).

Threads beyond 1024 (one full block) gracefully fall back to slot 0.
This covers the vast majority of GPU launch configs.

## Findings
### Q: How to get thread ID on nvptx64?
A: Inline PTX: `mov.u32 {}, %tid.x;` etc. Flat index =
`tid.x + tid.y * ntid.x + tid.z * ntid.x * ntid.y`. Compiles fine.
**Confidence**: high

### Q: Memory overhead?
A: 1024 × 4 bytes = 4 KB in `.global` static. Negligible.
**Confidence**: high
