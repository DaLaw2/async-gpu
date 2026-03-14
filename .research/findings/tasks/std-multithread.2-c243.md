# std-multithread.2: Implement ThreadLocal using tid-indexed array
**Cycle**: 243 | **Theme**: std-multithread | **Kind**: experiment | **Status**: done

## Summary
Implemented `gpu_threads.rs` — a new thread-local backend for `target_os = "cuda"` in the
patched std. Replaces `no_threads.rs` (which assumes single thread) with per-thread storage
via thread-ID-indexed arrays. All three components implemented: `EagerStorage<T>`,
`LazyStorage<T>`, `LocalPointer`. Compilation verified, PTX contains 405 `%tid` reads.

## Findings
### Q: Can we replace no_threads.rs with tid-indexed arrays for GPU multi-thread?
A: **Yes.** The implementation compiles cleanly and generates correct PTX.

**Key implementation details:**
- `EagerStorage<T>`: Template value + `[MaybeUninit<T>; 1024]` slots + `[bool; 1024]` init flags
- `LazyStorage<T>`: `[MaybeUninit<T>; 1024]` + `[State; 1024]` per-thread state tracking
- `LocalPointer`: `[*mut (); 1024]` array — `*mut ()` is Copy, direct array init works
- Array init for non-Copy types: `MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init()`
- Thread ID: inline PTX `%tid.x`, `%tid.y`, `%tid.z`, `%ntid.x`, `%ntid.y`
- Module routing: `cfg_select!` in `mod.rs` — CUDA branch placed BEFORE the no_threads branch

**Files modified:**
1. `patched-std/library/std/src/sys/thread_local/gpu_threads.rs` — NEW (210 lines)
2. `patched-std/library/std/src/sys/thread_local/mod.rs` — Route CUDA to gpu_threads
3. `crates/gpu-kernel-std/src/lib.rs` — Added `asm_experimental_arch` feature + test kernels

**Confidence**: high (compiles, PTX verified, ready for GPU test)

## Unexpected Discoveries
- `#![feature(asm_experimental_arch)]` needed in gpu-kernel-std for the `get_tid()` helper
  (inline PTX in kernel code). The patched std already has this feature enabled in lib.rs.

## Open Questions
None — implementation is straightforward.

## Impact on Downstream Tasks
- std-multithread.3: Can now test with multi-thread launch
