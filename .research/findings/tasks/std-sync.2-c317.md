# std-sync.2: GPU Mutex<T> via atomic CAS
**Cycle**: 317 | **Theme**: std-sync | **Kind**: experiment | **Status**: done

## Summary
Implemented `Mutex<T>` and `MutexGuard<T>` in `gpu-runtime::sync` module. Uses
`sys_cas_u32` spin-lock for acquisition, `sys_store_release_u32` for unlock, and
`sys_spin_load_acquire_u32` (with nanosleep yield) for the spin loop. Added to
the `prelude` for easy access.

## Changes Made

### 1. New `sync` module in `crates/core/gpu-runtime/src/lib.rs`
- `Mutex<T>`: `#[repr(C)]` struct with `UnsafeCell<u32>` lock word + `UnsafeCell<T>` data
- `MutexGuard<'a, T>`: RAII guard implementing `Deref`, `DerefMut`, `Drop`
- `Mutex::new(value)`: const constructor
- `Mutex::lock()`: spin-lock with CAS, panics after MUTEX_MAX_SPIN (10M iterations)
- `Mutex::try_lock()`: single CAS attempt, returns `Option<MutexGuard>`
- `Mutex::unlock()`: release-store UNLOCKED (called by guard Drop)
- `Send + Sync` impl for `Mutex<T: Send>`

### 2. Prelude update
- Added `pub use crate::sync::{Mutex, MutexGuard};` to the prelude

## Design Decisions
- **Spin-lock, not hostcall-based sleep**: Low latency for short critical sections,
  proven pattern (project uses spin-poll extensively).
- **No poisoning**: GPU panics trap the thread; no panic unwinding to detect.
- **System-scope atomics**: Uses `.sys` scope CAS for correctness across blocks.
  Could use `.gpu` scope for GPU-only data (future optimization).
- **MUTEX_MAX_SPIN = 10M**: Matches existing `GPU_MAX_SPIN` for consistency.
  Timeout traps the thread to aid deadlock debugging.
- **`#[repr(C)]`**: Ensures predictable layout for initialization from host side.

## Verification
- Compiles on x86_64 (stub path): `cargo check --target x86_64-pc-windows-msvc`
- GPU-side (nvptx64) verification requires patched toolchain — deferred to integration test

## Impact on Downstream Tasks
- **std-sync.3 (HashMap)**: Can use Mutex for bucket-level locking if needed
- **executor-impl.2 (WorkQueue)**: Mutex available as alternative to lock-free CAS
  (though lock-free is preferred for the work queue)
- **extended-std epic**: Criterion 1 ("GPU Mutex<T> via atomic CAS") met
