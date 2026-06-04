# sc-channel.4: Unified GpuChannel<T> enum — auto-selects transport

**Status**: DONE
**Kind**: experiment
**Theme**: sc-channel (structured concurrency channels)

## Summary

Implemented a unified channel API in `gpu_runtime::unified_channel` that auto-selects
block-scoped (shared memory, CTA-scope atomics) or grid-scoped (global memory,
system-scope atomics) transport depending on how the channel is created.

Users call `scope.oneshot::<T>()` or `gscope.oneshot::<T>()` and get back the same
`ScopedOneshotSender`/`ScopedOneshotReceiver` enum type — the transport is selected
at construction time.

## Files Created/Modified

- **NEW** `crates/core/gpu-runtime/src/unified_channel.rs` — Unified channel module
- **MOD** `crates/core/gpu-runtime/src/lib.rs` — Module registration
- **MOD** `crates/core/gpu-runtime/src/prelude.rs` — Re-exports for unified types
- **MOD** `crates/core/gpu-runtime/src/scope.rs` — Added `alloc_raw_bytes()` to both
  `BlockScope` and `GridScope` for allocating non-Copy types (channel storage contains
  `UnsafeCell`)

## Design Decisions

### Enum, not trait object
All unified types are enums dispatching to Block or Grid variants. No vtable, no
heap allocation. On GPU, the discriminant is set at construction and the branch
predictor eliminates overhead.

### Unified types implemented
1. **`ScopedOneshotSender<'scope, T>`** / **`ScopedOneshotReceiver<'scope, T>`** —
   enum over `BlockOneshotSender` (CTA atomics) and `GridOneshotSender` (sys atomics)
2. **`ScopedMpscSender<'scope, T, N>`** / **`ScopedMpscReceiver<'scope, T, N>`** —
   enum over `BlockMpscSender` and `GridMpscSender`
3. **`ScopedOneshotClosed`** — unified error type
4. **`ScopedMpscSendError<T>`** — unified error type (Closed/Full)

### Scope integration
- `BlockScope::oneshot<T>()` and `BlockScope::mpsc<T, N>()` — allocates from shared
  memory, returns block-variant unified types
- `GridScope::oneshot<T>()` and `GridScope::mpsc<T, N>()` — allocates from the global
  memory pool, returns grid-variant unified types

### alloc_raw_bytes() — new scope method
Channel storage types (`BlockOneshotSlot`, `OneshotSlot`, `BlockMpscChannel`,
`MpscChannel`) all contain `UnsafeCell`, which prevents them from implementing `Copy`.
The existing `alloc<T: Copy>()` / `alloc_uninit<T: Copy>()` methods cannot allocate
these types. Added `alloc_raw_bytes(size, align) -> *mut u8` to both `BlockScope` and
`GridScope` for this purpose.

### Clone vs Copy for MPSC senders
- `BlockMpscSender` is `Copy` (raw pointer, no overhead)
- `MpscSender` (global) is `Clone` only (not Copy)
- Therefore `ScopedMpscSender` implements `Clone` but not `Copy`
  (Block variant copies, Grid variant clones)

### shfl.sync excluded
Per sc-channel.1 findings, shuffle is a broadcast primitive, not a channel. The
unified type does not include a warp-level variant.

## Grid-scoped receiver approach
The grid-scoped oneshot receiver (`GridOneshotReceiver`) holds a raw `*const OneshotSlot<T>`
rather than wrapping the existing `OneshotReceiver` (which is a `Future`). This provides
`try_recv()` and `recv_spin()` methods matching the block-scoped API, without requiring
a `Future`-based polling model.

## Verification
```
bash scripts/ci-lint.sh
=> All CI lint checks passed!
```
- fmt: OK (all crates including gpu-runtime)
- clippy: OK (no warnings)
- doc: OK
- PTX kernel builds: OK (all nvptx64 targets compile)
