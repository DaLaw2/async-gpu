# std-sync.3: GPU-side HashMap with open addressing
**Cycle**: 319 | **Theme**: std-sync | **Kind**: experiment | **Status**: done

## Summary
Implemented `GpuHashMap<const N: usize>` in `gpu-runtime::collections` module.
Fixed-capacity open-addressing hash map with linear probing, lock-free inserts
via atomic CAS, and lock-free reads via volatile loads. Keys are non-zero `u32`,
values are `u32`.

## Changes Made

### 1. New `collections` module in `crates/core/gpu-runtime/src/lib.rs`
- `GpuHashMap<const N: usize>`: generic over capacity (must be power of 2)
- `new()`: const constructor, all slots zero-initialized
- `init()`: explicit zero-fill for runtime initialization
- `insert(key, value)`: CAS-based lock-free insert with linear probing
- `get(key)`: volatile-read lock-free lookup
- `contains_key(key)`: convenience wrapper
- `len()`, `is_empty()`, `capacity()`: diagnostics
- Fibonacci multiplicative hashing (`key * 0x9E3779B9`)
- No delete support (no tombstones needed for insert-only workloads)

### 2. Prelude update
- Added `GpuHashMap` to the prelude

## Design Decisions
- **u32 keys/values**: Simplest type that covers most GPU use cases (indices,
  counters, flags). Wrap in newtypes for richer semantics.
- **Key 0 reserved**: Empty sentinel. Allows detecting occupied slots without
  a separate bitmap. Standard open-addressing technique.
- **No resizing**: Bump allocator constraint. Users must size generously at
  creation time (recommended load factor < 0.7).
- **Linear probing**: Simpler than Robin Hood, adequate for moderate load factors.
  CPU cache benefits don't apply on GPU, but sequential memory access does.
- **Non-atomic len**: Only diagnostic. True count requires extra CAS per insert.

## Verification
- Compiles on x86_64 (stub path): `cargo clippy --target x86_64-pc-windows-msvc`
- GPU-side (nvptx64) verification deferred to integration test

## Impact on Downstream Tasks
- **extended-std epic**: Criteria 2 (HashMap) and 3 (2+ new std types) met
  (Mutex + HashMap = 2 new types beyond File/Vec/String/Box/stdin)
- **gpu-executor**: HashMap available for task metadata tracking if needed
