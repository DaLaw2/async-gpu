# unified-transfer.2: Implement GpuVec<T>

## Status: COMPLETE

## Summary

Implemented `GpuVec<T>` — a high-level zero-copy GPU buffer that wraps
`MappedBuffer<T>` and eliminates manual `cudaMemcpy` for the common case.

## What was built

### `GpuVec<T>` in `crates/core/gpu-host/src/memory.rs`

A safe, ergonomic wrapper around `MappedBuffer<T>` with zero-copy semantics:

- `GpuVec::from_vec(Vec<T>) -> Result<Self>` — copies data into pinned memory
- `GpuVec::zeroed(len) -> Result<Self>` — allocates zeroed pinned buffer
- `GpuVec::dev_ptr() -> u64` — device-visible pointer for kernel arguments
- `GpuVec::as_slice() -> &[T]` — zero-copy host read (no download needed)
- `GpuVec::as_mut_slice() -> &mut [T]` — zero-copy host write
- `GpuVec::len() -> usize` / `is_empty() -> bool`
- `GpuVec::into_vec(self) -> Vec<T>` — copies from pinned to owned Vec
- `TryFrom<Vec<T>>` and `TryFrom<&[T]>` trait implementations

### Re-exports

- `GpuVec` re-exported from `gpu_host` crate root (`pub use memory::GpuVec`)
- Added to key types documentation in `lib.rs`

## Design decisions

1. **`T: Copy` bound** — Required for `ptr::copy_nonoverlapping` in `from_vec()`
   and `into_vec()`. This matches the design doc's `T: Copy + Send + Sync` but
   relaxes the Send/Sync requirement since MappedBuffer already impls them for
   `T: Send` / `T: Sync`.

2. **`TryFrom` instead of `From`** — Allocation can fail, so panicking `From`
   would hide errors. `TryFrom` is the idiomatic Rust approach. Users who want
   panic-on-error can call `.unwrap()`.

3. **Safe `as_slice()`** — MappedBuffer's `as_slice()` is `unsafe` because of
   GPU synchronization concerns. GpuVec makes it safe because:
   - The `&self` borrow prevents Rust-side mutation
   - GPU synchronization is always the caller's responsibility anyway (same as
     reading any buffer after kernel launch)
   - Stale reads are a logic bug, not memory unsafety

4. **No `DeviceBuffer` variant yet** — Per the task scope, this is the foundation.
   `to_device()` / `to_mapped()` are deferred to unified-transfer.3.

## Build verification

- `cargo check -p gpu-host` — passes (0 errors, 0 warnings)
- `cargo clippy -p gpu-host -- -D warnings` — passes (0 warnings)

## Files changed

- `crates/core/gpu-host/src/memory.rs` — added `GpuVec<T>` type
- `crates/core/gpu-host/src/lib.rs` — re-export + doc update
