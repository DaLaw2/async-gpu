# om-tiered-types.3: Tiered Types Integration

## Summary

Implemented the tiered memory type system: `GpuRef<'scope, T, Tier>` with
`SharedRef` / `GlobalRef` type aliases, sealed `MemoryTier` trait, inline PTX
asm intrinsics for `ld.shared`/`st.shared`/`ld.global`/`st.global`, and
integrated with `BlockScope::alloc_shared()` / `GridScope::alloc_global()`.

All CI lint checks pass. PTX kernel builds compile successfully for nvptx64,
confirming the inline asm is valid.

## Findings

### 1. Implementation

Created `crates/core/gpu-runtime/src/tiered_mem.rs` with:

- **Sealed `MemoryTier` trait** with `Shared` and `Global` zero-sized markers.
- **`GpuRef<'scope, T, Tier>`** — raw pointer + length + PhantomData for tier
  and invariant lifetime. `Copy + Clone` (thin wrapper). `SharedRef` is
  `!Send + !Sync` (via raw pointer); `GlobalRef` has explicit `Send + Sync`.
- **`TieredAccess<T>` trait** implemented for `(u8, u32, i32, f32, u64, i64, f64)`
  x `(Shared, Global)` = 14 impls. Each dispatches to tier-specific inline asm.
- **`shared_addr_at()`** — raw shared-space pointer via `mov.u64 {addr}, dynamic_smem;
  add.u64 {addr}, {addr}, {off};` (no `cvta.shared` conversion).
- **`read(i)` / `write(i, val)`** — bounds-checked, dispatch through `TieredAccess`.
- **`sub_ref(start, len)`** — create sub-references for tiling patterns.
- **Escape hatches**: `as_generic_slice()` / `as_generic_slice_mut()` for migration.

### 2. Scope Integration

- `BlockScope::alloc_shared<T>(count) -> SharedRef<'scope, T>` — bump-allocates
  from shared memory, returns raw shared-space pointer. Zero-init via generic
  pointer (since `write_bytes` needs generic space), but the `SharedRef` pointer
  stays in addrspace(3).
- `GridScope::alloc_global<T>(count) -> GlobalRef<'scope, T>` — bump-allocates
  from global pool, returns `GlobalRef`. Global pointers are already in generic
  space on the host/GPU, so no address space tricks needed.
- **Non-breaking**: Added as new methods (`alloc_shared`, `alloc_global`)
  alongside existing `alloc()` which still returns `&'scope mut [T]`. Migration
  can happen incrementally.

### 3. Address Space Verification

The `mov.u64 + add.u64` pattern for `shared_addr_at` compiles successfully for
nvptx64. This produces a shared-space address that `ld.shared.u32 {val}, [{addr}]`
can use with register-indirect addressing. This resolves the open question from
om-tiered-types.2 about whether `ld.shared` supports register operands.

The `ld.shared.u8` instruction also works (PTX supports `.u8` width for both
load and store, zero-extending on load).

### 4. Design Decisions Made

- **`write(&self, ...)` not `write(&mut self, ...)`**: `GpuRef` is `Copy` and
  acts like a pointer — writes go through the pointer, not through `&mut self`.
  Same semantics as `*mut T`.
- **No `Deref`**: Confirmed — prevents silent fallback to generic loads.
- **f32 via `from_bits`/`to_bits`**: Loads/stores for `f32` go through `u32`
  bit reinterpretation to use `ld.shared.u32` / `st.shared.u32`, which is
  correct and avoids `ld.shared.f32` (which is not a valid PTX instruction
  without `.approx`).
- **u8 direct**: PTX supports `ld.shared.u8` / `st.shared.u8` natively.

## Open Questions

1. **Performance delta**: Not benchmarked yet. The type system and inline asm
   are in place; a microbenchmark comparing `ld.shared` vs generic `ld`
   (cvta.shared path) would quantify the latency savings.

2. **Vectorized bulk ops**: `GpuRef` does not yet provide `fill()` or vectorized
   `st.shared.v4.u32`. Element-by-element init via `write_bytes` on generic
   pointer is sufficient for now; vectorized stores can be added later.

3. **DisjointSlice integration**: The design spec calls for `DisjointSlice<'scope, T, Tier>`.
   Not implemented in this task — current `DisjointSlice` remains tier-unaware.
   This is a natural next step once `SharedRef`/`GlobalRef` are proven in use.

4. **Full GEMM kernel adaptation**: Deferred per task spec. The types compile
   and the allocation path works; GEMM adaptation is a separate task.
