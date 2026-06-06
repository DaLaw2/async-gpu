# om-tiered-types synthesis

## Current state
- `GpuRef<'scope, T, Tier>` design complete — single generic + `SharedRef`/`GlobalRef` aliases
- Sealed `MemoryTier` trait (`Shared`, `Global`) encodes address space at type level
- Inline PTX asm accessors (`ld.shared`/`st.shared`, `ld.global`/`st.global`) bypass `cvta.shared`
- `TieredAccess<T>` trait enables tier-generic code over both address spaces
- `DisjointSlice` gains `Tier` parameter; `get_mut` returns `GpuRef` instead of `&mut [T]`
- No `Deref` impl — forces explicit `.read(i)`/`.write(i,v)` to prevent silent generic fallback
- `as_generic_slice()` escape hatch for migration

## Key design decisions
1. Raw shared-space pointer (no `cvta.shared`) stored in `GpuRef::ptr`
2. `SharedRef` is `!Send` (per-block); `GlobalRef` is `Send + Sync` (grid-wide)
3. Breaking API: `alloc()` returns `GpuRef` not `&'scope mut [T]`
4. Compile errors for shared/global confusion, scope escape, cross-block send

## Blocking question
Verify `ld.shared.u32 %r, [%rd]` works with register-indirect addressing in PTX.
If not, inline-asm accessor approach needs redesign. Benchmark generic vs shared
load latency to confirm optimization justifies complexity.

## Next steps
- PTX compilation test: register-indirect `ld.shared` feasibility
- Microbenchmark: `ld.shared` vs generic `ld` on SM75/SM86
- Implementation: `GpuRef`, `MemoryTier`, intrinsics module, alloc migration
