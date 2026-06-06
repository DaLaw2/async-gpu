# om-tiered-types synthesis

## Current state
- `tiered_mem.rs` implemented: `GpuRef<'scope, T, Tier>` with `SharedRef`/`GlobalRef` aliases
- Sealed `MemoryTier` trait (`Shared`, `Global`) encodes address space at type level
- Inline PTX asm intrinsics for 7 types x 2 tiers (u8, u32, i32, f32, u64, i64, f64)
- `shared_addr_at()` uses `mov.u64 + add.u64` for raw addrspace(3) pointers
- `BlockScope::alloc_shared()` returns `SharedRef`; `GridScope::alloc_global()` returns `GlobalRef`
- Non-breaking: existing `alloc() -> &'scope mut [T]` preserved alongside new methods
- `as_generic_slice()` escape hatch for incremental migration
- All CI lint checks pass; PTX kernel builds compile successfully for nvptx64

## Key design decisions
1. Raw shared-space pointer (no `cvta.shared`) stored in `GpuRef::ptr`
2. `SharedRef` is `!Send` (per-block); `GlobalRef` is `Send + Sync` (grid-wide)
3. No `Deref` — forces `.read(i)` / `.write(i, val)` for address-space-aware access
4. `write(&self, ...)` not `&mut self` — `GpuRef` is `Copy`, acts like pointer

## Next steps
- Microbenchmark: `ld.shared` vs generic `ld` latency on SM75
- `DisjointSlice<'scope, T, Tier>` integration
- GEMM kernel adaptation using `SharedRef` for tile loading
