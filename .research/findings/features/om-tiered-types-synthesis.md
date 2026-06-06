# om-tiered-types synthesis

## Current state
- BlockScope::alloc → shared mem via `cvta.shared` → generic addrspace → `ld.b32`/`st.b32`
- GridScope::alloc → global mem pool → `ld.global`/`st.global`
- Both return `&'scope mut [T]` — no type-level memory tier distinction
- Zero `ld.shared`/`st.shared` in PTX output; all shared access via generic space
- `for<'scope>` HRTB + PhantomData invariance prevents reference escape (sound)

## Key insight
The `cvta.shared.u64` conversion erases the address space. Phantom-typed wrappers
(`SharedRef<'block, T>` / `GlobalRef<'grid, T>`) can restore type-level tracking,
but emitting `ld.shared`/`st.shared` requires inline asm since Rust/LLVM doesn't
expose addrspace qualifiers on pointer types.

## Design direction
1. Wrapper types with inline-asm accessors (skip `cvta.shared`, use raw shared addr)
2. BlockScope::alloc returns `SharedRef`, GridScope::alloc returns `GlobalRef`
3. DisjointSlice becomes tier-generic or gets shared/global variants
4. Register promotion: enforce via `Copy + !reference` marker trait for inner-loop values

## Risk
Inline asm for every shared load/store adds verbosity. Benchmark generic vs shared
latency on SM75 first — if hardware resolves generic→shared fast enough, the
address-space-specific instructions may not justify the complexity.
