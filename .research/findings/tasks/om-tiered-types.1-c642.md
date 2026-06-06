# om-tiered-types.1: BlockScope/GridScope Alloc Audit & LLVM Addrspace Semantics

## Summary

Audited BlockScope and GridScope allocation paths, PTX output, and LLVM address
space usage. Found that **all shared memory access goes through generic address
space** (addrspace(0)) after `cvta.shared.u64` conversion, losing the
opportunity for `ld.shared`/`st.shared` instructions. The `for<'scope>` HRTB
pattern successfully prevents reference escape, giving us a solid foundation for
phantom-typed memory tier wrappers.

## Findings

### 1. BlockScope allocation flow

- `BlockScope::alloc<T>(count)` calls `ALLOCATOR.alloc_raw(size, align)` which
  bumps a watermark offset, then calls `crate::block::shared_mem_at::<T>(offset)`.
- `shared_mem_at<T>(offset)` calls `shared_mem_ptr().add(offset) as *mut T`.
- `shared_mem_ptr()` uses inline asm: `cvta.shared.u64 {out}, dynamic_smem;`
- Returns `&'scope mut [T]` — a plain Rust mutable slice.
- The `ALLOCATOR` struct itself lives in **global memory** (Rust `static`),
  accessed via `st.global.b32`/`ld.global.b32`. Only the offsets it tracks point
  into shared memory.

### 2. GridScope allocation flow

- `GridScope::alloc<T>(count)` does a bump allocation on `pool_offset` (an
  `UnsafeCell<u32>` stored inline in the GridScope struct), then computes
  `pool_base.add(aligned) as *mut T`.
- `pool_base` is a `*mut u8` passed from the host — global device memory.
- Returns `&'scope mut [T]` — same type as BlockScope despite different memory.
- No address-space annotation at the type level.

### 3. LLVM address spaces in PTX output

**Critical finding: `cvta.shared.u64` converts a shared-address-space pointer to
a generic-address-space pointer.** After this conversion, all subsequent
load/store instructions use the generic address space:

```ptx
cvta.shared.u64 %rd58, dynamic_smem;    // shared → generic conversion
st.volatile.b32 [%rd58], 0;             // generic st (NOT st.shared)
...
cvta.shared.u64 %rd62, dynamic_smem;
add.s64 %rd5, %rd62, %rd4;
st.b8 [%rd5], 0;                        // generic st (NOT st.shared)
...
ld.b32 %r5, [%rd6];                     // generic ld (NOT ld.shared)
```

Zero `ld.shared` / `st.shared` instructions exist in the entire PTX output.
All shared memory access uses generic loads/stores (`ld.b32`, `st.b32`, etc.).

For global memory (GridScope pool, ALLOCATOR static), the PTX uses explicit
`ld.global` / `st.global` or `st.release.sys.global` / `ld.acquire.sys.global`.

**Address space inventory from PTX:**
- `addrspace(0)` / generic: All shared memory data access after `cvta.shared`
- `addrspace(1)` / `.global`: ALLOCATOR static, WARP_STATUS, SCRATCH, pool_base
  data, kernel params (`cvta.to.global.u64`)
- `addrspace(3)` / `.shared`: Only in declaration: `.extern .shared .align 4
  .b8 dynamic_smem[];` — never used for load/store
- `addrspace(5)` / `.local`: Stack spills (`__local_depot`, `cvta.local.u64`)

### 4. Performance implication

On SM75 (Turing), `ld.shared` / `st.shared` have ~2 cycle latency vs generic
`ld.b32` / `st.b32` which must go through address translation (~4-5 cycles for
shared, ~100 cycles if the pointer happens to land in global memory). The
hardware CAN resolve generic pointers to shared memory, but there is a latency
penalty vs direct shared-space instructions. For high-traffic inner loops, this
matters.

### 5. `for<'scope>` HRTB escape prevention

The pattern works correctly:

```rust
pub fn block_scope<F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
```

- `PhantomData<&'scope mut &'scope ()>` in `BlockScope` enforces lifetime
  invariance (prevents covariant escape).
- References returned by `alloc()` carry `'scope` and cannot be stored in any
  structure that outlives the closure.
- This is the same pattern as Rayon's `scope()` and crossbeam's scoped threads.
- **Already prevents cross-scope memory access at the Rust level.**

### 6. Return types: no memory-tier distinction

Both `BlockScope::alloc()` and `GridScope::alloc()` return `&'scope mut [T]`.
There is no type-level distinction between a reference to shared memory and one
to global memory. The compiler, LLVM, and PTX assembler all see the same generic
pointer type.

### 7. What phantom-typed wrappers would need

To create `SharedRef<'block, T>` / `GlobalRef<'grid, T>`:

1. **Wrapper types** carrying `PhantomData` for address space:
   ```rust
   pub struct SharedRef<'scope, T> { ptr: *mut T, _marker: ... }
   pub struct GlobalRef<'scope, T> { ptr: *mut T, _marker: ... }
   ```

2. **Address-space-aware intrinsics** for load/store:
   ```rust
   // For SharedRef: emit ld.shared / st.shared directly
   unsafe fn shared_load<T>(ptr: *const T) -> T;
   unsafe fn shared_store<T>(ptr: *mut T, val: T);
   ```

3. **Avoiding `cvta.shared`**: Instead of converting shared→generic, keep the
   shared-space address (`dynamic_smem + offset`) and use `ld.shared` /
   `st.shared` with that address directly. This requires inline asm that
   operates in `addrspace(3)`.

4. **LLVM constraint**: Rust's LLVM backend for nvptx64 does not expose address
   space qualifiers on pointer types. All `*mut T` pointers are `addrspace(0)`.
   To get `addrspace(3)` loads/stores, we must use inline asm (PTX) or modify
   the compiler's LLVM IR generation.

## Open Questions

1. **Can we avoid `cvta.shared` entirely?** The shared memory symbol
   `dynamic_smem` is in `addrspace(3)`. If we keep the raw shared-space
   address (before `cvta.shared` converts it to generic), we can use
   `ld.shared` / `st.shared` directly. But Rust's pointer types don't carry
   address space info, so this requires inline asm for every access.

2. **Performance delta**: What is the actual cycle cost of generic vs shared
   loads on SM75? Need microbenchmark: `ld.shared` vs `ld.b32` on known-shared
   pointer. The hardware may optimize the generic case well enough that the
   difference is negligible for many workloads.

3. **Register promotion feasibility**: For `move` semantics in inner loops,
   values should live in registers. Rust already promotes scalar locals to
   registers. The question is whether the type system can ENFORCE this (i.e.,
   prevent accidental spills to shared/global memory for types that should stay
   in registers).

4. **Cross-scope prevention gap**: The `for<'scope>` pattern prevents escape,
   but doesn't prevent passing a `&'scope mut [T]` from BlockScope into a
   GridScope closure (or vice versa). Phantom-typed wrappers could close this
   gap by making `SharedRef` and `GlobalRef` incompatible types.

5. **Compatibility with DisjointSlice**: `DisjointSlice` currently wraps
   `*mut T`. It would need to become generic over the memory tier
   (`DisjointSlice<'scope, T, Tier>`) or have separate shared/global variants.
