# om-tiered-types.2: SharedRef / GlobalRef Type Design

## Summary

Concrete type design for `SharedRef<'scope, T>` and `GlobalRef<'scope, T>` —
phantom-typed wrappers that encode memory address space at the type level. The
design uses inline PTX asm for address-space-specific loads/stores, preserves
the existing `for<'scope>` HRTB escape prevention, and makes shared-vs-global
a compile-time distinction that the user cannot accidentally cross.

## Findings / Design

### 1. Type Structure

```rust
/// Marker trait sealed to crate — shared vs global address space.
pub trait MemoryTier: sealed::Sealed {
    /// Human-readable name for error messages.
    const NAME: &'static str;
}

/// Shared memory (addrspace 3) — per-block, ~2-cycle latency.
pub struct Shared;
impl MemoryTier for Shared { const NAME: &'static str = "shared"; }

/// Global memory (addrspace 1) — grid-wide, ~100-cycle latency.
pub struct Global;
impl MemoryTier for Global { const NAME: &'static str = "global"; }

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Shared {}
    impl Sealed for super::Global {}
}
```

The reference types are unified under a single generic:

```rust
/// A lifetime-bounded, address-space-aware GPU memory reference.
///
/// `Tier` is `Shared` or `Global` — a zero-sized phantom that encodes
/// which PTX address space the pointer lives in.
///
/// Invariant: the inner `ptr` is a RAW address-space pointer (NOT converted
/// via `cvta.shared`). For Shared, this is the `dynamic_smem + offset` value
/// in addrspace(3). For Global, this is the global pointer from the host.
pub struct GpuRef<'scope, T: Copy, Tier: MemoryTier> {
    ptr: *mut T,
    len: usize,
    _tier: PhantomData<Tier>,
    _scope: PhantomData<&'scope mut &'scope ()>,  // invariant lifetime
}

/// Shared memory reference — bound to a BlockScope.
pub type SharedRef<'scope, T> = GpuRef<'scope, T, Shared>;

/// Global memory reference — bound to a GridScope.
pub type GlobalRef<'scope, T> = GpuRef<'scope, T, Global>;
```

**Key design decisions:**

- **Single generic type + type aliases**: Avoids duplicating methods. Trait
  impls can be specialized per `Tier` where needed (e.g., accessor intrinsics).
- **Raw address-space pointer**: The `ptr` field stores the address *before*
  `cvta.shared` conversion. For `SharedRef`, this is the shared-space address
  (`dynamic_smem + offset` without the `cvta.shared.u64` conversion). For
  `GlobalRef`, this is the plain global pointer. This is what enables
  `ld.shared` / `st.shared` emission.
- **`T: Copy` bound**: Same as existing `alloc()` — no Drop types in GPU memory.
- **Invariant `'scope`**: Same `PhantomData<&'scope mut &'scope ()>` pattern
  as `BlockScope` / `GridScope` — prevents covariant escape.

**Properties:**

- `!Send` for `SharedRef` (shared memory is per-block; sending to another
  block is meaningless). Achieved via `PhantomData<*const ()>` or explicit
  negative impl.
- `Send + Sync` for `GlobalRef` (global memory is grid-wide; safe to share
  across blocks, gated by system-scope atomics at the access layer).
- `Copy + Clone` for both (thin wrapper; same rationale as `DisjointSlice`).

### 2. Accessor API: Inline PTX Intrinsics

The core optimization: bypass `cvta.shared.u64` and use address-space-specific
load/store instructions directly.

```rust
// --- Low-level intrinsics (private, in a `mem_intrinsics` module) ---

/// Load a u32 from shared memory (addrspace 3).
/// `addr` is the raw shared-space address (NOT generic).
#[inline(always)]
unsafe fn shared_load_u32(addr: *const u32) -> u32 {
    let val: u32;
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "ld.shared.u32 {val}, [{addr}];",
        val = out(reg32) val,
        addr = in(reg64) addr as u64,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    { val = core::ptr::read(addr); }
    val
}

/// Store a u32 to shared memory (addrspace 3).
#[inline(always)]
unsafe fn shared_store_u32(addr: *mut u32, val: u32) {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.shared.u32 [{addr}], {val};",
        addr = in(reg64) addr as u64,
        val = in(reg32) val,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    core::ptr::write(addr, val);
}

// Analogous: shared_load_f32, shared_store_f32,
//            global_load_u32, global_store_u32, etc.
// Wider types (u64, f64) use b64 variants.
```

**PTX addressing note**: `ld.shared` and `st.shared` accept a generic register
as the address operand — the hardware interprets it as a shared-space offset.
The trick is to NOT convert via `cvta.shared.u64` (which produces a generic-
space pointer). Instead, we compute `dynamic_smem + byte_offset` directly
in addrspace(3) using PTX:

```rust
/// Get the raw shared-space address at a byte offset (no cvta.shared).
#[inline(always)]
unsafe fn shared_addr_at(offset: usize) -> *mut u8 {
    let addr: u64;
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        // Add offset to the shared memory symbol directly.
        // dynamic_smem is in addrspace(3); adding offset stays in addrspace(3).
        "mov.u64 {addr}, dynamic_smem;
         add.u64 {addr}, {addr}, {off};",
        addr = out(reg64) addr,
        off = in(reg64) offset as u64,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    { addr = 0; }
    addr as *mut u8
}
```

**User-facing API on `GpuRef`:**

```rust
impl<'scope, T: Copy> GpuRef<'scope, T, Shared> {
    /// Read element at index `i` via `ld.shared`.
    #[inline(always)]
    pub fn read(&self, i: usize) -> T {
        assert!(i < self.len, "SharedRef::read: index out of bounds");
        unsafe { shared_load::<T>(self.ptr.add(i)) }
    }

    /// Write element at index `i` via `st.shared`.
    #[inline(always)]
    pub fn write(&mut self, i: usize, val: T) {
        assert!(i < self.len, "SharedRef::write: index out of bounds");
        unsafe { shared_store::<T>(self.ptr.add(i), val); }
    }

    /// Returns a raw shared-space pointer (for advanced use / PTX interop).
    pub fn as_shared_ptr(&self) -> *const T { self.ptr }
    pub fn as_shared_mut_ptr(&mut self) -> *mut T { self.ptr }

    pub fn len(&self) -> usize { self.len }
}

impl<'scope, T: Copy> GpuRef<'scope, T, Global> {
    /// Read element at index `i` via `ld.global`.
    #[inline(always)]
    pub fn read(&self, i: usize) -> T {
        assert!(i < self.len, "GlobalRef::read: index out of bounds");
        unsafe { global_load::<T>(self.ptr.add(i)) }
    }

    /// Write element at index `i` via `st.global`.
    #[inline(always)]
    pub fn write(&mut self, i: usize, val: T) {
        assert!(i < self.len, "GlobalRef::write: index out of bounds");
        unsafe { global_store::<T>(self.ptr.add(i), val); }
    }

    pub fn as_global_ptr(&self) -> *const T { self.ptr }
    pub fn as_global_mut_ptr(&mut self) -> *mut T { self.ptr }

    pub fn len(&self) -> usize { self.len }
}
```

**Why NOT `Deref<Target = [T]>`**: Implementing `Deref` would return a plain
`&[T]` / `&mut [T]`, which are generic-space pointers — the very thing we are
trying to avoid. Users would silently fall back to generic loads. Instead, the
explicit `read(i)` / `write(i, val)` API forces address-space-aware access.

**Indexing sugar**: We can implement `Index<usize>` for immutable access
(returns `T` by value for `Copy` types, not `&T`). This requires a newtype
wrapper since `Index` must return a reference. Alternative: provide
`fn get(&self, i: usize) -> T` and skip `Index` trait entirely. The `read`/
`write` pair is clearer for GPU memory semantics.

### 3. BlockScope::alloc Changes

Current signature:
```rust
pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T]
```

New signature:
```rust
pub fn alloc<T: Copy>(&self, count: usize) -> SharedRef<'scope, T>
```

Implementation change in `BlockScope::alloc`:
```rust
pub fn alloc<T: Copy>(&self, count: usize) -> SharedRef<'scope, T> {
    let size = core::mem::size_of::<T>() * count;
    let align = core::mem::align_of::<T>();

    let offset = unsafe { &mut *ALLOCATOR.as_ptr() }
        .alloc_raw(size, align)
        .expect("BlockScope::alloc: shared memory exhausted");

    // Use shared_addr_at instead of shared_mem_at to get a raw
    // shared-space address (no cvta.shared conversion).
    let ptr = unsafe { shared_addr_at(offset as usize) as *mut T };

    // Zero-initialize via st.shared
    for i in 0..count {
        unsafe { shared_store(ptr.add(i), T::zeroed()); }
    }

    SharedRef {
        ptr,
        len: count,
        _tier: PhantomData,
        _scope: PhantomData,
    }
}
```

`GridScope::alloc` changes analogously:
```rust
pub fn alloc<T: Copy>(&self, count: usize) -> GlobalRef<'scope, T>
```

**Migration path**: This is a breaking API change. Existing code using
`scope.alloc::<f32>(n)` as a `&mut [T]` must switch to `.read(i)` / `.write(i, v)`.
To ease migration, provide:

```rust
impl<'scope, T: Copy, Tier: MemoryTier> GpuRef<'scope, T, Tier> {
    /// Escape hatch: convert to a plain slice (generic address space).
    /// This loses the address-space optimization.
    pub unsafe fn as_generic_slice(&self) -> &'scope [T] { ... }
    pub unsafe fn as_generic_slice_mut(&mut self) -> &'scope mut [T] { ... }
}
```

### 4. MemoryTier Trait

The sealed `MemoryTier` trait serves three purposes:

1. **Type-level discrimination**: `SharedRef` and `GlobalRef` are distinct
   types. Passing a `SharedRef` where a `GlobalRef` is expected is a compile
   error.

2. **Generic code over tiers**: Functions that work with any memory tier:
   ```rust
   fn process<'s, Tier: MemoryTier>(data: &GpuRef<'s, f32, Tier>) -> f32 {
       // Reads via the tier-appropriate intrinsic
       data.read(0)
   }
   ```
   This requires a `GpuRead` / `GpuWrite` trait (see below).

3. **Error message clarity**: `Tier::NAME` gives "shared" or "global" in
   panic messages.

**Tier-generic access trait** (needed for generic code over tiers):

```rust
/// Trait for tier-specific load/store. Implemented per (T, Tier) pair.
/// Sealed — users cannot add new tiers.
pub trait TieredAccess<T: Copy>: MemoryTier {
    unsafe fn load(ptr: *const T) -> T;
    unsafe fn store(ptr: *mut T, val: T);
}

impl TieredAccess<u32> for Shared {
    unsafe fn load(ptr: *const u32) -> u32 { shared_load_u32(ptr) }
    unsafe fn store(ptr: *mut u32, val: u32) { shared_store_u32(ptr, val) }
}

impl TieredAccess<u32> for Global {
    unsafe fn load(ptr: *const u32) -> u32 { global_load_u32(ptr) }
    unsafe fn store(ptr: *mut u32, val: u32) { global_store_u32(ptr, val) }
}
// ... repeat for f32, u64, f64, u8, etc.
```

Then `GpuRef::read` becomes:
```rust
impl<'scope, T: Copy, Tier: TieredAccess<T>> GpuRef<'scope, T, Tier> {
    pub fn read(&self, i: usize) -> T {
        assert!(i < self.len);
        unsafe { Tier::load(self.ptr.add(i)) }
    }
}
```

### 5. DisjointSlice Interaction

Current `DisjointSlice` wraps a raw `*mut T`. Two paths forward:

**Option A: Tier-generic DisjointSlice** (recommended):
```rust
pub struct DisjointSlice<'scope, T: Copy, Tier: MemoryTier> {
    ptr: *mut T,       // raw address-space pointer (shared or global)
    len: usize,
    _tier: PhantomData<Tier>,
    _scope: PhantomData<&'scope mut [T]>,
}
```

`get_mut` returns a `GpuRef<'scope, T, Tier>` instead of `&mut [T]`:
```rust
pub fn get_mut(&self, idx: &WarpIndex<'_>) -> GpuRef<'scope, T, Tier> {
    // ... partition computation same as before ...
    GpuRef { ptr: self.ptr.add(start), len: count, ... }
}
```

`BlockScope::disjoint_slice` produces `DisjointSlice<'scope, T, Shared>`,
`GridScope` would produce `DisjointSlice<'scope, T, Global>`.

**Option B: Separate types** — `SharedDisjointSlice`, `GlobalDisjointSlice`.
More verbose, no real benefit over Option A.

### 6. Compile-Time Error Design

The type system should catch these misuse patterns at compile time:

| Misuse | Error mechanism |
|--------|----------------|
| Pass `SharedRef` to fn expecting `GlobalRef` | Type mismatch (`Shared != Global`) |
| Store `SharedRef` past scope exit | `'scope` lifetime violation (existing HRTB) |
| Send `SharedRef` to another block | `!Send` on `SharedRef` |
| Mix shared/global in single `DisjointSlice` | Type param `Tier` must be uniform |
| Use `SharedRef` inside `GridScope::alloc` | `alloc` returns `GlobalRef`, not `SharedRef` |
| Accidentally use generic `&[T]` | No `Deref` impl; must call `.read()` / `.write()` |

Example error the user would see:
```
error[E0308]: mismatched types
  --> src/kernel.rs:15:20
   |
15 |     process_global(shared_data);
   |                    ^^^^^^^^^^^ expected `GpuRef<'_, f32, Global>`,
   |                                   found `GpuRef<'_, f32, Shared>`
```

### 7. Benchmarking Strategy (Pre-Implementation Gate)

Before implementing the full tiered type system, we should verify that
`ld.shared` / `st.shared` actually outperform generic `ld` / `st` on our
target hardware:

1. **Microbenchmark**: Two kernels doing identical work on shared memory.
   Kernel A: `cvta.shared` + generic ld/st (current behavior).
   Kernel B: raw shared addr + `ld.shared` / `st.shared` (proposed).
   Measure cycles per element for a tight loop (~10K iterations).

2. **Expected delta**: ~2 cycles (shared) vs ~4-5 cycles (generic resolving
   to shared) per access. On a tight inner loop with 1 load + 1 store per
   iteration, this is ~4 cycles saved per iteration — potentially 2x throughput
   for memory-bound kernels.

3. **If delta < 10%**: The tiered types are still valuable for type safety
   (preventing shared/global confusion), but the inline-asm accessor
   complexity may not be justified. In that case, use generic loads inside
   `GpuRef::read()`/`write()` and keep the type wrappers for safety only.

## Open Questions

1. **PTX `ld.shared` with register address**: Does `ld.shared.u32 %r, [%rd];`
   work when `%rd` holds a raw shared-space address (from `mov.u64 %rd,
   dynamic_smem; add.u64 %rd, %rd, offset`)? Or does PTX require a literal
   shared-memory symbol operand? This needs a PTX compilation test. If
   register-indirect `ld.shared` is not supported, the entire inline-asm
   accessor approach needs rethinking.

2. **Bulk operations**: `shared_store` element-by-element for zero-init is
   slow. Should `GpuRef` provide a `fill(val)` method that uses a vectorized
   `st.shared.v4.u32` for 16-byte chunks? Or rely on the user calling
   `spawn_all` for parallel init?

3. **Supported types**: The `TieredAccess` trait must be implemented for each
   `(T, Tier)` pair. Which types do we support initially? Minimum: `u8`, `u32`,
   `u64`, `f32`, `f64`. Compound types (`[f32; 4]`) would need vectorized
   load/store or fall back to element-wise access.

4. **alloc_raw_bytes migration**: `BlockScope::alloc_raw_bytes` returns `*mut u8`
   for channel storage (which contains `UnsafeCell`, not `Copy`). This cannot
   return a `SharedRef` since the types are not `Copy`. Keep as-is? Or add a
   `SharedPtr` (non-slice, raw pointer variant)?

5. **Backward compatibility**: The `alloc() -> &'scope mut [T]` API is used
   extensively. Should we keep the old API as `alloc_generic()` and make
   `alloc()` return `SharedRef`, or vice versa? A `#[deprecated]` shim could
   ease migration.
