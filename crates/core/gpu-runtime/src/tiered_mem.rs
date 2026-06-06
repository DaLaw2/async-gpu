//! Lifetime-parameterized memory tier types for GPU address spaces.
//!
//! Provides [`GpuRef`] — a lifetime-bounded, address-space-aware GPU memory
//! reference. The `Tier` type parameter (`Shared` or `Global`) encodes which
//! PTX address space the pointer lives in, enabling the compiler to emit
//! address-space-specific load/store instructions (`ld.shared`/`st.shared`
//! vs `ld.global`/`st.global`) instead of generic loads that must resolve
//! the address space at runtime.
//!
//! # Key Types
//!
//! - [`MemoryTier`] — sealed trait marking address spaces (`Shared`, `Global`).
//! - [`GpuRef<'scope, T, Tier>`] — address-space-aware pointer + length.
//! - [`SharedRef<'scope, T>`] — alias for `GpuRef<'scope, T, Shared>`.
//! - [`GlobalRef<'scope, T>`] — alias for `GpuRef<'scope, T, Global>`.
//!
//! # Design Principles
//!
//! - **No `Deref`**: Prevents silent fallback to generic address space loads.
//!   Users must call `.read(i)` / `.write(i, val)` explicitly.
//! - **Raw address-space pointers**: `SharedRef` stores the shared-space address
//!   directly (no `cvta.shared` conversion), enabling `ld.shared` emission.
//! - **Sealed trait**: Users cannot add new memory tiers.
//! - **`!Send` for `SharedRef`**: Shared memory is per-block; cross-block send
//!   is meaningless.
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::scope::block_scope;
//!
//! block_scope(|scope| {
//!     let buf: SharedRef<'_, f32> = scope.alloc_shared::<f32>(256);
//!     buf.write(0, 42.0);
//!     let val = buf.read(0); // emits ld.shared.b32
//! });
//! ```

use core::marker::PhantomData;

// ============================================================
// MemoryTier — sealed trait for address space discrimination
// ============================================================

/// Sealed marker trait for GPU memory address spaces.
///
/// Implemented only by [`Shared`] and [`Global`]. Users cannot implement
/// this trait for custom types (sealed via private super-trait).
pub trait MemoryTier: sealed::Sealed {
    /// Human-readable name for error/panic messages.
    const NAME: &'static str;
}

/// Shared memory (PTX address space 3) — per-block, ~2-cycle latency.
///
/// A zero-sized marker type used as the `Tier` parameter of [`GpuRef`]
/// to indicate the pointer lives in shared memory.
pub struct Shared;

impl MemoryTier for Shared {
    const NAME: &'static str = "shared";
}

/// Global memory (PTX address space 1) — grid-wide, ~100-cycle latency.
///
/// A zero-sized marker type used as the `Tier` parameter of [`GpuRef`]
/// to indicate the pointer lives in global memory.
pub struct Global;

impl MemoryTier for Global {
    const NAME: &'static str = "global";
}

mod sealed {
    /// Private super-trait that seals [`MemoryTier`](super::MemoryTier).
    pub trait Sealed {}
    impl Sealed for super::Shared {}
    impl Sealed for super::Global {}
}

// ============================================================
// TieredAccess — tier-specific load/store intrinsics
// ============================================================

/// Trait for tier-specific load/store operations on a given element type.
///
/// Implemented for each `(T, Tier)` pair where `T` is a supported GPU
/// primitive type. The trait is sealed via [`MemoryTier`]'s seal — users
/// cannot add new tiers.
///
/// # Safety
///
/// Implementations use inline PTX assembly. The `ptr` argument must be
/// a valid address in the corresponding address space (shared or global).
pub trait TieredAccess<T: Copy>: MemoryTier {
    /// Load a value from the given address using tier-specific instructions.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, aligned pointer in the correct address space.
    unsafe fn load(ptr: *const T) -> T;

    /// Store a value to the given address using tier-specific instructions.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, aligned pointer in the correct address space.
    unsafe fn store(ptr: *mut T, val: T);
}

// ============================================================
// Inline PTX intrinsics — private
// ============================================================

// --- Shared memory intrinsics ---

/// Load a u32 from shared memory via `ld.shared.u32`.
#[inline(always)]
unsafe fn shared_load_u32(addr: *const u32) -> u32 {
    let val: u32;
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "ld.shared.u32 {val}, [{addr}];",
            val = out(reg32) val,
            addr = in(reg64) addr as u64,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        val = core::ptr::read(addr);
    }
    val
}

/// Store a u32 to shared memory via `st.shared.u32`.
#[inline(always)]
unsafe fn shared_store_u32(addr: *mut u32, val: u32) {
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "st.shared.u32 [{addr}], {val};",
            addr = in(reg64) addr as u64,
            val = in(reg32) val,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::ptr::write(addr, val);
    }
}

/// Load a u64 from shared memory via `ld.shared.u64`.
#[inline(always)]
unsafe fn shared_load_u64(addr: *const u64) -> u64 {
    let val: u64;
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "ld.shared.u64 {val}, [{addr}];",
            val = out(reg64) val,
            addr = in(reg64) addr as u64,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        val = core::ptr::read(addr);
    }
    val
}

/// Store a u64 to shared memory via `st.shared.u64`.
#[inline(always)]
unsafe fn shared_store_u64(addr: *mut u64, val: u64) {
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "st.shared.u64 [{addr}], {val};",
            addr = in(reg64) addr as u64,
            val = in(reg64) val,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::ptr::write(addr, val);
    }
}

// --- Global memory intrinsics ---

/// Load a u32 from global memory via `ld.global.u32`.
#[inline(always)]
unsafe fn global_load_u32(addr: *const u32) -> u32 {
    let val: u32;
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "ld.global.u32 {val}, [{addr}];",
            val = out(reg32) val,
            addr = in(reg64) addr as u64,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        val = core::ptr::read(addr);
    }
    val
}

/// Store a u32 to global memory via `st.global.u32`.
#[inline(always)]
unsafe fn global_store_u32(addr: *mut u32, val: u32) {
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "st.global.u32 [{addr}], {val};",
            addr = in(reg64) addr as u64,
            val = in(reg32) val,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::ptr::write(addr, val);
    }
}

/// Load a u64 from global memory via `ld.global.u64`.
#[inline(always)]
unsafe fn global_load_u64(addr: *const u64) -> u64 {
    let val: u64;
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "ld.global.u64 {val}, [{addr}];",
            val = out(reg64) val,
            addr = in(reg64) addr as u64,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        val = core::ptr::read(addr);
    }
    val
}

/// Store a u64 to global memory via `st.global.u64`.
#[inline(always)]
unsafe fn global_store_u64(addr: *mut u64, val: u64) {
    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "st.global.u64 [{addr}], {val};",
            addr = in(reg64) addr as u64,
            val = in(reg64) val,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::ptr::write(addr, val);
    }
}

// ============================================================
// TieredAccess impls — (T, Tier) pairs
// ============================================================

// --- u32 ---

impl TieredAccess<u32> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> u32 {
        shared_load_u32(ptr)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u32, val: u32) {
        shared_store_u32(ptr, val)
    }
}

impl TieredAccess<u32> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> u32 {
        global_load_u32(ptr)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u32, val: u32) {
        global_store_u32(ptr, val)
    }
}

// --- i32 (reinterpret as u32 for PTX) ---

impl TieredAccess<i32> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> i32 {
        shared_load_u32(ptr as *const u32) as i32
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut i32, val: i32) {
        shared_store_u32(ptr as *mut u32, val as u32)
    }
}

impl TieredAccess<i32> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> i32 {
        global_load_u32(ptr as *const u32) as i32
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut i32, val: i32) {
        global_store_u32(ptr as *mut u32, val as u32)
    }
}

// --- f32 (reinterpret as u32 for PTX bit-level load/store) ---

impl TieredAccess<f32> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> f32 {
        let bits = shared_load_u32(ptr as *const u32);
        f32::from_bits(bits)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f32, val: f32) {
        shared_store_u32(ptr as *mut u32, val.to_bits())
    }
}

impl TieredAccess<f32> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> f32 {
        let bits = global_load_u32(ptr as *const u32);
        f32::from_bits(bits)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f32, val: f32) {
        global_store_u32(ptr as *mut u32, val.to_bits())
    }
}

// --- u64 ---

impl TieredAccess<u64> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const u64) -> u64 {
        shared_load_u64(ptr)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u64, val: u64) {
        shared_store_u64(ptr, val)
    }
}

impl TieredAccess<u64> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const u64) -> u64 {
        global_load_u64(ptr)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u64, val: u64) {
        global_store_u64(ptr, val)
    }
}

// --- i64 (reinterpret as u64 for PTX) ---

impl TieredAccess<i64> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const i64) -> i64 {
        shared_load_u64(ptr as *const u64) as i64
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut i64, val: i64) {
        shared_store_u64(ptr as *mut u64, val as u64)
    }
}

impl TieredAccess<i64> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const i64) -> i64 {
        global_load_u64(ptr as *const u64) as i64
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut i64, val: i64) {
        global_store_u64(ptr as *mut u64, val as u64)
    }
}

// --- f64 (reinterpret as u64 for PTX bit-level load/store) ---

impl TieredAccess<f64> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> f64 {
        let bits = shared_load_u64(ptr as *const u64);
        f64::from_bits(bits)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f64, val: f64) {
        shared_store_u64(ptr as *mut u64, val.to_bits())
    }
}

impl TieredAccess<f64> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> f64 {
        let bits = global_load_u64(ptr as *const u64);
        f64::from_bits(bits)
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f64, val: f64) {
        global_store_u64(ptr as *mut u64, val.to_bits())
    }
}

// --- u8 (promoted to u32 for PTX load/store) ---

impl TieredAccess<u8> for Shared {
    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> u8 {
        // PTX doesn't have ld.shared.u8; use generic read on non-nvptx,
        // and ld.shared.u8 on nvptx (PTX actually supports .u8 loads).
        #[cfg(target_arch = "nvptx64")]
        {
            let val: u32;
            core::arch::asm!(
                "ld.shared.u8 {val}, [{addr}];",
                val = out(reg32) val,
                addr = in(reg64) ptr as u64,
                options(nostack),
            );
            val as u8
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            core::ptr::read(ptr)
        }
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u8, val: u8) {
        #[cfg(target_arch = "nvptx64")]
        {
            core::arch::asm!(
                "st.shared.u8 [{addr}], {val};",
                addr = in(reg64) ptr as u64,
                val = in(reg32) val as u32,
                options(nostack),
            );
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            core::ptr::write(ptr, val);
        }
    }
}

impl TieredAccess<u8> for Global {
    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> u8 {
        #[cfg(target_arch = "nvptx64")]
        {
            let val: u32;
            core::arch::asm!(
                "ld.global.u8 {val}, [{addr}];",
                val = out(reg32) val,
                addr = in(reg64) ptr as u64,
                options(nostack),
            );
            val as u8
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            core::ptr::read(ptr)
        }
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut u8, val: u8) {
        #[cfg(target_arch = "nvptx64")]
        {
            core::arch::asm!(
                "st.global.u8 [{addr}], {val};",
                addr = in(reg64) ptr as u64,
                val = in(reg32) val as u32,
                options(nostack),
            );
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            core::ptr::write(ptr, val);
        }
    }
}

// ============================================================
// shared_addr_at — raw shared-space address (no cvta.shared)
// ============================================================

/// Get a raw shared-space pointer at the given byte offset.
///
/// Unlike [`crate::block::shared_mem_at`] which uses `cvta.shared.u64`
/// (converting to generic address space), this returns the raw
/// shared-space address by adding the offset to `dynamic_smem` directly.
/// This keeps the pointer in address space 3, enabling `ld.shared`/`st.shared`.
///
/// # Safety
///
/// - The byte offset must be within the allocated shared memory region.
/// - Alignment must be appropriate for the intended type.
/// - `dynamic_smem` must be declared (via `global_asm!` in the kernel crate).
#[inline(always)]
pub(crate) unsafe fn shared_addr_at(offset: usize) -> *mut u8 {
    #[cfg(target_arch = "nvptx64")]
    {
        let addr: u64;
        core::arch::asm!(
            "mov.u64 {addr}, dynamic_smem;
             add.u64 {addr}, {addr}, {off};",
            addr = out(reg64) addr,
            off = in(reg64) offset as u64,
            options(nostack),
        );
        addr as *mut u8
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        // On host, fall back to generic address (tests use regular memory).
        crate::block::shared_mem_at::<u8>(offset)
    }
}

// ============================================================
// GpuRef — lifetime-bounded, address-space-aware GPU memory reference
// ============================================================

/// A lifetime-bounded, address-space-aware GPU memory reference.
///
/// `Tier` is [`Shared`] or [`Global`] — a zero-sized phantom that encodes
/// which PTX address space the pointer lives in.
///
/// # Invariants
///
/// - The inner `ptr` is a RAW address-space pointer (NOT converted via
///   `cvta.shared`). For `Shared`, this is the `dynamic_smem + offset`
///   value in addrspace(3). For `Global`, this is the global pointer.
/// - `len` is the number of `T` elements (not bytes).
/// - The `'scope` lifetime is invariant, preventing covariant escape.
///
/// # Properties
///
/// - `Copy + Clone` — thin wrapper, same rationale as [`DisjointSlice`](crate::safety::DisjointSlice).
/// - `SharedRef` is `!Send + !Sync` (shared memory is per-block).
/// - `GlobalRef` is `Send + Sync` (global memory is grid-wide).
/// - No `Deref` impl — forces explicit `.read(i)` / `.write(i, val)`.
#[derive(Copy, Clone)]
pub struct GpuRef<'scope, T: Copy, Tier: MemoryTier> {
    ptr: *mut T,
    len: usize,
    _tier: PhantomData<Tier>,
    /// Invariant lifetime — prevents covariance from allowing escape.
    _scope: PhantomData<&'scope mut &'scope ()>,
}

// SharedRef is !Send + !Sync by default because *mut T is !Send + !Sync.
// We do NOT implement Send/Sync for GpuRef<'_, _, Shared>.

// GlobalRef needs Send + Sync for cross-block use.
// SAFETY: Global memory is accessible from all blocks. The scope lifetime
// ensures the allocation remains valid. Access via read/write uses
// address-space-specific atomics when needed.
unsafe impl<'scope, T: Copy> Send for GpuRef<'scope, T, Global> {}
unsafe impl<'scope, T: Copy> Sync for GpuRef<'scope, T, Global> {}

/// Shared memory reference — bound to a [`BlockScope`](crate::scope::BlockScope).
///
/// Created by [`BlockScope::alloc_shared()`](crate::scope::BlockScope::alloc_shared).
/// The `'scope` lifetime ties this reference to the enclosing block scope,
/// preventing it from escaping or being sent to another block.
pub type SharedRef<'scope, T> = GpuRef<'scope, T, Shared>;

/// Global memory reference — bound to a [`GridScope`](crate::scope::GridScope).
///
/// Created by [`GridScope::alloc_global()`](crate::scope::GridScope::alloc_global).
/// The `'scope` lifetime ties this reference to the enclosing grid scope.
/// Unlike `SharedRef`, this is `Send + Sync` since global memory is
/// accessible from all blocks.
pub type GlobalRef<'scope, T> = GpuRef<'scope, T, Global>;

impl<'scope, T: Copy, Tier: MemoryTier> GpuRef<'scope, T, Tier> {
    /// Create a new `GpuRef` from raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to `len` elements of `T` in the
    ///   address space corresponding to `Tier`.
    /// - The memory must remain valid for `'scope`.
    /// - For `SharedRef`, `ptr` must be a raw shared-space address (no `cvta.shared`).
    #[inline(always)]
    pub(crate) unsafe fn new(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _tier: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Returns the number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the reference is empty (zero elements).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// --- Tier-specific read/write methods ---
// These are on GpuRef where Tier: TieredAccess<T>, so they dispatch
// to the correct inline PTX intrinsic.

impl<'scope, T: Copy, Tier: TieredAccess<T>> GpuRef<'scope, T, Tier> {
    /// Read element at index `i` using tier-specific load instruction.
    ///
    /// For `SharedRef`, this emits `ld.shared`; for `GlobalRef`, `ld.global`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len`.
    #[inline(always)]
    pub fn read(&self, i: usize) -> T {
        assert!(
            i < self.len,
            "GpuRef<{}>::read: index {} out of bounds (len {})",
            Tier::NAME,
            i,
            self.len
        );
        unsafe { Tier::load(self.ptr.add(i)) }
    }

    /// Write `val` to element at index `i` using tier-specific store instruction.
    ///
    /// For `SharedRef`, this emits `st.shared`; for `GlobalRef`, `st.global`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len`.
    #[inline(always)]
    pub fn write(&self, i: usize, val: T) {
        assert!(
            i < self.len,
            "GpuRef<{}>::write: index {} out of bounds (len {})",
            Tier::NAME,
            i,
            self.len
        );
        unsafe { Tier::store(self.ptr.add(i), val) }
    }
}

// --- Shared-specific methods ---

impl<'scope, T: Copy> GpuRef<'scope, T, Shared> {
    /// Returns a raw shared-space pointer (for advanced PTX interop).
    ///
    /// The returned pointer is in PTX address space 3 (shared memory).
    /// Do NOT pass it to functions expecting generic-space pointers.
    #[inline(always)]
    pub fn as_shared_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable shared-space pointer (for advanced PTX interop).
    #[inline(always)]
    pub fn as_shared_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Escape hatch: convert to a plain slice via generic address space.
    ///
    /// Computes the byte offset of this `SharedRef` within shared memory
    /// and uses `cvta.shared.u64` (via [`crate::block::shared_mem_at`]) to
    /// produce a generic-space pointer. The resulting slice uses generic
    /// loads/stores (losing the address-space optimization).
    ///
    /// Use this for migration or interop with code that expects `&[T]`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure no concurrent writes to the same memory.
    /// - The `SharedRef` must have been created from a valid shared memory
    ///   allocation (so the offset computation is correct).
    #[inline(always)]
    pub unsafe fn as_generic_slice(&self) -> &'scope [T] {
        // Compute byte offset from shared memory base.
        let base = shared_addr_at(0);
        let offset = (self.ptr as *const u8).offset_from(base) as usize;
        let generic_ptr = crate::block::shared_mem_at::<T>(offset);
        core::slice::from_raw_parts(generic_ptr, self.len)
    }

    /// Escape hatch: convert to a mutable slice via generic address space.
    ///
    /// # Safety
    ///
    /// - The caller must ensure exclusive access to the memory region.
    /// - The `SharedRef` must have been created from a valid shared memory
    ///   allocation.
    #[inline(always)]
    pub unsafe fn as_generic_slice_mut(&self) -> &'scope mut [T] {
        let base = shared_addr_at(0);
        let offset = (self.ptr as *const u8).offset_from(base) as usize;
        let generic_ptr = crate::block::shared_mem_at::<T>(offset);
        core::slice::from_raw_parts_mut(generic_ptr, self.len)
    }
}

// --- Global-specific methods ---

impl<'scope, T: Copy> GpuRef<'scope, T, Global> {
    /// Returns a raw global-space pointer.
    #[inline(always)]
    pub fn as_global_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable global-space pointer.
    #[inline(always)]
    pub fn as_global_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Convert to a plain slice.
    ///
    /// Global pointers are in the generic address space on most architectures,
    /// so this does not require address space conversion.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent writes to the same memory.
    #[inline(always)]
    pub unsafe fn as_generic_slice(&self) -> &'scope [T] {
        core::slice::from_raw_parts(self.ptr, self.len)
    }

    /// Convert to a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive access to the memory region.
    #[inline(always)]
    pub unsafe fn as_generic_slice_mut(&self) -> &'scope mut [T] {
        core::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}

// ============================================================
// GpuRef: Sub-referencing (slicing into sub-regions)
// ============================================================

impl<'scope, T: Copy, Tier: MemoryTier> GpuRef<'scope, T, Tier> {
    /// Create a sub-reference to a contiguous range of elements.
    ///
    /// Returns a new `GpuRef` pointing to `self[start..start+len]`.
    /// The returned reference has the same `'scope` lifetime and `Tier`.
    ///
    /// # Panics
    ///
    /// Panics if `start + sub_len > self.len`.
    #[inline(always)]
    pub fn sub_ref(&self, start: usize, sub_len: usize) -> GpuRef<'scope, T, Tier> {
        assert!(
            start + sub_len <= self.len,
            "GpuRef<{}>::sub_ref: range [{}..{}] out of bounds (len {})",
            Tier::NAME,
            start,
            start + sub_len,
            self.len
        );
        unsafe { GpuRef::new(self.ptr.add(start), sub_len) }
    }
}
