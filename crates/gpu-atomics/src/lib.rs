//! gpu-atomics: System-scope GPU atomic primitives for nvptx64
//!
//! Provides correct system-scope (`.sys`) atomic operations and fences for
//! GPU-CPU shared memory communication. Uses `core::arch::asm!` with inline
//! PTX, which is confirmed to work on the nvptx64-nvidia-cuda target via
//! the `asm_experimental_arch` feature (verified 2026-03-11, nightly built
//! 2025-08-25, LLVM 19.x).
//!
//! All operations target `.sys` scope, which is required for GPU-CPU
//! communication via pinned (mapped) memory on SM70+ (SM86 / RTX 3060).
//!
//! IMPORTANT: These are `unsafe` because:
//! 1. They emit raw PTX without the Rust memory model guarantees.
//! 2. Correct use requires pointers to pinned/mapped global memory.
//! 3. No bounds checking or lifetime tracking.

#![no_std]
#![feature(asm_experimental_arch)]
#![feature(link_llvm_intrinsics)]

// ============================================================
// System-scope fence
// ============================================================

/// System-scope memory barrier (fence).
///
/// Emits `membar.sys;` PTX instruction.
///
/// Equivalent to `fence.sc.sys` for ordering purposes: all memory operations
/// issued before this fence in program order are guaranteed to be globally
/// visible (including to the host CPU) before any memory operation issued
/// after this fence.
///
/// Required: SM60+ (Pascal). Recommended: SM70+ (Volta) for full acquire-
/// release semantics with `.sys` scope atomics.
#[inline(always)]
pub unsafe fn membar_sys() {
    core::arch::asm!("membar.sys;", options(nostack));
}

// ============================================================
// System-scope stores
// ============================================================

/// System-scope release store of a u32.
///
/// Emits `st.release.sys.global.u32 [ptr], val;`
///
/// All prior memory operations (in program order) are guaranteed to be
/// visible to any thread in the system before this store becomes visible.
/// This is the correct instruction for writing a "data ready" flag to
/// host-accessible memory.
///
/// Safety: `ptr` must be a valid, aligned pointer to mapped (pinned) global
/// GPU memory accessible from both GPU and CPU.
#[inline(always)]
pub unsafe fn sys_store_release_u32(ptr: *mut u32, val: u32) {
    core::arch::asm!(
        "st.release.sys.global.u32 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
}

/// System-scope release store of a u64.
///
/// Emits `st.release.sys.global.u64 [ptr], val;`
#[inline(always)]
pub unsafe fn sys_store_release_u64(ptr: *mut u64, val: u64) {
    core::arch::asm!(
        "st.release.sys.global.u64 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg64) val,
        options(nostack),
    );
}

// ============================================================
// System-scope loads
// ============================================================

/// System-scope acquire load of a u32.
///
/// Emits `ld.acquire.sys.global.u32 result, [ptr];`
///
/// All subsequent memory operations (in program order) are guaranteed to
/// observe any stores that were visible before the matching release store
/// on the other side of this synchronization pair. This is the correct
/// instruction for reading a "data ready" flag from host-accessible memory
/// and then safely reading the data.
///
/// Safety: `ptr` must be a valid, aligned pointer to mapped (pinned) global
/// GPU memory accessible from both GPU and CPU.
///
/// NOTE: The `readonly` asm option tells LLVM this block does not write
/// memory. If used inside a GPU-side spin loop, LLVM may hoist the load
/// out of the loop. For spin-loop use cases, prefer calling this in a
/// function not marked `#[inline(always)]` or use a volatile fence.
#[inline(always)]
pub unsafe fn sys_load_acquire_u32(ptr: *const u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "ld.acquire.sys.global.u32 {result}, [{ptr}];",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        options(nostack, readonly),
    );
    result
}

/// System-scope acquire load of a u64.
///
/// Emits `ld.acquire.sys.global.u64 result, [ptr];`
#[inline(always)]
pub unsafe fn sys_load_acquire_u64(ptr: *const u64) -> u64 {
    let result: u64;
    core::arch::asm!(
        "ld.acquire.sys.global.u64 {result}, [{ptr}];",
        result = out(reg64) result,
        ptr = in(reg64) ptr,
        options(nostack, readonly),
    );
    result
}

// ============================================================
// System-scope compare-and-swap
// ============================================================

/// System-scope compare-and-swap (CAS) on a u32.
///
/// Emits `atom.cas.sys.global.b32 result, [ptr], expected, desired;`
///
/// Atomically: if `*ptr == expected`, sets `*ptr = desired`. Returns the
/// original value of `*ptr` regardless of whether the swap occurred.
/// The operation is atomic at system scope — visible to all threads
/// including the host CPU.
///
/// Safety: `ptr` must be a valid, aligned pointer to mapped (pinned) global
/// GPU memory accessible from both GPU and CPU.
#[inline(always)]
pub unsafe fn sys_cas_u32(ptr: *mut u32, expected: u32, desired: u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "atom.cas.sys.global.b32 {result}, [{ptr}], {expected}, {desired};",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        expected = in(reg32) expected,
        desired = in(reg32) desired,
        options(nostack),
    );
    result
}

// ============================================================
// System-scope atomic add
// ============================================================

/// System-scope atomic fetch-and-add on a u32.
///
/// Emits `atom.add.sys.global.u32 result, [ptr], val;`
///
/// Atomically adds `val` to `*ptr` and returns the original value.
/// System-scope: visible to all threads including host CPU.
/// NOTE: This does NOT have a `.sem` qualifier (no acquire/release).
/// For ordering guarantees, surround with `membar_sys()`.
#[inline(always)]
pub unsafe fn sys_fetch_add_u32(ptr: *mut u32, val: u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "atom.add.sys.global.u32 {result}, [{ptr}], {val};",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
    result
}

// ============================================================
// Helper: store to global address space
// ============================================================

/// Plain store to global address space (no ordering qualifier).
///
/// Emits `st.global.b32 [ptr], val;`
///
/// Use this instead of a Rust pointer dereference (`*ptr = val`) when
/// you need an explicit `.global` address space qualifier in PTX.
#[inline(always)]
pub unsafe fn st_global_u32(ptr: *mut u32, val: u32) {
    core::arch::asm!(
        "st.global.b32 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
}

// ============================================================
// NVVM intrinsic fallbacks (internal, for comparison / older SM)
// ============================================================
// These are kept for testing and as a fallback path if inline asm
// regresses in a future LLVM version. Not part of the public API.

extern "C" {
    /// LLVM NVPTX intrinsic for system-scope memory barrier.
    /// Emits `membar.sys;` (same as membar_sys() above but via intrinsic path).
    #[link_name = "llvm.nvvm.membar.sys"]
    pub(crate) fn nvvm_membar_sys();

    /// LLVM NVPTX intrinsic for system-scope atomic add on i32.
    /// Emits `atom.sys.add.s32 result, [ptr], val;`
    /// Note: `.sys` scope but no `.sem` qualifier — relaxed ordering at sys scope.
    #[link_name = "llvm.nvvm.atomic.add.gen.i.sys.i32.p0i32"]
    pub(crate) fn nvvm_atomic_add_sys_i32(ptr: *mut i32, val: i32) -> i32;
}
