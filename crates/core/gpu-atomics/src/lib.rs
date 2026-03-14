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
#![allow(clippy::missing_safety_doc)]
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))]
// Note: link_llvm_intrinsics feature removed — NVVM intrinsics
// (llvm.nvvm.atomic.add.gen.i.sys) fail with "Cannot select" on
// nightly 2025-08-25 LLVM. All operations now use inline PTX asm.
//
// On non-nvptx64 targets (e.g., x86_64), stub implementations are provided
// so that `cargo doc` and dependent crate compilation works. The stubs panic
// at runtime — they exist only for documentation and type-checking.

/// Stub body for non-GPU targets (panics at runtime).
#[cfg(not(target_arch = "nvptx64"))]
macro_rules! gpu_stub {
    () => {
        panic!("gpu-atomics: called on non-GPU target")
    };
    ($t:ty) => {{
        panic!("gpu-atomics: called on non-GPU target")
    }};
}

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
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!("membar.sys;", options(nostack));
    #[cfg(not(target_arch = "nvptx64"))]
    gpu_stub!();
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
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.release.sys.global.u32 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!();
    }
}

/// System-scope release store of a u64.
///
/// Emits `st.release.sys.global.u64 [ptr], val;`
#[inline(always)]
pub unsafe fn sys_store_release_u64(ptr: *mut u64, val: u64) {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.release.sys.global.u64 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg64) val,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!();
    }
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
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "ld.acquire.sys.global.u32 {result}, [{ptr}];",
            result = out(reg32) result,
            ptr = in(reg64) ptr,
            options(nostack, readonly),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = ptr;
        gpu_stub!()
    }
}

/// System-scope acquire load of a u64.
///
/// Emits `ld.acquire.sys.global.u64 result, [ptr];`
#[inline(always)]
pub unsafe fn sys_load_acquire_u64(ptr: *const u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u64;
        core::arch::asm!(
            "ld.acquire.sys.global.u64 {result}, [{ptr}];",
            result = out(reg64) result,
            ptr = in(reg64) ptr,
            options(nostack, readonly),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = ptr;
        gpu_stub!()
    }
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
    #[cfg(target_arch = "nvptx64")]
    {
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
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, expected, desired);
        gpu_stub!()
    }
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
    #[cfg(target_arch = "nvptx64")]
    {
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
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!()
    }
}

// ============================================================
// System-scope compare-and-swap (u64)
// ============================================================

/// System-scope compare-and-swap (CAS) on a u64.
///
/// Emits `atom.cas.sys.global.b64 result, [ptr], expected, desired;`
///
/// Atomically: if `*ptr == expected`, sets `*ptr = desired`. Returns the
/// original value of `*ptr`. System-scope: visible to all threads including CPU.
/// Required for tagged-pointer operations in the hostcall protocol.
///
/// Safety: `ptr` must be a valid, 8-byte aligned pointer to mapped (pinned)
/// global GPU memory accessible from both GPU and CPU.
#[inline(always)]
pub unsafe fn sys_cas_u64(ptr: *mut u64, expected: u64, desired: u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u64;
        core::arch::asm!(
            "atom.cas.sys.global.b64 {result}, [{ptr}], {expected}, {desired};",
            result = out(reg64) result,
            ptr = in(reg64) ptr,
            expected = in(reg64) expected,
            desired = in(reg64) desired,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, expected, desired);
        gpu_stub!()
    }
}

// ============================================================
// System-scope atomic add (u64)
// ============================================================

/// System-scope atomic fetch-and-add on a u64.
///
/// Emits `atom.add.sys.global.u64 result, [ptr], val;`
///
/// Atomically adds `val` to `*ptr` and returns the original value.
/// System-scope: visible to all threads including host CPU.
/// NOTE: No `.sem` qualifier — use `membar_sys()` for ordering.
#[inline(always)]
pub unsafe fn sys_fetch_add_u64(ptr: *mut u64, val: u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u64;
        core::arch::asm!(
            "atom.add.sys.global.u64 {result}, [{ptr}], {val};",
            result = out(reg64) result,
            ptr = in(reg64) ptr,
            val = in(reg64) val,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!()
    }
}

// ============================================================
// System-scope atomic exchange (u64)
// ============================================================

/// System-scope atomic exchange on a u64.
///
/// Emits `atom.exch.sys.global.b64 result, [ptr], val;`
///
/// Atomically sets `*ptr = val` and returns the previous value.
/// System-scope: visible to all threads including host CPU.
#[inline(always)]
pub unsafe fn sys_exchange_u64(ptr: *mut u64, val: u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u64;
        core::arch::asm!(
            "atom.exch.sys.global.b64 {result}, [{ptr}], {val};",
            result = out(reg64) result,
            ptr = in(reg64) ptr,
            val = in(reg64) val,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!()
    }
}

// ============================================================
// Spin-loop acquire loads (LLVM-hoist-safe)
// ============================================================
// These variants do NOT use `options(readonly)` which is the primary
// defense against LLVM's LICM pass hoisting the load out of a spin loop.
// They also include `nanosleep.u32 64` to yield the warp slot to the
// hardware scheduler during spinning.
//
// NOTE: Originally designed as `#[inline(never)]` for a second LICM defense,
// but nvptx64's PTX backend cannot link cross-crate non-inline functions
// (they appear as unresolved `.extern .func`). Changed to `#[inline(always)]`
// to ensure the function body is emitted in the final PTX. The absence of
// `options(readonly)` remains the effective LICM prevention.
//
// Use these ONLY inside spin/poll loops. For single-shot reads,
// prefer `sys_load_acquire_u32/u64` (with readonly, inlineable).

/// Spin-loop-safe system-scope acquire load of a u32.
///
/// Emits `ld.acquire.sys.global.u32` followed by `nanosleep.u32 64`.
///
/// Unlike `sys_load_acquire_u32`, this function:
/// - Does NOT use `options(readonly)` — prevents LLVM LICM hoisting
/// - Includes `nanosleep` — yields warp slot during spin
#[inline(always)]
pub unsafe fn sys_spin_load_acquire_u32(ptr: *const u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "ld.acquire.sys.global.u32 {result}, [{ptr}];",
            "nanosleep.u32 64;",
            result = out(reg32) result,
            ptr = in(reg64) ptr,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = ptr;
        gpu_stub!()
    }
}

/// Spin-loop-safe system-scope acquire load of a u64.
///
/// Emits `ld.acquire.sys.global.u64` followed by `nanosleep.u32 64`.
#[inline(always)]
pub unsafe fn sys_spin_load_acquire_u64(ptr: *const u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u64;
        core::arch::asm!(
            "ld.acquire.sys.global.u64 {result}, [{ptr}];",
            "nanosleep.u32 64;",
            result = out(reg64) result,
            ptr = in(reg64) ptr,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = ptr;
        gpu_stub!()
    }
}

// ============================================================
// Warp intrinsics
// ============================================================

/// Returns the active lane mask for the current warp.
///
/// Emits `activemask.b32 result;` (PTX ISA 6.2+, SM 7.0+).
///
/// Returns a 32-bit bitmask where bit N is set if lane N is currently
/// executing (not predicated off by a conditional). Use this to fill
/// `packet.header.active_mask` in the hostcall protocol.
///
/// IMPORTANT: Do NOT hardcode 0xFFFFFFFF — partial warps at kernel
/// grid edges have fewer than 32 active lanes.
#[inline(always)]
pub unsafe fn activemask() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask: u32;
        core::arch::asm!(
            "activemask.b32 {mask};",
            mask = out(reg32) mask,
            options(nostack),
        );
        mask
    }
    #[cfg(not(target_arch = "nvptx64"))]
    gpu_stub!()
}

/// Returns the lane ID (0..31) within the current warp.
///
/// Emits `mov.u32 result, %laneid;`
#[inline(always)]
pub unsafe fn lane_id() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let id: u32;
        core::arch::asm!(
            "mov.u32 {id}, %laneid;",
            id = out(reg32) id,
            options(nostack, readonly),
        );
        id
    }
    #[cfg(not(target_arch = "nvptx64"))]
    gpu_stub!()
}

/// Warp barrier: synchronize all lanes specified in `mask`.
///
/// Emits `bar.warp.sync mask;` (PTX ISA 6.0+, SM 7.0+).
///
/// All lanes in `mask` must reach this barrier before any of them can proceed.
/// On SM70+ with Independent Thread Scheduling, this is required to force
/// reconvergence. Cost: 0-2 cycles when already converged (effectively free).
///
/// **CRITICAL**: Using `mask = 0xFFFFFFFF` on a partial warp (fewer than 32
/// active lanes) causes deadlock — inactive lanes never reach the barrier.
/// Always use `activemask()` for the mask unless you are certain all 32 lanes
/// are active.
#[inline(always)]
pub unsafe fn syncwarp(mask: u32) {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "bar.warp.sync {mask};",
        mask = in(reg32) mask,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = mask;
        gpu_stub!();
    }
}

/// Warp shuffle: broadcast a u32 value from `src_lane` to all lanes.
///
/// Emits `shfl.sync.idx.b32 result, val, src_lane, 0x1f, mask;`
/// (PTX ISA 6.0+, SM 3.0+ for shfl, SM 7.0+ for shfl.sync).
///
/// Each lane provides `val` but receives the value from `src_lane`.
/// The `0x1f` parameter means the warp width is 32 (clamp = 31).
/// Commonly used with `src_lane = 0` to broadcast lane 0's value to all.
///
/// Use cases:
/// - Broadcasting state discriminant from lane 0 (WarpFuture)
/// - Warp-cooperative reductions and scans
/// - Sharing packet indices across lanes
#[inline(always)]
pub unsafe fn shfl_sync_idx_u32(mask: u32, val: u32, src_lane: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "shfl.sync.idx.b32 {result}, {val}, {src}, 0x1f, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            src = in(reg32) src_lane,
            mask = in(reg32) mask,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (mask, val, src_lane);
        gpu_stub!()
    }
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
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.global.b32 [{ptr}], {val};",
        ptr = in(reg64) ptr,
        val = in(reg32) val,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (ptr, val);
        gpu_stub!();
    }
}

// NVVM intrinsic fallbacks removed — llvm.nvvm.atomic.add.gen.i.sys
// fails with "Cannot select" on current nightly LLVM. All operations
// use inline PTX asm which is confirmed working.
