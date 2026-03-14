//! Device-side memory functions (memcpy, memset, memcmp, malloc, free, realloc).
//!
//! These run entirely on the GPU without hostcall overhead.
//!
//! ## Allocation Strategy
//!
//! Uses an atomic bump allocator in GPU global memory. The bump pointer
//! is advanced via `AtomicU64` CAS, making concurrent `malloc()` from
//! multiple CUDA threads safe. The allocator state must be initialized
//! by the host before kernel launch via a dedicated global memory region.
//!
//! For this initial implementation, `free` is a no-op (bump allocator).
//! A freelist allocator can be added later for long-running kernels.

use crate::errno::*;
use crate::types::*;
use core::sync::atomic::{AtomicU64, Ordering};

// ============================================================
// Memory operations (device-side, no hostcall needed)
// ============================================================

/// Copy `n` bytes from `src` to `dest`. Regions must not overlap.
///
/// Note: compiler-builtins already provides memcpy when
/// `compiler-builtins-mem` feature is enabled. This is a fallback
/// that will be used if the compiler-builtins version is not linked.
#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, n: size_t) -> *mut c_void {
    let d = dest as *mut u8;
    let s = src as *const u8;
    let mut i: usize = 0;
    while i < n {
        *d.add(i) = *s.add(i);
        i += 1;
    }
    dest
}

/// Fill `n` bytes of `s` with byte value `c`.
#[no_mangle]
pub unsafe extern "C" fn memset(s: *mut c_void, c: c_int, n: size_t) -> *mut c_void {
    let d = s as *mut u8;
    let byte = c as u8;
    let mut i: usize = 0;
    while i < n {
        *d.add(i) = byte;
        i += 1;
    }
    s
}

/// Compare `n` bytes of `s1` and `s2`.
/// Returns 0 if equal, negative if s1 < s2, positive if s1 > s2.
#[no_mangle]
pub unsafe extern "C" fn memcmp(s1: *const c_void, s2: *const c_void, n: size_t) -> c_int {
    let a = s1 as *const u8;
    let b = s2 as *const u8;
    let mut i: usize = 0;
    while i < n {
        let diff = (*a.add(i) as c_int) - (*b.add(i) as c_int);
        if diff != 0 {
            return diff;
        }
        i += 1;
    }
    0
}

/// Copy `n` bytes from `src` to `dest`, handling overlapping regions.
#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut c_void, src: *const c_void, n: size_t) -> *mut c_void {
    let d = dest as *mut u8;
    let s = src as *const u8;
    if (d as usize) < (s as usize) {
        // Copy forward
        let mut i: usize = 0;
        while i < n {
            *d.add(i) = *s.add(i);
            i += 1;
        }
    } else if (d as usize) > (s as usize) {
        // Copy backward
        let mut i = n;
        while i > 0 {
            i -= 1;
            *d.add(i) = *s.add(i);
        }
    }
    dest
}

// ============================================================
// Bump allocator for GPU global memory
// ============================================================

/// Atomic bump allocator state.
///
/// The bump pointer is an `AtomicU64` so that concurrent `malloc()` calls
/// from multiple CUDA threads are safe (each thread atomically claims its
/// own region via compare-and-swap). The end pointer is also `AtomicU64`
/// but is only written once during init and read-only thereafter.
///
/// `free` is a no-op. This is suitable for short-lived kernels where
/// the entire heap is freed after the kernel completes.
struct BumpState {
    current: AtomicU64,
    end: AtomicU64,
}

/// Global bump allocator state.
/// Must be initialized via `gpu_heap_init` before any allocations.
static BUMP_STATE: BumpState = BumpState {
    current: AtomicU64::new(0),
    end: AtomicU64::new(0),
};

/// Initialize the bump allocator with a memory region.
/// Called by the kernel entry point before any allocations.
///
/// # Safety
/// - `heap_start` must point to a valid GPU global memory region
/// - `heap_size` must be the size of that region in bytes
/// - Must be called exactly once, before any malloc/free calls
/// - All threads must see this init before calling malloc (use a barrier
///   or call from a single thread before launching multi-thread work)
#[no_mangle]
pub unsafe extern "C" fn gpu_heap_init(heap_start: *mut u8, heap_size: usize) {
    BUMP_STATE
        .end
        .store(heap_start.add(heap_size) as u64, Ordering::Relaxed);
    BUMP_STATE
        .current
        .store(heap_start as u64, Ordering::Release);
}

/// Allocate `size` bytes with default alignment (MALLOC_ALIGN).
///
/// Thread-safe: uses atomic CAS on the bump pointer so multiple CUDA
/// threads can allocate concurrently without data races.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    if size == 0 {
        return core::ptr::null_mut();
    }

    let align = MALLOC_ALIGN;
    let end = BUMP_STATE.end.load(Ordering::Relaxed);

    loop {
        let current = BUMP_STATE.current.load(Ordering::Relaxed);
        // Align up
        let aligned = (current as usize + align - 1) & !(align - 1);
        let new_current = aligned + size;

        if new_current as u64 > end {
            // Out of memory
            set_errno(ENOMEM);
            return core::ptr::null_mut();
        }

        // Atomically advance the bump pointer
        match BUMP_STATE.current.compare_exchange_weak(
            current,
            new_current as u64,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return aligned as *mut c_void,
            Err(_) => continue, // Another thread won the race, retry
        }
    }
}

/// Allocate `nmemb * size` bytes, zero-initialized.
#[no_mangle]
pub unsafe extern "C" fn calloc(nmemb: size_t, size: size_t) -> *mut c_void {
    let total = nmemb.checked_mul(size);
    match total {
        Some(0) | None => {
            if total.is_none() {
                set_errno(ENOMEM);
            }
            core::ptr::null_mut()
        }
        Some(total) => {
            let ptr = malloc(total);
            if !ptr.is_null() {
                memset(ptr, 0, total);
            }
            ptr
        }
    }
}

/// Free a previously allocated block.
/// No-op for bump allocator — memory is reclaimed when the kernel exits.
#[no_mangle]
pub unsafe extern "C" fn free(_ptr: *mut c_void) {
    // No-op for bump allocator
}

/// Resize an allocation. For bump allocator, this always allocates new
/// memory and copies the data (cannot shrink in place).
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: size_t) -> *mut c_void {
    if ptr.is_null() {
        return malloc(new_size);
    }
    if new_size == 0 {
        free(ptr);
        return core::ptr::null_mut();
    }

    let new_ptr = malloc(new_size);
    if new_ptr.is_null() {
        return core::ptr::null_mut();
    }

    // Copy old data. We don't know the old size, so we copy new_size bytes.
    // This is safe because:
    // - If shrinking, we only copy new_size bytes (correct)
    // - If growing, we copy new_size bytes from the old region. The old
    //   allocation must have been at least old_size bytes, and the caller
    //   is responsible for ensuring new_size >= old_size for the valid region.
    //
    // Note: In a real implementation, we'd track allocation sizes.
    // For now, the bump allocator makes this acceptable.
    memcpy(new_ptr, ptr as *const c_void, new_size);
    new_ptr
}

/// Allocate memory with specific alignment.
/// Returns 0 on success, ENOMEM on failure. Result stored in *memptr.
///
/// Thread-safe: uses atomic CAS on the bump pointer.
#[no_mangle]
pub unsafe extern "C" fn posix_memalign(
    memptr: *mut *mut c_void,
    align: size_t,
    size: size_t,
) -> c_int {
    if size == 0 {
        *memptr = core::ptr::null_mut();
        return 0;
    }

    // Validate alignment: must be power of 2 and multiple of sizeof(void*)
    if align == 0 || (align & (align - 1)) != 0 || align % core::mem::size_of::<*mut c_void>() != 0
    {
        return EINVAL;
    }

    let end = BUMP_STATE.end.load(Ordering::Relaxed);

    loop {
        let current = BUMP_STATE.current.load(Ordering::Relaxed);
        let aligned = (current as usize + align - 1) & !(align - 1);
        let new_current = aligned + size;

        if new_current as u64 > end {
            return ENOMEM;
        }

        match BUMP_STATE.current.compare_exchange_weak(
            current,
            new_current as u64,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                *memptr = aligned as *mut c_void;
                return 0;
            }
            Err(_) => continue,
        }
    }
}
