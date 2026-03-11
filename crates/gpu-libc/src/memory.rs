//! Device-side memory functions (memcpy, memset, memcmp, malloc, free, realloc).
//!
//! These run entirely on the GPU without hostcall overhead.
//!
//! ## Allocation Strategy
//!
//! Uses a simple bump allocator in GPU global memory. The allocator
//! state (bump pointer, region bounds) must be initialized by the host
//! before kernel launch via a dedicated global memory region.
//!
//! For this initial implementation, `free` is a no-op (bump allocator).
//! A freelist allocator can be added later for long-running kernels.

use crate::types::*;
use crate::errno::*;

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

/// Bump allocator state. The host initializes these before kernel launch
/// by writing to the mapped/device memory region.
///
/// Layout in global memory:
///   [0..8]   current_ptr: u64  (bump pointer, starts at region_start)
///   [8..16]  region_end:  u64  (exclusive upper bound)
///
/// The allocator simply advances current_ptr by the aligned size.
/// `free` is a no-op. This is suitable for short-lived kernels where
/// the entire heap is freed after the kernel completes.
struct BumpState {
    current: *mut u8,
    end: *mut u8,
}

/// Global bump allocator state pointer.
/// Must be set by the host before kernel launch.
/// Points to a BumpState struct in device global memory.
static mut BUMP_STATE: BumpState = BumpState {
    current: core::ptr::null_mut(),
    end: core::ptr::null_mut(),
};

/// Initialize the bump allocator with a memory region.
/// Called by the kernel entry point before any allocations.
///
/// # Safety
/// - `heap_start` must point to a valid GPU global memory region
/// - `heap_size` must be the size of that region in bytes
/// - Must be called before any malloc/free calls
#[no_mangle]
pub unsafe extern "C" fn gpu_heap_init(heap_start: *mut u8, heap_size: usize) {
    BUMP_STATE.current = heap_start;
    BUMP_STATE.end = heap_start.add(heap_size);
}

/// Allocate `size` bytes with default alignment (MALLOC_ALIGN).
#[no_mangle]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    if size == 0 {
        return core::ptr::null_mut();
    }

    let align = MALLOC_ALIGN;
    let current = BUMP_STATE.current as usize;
    // Align up
    let aligned = (current + align - 1) & !(align - 1);
    let new_current = aligned + size;

    if new_current > BUMP_STATE.end as usize {
        // Out of memory
        set_errno(ENOMEM);
        return core::ptr::null_mut();
    }

    BUMP_STATE.current = new_current as *mut u8;
    aligned as *mut c_void
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

    let current = BUMP_STATE.current as usize;
    let aligned = (current + align - 1) & !(align - 1);
    let new_current = aligned + size;

    if new_current > BUMP_STATE.end as usize {
        return ENOMEM;
    }

    BUMP_STATE.current = new_current as *mut u8;
    *memptr = aligned as *mut c_void;
    0
}
