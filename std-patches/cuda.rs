use crate::alloc::{GlobalAlloc, Layout, System};
use core::sync::atomic::{AtomicUsize, Ordering};

// Simple bump allocator for GPU. Uses a statically-sized heap in global memory.
// This allocator never deallocates — suitable for short-lived GPU kernel invocations.
const GPU_HEAP_SIZE: usize = 1024 * 1024; // 1 MB

#[repr(align(16))]
struct GpuHeap {
    data: [u8; GPU_HEAP_SIZE],
}

static mut GPU_HEAP: GpuHeap = GpuHeap { data: [0u8; GPU_HEAP_SIZE] };
static GPU_HEAP_POS: AtomicUsize = AtomicUsize::new(0);

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        loop {
            let pos = GPU_HEAP_POS.load(Ordering::Relaxed);
            let aligned = (pos + align - 1) & !(align - 1);
            let new_pos = aligned + size;

            if new_pos > GPU_HEAP_SIZE {
                return core::ptr::null_mut();
            }

            match GPU_HEAP_POS.compare_exchange_weak(
                pos,
                new_pos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return unsafe { (core::ptr::addr_of_mut!(GPU_HEAP.data) as *mut u8).add(aligned) };
                }
                Err(_) => continue,
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator: no deallocation
    }
}
