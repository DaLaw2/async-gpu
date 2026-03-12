use crate::alloc::{GlobalAlloc, Layout, System};
use core::sync::atomic::{AtomicU32, Ordering};

// Slab+bitmap allocator for GPU. Supports allocation AND deallocation.
//
// Design: 8 size classes (16B to 4096B), each backed by a fixed pool of blocks
// and an atomic bitmap. Allocations > 4096B fall back to a bump region.
//
// Thread safety: uses AtomicU32 CAS on bitmaps (GPU-internal, no .sys scope needed).
// Lock-free: multiple GPU threads can alloc/dealloc concurrently.

const GPU_HEAP_SIZE: usize = 1024 * 1024; // 1 MB total

// Size classes and block counts.
// Total slab data: 32K+32K+32K+32K+32K+32K+64K+128K = 384K
// Bitmap overhead: ~516 bytes
// Overflow bump region: ~640K
const NUM_CLASSES: usize = 8;
const CLASS_SIZES: [usize; NUM_CLASSES] = [16, 32, 64, 128, 256, 512, 1024, 4096];
const CLASS_COUNTS: [usize; NUM_CLASSES] = [2048, 1024, 512, 256, 128, 64, 64, 32];
const CLASS_BITMAP_WORDS: [usize; NUM_CLASSES] = [64, 32, 16, 8, 4, 2, 2, 1];

// Precomputed byte offsets for each class's data pool within the heap.
// Class 0 starts at offset 0, class 1 at offset CLASS_SIZES[0]*CLASS_COUNTS[0], etc.
const fn class_data_offset(class: usize) -> usize {
    let mut offset = 0;
    let mut i = 0;
    while i < class {
        offset += CLASS_SIZES[i] * CLASS_COUNTS[i];
        i += 1;
    }
    offset
}

const fn total_slab_size() -> usize {
    class_data_offset(NUM_CLASSES)
}

// The slab data region is at the start of the heap.
// After slab data comes the overflow bump region.
const SLAB_DATA_SIZE: usize = total_slab_size(); // 384K
const OVERFLOW_OFFSET: usize = SLAB_DATA_SIZE;
const OVERFLOW_SIZE: usize = GPU_HEAP_SIZE - SLAB_DATA_SIZE; // ~640K

// Total bitmap words across all classes.
const fn total_bitmap_words() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < NUM_CLASSES {
        total += CLASS_BITMAP_WORDS[i];
        i += 1;
    }
    total
}

const TOTAL_BITMAP_WORDS: usize = total_bitmap_words(); // 129

// Bitmap word offset for each class within the BITMAPS array.
const fn bitmap_offset(class: usize) -> usize {
    let mut offset = 0;
    let mut i = 0;
    while i < class {
        offset += CLASS_BITMAP_WORDS[i];
        i += 1;
    }
    offset
}

// The heap data.
#[repr(align(4096))]
struct GpuHeap {
    data: [u8; GPU_HEAP_SIZE],
}

static mut GPU_HEAP: GpuHeap = GpuHeap { data: [0u8; GPU_HEAP_SIZE] };

// Bitmaps: bit=1 means block is allocated.
// Using a fixed array of AtomicU32. Each word covers 32 blocks.
static BITMAPS: [AtomicU32; TOTAL_BITMAP_WORDS] = {
    // const init: all zeros (all blocks free)
    const ZERO: AtomicU32 = AtomicU32::new(0);
    [ZERO; TOTAL_BITMAP_WORDS]
};

// Overflow bump allocator position (for allocations > 4096 bytes).
static OVERFLOW_POS: AtomicU32 = AtomicU32::new(0);

/// Find the size class for a given (size, align) pair.
/// Returns None if the allocation doesn't fit any class.
#[inline]
fn find_class(size: usize, align: usize) -> Option<usize> {
    // The effective size must satisfy both size and alignment.
    let effective = if align > size { align } else { size };
    let mut i = 0;
    while i < NUM_CLASSES {
        if effective <= CLASS_SIZES[i] {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Find the class that a pointer belongs to, based on its offset in the heap.
#[inline]
fn class_from_ptr(offset: usize) -> Option<usize> {
    if offset >= SLAB_DATA_SIZE {
        return None; // In overflow region
    }
    let mut i = 0;
    while i < NUM_CLASSES {
        let start = class_data_offset(i);
        let end = start + CLASS_SIZES[i] * CLASS_COUNTS[i];
        if offset >= start && offset < end {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        // Try slab allocation for small sizes.
        if let Some(class) = find_class(size, align) {
            if let Some(ptr) = slab_alloc(class) {
                return ptr;
            }
            // Slab full for this class — fall through to overflow.
        }

        // Overflow bump allocation for large sizes or full slabs.
        overflow_alloc(size, align)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let heap_base = unsafe { core::ptr::addr_of_mut!(GPU_HEAP.data) as *mut u8 };
        let offset = (ptr as usize).wrapping_sub(heap_base as usize);

        // If in slab region, free the block.
        if let Some(class) = class_from_ptr(offset) {
            slab_dealloc(class, offset);
        }
        // Overflow region: no deallocation (bump).
    }
}

/// Allocate a block from the slab for the given class.
#[inline]
fn slab_alloc(class: usize) -> Option<*mut u8> {
    let bm_start = bitmap_offset(class);
    let bm_count = CLASS_BITMAP_WORDS[class];
    let block_size = CLASS_SIZES[class];
    let data_start = class_data_offset(class);

    // Scan bitmap words for a free bit.
    for word_idx in 0..bm_count {
        let bm = &BITMAPS[bm_start + word_idx];

        loop {
            let old = bm.load(Ordering::Relaxed);
            if old == 0xFFFFFFFF {
                break; // All 32 bits set, try next word.
            }

            // Find first zero bit.
            let bit = (!old).trailing_zeros();
            if bit >= 32 {
                break; // Should not happen if old != 0xFFFFFFFF.
            }

            let mask = 1u32 << bit;
            // CAS to claim the bit.
            match bm.compare_exchange_weak(old, old | mask, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {
                    // Success! Compute pointer.
                    let block_index = (word_idx as u32 * 32 + bit) as usize;
                    let offset = data_start + block_index * block_size;
                    let heap_base = unsafe { core::ptr::addr_of_mut!(GPU_HEAP.data) as *mut u8 };
                    return Some(unsafe { heap_base.add(offset) });
                }
                Err(_) => continue, // CAS failed, retry with fresh load.
            }
        }
    }

    None // Class is full.
}

/// Free a block back to the slab.
#[inline]
fn slab_dealloc(class: usize, offset: usize) {
    let data_start = class_data_offset(class);
    let block_size = CLASS_SIZES[class];
    let block_index = (offset - data_start) / block_size;

    let word_idx = block_index / 32;
    let bit = block_index % 32;
    let mask = 1u32 << bit;

    let bm_start = bitmap_offset(class);
    let bm = &BITMAPS[bm_start + word_idx];

    // Clear the bit with CAS loop.
    loop {
        let old = bm.load(Ordering::Relaxed);
        let new = old & !mask;
        match bm.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(_) => continue,
        }
    }
}

/// Overflow bump allocation for sizes > 4096 or full slabs.
#[inline]
fn overflow_alloc(size: usize, align: usize) -> *mut u8 {
    loop {
        let pos = OVERFLOW_POS.load(Ordering::Relaxed) as usize;
        let base = OVERFLOW_OFFSET + pos;
        let aligned = (base + align - 1) & !(align - 1);
        let new_end = aligned + size;

        if new_end > GPU_HEAP_SIZE {
            return core::ptr::null_mut(); // OOM
        }

        let new_pos = (new_end - OVERFLOW_OFFSET) as u32;
        match OVERFLOW_POS.compare_exchange_weak(
            pos as u32,
            new_pos,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let heap_base = unsafe { core::ptr::addr_of_mut!(GPU_HEAP.data) as *mut u8 };
                return unsafe { heap_base.add(aligned) };
            }
            Err(_) => continue,
        }
    }
}
