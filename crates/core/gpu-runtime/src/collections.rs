use core::cell::UnsafeCell;
use gpu_atomics::sys_cas_u32;

/// Empty key sentinel — slot is unoccupied.
const EMPTY_KEY: u32 = 0;

/// A fixed-capacity hash map for GPU global memory.
///
/// Keys and values are `u32`. Key `0` is reserved as the empty sentinel.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::collections::GpuHashMap;
///
/// // Allocate in global memory (e.g., via MappedBuffer from host)
/// let map: &GpuHashMap<64> = unsafe { &*(ptr as *const GpuHashMap<64>) };
/// unsafe { map.init() };
///
/// // Insert (lane 0 only for correctness, or any thread with unique keys)
/// unsafe { map.insert(42, 100) };
///
/// // Lookup (any thread, lock-free)
/// let val = unsafe { map.get(42) }; // Some(100)
/// ```
#[repr(C)]
pub struct GpuHashMap<const N: usize> {
    /// Parallel arrays: keys[i] and values[i] form a slot.
    keys: [UnsafeCell<u32>; N],
    values: [UnsafeCell<u32>; N],
    /// Number of occupied slots (approximate, for diagnostics).
    len: UnsafeCell<u32>,
}

// SAFETY: Concurrent access is mediated by atomic CAS on keys.
unsafe impl<const N: usize> Send for GpuHashMap<N> {}
unsafe impl<const N: usize> Sync for GpuHashMap<N> {}

#[allow(clippy::new_without_default)]
impl<const N: usize> GpuHashMap<N> {
    /// Create a new empty hash map.
    ///
    /// All keys are initialized to `EMPTY_KEY` (0).
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: UnsafeCell<u32> = UnsafeCell::new(0);
        Self {
            keys: [ZERO; N],
            values: [ZERO; N],
            len: UnsafeCell::new(0),
        }
    }

    /// Initialize all slots to empty. Call once before use.
    ///
    /// # Safety
    /// Must be called by exactly one thread.
    pub unsafe fn init(&self) {
        for i in 0..N {
            core::ptr::write_volatile(self.keys[i].get(), EMPTY_KEY);
            core::ptr::write_volatile(self.values[i].get(), 0);
        }
        core::ptr::write_volatile(self.len.get(), 0);
    }

    /// Hash a key to a bucket index. Simple multiplicative hash.
    #[inline(always)]
    fn hash(key: u32) -> usize {
        // Fibonacci hashing: multiply by golden ratio, take upper bits
        let h = key.wrapping_mul(0x9E3779B9);
        (h >> (32 - Self::log2_n())) as usize
    }

    /// Compute log2(N) at compile time for hash shift.
    #[inline(always)]
    const fn log2_n() -> u32 {
        let mut n = N;
        let mut log = 0u32;
        while n > 1 {
            n >>= 1;
            log += 1;
        }
        log
    }

    /// Insert a key-value pair. Returns `true` if inserted, `false` if
    /// the key already exists or the map is full.
    ///
    /// Key must be non-zero (0 is the empty sentinel).
    ///
    /// # Safety
    /// - `self` must reside in global memory
    /// - Multiple threads can insert concurrently if they use distinct keys
    /// - Inserting the same key from multiple threads is safe (one wins)
    #[inline(always)]
    pub unsafe fn insert(&self, key: u32, value: u32) -> bool {
        debug_assert!(key != EMPTY_KEY, "key 0 is reserved as empty sentinel");
        if key == EMPTY_KEY {
            return false;
        }

        let mut idx = Self::hash(key);
        for _ in 0..N {
            let slot_key = core::ptr::read_volatile(self.keys[idx].get());

            if slot_key == key {
                // Key already exists — update value
                core::ptr::write_volatile(self.values[idx].get(), value);
                return true;
            }

            if slot_key == EMPTY_KEY {
                // Try to claim this slot
                let old = sys_cas_u32(self.keys[idx].get(), EMPTY_KEY, key);
                if old == EMPTY_KEY {
                    // We won the slot — write value
                    core::ptr::write_volatile(self.values[idx].get(), value);
                    // Approximate count increment (not atomic, just diagnostic)
                    let count = core::ptr::read_volatile(self.len.get());
                    core::ptr::write_volatile(self.len.get(), count.wrapping_add(1));
                    return true;
                }
                if old == key {
                    // Another thread inserted our key — update value
                    core::ptr::write_volatile(self.values[idx].get(), value);
                    return true;
                }
                // Someone else took the slot with a different key — probe
            }

            idx = (idx + 1) % N;
        }

        // Map is full
        false
    }

    /// Look up a key. Returns `Some(value)` if found, `None` if not.
    ///
    /// Lock-free: uses volatile reads only.
    ///
    /// # Safety
    /// `self` must reside in global memory.
    #[inline(always)]
    pub unsafe fn get(&self, key: u32) -> Option<u32> {
        if key == EMPTY_KEY {
            return None;
        }

        let mut idx = Self::hash(key);
        for _ in 0..N {
            let slot_key = core::ptr::read_volatile(self.keys[idx].get());

            if slot_key == key {
                return Some(core::ptr::read_volatile(self.values[idx].get()));
            }

            if slot_key == EMPTY_KEY {
                // Empty slot means key was never inserted
                // (no deletions, so probe chain is intact)
                return None;
            }

            idx = (idx + 1) % N;
        }

        // Probed entire table without finding key
        None
    }

    /// Check if a key exists. Lock-free.
    ///
    /// # Safety
    /// `self` must reside in global memory.
    #[inline(always)]
    pub unsafe fn contains_key(&self, key: u32) -> bool {
        self.get(key).is_some()
    }

    /// Get the approximate number of entries (diagnostic, not atomic).
    pub unsafe fn len(&self) -> u32 {
        core::ptr::read_volatile(self.len.get())
    }

    /// Check if the map is empty (approximate).
    pub unsafe fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity.
    pub const fn capacity(&self) -> usize {
        N
    }
}
