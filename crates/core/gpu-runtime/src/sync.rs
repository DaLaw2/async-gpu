use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use gpu_atomics::{sys_cas_u32, sys_spin_load_acquire_u32, sys_store_release_u32};

/// Maximum spin iterations before `lock()` panics with a timeout.
/// Same as the hostcall protocol's GPU_MAX_SPIN (10M iterations).
pub const MUTEX_MAX_SPIN: u32 = 10_000_000;

/// Lock states.
const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;

/// A mutual exclusion primitive for GPU global memory.
///
/// Protects shared data with a spin-lock using system-scope atomic CAS.
/// Works correctly across warps and blocks. The lock word and data must
/// reside in global memory (device or mapped) — not shared or local memory.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::sync::Mutex;
///
/// // In global memory (e.g., passed as kernel argument pointer)
/// let mutex: &Mutex<u32> = unsafe { &*(ptr as *const Mutex<u32>) };
///
/// // Lock, modify, auto-unlock via Drop
/// {
///     let mut guard = unsafe { mutex.lock() };
///     *guard += 1;
/// } // guard dropped here → unlock
/// ```
#[repr(C)]
pub struct Mutex<T> {
    lock_word: UnsafeCell<u32>,
    data: UnsafeCell<T>,
}

// SAFETY: GPU threads are not OS threads. The Mutex provides the necessary
// synchronization for cross-warp/cross-block access. Marking as Send+Sync
// allows Mutex to be used in static/global contexts on GPU.
unsafe impl<T: Send> Send for Mutex<T> {}
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create a new unlocked Mutex wrapping the given value.
    ///
    /// The Mutex must reside in global memory for cross-warp/cross-block use.
    /// Typically you'll initialize a Mutex in mapped memory from the host side
    /// by zeroing the lock word and writing the initial value.
    pub const fn new(value: T) -> Self {
        Self {
            lock_word: UnsafeCell::new(UNLOCKED),
            data: UnsafeCell::new(value),
        }
    }

    /// Acquire the lock, spinning until it becomes available.
    ///
    /// Returns a `MutexGuard` that automatically releases the lock on drop.
    /// Panics (traps) if the lock is not acquired within `MUTEX_MAX_SPIN`
    /// iterations, indicating likely deadlock.
    ///
    /// # Safety
    ///
    /// - The Mutex must reside in global memory (device or mapped).
    /// - Must not be called from within the same warp that already holds
    ///   the lock (will deadlock on pre-Volta GPUs, may stall on Volta+).
    #[inline(always)]
    pub unsafe fn lock(&self) -> MutexGuard<'_, T> {
        let lock_ptr = self.lock_word.get();
        let mut spins: u32 = 0;
        loop {
            // Try to acquire: CAS(ptr, UNLOCKED, LOCKED)
            // If returns UNLOCKED, we won the lock.
            let old = sys_cas_u32(lock_ptr, UNLOCKED, LOCKED);
            if old == UNLOCKED {
                return MutexGuard { mutex: self };
            }
            // Spin with nanosleep yield (prevents warp starvation)
            let _ = sys_spin_load_acquire_u32(lock_ptr as *const u32);
            spins += 1;
            if spins >= MUTEX_MAX_SPIN {
                // Likely deadlock — trap
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("trap;", options(noreturn, nostack));
                #[cfg(not(target_arch = "nvptx64"))]
                panic!("GPU Mutex: spin timeout (likely deadlock)");
            }
        }
    }

    /// Try to acquire the lock without spinning.
    ///
    /// Returns `Some(MutexGuard)` if the lock was acquired, `None` if
    /// it's currently held by another thread.
    ///
    /// # Safety
    ///
    /// Same requirements as `lock()`.
    #[inline(always)]
    pub unsafe fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        let lock_ptr = self.lock_word.get();
        let old = sys_cas_u32(lock_ptr, UNLOCKED, LOCKED);
        if old == UNLOCKED {
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }

    /// Release the lock.
    ///
    /// Normally called automatically via `MutexGuard::drop()`. Only call
    /// this directly if you need to release without a guard.
    ///
    /// # Safety
    ///
    /// Caller must hold the lock.
    #[inline(always)]
    unsafe fn unlock(&self) {
        sys_store_release_u32(self.lock_word.get(), UNLOCKED);
    }
}

/// RAII guard for a locked Mutex. Releases the lock on drop.
pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: We hold the lock, so exclusive access is guaranteed.
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We hold the lock, so exclusive access is guaranteed.
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T> Drop for MutexGuard<'a, T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: Guard exists only when lock is held.
        unsafe { self.mutex.unlock() };
    }
}
