//! Thread-ID-indexed thread-local storage for GPU (nvptx64) targets.
//!
//! GPU "threads" are hardware lanes within a block. Unlike `no_threads.rs` which
//! assumes a single thread, this module provides per-thread storage by indexing
//! into statically-allocated arrays using the flat thread ID within the block.
//!
//! Supports up to `MAX_GPU_THREADS` concurrent threads per block (default 1024).
//! Threads beyond this limit share slot 0 (graceful degradation).

use crate::cell::{Cell, UnsafeCell};
use crate::mem::MaybeUninit;
use crate::ptr;

/// Maximum concurrent GPU threads with independent thread-local storage.
/// 1024 = one full CUDA block (32 warps × 32 lanes).
const MAX_GPU_THREADS: usize = 1024;

/// Read the flat thread index within the current block via inline PTX.
///
/// Returns `threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`.
/// Clamped to `MAX_GPU_THREADS - 1` to prevent out-of-bounds access.
#[inline(always)]
fn gpu_tid() -> usize {
    let tid_x: u32;
    let tid_y: u32;
    let tid_z: u32;
    let ntid_x: u32;
    let ntid_y: u32;
    unsafe {
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid_x);
        core::arch::asm!("mov.u32 {}, %tid.y;", out(reg32) tid_y);
        core::arch::asm!("mov.u32 {}, %tid.z;", out(reg32) tid_z);
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid_x);
        core::arch::asm!("mov.u32 {}, %ntid.y;", out(reg32) ntid_y);
    }
    let tid = (tid_x + tid_y * ntid_x + tid_z * ntid_x * ntid_y) as usize;
    if tid < MAX_GPU_THREADS { tid } else { 0 }
}

// ============================================================
// thread_local_inner! macro
// ============================================================

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semiopaque"]
pub macro thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, $(#[$align_attr:meta])*, const $init:expr) => {{
        const __RUST_STD_INTERNAL_INIT: $t = $init;

        unsafe {
            $crate::thread::LocalKey::new(|_| {
                $(#[$align_attr])*
                static __RUST_STD_INTERNAL_VAL: $crate::thread::local_impl::EagerStorage<$t> =
                    $crate::thread::local_impl::EagerStorage::new(__RUST_STD_INTERNAL_INIT);
                __RUST_STD_INTERNAL_VAL.get()
            })
        }
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $(#[$align_attr:meta])*, $init:expr) => {{
        #[inline]
        fn __rust_std_internal_init_fn() -> $t { $init }

        unsafe {
            $crate::thread::LocalKey::new(|__rust_std_internal_init| {
                $(#[$align_attr])*
                static __RUST_STD_INTERNAL_VAL: $crate::thread::local_impl::LazyStorage<$t> = $crate::thread::local_impl::LazyStorage::new();
                __RUST_STD_INTERNAL_VAL.get(__rust_std_internal_init, __rust_std_internal_init_fn)
            })
        }
    }},
}

// ============================================================
// EagerStorage — const-initialized, per-thread via tid-indexed array
// ============================================================

#[allow(missing_debug_implementations)]
pub struct EagerStorage<T> {
    /// Template value — used to initialize each thread's slot on first access.
    value: T,
    /// Per-thread storage slots. Initialized lazily from `value` via memcpy.
    slots: UnsafeCell<[MaybeUninit<T>; MAX_GPU_THREADS]>,
    /// Per-thread initialization flags.
    inited: UnsafeCell<[bool; MAX_GPU_THREADS]>,
}

// SAFETY: Each thread accesses only its own slot, indexed by hardware thread ID.
// No two threads within a block share the same threadIdx, so no data races occur.
unsafe impl<T> Sync for EagerStorage<T> {}

impl<T> EagerStorage<T> {
    pub const fn new(value: T) -> EagerStorage<T> {
        EagerStorage {
            value,
            // SAFETY: MaybeUninit<T> is valid when uninitialized. An array of
            // uninitialized MaybeUninit is sound — this is a well-known pattern
            // used throughout the standard library.
            slots: UnsafeCell::new(unsafe {
                MaybeUninit::<[MaybeUninit<T>; MAX_GPU_THREADS]>::uninit().assume_init()
            }),
            inited: UnsafeCell::new([false; MAX_GPU_THREADS]),
        }
    }

    /// Returns a reference to the current thread's slot, initializing it from
    /// the template value on first access.
    #[inline]
    pub fn get(&'static self) -> &T {
        let tid = gpu_tid();
        unsafe {
            let inited = &mut *self.inited.get();
            if !inited[tid] {
                let slots = &mut *self.slots.get();
                // Bitwise copy from template to this thread's slot.
                // Safe because:
                // - `self.value` is a static and will never be dropped
                // - The destination is MaybeUninit, so overwriting is sound
                // - T is the same type, so alignment and layout match
                core::ptr::copy_nonoverlapping(
                    &self.value as *const T as *const u8,
                    slots[tid].as_mut_ptr() as *mut u8,
                    core::mem::size_of::<T>(),
                );
                inited[tid] = true;
            }
            &*(*self.slots.get())[tid].as_ptr()
        }
    }
}

// ============================================================
// LazyStorage — runtime-initialized, per-thread via tid-indexed array
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Initial,
    Alive,
    Destroying,
}

#[allow(missing_debug_implementations)]
pub struct LazyStorage<T> {
    /// Per-thread storage slots.
    values: UnsafeCell<[MaybeUninit<T>; MAX_GPU_THREADS]>,
    /// Per-thread initialization state.
    states: UnsafeCell<[State; MAX_GPU_THREADS]>,
}

// SAFETY: Each thread accesses only its own slot, indexed by hardware thread ID.
unsafe impl<T> Sync for LazyStorage<T> {}

impl<T> LazyStorage<T> {
    pub const fn new() -> LazyStorage<T> {
        LazyStorage {
            // SAFETY: same pattern as EagerStorage — array of uninitialized MaybeUninit.
            values: UnsafeCell::new(unsafe {
                MaybeUninit::<[MaybeUninit<T>; MAX_GPU_THREADS]>::uninit().assume_init()
            }),
            states: UnsafeCell::new([State::Initial; MAX_GPU_THREADS]),
        }
    }

    /// Gets a pointer to the current thread's TLS value, potentially initializing
    /// it with the provided parameters.
    #[inline]
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let tid = gpu_tid();
        unsafe {
            let states = &*self.states.get();
            if states[tid] == State::Alive {
                (*self.values.get())[tid].as_ptr()
            } else {
                self.initialize(tid, i, f)
            }
        }
    }

    #[cold]
    fn initialize(
        &'static self,
        tid: usize,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        let value = i.and_then(Option::take).unwrap_or_else(f);

        unsafe {
            let states = &mut *self.states.get();
            let values = &mut *self.values.get();

            // Destroy the old value if it is initialized
            if states[tid] == State::Alive {
                states[tid] = State::Destroying;
                ptr::drop_in_place(values[tid].as_mut_ptr());
                states[tid] = State::Initial;
            }

            // Guard against initialization during drop
            if states[tid] == State::Destroying {
                panic!("Attempted to initialize thread-local while it is being dropped");
            }

            values[tid] = MaybeUninit::new(value);
            states[tid] = State::Alive;

            values[tid].as_ptr()
        }
    }
}

// ============================================================
// LocalPointer — per-thread raw pointer storage
// ============================================================

#[rustc_macro_transparency = "semiopaque"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        $vis static $name: $crate::sys::thread_local::LocalPointer = $crate::sys::thread_local::LocalPointer::__new();
        $crate::sys::thread_local::local_pointer! { $($rest)* }
    },
}

pub(crate) struct LocalPointer {
    ptrs: UnsafeCell<[*mut (); MAX_GPU_THREADS]>,
}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer { ptrs: UnsafeCell::new([ptr::null_mut(); MAX_GPU_THREADS]) }
    }

    pub fn get(&self) -> *mut () {
        let tid = gpu_tid();
        unsafe { (*self.ptrs.get())[tid] }
    }

    pub fn set(&self, p: *mut ()) {
        let tid = gpu_tid();
        unsafe {
            (*self.ptrs.get())[tid] = p;
        }
    }
}

// SAFETY: Each thread accesses only its own slot, indexed by hardware thread ID.
unsafe impl Sync for LocalPointer {}
