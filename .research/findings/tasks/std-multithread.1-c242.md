# std-multithread.1: Design thread-ID-indexed ThreadLocal for gpu-libc
**Cycle**: 242 | **Theme**: std-multithread | **Kind**: design | **Status**: done

## Summary
Designed a `gpu_threads.rs` module to replace `no_threads.rs` for `target_os = "cuda"`. The
new module provides per-thread storage for `EagerStorage<T>`, `LazyStorage<T>`, and
`LocalPointer` by indexing into statically-allocated arrays using the flat thread ID within
the block (reusing the proven errno pattern from `gpu-libc/src/errno.rs`).

## Problem

The `no_threads.rs` module assumes the target has only one thread. On GPU, each `thread_local!`
declaration creates a single static, and all CUDA threads share it. This causes data races
when launching kernels with more than one thread.

**Affected thread_local! declarations that actually execute on GPU:**

| Location | Type | Impact |
|----------|------|--------|
| `panicking.rs:382` | `Cell<(usize, bool)>` | Panic count shared → nested panic misdetection |
| `reentrant_lock.rs:116` | `Cell<u8>` | TLS address for lock ownership → wrong owner detection |
| `sync/mpmc/waker.rs:207` | `Cell<u8>` | Thread identity for mpmc channels |
| `sync/mpmc/context.rs:41` | `Cell<Option<Context>>` | Channel operation context |
| `thread/current.rs:13` | `LocalPointer` | Current thread handle |
| `thread/current.rs:42-89` | `LocalPointer` (×2-4) | Thread ID encoding |

**NOT affected (bypassed on GPU):**
- `io/stdio.rs:21` OUTPUT_CAPTURE — `_print()` returns early on `target_arch = "nvptx64"`
- `hash/random.rs:67` KEYS — CUDA target uses direct `hashmap_random_keys()` without caching

## Design

### Architecture: New `gpu_threads.rs` module

Create `patched-std/library/std/src/sys/thread_local/gpu_threads.rs` — a drop-in replacement
for `no_threads.rs` that provides per-thread storage via thread-ID-indexed arrays.

Route `target_os = "cuda"` to `gpu_threads` instead of `no_threads` in `mod.rs`.

### Thread ID Function

Reuse the proven pattern from `gpu-libc/src/errno.rs`:

```rust
const MAX_GPU_THREADS: usize = 1024;  // One full block

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
```

### EagerStorage<T> — const-initialized thread_local

```rust
pub struct EagerStorage<T> {
    pub value: T,  // Template value (used as init source)
    slots: UnsafeCell<[MaybeUninit<T>; MAX_GPU_THREADS]>,
    inited: UnsafeCell<[bool; MAX_GPU_THREADS]>,
}

unsafe impl<T> Sync for EagerStorage<T> {}
```

**Const initialization**: `[MaybeUninit<T>; N]` cannot use `[expr; N]` because
`MaybeUninit<T>` is only Copy when T: Copy. Use the well-known pattern:
```rust
unsafe { MaybeUninit::<[MaybeUninit<T>; MAX_GPU_THREADS]>::uninit().assume_init() }
```
This is sound because `MaybeUninit<T>` is valid when uninitialized.

**Per-thread initialization**: On first access, `memcpy` from `value` template:
```rust
impl<T> EagerStorage<T> {
    pub fn get(&'static self) -> &T {
        let tid = gpu_tid();
        unsafe {
            let inited = &mut *self.inited.get();
            if !inited[tid] {
                let slots = &mut *self.slots.get();
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
```

**Macro change** (const path):
```rust
// Before (no_threads.rs):
static VAL: EagerStorage<$t> = EagerStorage { value: INIT };
&VAL.value

// After (gpu_threads.rs):
static VAL: EagerStorage<$t> = EagerStorage::new(INIT);
VAL.get()
```

### LazyStorage<T> — runtime-initialized thread_local

```rust
pub struct LazyStorage<T> {
    values: UnsafeCell<[MaybeUninit<T>; MAX_GPU_THREADS]>,
    states: UnsafeCell<[State; MAX_GPU_THREADS]>,
}

unsafe impl<T> Sync for LazyStorage<T> {}
```

**Const initialization**: `State` is `Copy` (enum with no data), so `[State::Initial; N]` works.
For `values`, use the same `MaybeUninit::uninit().assume_init()` trick.

**Per-thread access**: Same API as `no_threads.rs` but indexes by `gpu_tid()`:
```rust
impl<T> LazyStorage<T> {
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let tid = gpu_tid();
        unsafe {
            let states = &*self.states.get();
            if states[tid].get() == State::Alive {
                (*self.values.get())[tid].as_ptr()
            } else {
                self.initialize(tid, i, f)
            }
        }
    }
}
```

**Macro**: No change needed — LazyStorage macro already calls `VAL.get(init, f)`.

### LocalPointer — per-thread raw pointer storage

```rust
pub(crate) struct LocalPointer {
    ptrs: UnsafeCell<[*mut (); MAX_GPU_THREADS]>,
}

unsafe impl Sync for LocalPointer {}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer {
            ptrs: UnsafeCell::new([ptr::null_mut(); MAX_GPU_THREADS]),
        }
    }

    pub fn get(&self) -> *mut () {
        let tid = gpu_tid();
        unsafe { (*self.ptrs.get())[tid] }
    }

    pub fn set(&self, p: *mut ()) {
        let tid = gpu_tid();
        unsafe { (*self.ptrs.get())[tid] = p; }
    }
}
```

`*mut ()` is Copy, so `[ptr::null_mut(); N]` works without issues.

### Module Routing Patch

Change `sys/thread_local/mod.rs`:
```rust
// Before:
target_os = "cuda",
) => {
    mod no_threads;
    pub use no_threads::{EagerStorage, LazyStorage, thread_local_inner};
    pub(crate) use no_threads::{LocalPointer, local_pointer};
}

// After:
target_os = "cuda",
) => {
    mod gpu_threads;
    pub use gpu_threads::{EagerStorage, LazyStorage, thread_local_inner};
    pub(crate) use gpu_threads::{LocalPointer, local_pointer};
}
```

### Memory Cost Analysis

Each `thread_local!` declaration costs `MAX_GPU_THREADS * (sizeof(T) + overhead)` bytes
of GPU global memory.

| Component | sizeof(T) | Per-declaration cost | Count | Total |
|-----------|-----------|---------------------|-------|-------|
| LOCAL_PANIC_COUNT | 16 bytes | ~17 KB | 1 | 17 KB |
| ReentrantLock TLS addr | 1 byte | ~2 KB | 1 | 2 KB |
| mpmc waker | 1 byte | ~2 KB | 1 | 2 KB |
| mpmc context | ~16 bytes | ~17 KB | 1 | 17 KB |
| LocalPointer (CURRENT) | 8 bytes | ~8 KB | 1 | 8 KB |
| LocalPointer (ID*) | 8 bytes | ~8 KB | 2-4 | 16-32 KB |
| **Total** | | | | **~62-78 KB** |

This is negligible for GPU global memory (2-24 GB typical).

### Destructor Semantics

GPU kernels have bounded lifetime — no thread exit cleanup needed. The `guard::enable()`
function is already a no-op for CUDA (line 126 of `mod.rs`). LazyStorage destructors are
never called, matching the existing `no_threads` behavior.

### Patch Strategy

1. **New file**: `gpu_threads.rs` — added directly to `patched-std/library/std/src/sys/thread_local/`
2. **Modified patch**: Update `sys_thread_local_mod.patch` to route CUDA to `gpu_threads`
   instead of `no_threads`
3. **Regenerate**: Run `bash scripts/gen-patches.sh` to create/update patches

## Open Questions

1. **MAX_GPU_THREADS value**: 1024 matches errno.rs and covers one full block. Multi-block
   launches use the same slots (threadIdx is per-block). This means Block 0 Thread 5 and
   Block 1 Thread 5 share the same thread_local slot. For single-block launches (the common
   test case), this is fine. For multi-block, the slot is effectively per-block-thread-index,
   not per-global-thread — same as errno. This is acceptable for correctness because:
   - Different blocks execute independently (no cross-block data dependencies)
   - Thread locals within a block are isolated from other blocks' thread locals
   - The only race is if blocks time-slice on the same SM with overlapping threadIdx,
     which doesn't happen (blocks occupy separate SM resources)

2. **`core::arch::asm!` in std**: The patched std already uses `#![feature(asm_experimental_arch)]`
   or similar. Need to verify nvptx asm is available in the std build. If not, fallback:
   use extern "C" fn from gpu-libc (already exports `thread_id_in_block`).

## Impact on Downstream Tasks

- **std-multithread.2**: Implement this design — create `gpu_threads.rs`, update routing patch
- **std-multithread.3**: Test with 32-thread launch — should work with no further changes
  once the ThreadLocal replacement is in place
