# sc-resource.3 — GridScope Global Memory Allocator

## Status: done
## Summary: Implemented `GridScope<'scope>` and `grid_scope()` in `scope.rs`. GridScope provides a grid-level structured concurrency scope that coordinates across GPU blocks using global memory with system-scope atomics. It features a bump allocator over a caller-owned global memory pool, an atomic completion counter for block synchronization, and a cooperative cancellation flag. The `for<'scope>` HRTB prevents global memory references from escaping the scope, and `Drop` resets the pool offset on scope exit.

## Implementation

### GridScope struct
- `pool_base: *mut u8` — caller-owned global memory pool base
- `pool_offset: UnsafeCell<u32>` — bump allocator offset (interior mutability for `&self` alloc)
- `pool_capacity: u32` — pool size in bytes
- `completion_counter: *mut u32` — system-scope atomic counter at pool offset 0
- `expected_completions: UnsafeCell<u32>` — set by user, checked at scope exit
- `cancel_flag: *mut u32` — system-scope atomic flag at pool offset 4
- `_marker: PhantomData<&'scope mut &'scope ()>` — invariant lifetime

### Pool header layout (8 bytes)
- Bytes 0..4: `completion_counter` (u32, initialized to 0 via `sys_store_release_u32`)
- Bytes 4..8: `cancel_flag` (u32, initialized to 0 via `sys_store_release_u32`)
- User allocations start at byte 8+

### Methods
- `alloc<T: Copy>(count) -> &'scope mut [T]` — zero-initialized bump allocation
- `alloc_uninit<T: Copy>(count) -> &'scope mut [T]` — uninitialized allocation (unsafe)
- `alloc_val<T: Copy>(val) -> &'scope mut T` — single-value allocation
- `available_bytes() -> usize` — remaining pool space
- `cancel()` — system-scope release store to cancel flag
- `is_cancelled() -> bool` — system-scope acquire load of cancel flag
- `wait_for_completions(expected)` — spin-loop with `sys_spin_load_acquire_u32`
- `completion_counter_ptr() -> *mut u32` — for blocks to `sys_fetch_add_u32`
- `cancel_flag_ptr() -> *const u32` — for blocks to check cancellation
- `expected_completions() -> u32` / `set_expected_completions(n)` — configure exit condition

### Drop behavior
- Waits for `expected_completions` via `wait_for_completions` (if > 0)
- Resets `pool_offset` to 0 (logical free, pool is caller-owned)

### grid_scope() entry function
- `unsafe fn grid_scope<F, R>(pool: *mut u8, pool_size: u32, f: F) -> R`
- Asserts pool_size >= 8 (header size)
- Initializes header with system-scope stores
- Constructs GridScope with offset past header
- Calls closure with `&GridScope<'scope>`
- Drop handles cleanup (wait + reset)

### Key design decisions
- Used `UnsafeCell<u32>` for `pool_offset` and `expected_completions` to satisfy Rust's `invalid_reference_casting` lint while allowing mutation through `&self`
- System-scope atomics (not CTA-scope) because GridScope spans multiple blocks
- `sys_spin_load_acquire_u32` for the completion wait loop (includes nanosleep, avoids LLVM LICM hoisting)
- No nesting support (unlike BlockScope's 4-level watermark stack) — GridScopes are top-level coordination primitives

## Files Changed
- `crates/core/gpu-runtime/src/scope.rs` — added GridScope struct, impl, Drop, grid_scope() entry function; updated module doc
