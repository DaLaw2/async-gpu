# sc-resource.1 — Rust Lifetime Mechanics for GPU Memory Allocation

## Status: done
## Summary: Rust's borrow checker and lifetime enforcement work identically on nvptx64 — there are no target-specific relaxations. `Drop` impls DO run on GPU for stack-allocated types (proven by `MutexGuard` and `OneshotSender` in the codebase), so `BlockScope` can use `Drop` for cleanup. The `cvta.shared` instruction converts shared-memory pointers to generic address space, making `&mut [T]` references into shared memory fully valid, though the allocator must enforce alignment manually. Five specific implementation caveats are identified.

## 1. Lifetime Enforcement on nvptx64

**Finding: The borrow checker works identically. No target-specific relaxations.**

Rust's borrow checker operates entirely at the MIR (Mid-level Intermediate Representation) level, well before any target-specific code generation. The `for<'scope>` HRTB mechanism, `PhantomData<&'scope mut &'scope ()>` invariance, and all lifetime inference run in `rustc_borrowck`, which is target-independent. The nvptx64 backend only receives already-validated MIR.

Verified by examining the patched rustc codegen: `rustc_codegen_llvm` handles nvptx64 address spaces (`AddressSpace::GPU_WORKGROUP` for shared memory) but this is purely at the LLVM IR level — pointer address spaces are invisible to the borrow checker.

**Edge cases investigated:**

1. **`'static` on GPU**: `static` variables on nvptx64 are placed in `.global` memory (LLVM address space 1). The `'static` lifetime works as expected — variables live for the kernel's entire execution. The existing codebase uses `static WARP_STATUS: [AtomicU32; MAX_WARPS]` extensively in `thread.rs`, confirming this works.

2. **Pointer provenance across address spaces**: When `cvta.shared.u64` converts a shared-memory address (addrspace 3) to a generic pointer (addrspace 0), Rust sees only a `*mut u8` — the address space conversion is invisible to the type system. This is by design; the `gpu_launch_sized_workgroup_mem` intrinsic in the compiler performs an `addrspacecast` at the LLVM IR level, returning a plain `ptr` (addrspace 0) to Rust.

3. **No unsafe escape hatches**: There is no nvptx64-specific mechanism that could bypass lifetime checks. Even `core::arch::asm!` blocks that touch shared memory operate through raw pointers, not references — so they don't interact with the borrow checker at all.

**Conclusion**: The `for<'scope> FnOnce(&'scope BlockScope<'scope>) -> R` pattern will enforce lifetime safety on nvptx64 exactly as it does on x86_64. No adaptation needed.

## 2. Drop Semantics on GPU

**Finding: Drop DOES work on GPU. BlockScope can use Drop for guaranteed cleanup.**

### 2.1 Proof from existing codebase

Two types in gpu-runtime already use `Drop` on nvptx64:

1. **`MutexGuard<'a, T>`** (`sync.rs:148`): `impl<'a, T> Drop for MutexGuard<'a, T>` calls `self.mutex.unlock()` which emits `st.release.sys.global.u32`. This runs correctly on GPU — the Mutex tests pass.

2. **`OneshotSender<T>`** (`channel.rs:119`): `impl<T: Copy> Drop for OneshotSender<T>` calls `sys_store_release_u32` to set the channel state to CLOSED. The `send()` method calls `core::mem::forget(self)` to suppress the Drop, confirming the Drop path is real and active.

Both prove that `Drop` impls compile and execute correctly on nvptx64 for stack-allocated types.

### 2.2 Using Drop for BlockScope cleanup

**Recommended approach**: Implement `Drop` for `BlockScope` as a safety net.

```rust
impl<'scope> Drop for BlockScope<'scope> {
    fn drop(&mut self) {
        // 1. Join all spawned warps (poll WARP_STATUS)
        // 2. Pop watermark: ALLOCATOR.pop()
        // Both are warp-0-only operations (safe — only warp 0 holds the scope)
    }
}
```

However, the primary cleanup path should still be in `block_scope()` after the closure returns, because:

- `Drop` is the **fallback** path (early return, `?` operator, panic-catch).
- The explicit path after the closure returns is the **normal** path and can return a value.
- The `Drop` impl should be idempotent — check `self.spawn_count > 0` before joining, check `self.saved_watermark != INVALID` before popping.

### 2.3 Drop and GPU trap (warp panic)

**Critical finding: `trap;` does NOT run destructors.**

When a GPU thread executes `trap;` (PTX instruction), the warp terminates immediately. No Rust stack unwinding occurs. This means:

- If warp 0 (the scope owner) traps, `BlockScope::drop()` will NOT run.
- The watermark will not be popped, spawned warps will not be joined.
- However, this is a non-issue in practice: `trap;` terminates the entire kernel on NVIDIA GPUs (all warps in the block are killed). There is no partial recovery to worry about.

The existing panic handler (`lib.rs:206-232`) confirms this: after `send_panic_hostcall`, it executes `core::arch::asm!("trap;", options(noreturn))`. No cleanup happens between the panic message and the trap.

**Summary**:
- Normal exit: `block_scope()` performs explicit cleanup after closure returns.
- Early return / `?`: `Drop` runs and performs cleanup. Works.
- Panic: `Drop` runs (Rust panic handler runs destructors before aborting). The panic handler sends the message, then traps.
- Direct `trap;` in unsafe code: No cleanup. Kernel dies entirely. Acceptable.

### 2.4 T: Copy constraint

The `T: Copy` bound on `scope.alloc()` is correct and sufficient. `Copy` types have no `Drop` impl, so the watermark-pop approach (bulk deallocation without per-element destruction) is sound. No adaptation needed.

## 3. Interior Mutability (UnsafeCell on GPU)

**Finding: UnsafeCell compiles and works correctly on nvptx64. Already proven in the codebase.**

### 3.1 UnsafeCell in global memory statics

`core::cell::UnsafeCell` is a zero-cost wrapper (`#[repr(transparent)]`). It compiles to the same LLVM IR on nvptx64 as on x86_64. The codebase already uses this pattern extensively:

- `Mutex<T>` (`sync.rs:34-37`): Contains `lock_word: UnsafeCell<u32>` and `data: UnsafeCell<T>`.
- `OneshotSlot<T>` (`channel.rs:28-32`): Contains `state: UnsafeCell<u32>` and `value: UnsafeCell<MaybeUninit<T>>`.
- `MpscChannel<T>` (`channel.rs:258-269`): Contains multiple `UnsafeCell` fields.

All of these work correctly in global memory on GPU.

### 3.2 SharedMemAllocator as a global static with UnsafeCell

The proposed pattern for the allocator:

```rust
// Allocator state in global memory (static), tracking offsets into shared memory
static ALLOCATOR: UnsafeCell<SharedMemAllocator> = UnsafeCell::new(SharedMemAllocator::uninit());
```

This requires `unsafe impl Sync for SharedMemAllocator` (or a wrapper type). The safety justification is sound: only warp 0 ever calls `alloc()`, and only from `block_scope` (which asserts `warp_id() == 0`). This is the same single-writer pattern used by `PANIC_BUF` (`panic.rs:14`) and `NUM_WARPS` (`thread.rs:56`).

**Important**: Use `UnsafeCell`, not `Cell` or `RefCell`. `Cell` requires `T: Copy` for `get()` and doesn't add safety value here. `RefCell` adds runtime borrow checking overhead that is unnecessary when we have the warp-0-only invariant.

### 3.3 Visibility to other warps

**Finding: No synchronization needed for the allocator itself, but shared memory writes need `bar.sync` or `__threadfence_block()`.**

The allocator state (watermark, stack, depth) lives in global memory and is only written by warp 0. Other warps never read it — they receive pointers to shared memory from warp 0 via the trampoline mechanism (closure captures).

For the actual shared memory content (data written by `scope.alloc()` during zero-initialization), visibility to other warps depends on:

1. **Within `spawn_all`**: Warp 0 allocates and initializes before `spawn_all` is called. The `spawn_all` implementation writes to `WARP_STATUS` with `Release` ordering, and worker warps read with `Acquire`. This forms a happens-before relationship — the shared memory writes by warp 0 are visible to workers.

2. **Within `scope.spawn`**: Same mechanism — warp 0 calls `alloc()`, captures the reference in a closure, and publishes via `WARP_STATUS`. The `Release`/`Acquire` pair ensures visibility.

**Caveat**: The allocator's zero-initialization of shared memory (`core::ptr::write_bytes`) needs to complete before the `Release` store. Since `alloc()` is called sequentially from warp 0 before `spawn()`, this is guaranteed by program order.

## 4. Shared Memory Address Space and References

**Finding: `&mut [T]` references into shared memory work correctly. The `cvta.shared` instruction bridges the address space gap.**

### 4.1 Address space mechanics

CUDA shared memory is LLVM address space 3. Rust pointers are address space 0 (generic). The `cvta.shared.u64` instruction converts a shared-memory address to a generic address. From `block.rs:26`:

```asm
cvta.shared.u64 {out}, dynamic_smem;
```

This returns a generic-address-space pointer. When the GPU hardware accesses this pointer, it routes to the shared memory hardware based on the address range. The generic pointer IS a valid address — it just happens to point into shared memory.

### 4.2 Borrow checker and address spaces

**The borrow checker does not know or care about CUDA address spaces.** It operates on Rust's type system, where `&mut [f32]` is `&mut [f32]` regardless of which hardware memory backs it. The address space distinction exists only at the LLVM IR / PTX level.

The `gpu_launch_sized_workgroup_mem` intrinsic in `patched-rustc/compiler/rustc_codegen_llvm/src/intrinsic.rs:653-684` confirms this: it creates a global in `AddressSpace::GPU_WORKGROUP` (addrspace 3), then performs an `addrspacecast` to return a generic pointer to Rust code.

### 4.3 Existing usage validates the approach

The codebase already creates typed references into shared memory:

- `block::shared_mem_at::<T>(offset) -> *mut T` (`block.rs:43-45`): Returns a typed pointer into shared memory.
- `block::reduce_sum_f32` (`block.rs:59-77`): Uses `shared_mem_at::<f32>(smem_offset)` and reads/writes via raw pointer dereferencing. All warps access the same shared memory through these pointers.

The `scope.alloc()` design simply wraps this pattern with `core::slice::from_raw_parts_mut`, adding lifetime safety.

### 4.4 Alignment concerns

**Finding: Alignment must be manually enforced in the bump allocator.**

CUDA shared memory has no inherent alignment — the base pointer from `cvta.shared` is aligned to whatever the `global_asm!` declaration specifies (currently `.align 4` in `gpu-kernel-std/src/lib.rs:22`). Individual allocations within the shared memory region must be explicitly aligned by the bump allocator.

The `SharedMemAllocator::alloc_raw` design already addresses this:

```rust
fn alloc_raw(&mut self, size: usize, align: usize) -> Option<u32> {
    let aligned_offset = (self.watermark as usize + align - 1) & !(align - 1);
    // ...
}
```

**Practical alignment requirements**:
- `f32`: 4-byte aligned (matches `.align 4` base)
- `f64`: 8-byte aligned (needs padding if previous allocation ended at non-8-byte boundary)
- `u128` / SIMD types: 16-byte aligned
- The base `.align 4` in the `global_asm!` declaration should be increased to `.align 16` to support all common types without base-offset complications.

**Recommendation**: Change the shared memory declaration from `.align 4` to `.align 16`:
```rust
core::arch::global_asm!(".extern .shared .align 16 .b8 dynamic_smem[];");
```

This ensures the base pointer is 16-byte aligned, simplifying the bump allocator's alignment math.

## 5. Watermark Allocator Design Validation

### 5.1 Alignment handling

The bump allocator must round up the current watermark to `align_of::<T>()` before each allocation. Implementation:

```rust
fn alloc_raw(&mut self, size: usize, align: usize) -> Option<u32> {
    // Round up watermark to required alignment
    let aligned = (self.watermark as usize + align - 1) & !(align - 1);
    let new_watermark = aligned + size;
    if new_watermark > self.capacity as usize {
        return None; // OOM
    }
    self.watermark = new_watermark as u32;
    Some(aligned as u32)
}
```

This is standard bump allocator practice. The wasted padding between allocations is typically 0-15 bytes — negligible in a 48KB region.

### 5.2 Overflow detection

48KB (49152 bytes) is the shared memory limit on SM75 (Turing). The allocator checks `new_watermark > self.capacity` and returns `None` (or panics in the public `alloc()` method). This is correct.

**Recommendation**: The `capacity` should be set from the kernel launch config's `shared_mem_bytes` value, not hardcoded. The existing `block::shared_mem_ptr()` approach doesn't know the capacity — it must be passed in or queried.

**Practical approach**: Pass `shared_mem_bytes` as a kernel argument, or store it in a device global (like `__HOSTCALL_BUF`). The allocator's `init(capacity)` method sets this once at kernel entry.

### 5.3 Nested scope watermark stack

The design specifies a fixed-depth stack of 4 watermarks. This is sufficient:

- Depth 0: Initial state (watermark = 0)
- Depth 1: First `block_scope` (most kernels only go this deep)
- Depth 2: Nested `block_scope` (used for temporary scratch space)
- Depth 3-4: Deeply nested scopes (rare but supported)

The implementation is straightforward:

```rust
fn push(&mut self) -> u32 {
    assert!(self.depth < 4, "SharedMemAllocator: nesting depth exceeded");
    self.stack[self.depth as usize] = self.watermark;
    self.depth += 1;
    self.watermark
}

fn pop(&mut self) {
    assert!(self.depth > 0, "SharedMemAllocator: pop without push");
    self.depth -= 1;
    self.watermark = self.stack[self.depth as usize];
}
```

**Correctness property**: `pop()` restores the watermark to the value at `push()` time, not to the value before any allocations in the scope. This means inner scope allocations are freed, but outer scope allocations are preserved. This is correct for nested scopes.

### 5.4 Non-warp-0 access prevention

The design specifies that `scope.alloc()` asserts `warp_id() == 0`. This is a runtime check. Compile-time enforcement is not feasible because warp ID is a runtime value on GPU.

**Recommended implementation**:

```rust
pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T] {
    debug_assert_eq!(
        crate::index::thread_idx_x() / 32, 0,
        "scope.alloc() can only be called from warp 0"
    );
    // ... allocate ...
}
```

Use `debug_assert!` (compiled out in release) rather than `assert!` — the warp-0-only invariant is structurally enforced by the API design (only warp 0 enters `block_scope()`), so the assert is a safety net during development. In release builds, the check is unnecessary overhead.

### 5.5 Reserved space for bookkeeping

The cancel flag (4 bytes) is allocated from the shared memory pool at scope entry. This reduces available space by 4 bytes per scope level. For 4 nesting levels, that's 16 bytes — negligible.

**Recommendation**: Also reserve space for the completion/error bitmask (future work from sc-design.2 open question 10.3). Pre-allocating a `u32` error bitmask per scope costs another 4 bytes and would enable `STATUS_TRAPPED` detection without additional allocation.

## 6. Recommendations and Caveats

### 6.1 Implementation plan (no blockers found)

All investigated mechanisms work on nvptx64 as designed. No fundamental changes needed. Proceed to implementation with these specific notes:

### 6.2 Specific caveats

1. **Shared memory alignment**: Change `global_asm!` from `.align 4` to `.align 16` in kernel crates that use scoped allocation. The bump allocator must still pad per-allocation.

2. **Drop for BlockScope**: Implement as safety net, but make the primary cleanup path explicit in `block_scope()`. The `Drop` path should be idempotent (check for already-cleaned-up state).

3. **Capacity discovery**: The allocator needs to know `shared_mem_bytes` from the launch config. Either pass as kernel arg or use a device global. This is an API design choice for `kernel_entry` integration.

4. **trap; skips Drop**: Document that GPU panics (`trap;`) bypass all destructors. This is inherent to GPU execution and cannot be worked around. It matches existing behavior (Mutex lock held during trap is never released — the kernel dies).

5. **Zero-initialization cost**: `scope.alloc()` zero-initializes via `core::ptr::write_bytes`. For large allocations (e.g., 48KB), this takes ~1500 cycles on warp 0 alone. Consider `alloc_uninit()` as the fast path and document the safety requirement (caller must initialize before reading). The design already includes this.

6. **No `Send` needed for allocator**: The `SharedMemAllocator` does not need `Send` or `Sync` bounds on its contents because only warp 0 accesses it. The `unsafe impl Sync` is justified by the single-writer invariant.

## Files Changed: none
