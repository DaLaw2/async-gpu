# safety-tiers.1 — 3-Tier Safety Model for async-gpu

**Task**: Design — categorize all GPU operations into safety tiers
**Status**: Complete
**Cycle**: 623
**Depends on**: safety-types.1 (cuda-oxide investigation)

## Summary

This document defines async-gpu's 3-tier safety model, adapted from cuda-oxide's
approach to async-gpu's warp-cooperative execution model. Every public operation
in `gpu-runtime` is categorized into one of three tiers based on the degree of
compile-time safety the type system can enforce. The design introduces two new
witness types (`WarpIndex<'scope>` and `WarpHandle<'scope>`) and one new
memory type (`DisjointSlice<'scope, T>`) to lift common GPU patterns from
`unsafe` to safe Rust.

## Tier Definitions (adapted to async-gpu)

### Tier 1 — Safe by Construction

Operations that are **provably race-free by the type system alone**. No `unsafe`
block is needed by the user. Race freedom comes from one of:

- **Disjoint ownership**: Each warp gets exclusive access to a non-overlapping
  partition, mediated by a `WarpIndex<'scope>` witness that cannot be forged,
  copied, or sent across warp boundaries.
- **Read-only access**: Pure functions with no shared mutable state.
- **Encapsulated partitioning**: Terminal operations (par_iter) where the
  library internally guarantees correct partitioning.

**Key property**: If it compiles, there are no data races.

### Tier 2 — Scoped Unsafe (Witness-Mediated)

Operations that are **safe when used correctly within a scope**, but whose
safety contracts cannot be fully verified by the type system. The user writes
`unsafe` blocks (or uses witness types that encapsulate the unsafety).

Categories:
- **Shared mutable memory**: Multiple warps read/write overlapping regions
  with explicit synchronization (barriers, atomics).
- **Warp-cooperative intrinsics**: Operations requiring all 32 lanes to
  participate (shuffles, ballot, vote, reductions).
- **Channel communication**: Cross-warp message passing with atomic state
  machines.
- **Concurrency primitives**: Mutexes, cooperative execution with manual
  partitioning.

**Key property**: Correctness relies on documented invariants the user must
uphold; the `'scope` lifetime prevents memory from escaping.

### Tier 3 — Raw Hardware

Direct hardware access with **no library safety net**. Raw PTX inline assembly,
raw pointer arithmetic, direct shared/global memory manipulation without
scope mediation.

**Key property**: Full control, full responsibility. Future home of TMA,
WGMMA, tcgen05, cluster programming.

---

## Full Operation Categorization

### Module: `index` — Thread/Block/Grid Indexing

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `thread_idx_x()` | 1 | Pure read of hardware register, no shared state |
| `thread_idx_y()` | 1 | Pure read of hardware register |
| `thread_idx_z()` | 1 | Pure read of hardware register |
| `block_idx_x()` | 1 | Pure read of hardware register |
| `block_idx_y()` | 1 | Pure read of hardware register |
| `block_idx_z()` | 1 | Pure read of hardware register |
| `block_dim_x()` | 1 | Pure read of hardware register |
| `block_dim_y()` | 1 | Pure read of hardware register |
| `block_dim_z()` | 1 | Pure read of hardware register |
| `grid_dim_x()` | 1 | Pure read of hardware register |
| `grid_dim_y()` | 1 | Pure read of hardware register |
| `grid_dim_z()` | 1 | Pure read of hardware register |
| `global_thread_idx()` | 1 | Pure computation from hardware registers |
| `global_thread_count()` | 1 | Pure computation from hardware registers |
| `clock_nanos()` | 1 | Pure read of hardware timer |

### Module: `math` — GPU Math Intrinsics

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `sqrt_f32()` | 1 | Pure function, no shared state, single PTX instruction |
| `rsqrt_f32()` | 1 | Pure function |
| `exp_f32()` | 1 | Pure function |
| `log_f32()` | 1 | Pure function |
| `sin_f32()` | 1 | Pure function |
| `cos_f32()` | 1 | Pure function |
| `abs_f32()` | 1 | Pure function |
| `min_f32()` | 1 | Pure function |
| `max_f32()` | 1 | Pure function |
| `fma_f32()` | 1 | Pure function |
| `tanh_f32()` | 1 | Pure function (composition of Tier 1 ops) |
| `sigmoid_f32()` | 1 | Pure function |

### Module: `nn` — Neural Network Building Blocks

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `gelu_f32()` | 1 | Pure function (composition of Tier 1 math) |
| `relu_f32()` | 1 | Pure function |
| `leaky_relu_f32()` | 1 | Pure function |
| `silu_f32()` | 1 | Pure function |
| `warp_softmax_f32()` | 2 | **unsafe**: requires all 32 lanes active, calls warp reductions |
| `warp_layer_norm_f32()` | 2 | **unsafe**: requires all 32 lanes active, calls warp reductions |

### Module: `warp` — Warp-Level Compute Primitives

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `reduce_sum_f32()` | 2 | **unsafe**: all 32 lanes must call; uses `shfl.sync.bfly` |
| `reduce_sum_u32()` | 2 | **unsafe**: all 32 lanes must call |
| `reduce_max_f32()` | 2 | **unsafe**: all 32 lanes must call |
| `reduce_min_f32()` | 2 | **unsafe**: all 32 lanes must call |
| `shfl_bfly_u32()` | 2 | **unsafe**: all lanes in mask must call |
| `shfl_down_u32()` | 2 | **unsafe**: all lanes in mask must call |
| `shfl_up_u32()` | 2 | **unsafe**: all lanes in mask must call |
| `ballot()` | 2 | **unsafe**: all lanes in mask must call |
| `all()` | 2 | **unsafe**: all lanes in mask must call |
| `any()` | 2 | **unsafe**: all lanes in mask must call |

### Module: `thread` — Warp Thread Pool

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `gpu_main()` | 2 | Entry point; sets up warp pool; requires correct launch config |
| `gpu_main_poll()` | 2 | Same as gpu_main with polling sync |
| `spawn()` | 2 | Spawns closure on idle warp; closure must be `Send + 'static`; no compile-time race check on captured data |
| `JoinHandle::join()` | 2 | Blocks until warp completes; spin-wait protocol |
| `available_parallelism()` | 1 | Pure read of global counter |
| `current_id()` | 1 | Pure read of warp ID |
| `yield_now()` | 1 | Side-effect free nanosleep |
| `sleep_nanos()` | 1 | Side-effect free nanosleep |
| `cooperative()` | 2 | **unsafe**: caller must ensure no data races across warps |
| `cooperative_map()` | 2 | Closure receives `(src, dst, len)` as raw pointers; caller must ensure correct partitioning |
| `cooperative_reduce()` | 2 | Same as cooperative_map but for reductions |
| `cooperative_map_with_params()` | 2 | Same as cooperative_map with extra params |
| `CoopMapArgs` | 1 | Data struct, no safety implications |
| `CoopReduceArgs` | 1 | Data struct |
| `CoopMapExtArgs` | 1 | Data struct |
| `gpu_thread_spawn_raw()` | 3 | Raw C FFI: takes raw trampoline fn ptr + raw data ptr |
| `gpu_thread_join_warp()` | 3 | Raw C FFI: joins by raw warp ID |
| `gpu_thread_available_parallelism()` | 1 | Pure read |
| `gpu_thread_current_id()` | 1 | Pure read |

### Module: `block` — Block-Level Primitives

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `sync()` | 2 | **unsafe**: all threads in block must call (deadlock otherwise) |
| `shared_mem_ptr()` | 3 | **unsafe**: returns raw `*mut u8` to shared memory |
| `shared_mem_at::<T>()` | 3 | **unsafe**: raw typed pointer into shared memory at offset |
| `reduce_sum_f32()` | 2 | **unsafe**: all threads must participate; uses shared memory + barriers |
| `reduce_max_f32()` | 2 | **unsafe**: same |
| `reduce_min_f32()` | 2 | **unsafe**: same |

### Module: `scope` — Structured Concurrency Scopes

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `block_scope()` | 1→2 | Entry point is safe; provides `BlockScope` for Tier 2 operations |
| `block_scope_with_parent()` | 2 | Takes raw cancel pointer |
| `init_shared_mem_allocator()` | 2 | **unsafe**: must be called from warp 0 |
| `BlockScope::alloc()` | 1 | Safe: returns `&'scope mut [T]` with lifetime bound; zero-initialized |
| `BlockScope::alloc_uninit()` | 2 | **unsafe**: caller must init before reading |
| `BlockScope::alloc_val()` | 1 | Safe: single value, initialized |
| `BlockScope::alloc_raw_bytes()` | 3 | **unsafe**: raw bytes, caller must initialize |
| `BlockScope::available_bytes()` | 1 | Pure read |
| `BlockScope::spawn()` | 2 | Spawns on idle warp; closure bounded by `'scope`; no compile-time race check |
| `BlockScope::spawn_all()` | 2 | Cooperative dispatch to all warps; closures receive `(wid, n_warps)` as plain `u32` — no type witness |
| `BlockScope::cancel()` | 1 | Sets volatile flag, no races |
| `BlockScope::is_cancelled()` | 1 | Reads volatile flag(s) |
| `BlockScope::cancel_ptr()` | 2 | Returns raw pointer |
| `BlockScope::propagate_cancel()` | 1 | Reads and writes volatile flags |
| `BlockScope::join_all()` | 2 | Spin-waits on warp status; handles trapped warps |
| `BlockScope::error_mask()` | 1 | Pure read |
| `BlockScope::has_errors()` | 1 | Pure read |
| `ScopeJoinHandle::join()` | 2 | Spin-waits; panics if warp trapped |
| `grid_scope()` | 2 | **unsafe**: takes raw pool pointer + size |
| `GridScope::alloc()` | 1 | Safe: returns `&'scope mut [T]` with lifetime bound |
| `GridScope::alloc_uninit()` | 2 | **unsafe**: caller must init |
| `GridScope::alloc_val()` | 1 | Safe: initialized |
| `GridScope::alloc_raw_bytes()` | 3 | **unsafe**: raw bytes |
| `GridScope::available_bytes()` | 1 | Pure read |
| `GridScope::cancel()` | 1 | System-scope release store |
| `GridScope::is_cancelled()` | 1 | System-scope acquire load |
| `GridScope::wait_for_completions()` | 2 | Spin-waits on global memory counter |
| `GridScope::completion_counter_ptr()` | 3 | Returns raw `*mut u32` |
| `GridScope::cancel_flag_ptr()` | 3 | Returns raw `*const u32` |
| `GridScope::expected_completions()` | 1 | Pure read |
| `GridScope::alloc_work_slots()` | 2 | Allocates and initializes via system-scope atomics |
| `GridScope::dispatch_work_to_slot()` | 2 | **unsafe**: writes to global memory work slot |
| `GridScope::set_expected_completions()` | 1 | Single-writer mutation |

### Module: `par_iter` — Parallel Iterators

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `GpuSlice::from_raw_parts()` | 3 | **unsafe**: constructs from raw pointer |
| `GpuSliceMut::from_raw_parts()` | 3 | **unsafe**: constructs from raw mutable pointer |
| `GpuSlice::len()` | 1 | Pure read |
| `GpuSlice::is_empty()` | 1 | Pure read |
| `GpuSlice::as_ptr()` | 1 | Returns raw pointer (no deref) |
| `GpuSlice::par_iter()` | 1 | Creates lazy iterator (no execution) |
| `GpuSliceMut::len()` | 1 | Pure read |
| `GpuSliceMut::is_empty()` | 1 | Pure read |
| `GpuSliceMut::as_ptr()` | 1 | Returns raw pointer |
| `GpuSliceMut::as_mut_ptr()` | 1 | Returns raw pointer |
| `SendPtr::new()` | 2 | Wraps raw pointer as `Send + Sync` — user asserts validity |
| `SendPtr::as_ptr()` | 1 | Unwraps pointer |
| `SendPtrMut::new()` | 2 | Wraps raw mut pointer as `Send + Sync` |
| `SendPtrMut::as_ptr()` | 1 | Unwraps |
| `SendPtrMut::as_mut_ptr()` | 1 | Unwraps |
| `.map()` | 1 | Lazy adapter, no execution |
| `.enumerate()` | 1 | Lazy adapter |
| `.zip()` | 1 | Lazy adapter |
| `.filter()` | 1 | Lazy adapter |
| `.for_each()` | **1** | Terminal: partitioning is handled internally by `spawn_all` round-robin |
| `.fold()` | **1** | Terminal: internal partitioning + cross-warp reduction |
| `.collect_into()` | **1** | Terminal: internal partitioning, each warp writes distinct indices |
| `.sum()` | **1** | Convenience over fold |
| `.product()` | **1** | Convenience over fold |
| `.min()` | **1** | Convenience over fold |
| `.max()` | **1** | Convenience over fold |
| `.count()` | **1** | Pure computation |
| `GpuFilter::for_each()` | **1** | Internal partitioning with predicate skip |
| `GpuFilter::fold()` | **1** | Internal partitioning |
| `GpuFilter::collect_into()` | 2 | Uses atomic counter for cross-warp output coordination |
| `GpuFilter::collect_count()` | 2 | Same as collect_into |
| `GpuFilter::count()` | **1** | Internal partitioning, read-only reduction |
| `GpuFilter::sum/product/min/max()` | **1** | Built on fold |
| `GpuFilterMap::*` | same as GpuFilter | Same patterns |
| `par_iter()` (free fn) | 1 | Convenience wrapper |

### Module: `sync` — GPU Synchronization Primitives

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `Mutex::new()` | 1 | Construction is safe |
| `Mutex::lock()` | 2 | **unsafe**: must be in global memory; must not re-lock from same warp |
| `Mutex::try_lock()` | 2 | **unsafe**: same constraints |
| `MutexGuard::deref/deref_mut` | 2 | Safe while guard is held (RAII) but Mutex::lock is unsafe entry |
| `MutexGuard::drop` | 2 | Releases lock via system-scope atomic |

### Module: `channel` — Global-Memory Channels

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `OneshotSlot::new()` | 1 | Construction |
| `OneshotSlot::reset()` | 2 | **unsafe**: must be called before channel use |
| `oneshot()` | 2 | **unsafe**: slot must be in global memory, not shared |
| `OneshotSender::send()` | 2 | **unsafe**: slot must be in global memory |
| `OneshotReceiver` (Future impl) | 2 | Polls atomic state; system-scope atomics |
| `MpscChannel::init()` | 2 | **unsafe**: single-thread init |
| `MpscChannel::close()` | 2 | **unsafe** |
| `MpscChannel::is_closed()` | 2 | **unsafe** |
| `MpscChannel::try_send()` | 2 | **unsafe**: CAS-based send |
| `MpscChannel::try_recv()` | 2 | **unsafe**: single-consumer |
| `mpsc()` | 2 | **unsafe**: slot in global memory |
| `MpscSender::try_send()` | 2 | **unsafe** |
| `MpscReceiver::try_recv()` | 2 | **unsafe** |
| `MpscReceiver::is_terminated()` | 2 | **unsafe** |
| `MpscReceiver::recv()` | 2 | **unsafe**: creates Future |

### Module: `block_channel` — Block-Scoped Channels

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `BlockOneshotSlot::reset()` | 2 | **unsafe**: must be in shared memory |
| `block_oneshot()` | 2 | **unsafe**: slot must be in shared memory |
| `BlockOneshotSender::send()` | 2 | **unsafe**: CTA-scope atomics |
| `BlockOneshotReceiver::try_recv()` | 2 | **unsafe** |
| `BlockOneshotReceiver::recv_spin()` | 2 | **unsafe** |
| `BlockMpscChannel::init()` | 2 | **unsafe** |
| `BlockMpscChannel::close()` | 2 | **unsafe** |
| `BlockMpscChannel::try_send()` | 2 | **unsafe** |
| `BlockMpscChannel::try_recv()` | 2 | **unsafe** |
| `block_mpsc()` | 2 | **unsafe** |
| `BlockMpscSender::try_send()` | 2 | **unsafe** |
| `BlockMpscReceiver::try_recv()` | 2 | **unsafe** |
| `BlockMpscReceiver::recv_spin()` | 2 | **unsafe** |
| `BlockMpscReceiver::is_terminated()` | 2 | **unsafe** |

### Module: `unified_channel` — Scope-Dispatched Channels

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `BlockScope::oneshot()` | 2 | Allocates from shared memory; returns scoped sender/receiver |
| `BlockScope::mpsc()` | 2 | Same |
| `GridScope::oneshot()` | 2 | Allocates from global pool |
| `GridScope::mpsc()` | 2 | Same |
| `ScopedOneshotSender::send()` | 2 | **unsafe** |
| `ScopedOneshotReceiver::try_recv()` | 2 | **unsafe** |
| `ScopedOneshotReceiver::recv_spin()` | 2 | **unsafe** |
| `ScopedMpscSender::try_send()` | 2 | **unsafe** |
| `ScopedMpscReceiver::try_recv()` | 2 | **unsafe** |
| `ScopedMpscReceiver::recv_spin()` | 2 | **unsafe** |

### Module: `collections` — GPU Hash Map

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `GpuHashMap::new()` | 1 | Construction |
| `GpuHashMap::init()` | 2 | **unsafe**: single-thread init |
| `GpuHashMap::insert()` | 2 | **unsafe**: CAS-based concurrent insert; must be in global memory |
| `GpuHashMap::get()` | 2 | **unsafe**: volatile reads from global memory |
| `GpuHashMap::contains_key()` | 2 | **unsafe**: delegates to get() |
| `GpuHashMap::len()` | 2 | **unsafe**: volatile read |
| `GpuHashMap::is_empty()` | 2 | **unsafe**: volatile read |
| `GpuHashMap::capacity()` | 1 | Const computation |

### Module: `executor` — Async Task Executor

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `GpuExecutor::init()` | 2 | **unsafe**: must be in global memory |
| `GpuExecutor::spawn()` | 2 | Spawns type-erased future into work queue |
| `GpuExecutor::run()` | 2 | Warp enters executor loop; all 32 lanes must be active |
| `GpuExecutor::shutdown()` | 2 | Sets shutdown flag via system-scope atomic |
| `TaskId` | 1 | Opaque handle |
| `ExecutorStats` | 1 | Data struct |
| `ExecutorError` | 1 | Error enum |

### Module: `hostcall` — GPU-Host Communication

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `gpu_hostcall_request()` | 2 | **unsafe**: raw buffer pointer, lock-free protocol |
| `gpu_hostcall_request_with_timeout()` | 2 | **unsafe** |
| `gpu_hostcall_print()` | 2 | **unsafe**: raw buffer + message pointers |
| `gpu_hostcall_trace()` | 2 | **unsafe** |
| `gpu_hostcall_assert()` | 2 | **unsafe**: traps after sending |
| `gpu_hostcall_release()` | 2 | **unsafe**: returns packet to free stack |
| `hc_pop_free()` | 3 | Internal: raw stack operations |
| `hc_push()` | 3 | Internal: raw stack operations |

### Module: `warp_future` / `warp_cooperative` / `warp_sequential` / `warp_result`

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `WarpFuture` trait | 2 | Requires all 32 lanes converged |
| `WarpExecutor::run()` | 2 | All lanes must participate |
| `WarpContext` | 1 | Data struct |
| `WarpPoll` | 1 | Enum (Ready/Pending) |
| `WarpCooperativeFuture::new()` | 2 | Wraps inner future for warp-cooperative polling |
| Warp-cooperative polling | 2 | Uses shfl.sync for broadcast |

### Module: `entry` — Kernel Entry

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `auto_init()` | 2 | Reads device global, initializes subsystems |

### Module: `grid_work` — Cross-Block Work Coordination

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `init_work_slots()` | 2 | **unsafe**: system-scope release stores |
| `dispatch_work()` | 2 | **unsafe**: writes to global memory work slot |
| `wait_slot_completed()` | 2 | Spin-waits on system-scope atomic |
| `read_result()` | 2 | **unsafe**: reads from global memory |
| `grid_worker_loop()` | 2 | **unsafe**: worker enters polling loop |
| `BlockWorkSlot` | 1 | Data struct (layout type) |

### Module: `panic` / `print_buffer` / `sideband` / `stdio` / `cmd` / `flight_recorder`

| Operation | Tier | Rationale |
|-----------|------|-----------|
| `gpu_panic_init()` | 2 | **unsafe**: sets global buffer pointer |
| `gpu_result_init()` | 2 | **unsafe**: sets global result pointer |
| `panic_handler!()` | 2 | Installs panic handler with raw hostcall |
| `print_buffer::init()` | 2 | **unsafe**: raw buffer setup |
| `print_buffer::print()` | 2 | **unsafe**: raw buffer writes |
| `print_buffer::flush()` | 2 | **unsafe**: hostcall |
| `sideband_alloc()` | 2 | **unsafe**: atomic bump allocation |
| `gpu_bulk_read/write()` | 2 | **unsafe**: hostcall with sideband |
| `stdio_init()` | 2 | **unsafe**: global init |
| `cmd_poll()` / `cmd_ack()` | 2 | **unsafe**: mapped memory polling |
| `flight_recorder::*` | 2 | **unsafe**: mapped memory ring buffer |

---

## New Types for Safety Tier Promotion

### WarpIndex<'scope> — Tier 1 Witness

The analog of cuda-oxide's `ThreadIndex`, adapted for warp-granularity:

```rust
/// Opaque witness proving this code runs on a specific warp within a scope.
///
/// Cannot be constructed, copied, sent, or stored outside the scope.
/// Provided by BlockScope::spawn_all_indexed() and cooperative_indexed().
pub struct WarpIndex<'scope> {
    warp_id: u32,
    n_warps: u32,
    _not_send: PhantomData<*const ()>,          // !Send
    _scope: PhantomData<&'scope mut &'scope ()>, // invariant lifetime
}

// Explicitly NOT: Send, Sync, Copy, Clone

impl<'scope> WarpIndex<'scope> {
    /// This warp's ID (0..n_warps).
    pub fn warp_id(&self) -> u32 { self.warp_id }

    /// Total number of participating warps.
    pub fn n_warps(&self) -> u32 { self.n_warps }

    /// Compute the global index for local element `local_i`
    /// using round-robin striding: `warp_id + local_i * n_warps`.
    pub fn global_index(&self, local_i: usize) -> usize {
        self.warp_id as usize + local_i * self.n_warps as usize
    }
}
```

**Construction**: Only by trusted entry points:
- `BlockScope::spawn_all_indexed(|widx: WarpIndex<'scope>| { ... })`
- `cooperative_indexed(|widx: WarpIndex<'_>| { ... })`

**Why !Send + !Copy + !Clone**:
- `!Copy + !Clone`: Prevents duplicating the witness (would allow aliased access).
- `!Send`: Prevents sending to another warp (would break the 1:1 mapping).
- Invariant `'scope`: Prevents storing in a &'static or escaping the scope.

### DisjointSlice<'scope, T> — Tier 1 Memory

A slice where each warp gets exclusive access to a non-overlapping partition:

```rust
/// A slice with per-warp exclusive partitions.
///
/// Created from scope-allocated memory. Access requires a WarpIndex witness
/// that proves the caller is the rightful owner of one partition.
pub struct DisjointSlice<'scope, T: Copy> {
    ptr: *mut T,
    len: usize,
    _scope: PhantomData<&'scope mut [T]>,
}

// !Send + !Sync — cannot escape the scope

impl<'scope, T: Copy> DisjointSlice<'scope, T> {
    /// Get this warp's partition as a mutable iterator of (global_index, &mut T).
    ///
    /// Partitioning is round-robin: warp `w` owns indices
    /// `w, w + n_warps, w + 2*n_warps, ...`.
    /// The WarpIndex witness guarantees exclusive access.
    pub fn partition_mut<'a>(
        &'a mut self,
        idx: &WarpIndex<'scope>,
    ) -> DisjointPartitionMut<'a, T> { ... }

    /// Get this warp's partition as a read-only iterator.
    pub fn partition<'a>(
        &'a self,
        idx: &WarpIndex<'scope>,
    ) -> DisjointPartition<'a, T> { ... }

    /// Total length of the underlying slice.
    pub fn len(&self) -> usize { self.len }
}
```

**How it provides Tier 1 safety**:
1. `DisjointSlice` is created from `BlockScope::disjoint_slice()` — the scope
   owns the memory, `'scope` prevents escape.
2. `partition_mut()` requires a `&WarpIndex<'scope>` — the witness proves
   the caller is a specific warp within the scope.
3. The partition is computed internally (round-robin) — the user cannot
   override the partitioning.
4. `WarpIndex` is `!Copy + !Clone` — only one exists per warp per scope,
   so only one `partition_mut()` can be active at a time.

### WarpHandle<'scope> — Tier 2 Safe Warp Ops

A witness that all 32 lanes are active and converged, enabling safe access
to warp-level intrinsics:

```rust
/// Witness that all 32 lanes are active and converged.
///
/// Constructed only by trusted entry points (spawn_all, cooperative).
/// Enables safe (non-unsafe) warp shuffle/ballot/reduce calls.
pub struct WarpHandle<'scope> {
    _scope: PhantomData<&'scope ()>,
    _not_send: PhantomData<*const ()>,
}

impl<'scope> WarpHandle<'scope> {
    // Warp reductions — safe because WarpHandle guarantees all lanes active
    pub fn reduce_sum_f32(&self, val: f32) -> f32 {
        unsafe { crate::warp::reduce_sum_f32(val) }
    }

    pub fn reduce_sum_u32(&self, val: u32) -> u32 {
        unsafe { crate::warp::reduce_sum_u32(val) }
    }

    pub fn reduce_max_f32(&self, val: f32) -> f32 {
        unsafe { crate::warp::reduce_max_f32(val) }
    }

    pub fn reduce_min_f32(&self, val: f32) -> f32 {
        unsafe { crate::warp::reduce_min_f32(val) }
    }

    // Shuffles
    pub fn shfl_bfly_u32(&self, val: u32, offset: u32) -> u32 {
        unsafe { crate::warp::shfl_bfly_u32(0xFFFF_FFFF, val, offset) }
    }

    pub fn shfl_down_u32(&self, val: u32, delta: u32) -> u32 {
        unsafe { crate::warp::shfl_down_u32(0xFFFF_FFFF, val, delta) }
    }

    pub fn shfl_up_u32(&self, val: u32, delta: u32) -> u32 {
        unsafe { crate::warp::shfl_up_u32(0xFFFF_FFFF, val, delta) }
    }

    // Vote ops
    pub fn ballot(&self, predicate: bool) -> u32 {
        unsafe { crate::warp::ballot(0xFFFF_FFFF, predicate) }
    }

    pub fn all(&self, predicate: bool) -> bool {
        unsafe { crate::warp::all(0xFFFF_FFFF, predicate) }
    }

    pub fn any(&self, predicate: bool) -> bool {
        unsafe { crate::warp::any(0xFFFF_FFFF, predicate) }
    }
}
```

**Construction**: Only by trusted entry points that guarantee all lanes are
active. The closure in `spawn_all_indexed` receives both `WarpIndex` and
`WarpHandle`:

```rust
scope.spawn_all_indexed(|widx: WarpIndex<'scope>, warp: WarpHandle<'scope>| {
    // widx: Tier 1 — safe disjoint access
    // warp: Tier 2 → safe — warp ops without unsafe
    let my_sum = warp.reduce_sum_f32(my_val);
    let my_partition = output.partition_mut(&widx);
    // ...
});
```

**Why this is only "Tier 2 safe" and not "Tier 1"**: The warp operations
themselves are correct (all lanes guaranteed active), but the data they operate
on may still have shared-access concerns. The `WarpHandle` makes the *intrinsic
call* safe, but it does not prove that `val` was computed without races.

---

## Tier Transitions

### Tier 3 → Tier 2: Scope Wrapping

Raw operations become scoped-unsafe by wrapping in `BlockScope` / `GridScope`:

```
// Tier 3: raw shared memory
let ptr = unsafe { block::shared_mem_ptr() };

// Tier 2: scope-managed shared memory
block_scope(|scope| {
    let buf = scope.alloc::<f32>(256); // 'scope lifetime prevents escape
});
```

### Tier 2 → Tier 1: Witness Types

Scoped-unsafe operations become safe-by-construction with witness types:

```
// Tier 2: manual partitioning, no compile-time race check
scope.spawn_all(|wid, n_warps| {
    let mut i = wid as usize;
    while i < len {
        output[i] = input[i] * 2.0; // could be wrong if partitioning is wrong
        i += n_warps as usize;
    }
});

// Tier 1: type-enforced partitioning
scope.spawn_all_indexed(|widx, _warp| {
    let out = output.partition_mut(&widx);
    for (i, slot) in out.iter_mut() {
        *slot = input[i] * 2.0; // partition is computed by DisjointSlice
    }
});
```

### Tier 2 → Tier 1 for warp ops (via WarpHandle):

```
// Tier 2: raw unsafe warp reduce
let sum = unsafe { warp::reduce_sum_f32(my_val) };

// Tier 1-ish (safe call, still Tier 2 concept):
scope.spawn_all_indexed(|_widx, warp| {
    let sum = warp.reduce_sum_f32(my_val); // no unsafe block needed
});
```

---

## Current Codebase Mapping

### What is already Tier 1

- `index::*` — pure hardware register reads
- `math::*` — pure math intrinsics
- `nn::{gelu, relu, leaky_relu, silu}` — pure activation functions
- `par_iter` adapters (`.map()`, `.enumerate()`, `.zip()`, `.filter()`)
- `par_iter` indexed terminals (`.for_each()`, `.fold()`, `.collect_into()`,
  `.sum()`, `.product()`, `.min()`, `.max()`)
- `BlockScope::alloc()`, `GridScope::alloc()` — safe allocation with lifetime

### What is Tier 2 today but could be partially promoted to Tier 1

| Current API | Current Tier | Promotion Path |
|-------------|-------------|----------------|
| `BlockScope::spawn_all(wid, n_warps)` | 2 | Add `spawn_all_indexed(WarpIndex)` variant → Tier 1 when used with DisjointSlice |
| `thread::cooperative(f)` | 2 (unsafe) | Add `cooperative_indexed(f: Fn(WarpIndex))` → Tier 1 with DisjointSlice |
| `warp::reduce_sum_f32()` etc. | 2 (unsafe) | Safe via `WarpHandle::reduce_sum_f32()` when called from indexed entry points |
| `GpuFilter::collect_into()` | 2 (atomic counter) | Remains Tier 2 — output ordering is data-dependent |
| `GpuSliceMut` (Copy + Send + Sync) | escape hatch | Replace with DisjointSlice in indexed terminals |

### What stays Tier 2

- All channel operations (oneshot, mpsc, block channels, unified channels)
- `Mutex::lock()` / `Mutex::try_lock()`
- `block::sync()` (barrier)
- `executor::*` (async task spawning)
- `hostcall::*` (raw hostcall protocol)
- `grid_work::*` (cross-block dispatch)
- `thread::spawn()` / `scope.spawn()` (individual warp spawn)

### What stays Tier 3

- `block::shared_mem_ptr()` / `block::shared_mem_at()`
- `GpuSlice::from_raw_parts()` / `GpuSliceMut::from_raw_parts()`
- `scope.alloc_raw_bytes()`
- `GridScope::completion_counter_ptr()` / `cancel_flag_ptr()`
- `gpu_thread_spawn_raw()` / `gpu_thread_join_warp()` (C FFI)
- `hc_pop_free()` / `hc_push()` (internal raw stack ops)

---

## API Design for Each Tier

### Tier 1 API Surface (user writes no `unsafe`)

```rust
use gpu_runtime::prelude::*;
use gpu_runtime::scope::block_scope;

// Kernel body:
block_scope(|scope| {
    let input = scope.alloc::<f32>(1024);
    // ... host fills input ...

    let mut output = scope.disjoint_slice::<f32>(1024); // NEW

    scope.spawn_all_indexed(|widx, warp| {                // NEW
        let out_part = output.partition_mut(&widx);
        for (global_i, slot) in out_part {
            *slot = math::sqrt_f32(input[global_i]);      // Tier 1 math
        }
        let my_sum = warp.reduce_sum_f32(*out_part.first()); // safe warp op
    });
});
```

Also, par_iter already provides a Tier 1 surface:

```rust
data.par_iter()
    .map(|x| x * 2.0)
    .map(|x| x + 1.0)
    .collect_into(output);
// No unsafe needed — partitioning is internal
```

### Tier 2 API Surface (user writes `unsafe` or uses scoped APIs)

```rust
block_scope(|scope| {
    let buf = scope.alloc::<f32>(256);
    let (tx, rx) = scope.oneshot::<u32>();

    scope.spawn(move || {
        // Manual computation — no WarpIndex
        unsafe { tx.send(42) };
        42u32
    });

    let val = unsafe { rx.recv_spin() };

    // Or: cooperative with manual partitioning
    unsafe {
        thread::cooperative(|| {
            let wid = thread::current_id() as usize;
            let n = thread::available_parallelism() + 1;
            // ... manual partitioning ...
        });
    }
});
```

### Tier 3 API Surface (raw hardware access)

```rust
// Direct shared memory access
unsafe {
    let smem = block::shared_mem_ptr();
    let val = core::ptr::read_volatile(smem.add(offset) as *const f32);
}

// Raw PTX assembly
unsafe {
    core::arch::asm!("shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, 0xffffffff;",
        dst = out(reg32) result,
        src = in(reg32) val,
        off = in(reg32) offset,
    );
}
```

---

## Migration Path

### Phase 1: Add New Types (non-breaking)

1. Add `WarpIndex<'scope>` to `gpu-runtime/src/scope.rs` (or new `safety.rs`)
2. Add `DisjointSlice<'scope, T>` alongside scope types
3. Add `WarpHandle<'scope>` alongside warp types
4. Add `BlockScope::spawn_all_indexed()` — new method, does not change existing `spawn_all()`
5. Add `BlockScope::disjoint_slice()` — new method
6. Add `cooperative_indexed()` — new function in `thread` module

**No existing code changes. All new APIs are additive.**

### Phase 2: Promote par_iter Internals (non-breaking)

1. Internally, `default_for_each` / `default_fold` / `default_collect_into`
   already use correct round-robin partitioning. Document them as Tier 1.
2. Add `collect_into_disjoint()` terminal that takes `DisjointSlice` instead
   of `GpuSliceMut` — provides compile-time proof of non-overlap.
3. The existing `collect_into(GpuSliceMut)` remains for backward compat.

### Phase 3: Deprecate Raw Variants (soft migration)

1. Add `#[deprecated(note = "use spawn_all_indexed for type-safe partitioning")]`
   to `spawn_all(|wid, n_warps|)` — eventually, but not yet.
2. Add `#[deprecated(note = "use cooperative_indexed")]` to `cooperative()`.
3. Keep deprecated variants indefinitely — they are still useful for cases
   where DisjointSlice does not apply (e.g., scatter patterns).

### Phase 4: Tighten GpuSliceMut (breaking, future)

1. Consider removing `Copy` from `GpuSliceMut<T>` — currently it is Copy + Send
   + Sync, which means it can be aliased freely. This is the biggest safety hole.
2. Alternatively, keep `GpuSliceMut` as-is for Tier 2/3 use and push users
   toward `DisjointSlice` for Tier 1. This is the pragmatic approach.

---

## Open Questions

### 1. Should WarpHandle carry the lane mask?

Currently proposed: `WarpHandle` implies all 32 lanes active (mask = 0xFFFFFFFF).
Alternative: `WarpHandle` carries an explicit mask, allowing partial-warp operations.

**Recommendation**: Start with full-warp only. async-gpu's model has lane 0
controlling execution with all 32 lanes participating in warp ops. Partial masks
are an advanced pattern that can be added later as `WarpHandlePartial<'scope>`.

### 2. Contiguous vs round-robin partitioning in DisjointSlice?

cuda-oxide uses contiguous partitions (warp 0 gets elements 0..N/W, warp 1 gets
N/W..2N/W). async-gpu's par_iter uses round-robin striding (warp w gets
w, w+W, w+2W, ...).

**Recommendation**: Default to round-robin (matches existing par_iter semantics
and provides better memory coalescing on GPU). Offer a `DisjointSlice::contiguous()`
constructor for cases where contiguous partitions are preferred.

### 3. Should channels be promotable to Tier 1?

Scoped channels (oneshot, mpsc) are currently Tier 2. Could they be promoted
if combined with witness types?

**Answer**: No. Channels are fundamentally about shared mutable state between
concurrent actors. The sender/receiver split provides ownership-like guarantees,
but the timing of sends/receives and the possibility of dropped senders are
not expressible in Tier 1's type constraints. Channels are the canonical Tier 2
pattern — safe within their protocol, but requiring discipline.

### 4. How does DisjointSlice interact with async/await?

A `WarpIndex` is `!Send`, preventing it from being held across `.await` points
in a multi-warp executor. Within a single scope entry (which is synchronous),
the index is valid throughout. If async patterns inside scopes are needed in
the future, `WarpIndex` would need a pin-based pattern or the executor would
need to guarantee warp affinity.

**Recommendation**: For now, `WarpIndex` is scoped to synchronous scope closures.
Async interaction is a future consideration when the executor integrates with
scopes.

### 5. Should we add a SafeSharedSlice for read-only shared access?

cuda-oxide's Tier 2 includes shared memory with explicit barriers. async-gpu
could offer a `SharedReadSlice<'scope, T>` that allows all warps to read
(after a barrier) but not write.

**Recommendation**: Defer. The current `scope.alloc()` returns `&'scope mut [T]`
which is exclusive to warp 0. For shared reads, users currently pass pointers
through closures. A `SharedReadSlice` would be a nice Tier 1 addition (immutable
shared access is race-free) but is not urgently needed.

---

## Comparison with cuda-oxide

| Aspect | cuda-oxide | async-gpu |
|--------|-----------|-----------|
| Unit of parallelism | Lane (1 of 32 SIMT lanes) | Warp (32 lanes as 1 logical thread) |
| Tier 1 witness | `ThreadIndex<'kernel, IS>` | `WarpIndex<'scope>` |
| Tier 1 memory | `DisjointSlice<T, IS>` | `DisjointSlice<'scope, T>` |
| Index space | Compile-time `Index1D`, `Index2D<S>` | Runtime round-robin (n_warps known at scope entry) |
| Lifetime source | `'kernel` from `#[kernel]` macro | `'scope` from `block_scope()` / `grid_scope()` HRTB |
| Tier 2 witness | None (raw `unsafe`) | `WarpHandle<'scope>` (safe warp ops) |
| Shared memory | `SharedArray<T, N>` (static, unsafe) | `BlockScope::alloc()` (dynamic, scoped) |
| Warp ops | Lane-level: each lane independent | Warp-level: lane 0 controls, all participate |
| Async support | None (synchronous kernel) | WarpFuture, executor, channels |
| Iterator API | None | `par_iter` (Tier 1 terminals) |

---

## Summary Table

| Tier | Enforcement | Example Operations | Witness/Gate |
|------|-------------|-------------------|--------------|
| 1 | Type system — compile-time race freedom | DisjointSlice + WarpIndex, par_iter terminals, math, index | WarpIndex<'scope>, DisjointSlice<'scope, T> |
| 2 | Scoped unsafe — documented invariants | Channels, Mutex, cooperative(), spawn(), warp shuffles, barriers | WarpHandle<'scope>, `'scope` lifetime, `unsafe` blocks |
| 3 | Raw hardware — full responsibility | Raw shared_mem_ptr, from_raw_parts, PTX asm, C FFI | None (raw `unsafe`) |

**Tier 1 coverage today**: ~40% of the public API (index, math, nn activations,
par_iter adapters+terminals, scope alloc).

**Tier 1 coverage after WarpIndex/DisjointSlice**: ~55% (adds spawn_all_indexed,
cooperative_indexed, safe warp ops via WarpHandle).

**Tier 2 remains**: ~40% (channels, mutex, executor, hostcall, grid_work,
manual cooperative).

**Tier 3 remains**: ~5% (raw pointers, C FFI, raw PTX asm).
