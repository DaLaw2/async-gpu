# sc-channel.1 — Transport Selection Heuristics for GPU Channels

## Status: done
## Summary: Investigated three transport tiers (shfl, shared memory, global memory) for GPU channel communication and evaluated four auto-selection strategies. Shuffle is unsuitable as a general channel primitive (it is a collective, not point-to-point), but is ideal for warp-internal broadcast/reduce within the executor. Shared-memory channels reuse the existing OneshotSlot/MpscChannel designs with CTA-scope atomics (~2-6 cycles vs ~100 cycles for system-scope), yielding 20-50x latency improvement for intra-block communication. The recommended architecture is scope-based selection: BlockScope channels use shared memory, GridScope channels use global memory, with a unified `Channel<T>` enum for cases where the scope is unknown at compile time.

## 1. Warp-Level Transport (shfl.sync)

### 1.1 When shuffle-based communication is appropriate

Shuffle is appropriate ONLY when:
- Both sender and receiver are lanes within the **same warp** (same 32-lane SIMT group).
- Both are at a **matching execution point** (the shuffle instruction is collectively executed).
- The data fits in a **32-bit register** per lane (or can be decomposed into 32-bit chunks).

The existing codebase uses shuffle for exactly these patterns:
- `warp::reduce_sum_f32` — butterfly reduction across 32 lanes via `shfl.sync.bfly`
- `warp::shfl_bfly_u32`, `shfl_down_u32`, `shfl_up_u32` — primitive data exchange
- `gpu_atomics::shfl_sync_idx_u32` — broadcast from a source lane to all lanes

### 1.2 Can we build a channel abstraction over shuffle?

**No — shuffle is fundamentally a different communication pattern.** A channel is point-to-point (or multi-producer single-consumer) with decoupled send/recv: the sender sends at time T, the receiver reads at time T+delta. Shuffle is a **synchronous collective**: all participating lanes execute the instruction simultaneously, and data is exchanged in a single cycle.

Specific mismatches with the channel abstraction:
1. **No temporal decoupling.** There is no "send now, receive later." Both endpoints must be at the shuffle instruction simultaneously.
2. **No buffering.** Shuffle is register-to-register with zero storage. A channel requires a buffer (even a single slot).
3. **All-to-all, not point-to-point.** Shuffle exchanges data between ALL lanes in the mask simultaneously. There is no concept of "lane 3 sends to lane 7."
4. **Cannot cross warp boundary.** Shuffle is warp-internal only. A channel passed from one warp to another is meaningless for shuffle transport.

**What shuffle IS useful for within the channel system:**
- **Warp-cooperative polling:** The executor already uses `shfl_sync_idx_u32` to broadcast poll results from lane 0 to all lanes (in `warp_future.rs`). This accelerates the Future polling loop.
- **Warp-internal broadcast of channel state:** When a warp polls a shared-memory channel, lane 0 can load the state and broadcast it to all 31 other lanes via shuffle, avoiding 31 redundant shared memory loads.
- **Reducing contention:** In an MPSC channel, a warp of 32 producers can use `shfl_down` + `ballot` to elect a single lane to perform the CAS on the head pointer, reducing atomic contention from 32 CAS operations to 1.

### 1.3 Latency and throughput on SM75

| Operation | Latency | Throughput |
|-----------|---------|------------|
| `shfl.sync.*.b32` | ~1 cycle | 32 values/cycle (one per lane) |
| `shfl.sync` with predicated mask | ~2 cycles | Same, but inactive lanes get undefined |
| Multiple shuffles for >32-bit data | N cycles for N*4 bytes | Linear in word count |

For a 64-bit value: 2 shuffle instructions = ~2 cycles. For a 128-bit value: 4 shuffles = ~4 cycles.

### 1.4 Limitations

- **32-bit per instruction.** Transferring a `T` larger than 4 bytes requires multiple shuffles. A 16-byte struct = 4 shuffles.
- **Requires active mask.** All lanes in `mask` must execute the shuffle simultaneously. On SM75 (Turing, independent thread scheduling), diverged lanes must be reconverged first via `syncwarp`.
- **Warp-local only.** Cannot communicate across warps. Period.
- **No async/Future integration.** Shuffle cannot return `Poll::Pending` — it completes immediately or deadlocks.

### 1.5 Recommendation for warp-level

Do NOT create a `WarpChannel<T>`. Instead, expose shuffle as what it is: a **warp-cooperative primitive** used internally by higher-level channel implementations to optimize polling and contention. Specifically:

```rust
// Internal optimization: broadcast channel state from lane 0
fn poll_channel_warp_cooperative<T>(slot: &OneshotSlot<T>) -> Option<T> {
    let state = if lane_id() == 0 {
        load_acquire(slot.state_ptr())
    } else {
        0
    };
    // Broadcast state to all lanes — avoids 31 redundant loads
    let state = shfl_sync_idx_u32(FULL_MASK, state, 0);
    if state == ONESHOT_SENT {
        // Only lane 0 reads the value
        if lane_id() == 0 { Some(read_volatile(slot.value_ptr())) }
        else { None }
    } else {
        None
    }
}
```

## 2. Block-Level Transport (Shared Memory)

### 2.1 Design: shared-memory oneshot channel

The existing `OneshotSlot<T>` design can be adapted to shared memory by changing the atomic scope from `.sys` (system) to `.cta` (block-local):

```rust
/// Oneshot channel slot backed by shared memory.
/// Allocated via BlockScope::alloc().
///
/// Layout identical to global-memory OneshotSlot but uses
/// CTA-scope atomics instead of system-scope.
#[repr(C)]
pub struct BlockOneshotSlot<T: Copy> {
    state: UnsafeCell<u32>,   // 4 bytes
    _pad: u32,                // 4 bytes (alignment)
    value: UnsafeCell<MaybeUninit<T>>,  // size_of::<T>() bytes
}
// Total: 8 + size_of::<T>() bytes, rounded up to align_of::<T>()
```

**Synchronization options for shared memory:**

Option A: **CTA-scope atomics** (recommended)
- PTX: `ld.acquire.cta.shared.u32`, `st.release.cta.shared.u32`, `atom.cas.cta.shared.b32`
- Latency: ~2-4 cycles per atomic on SM75
- Advantage: Matches existing OneshotSlot protocol exactly — only the PTX scope qualifier changes
- Works correctly with warps at different execution points (no `bar.sync` required)

Option B: **bar.sync-based signaling**
- Producer writes value to shared memory, then all threads execute `bar.sync`
- Latency: ~4 cycles for `bar.sync`
- Disadvantage: Requires ALL threads in the block to participate — incompatible with the thread pool model where warps are at different execution points
- Only viable for `spawn_all` patterns where all warps are cooperating

**Recommendation:** Option A (CTA-scope atomics). The existing acquire/release protocol from `OneshotSlot` translates directly. We need new atomic primitives in `gpu-atomics`:

```rust
// New primitives needed (CTA-scope, shared memory):
pub unsafe fn cta_store_release_shared_u32(ptr: *mut u32, val: u32);
  // PTX: st.release.cta.shared.u32 [ptr], val;

pub unsafe fn cta_load_acquire_shared_u32(ptr: *const u32) -> u32;
  // PTX: ld.acquire.cta.shared.u32 result, [ptr];

pub unsafe fn cta_cas_shared_u32(ptr: *mut u32, expected: u32, desired: u32) -> u32;
  // PTX: atom.cas.cta.shared.b32 result, [ptr], expected, desired;
```

### 2.2 Design: shared-memory MPSC ring buffer

The existing `MpscChannel<T, N>` can also be adapted to shared memory:

```rust
/// MPSC channel backed by shared memory.
///
/// Storage layout (in shared memory, allocated by BlockScope):
///   [head: u32] [tail: u32] [closed: u32] [waker_set: u32]
///   [_waker_pad: u32] [waker_bytes: 16B]
///   [slots: N * MpscSlot<T>]
///
/// Total size: 36 + N * (8 + size_of::<T>()) bytes
///
/// For N=8, T=u32: 36 + 8*12 = 132 bytes (fits easily in 48KB)
/// For N=64, T=u32: 36 + 64*12 = 804 bytes
pub struct BlockMpscChannel<T: Copy, const N: usize> { /* same fields as MpscChannel */ }
```

**Contention under multiple warps:** When K warps contend on the MPSC head pointer:
- System-scope CAS on global memory: ~100 cycles per attempt, K warps retry => ~100*K cycles worst case
- CTA-scope CAS on shared memory: ~2-6 cycles per attempt, K warps retry => ~6*K cycles worst case
- For K=8 (common block configuration): shared memory MPSC is **~15x faster** than global memory MPSC

### 2.3 Avoiding bank conflicts

Shared memory on SM75 has 32 banks, each 4 bytes wide. Consecutive 4-byte words map to consecutive banks. Bank conflicts cause serialized access.

**Channel slot layout analysis:**
- `OneshotSlot` is 8 + sizeof(T) bytes. For T=u32 (4 bytes), total = 12 bytes = 3 banks. No conflict for a single slot.
- `MpscChannel` head/tail/closed are at offsets 0, 4, 8. Head and tail are the hot fields. They map to banks 0 and 1 respectively — **no conflict** between producer (reads head) and consumer (reads tail).
- Ring buffer slots: each slot is 8 + sizeof(T) bytes. For T=u32, slots start at offset 36 and are 12 bytes apart. Slot 0 at bank 9, slot 1 at bank 12, slot 2 at bank 15... No two consecutive slots share a bank.

**Recommendation:** The natural layout is bank-conflict-free for the common case (T <= 8 bytes). For larger T, add padding to ensure each slot starts at a 32-byte boundary (8 banks):

```rust
// For T where size_of::<T>() + 8 > 32, pad to 32-byte boundary:
const SLOT_STRIDE: usize = ((8 + core::mem::size_of::<T>() + 31) / 32) * 32;
```

### 2.4 Latency comparison

| Operation | Global Memory (system-scope) | Shared Memory (CTA-scope) | Speedup |
|-----------|------------------------------|---------------------------|---------|
| Oneshot send (store release) | ~100 cycles (L2 miss) / ~50 cycles (L2 hit) | ~2-4 cycles | 25-50x |
| Oneshot recv (load acquire) | ~100 cycles / ~50 cycles | ~2-4 cycles | 25-50x |
| MPSC send (CAS + store) | ~150-200 cycles | ~6-10 cycles | 15-30x |
| MPSC recv (load + store) | ~100-150 cycles | ~4-8 cycles | 15-25x |

These numbers assume:
- SM75 (GTX 1660): L1 cache 32KB (unified with shared memory), L2 cache 1.5MB
- Global memory latency: ~200 cycles for DRAM, ~50 cycles for L2 hit
- System-scope atomics add ~50 cycles overhead vs device-scope due to cache invalidation protocol
- CTA-scope shared memory atomics: ~2 cycles base + ~2-4 cycles for CAS retry

### 2.5 Reusing existing designs

The key insight: **the protocol is identical; only the memory scope changes.** The state machine (EMPTY -> SENT -> consumed, or EMPTY -> CLOSED) is the same. The sequence-number-based MPSC ring buffer protocol is the same. Only these things change:

1. **Atomic instructions**: `sys_*` -> `cta_*` with `.shared` address space
2. **Memory backing**: global memory pointer -> shared memory offset (from `BlockScope::alloc`)
3. **Scope constraint**: `Send + Sync` becomes block-local (cannot escape `BlockScope<'scope>`)

This means we can factor the channel logic into a generic core parameterized by atomic operations:

```rust
trait ChannelAtomics {
    unsafe fn store_release_u32(ptr: *mut u32, val: u32);
    unsafe fn load_acquire_u32(ptr: *const u32) -> u32;
    unsafe fn cas_u32(ptr: *mut u32, expected: u32, desired: u32) -> u32;
}

struct SysAtomics;  // system-scope, global memory
struct CtaAtomics;  // CTA-scope, shared memory

// Then: OneshotSlot<T, A: ChannelAtomics> works for both transports
```

However, this adds a generic parameter to every channel type. Alternative: two concrete implementations (`BlockOneshotSlot` and `OneshotSlot`) sharing the same state machine logic via macros or inline helper functions. **Recommend the macro approach** for zero-cost abstraction without generic parameter pollution in the user API.

## 3. Global Memory Transport (Existing Channels)

### 3.1 Current implementation analysis

The existing channels are well-designed for global memory:
- `OneshotSlot<T>`: 8 + sizeof(T) bytes, acquire/release protocol, zero contention (single sender, single receiver)
- `MpscChannel<T, N>`: CAS-based head advancement, sequence-number publication, waker integration
- Both use `sys_*` (system-scope) atomics for maximum visibility (including GPU-CPU)

### 3.2 Latency profile on SM75

| Scenario | Latency |
|----------|---------|
| L2 cache hit (intra-SM, recent access) | ~30-50 cycles |
| L2 cache hit (cross-SM, different L1 partition) | ~50-80 cycles |
| L2 cache miss (DRAM) | ~200-400 cycles |
| System-scope atomic CAS (L2 hit) | ~80-120 cycles |
| System-scope atomic CAS (L2 miss) | ~250-400 cycles |

The `.sys` scope forces cache line invalidation across the entire memory hierarchy, including the CPU's cache coherence domain. This is necessary for GPU-CPU communication but **overkill for GPU-GPU communication**.

### 3.3 Optimization opportunities

**Opportunity 1: Use device-scope instead of system-scope for GPU-only channels.**

If both sender and receiver are GPU threads (no host involvement), we can use `.gpu` (device-scope) instead of `.sys`:
- PTX: `atom.cas.gpu.global.b32` instead of `atom.cas.sys.global.b32`
- Expected improvement: ~20-40% latency reduction (no CPU cache coherence overhead)
- Requires new primitives: `gpu_store_release_u32`, `gpu_load_acquire_u32`, `gpu_cas_u32`

**Opportunity 2: L2 cache residency hints.**

On SM75+, `ld.global.L2::cache_hint` can bias L2 eviction policy. For frequently-accessed channel metadata (head/tail pointers), the `.L2::128B` hint keeps the line resident. Not available in standard PTX ISA for SM75 — this is a Hopper (SM90) feature. **Not applicable to our GTX 1660.**

**Opportunity 3: Reduce system-scope atomic usage in MPSC.**

Currently `try_send` performs:
1. `sys_load_acquire` on `closed` flag
2. `sys_load_acquire` on `head`
3. `sys_load_acquire` on `tail`
4. `sys_cas_u32` on `head`
5. `sys_store_release` on slot sequence

If the channel is GPU-only, steps 1-4 can use device-scope, saving ~20 cycles each. Only the final publication (step 5) needs system-scope if the host reads.

**Recommendation:** Add `DeviceAtomics` alongside `SysAtomics` and `CtaAtomics`. For channels created within a `GridScope` (GPU-only), use device-scope. For channels visible to the host, use system-scope. This is a transparent optimization within the channel implementation.

## 4. Auto-Selection Heuristics

### 4.1 Option comparison

#### Option A: Runtime detection — check block_idx at channel creation

```rust
fn channel<T: Copy>(slot: &mut ChannelSlot<T>) -> (Sender<T>, Receiver<T>) {
    // At creation time, record the creator's block_idx
    let creator_block = block_idx_x();
    // ... later, at send/recv, check if current block matches
}
```

**Pros:**
- Maximum flexibility — channel can be created anywhere and transport is chosen at first use
- Single API surface: just `channel()`

**Cons:**
- **Runtime overhead on every send/recv:** Must check block_idx and dispatch to appropriate atomic instructions
- **Branch divergence:** The transport dispatch is a runtime branch that all lanes take — costs ~4 cycles per operation even when predictable
- **Cannot allocate in shared memory at runtime:** If the channel is created before knowing the scope, it must pessimistically use global memory (shared memory requires compile-time allocation via `BlockScope::alloc`)
- **Violates the lifetime model:** A shared-memory channel that gets passed to another block causes undefined behavior. Runtime checking can catch this (panic), but the damage may already be done.

**Verdict: Rejected.** The shared-memory allocation constraint makes this fundamentally unworkable. You cannot retroactively move a channel from global to shared memory.

#### Option B: Type-level — distinct types with enum wrapper

```rust
// Concrete types for each transport
struct BlockChannel<T: Copy> { /* shared memory backed */ }
struct GridChannel<T: Copy>  { /* global memory backed */ }

// Unified enum for polymorphic use
enum Channel<T: Copy> {
    Block(BlockChannel<T>),
    Grid(GridChannel<T>),
}
```

**Pros:**
- Zero-cost: no runtime dispatch for concrete types
- The compiler can optimize block-channel operations fully (known shared-memory atomics, inlineable)
- Type safety: `BlockChannel<T>` cannot be created outside a `BlockScope`

**Cons:**
- Two types for the user to learn
- Code that is generic over channel type requires the `Channel<T>` enum, adding one branch per operation
- Cannot change transport after creation

**Verdict: Viable as the low-level layer.** The concrete types provide zero-cost transport-specific channels for performance-critical code.

#### Option C: Scope-based — channel transport determined by creation scope

```rust
impl<'scope> BlockScope<'scope> {
    // Creates a shared-memory-backed channel
    fn oneshot<T: Copy>(&self) -> (Sender<'scope, T>, Receiver<'scope, T>);
    fn mpsc<T: Copy, const N: usize>(&self) -> (MpscSender<'scope, T, N>, MpscReceiver<'scope, T, N>);
}

impl<'scope> GridScope<'scope> {
    // Creates a global-memory-backed channel
    fn oneshot<T: Copy>(&self) -> (Sender<'scope, T>, Receiver<'scope, T>);
    fn mpsc<T: Copy, const N: usize>(&self) -> (MpscSender<'scope, T, N>, MpscReceiver<'scope, T, N>);
}
```

**Pros:**
- **Natural integration with structured concurrency:** The scope already knows the memory tier. `BlockScope` has shared memory; `GridScope` has global memory.
- **Lifetime safety for free:** `'scope` on sender/receiver prevents shared-memory references from escaping the block scope. The Rust borrow checker enforces transport correctness.
- **Zero API overhead:** The user calls `scope.oneshot()` — no need to specify transport. The scope picks the right one.
- **Composability:** A `BlockScope` inside a `GridScope` creates block-local channels; the `GridScope` itself creates grid-wide channels. Natural hierarchy.
- **Memory allocation is correct by construction:** `BlockScope::oneshot()` allocates from shared memory (via watermark allocator). `GridScope::oneshot()` allocates from global memory pool.

**Cons:**
- Channels cannot be created outside a scope (but standalone `oneshot()` and `mpsc()` remain for global-memory channels created outside any scope)
- Cannot upgrade a block channel to a grid channel if the scope changes

**Verdict: Recommended.** This is the cleanest design and integrates naturally with the structured concurrency model from sc-design.2.

#### Option D: Hybrid — scope-based creation + type-erased Channel<T>

Combine C with a `Channel<T>` enum for rare cases where the transport is not known statically:

```rust
enum ChannelTransport<T: Copy> {
    Block(BlockOneshotSlot<T>),
    Grid(OneshotSlot<T>),
}
```

This adds one branch per send/recv but provides a fallback for generic code.

**Verdict: Not needed initially.** The scope-based approach (Option C) covers all structured concurrency use cases. The `Channel<T>` enum can be added later if demand arises.

### 4.2 Recommended approach

**Option C (scope-based) as the primary API**, with these specifics:

1. `BlockScope::oneshot<T>()` -> shared-memory backed, CTA-scope atomics, `'scope`-bounded
2. `GridScope::oneshot<T>()` -> global-memory backed, device-scope atomics, `'scope`-bounded
3. Standalone `channel::oneshot()` -> global-memory backed, system-scope atomics (backward-compatible)
4. Same pattern for MPSC

**Why this handles the "channel passed to another block" question:**
- A `BlockScope`-created channel has lifetime `'scope` tied to the `BlockScope`.
- `BlockScope<'scope>` is block-local — the `'scope` lifetime does not outlive the block.
- If code tries to pass a `Sender<'scope, T>` from a `BlockScope` to another block (via `GridScope::spawn_block`), the closure bound `F: Send + 'scope` combined with `'scope` being shorter than `'static` causes a **compile error**. The Rust lifetime system prevents this misuse automatically.
- This is the same mechanism that prevents shared-memory references from escaping in sc-design.2.

```rust
// COMPILE ERROR: cannot pass block-scoped channel across block boundary
grid_scope(pool, pool_size, |gscope| {
    block_scope(|bscope| {
        let (tx, rx) = bscope.oneshot::<u32>();  // 'bscope lifetime

        gscope.spawn_block((), move |_| {
            // ERROR: tx has lifetime 'bscope, which doesn't outlive 'static
            // (spawn_block requires the closure to be Send + 'gscope,
            //  and 'bscope is shorter than 'gscope)
            unsafe { tx.send(42); }
        });
    });
});
```

## 5. Practical Concerns

### 5.1 Shared memory budget

GTX 1660 (SM75) has **48KB** (49152 bytes) of shared memory per block.

**Channel memory costs:**
| Channel Type | Memory (T=u32) | Memory (T=u64) | Memory (T=[f32;4]) |
|---|---|---|---|
| BlockOneshotSlot | 12 bytes | 16 bytes | 24 bytes |
| BlockMpscChannel, N=8 | 132 bytes | 196 bytes | 292 bytes |
| BlockMpscChannel, N=16 | 228 bytes | 356 bytes | 548 bytes |
| BlockMpscChannel, N=64 | 804 bytes | 1316 bytes | 2084 bytes |

**Budget allocation recommendation:**
- Reserve **4KB** (8% of 48KB) for channel infrastructure within a BlockScope
- This supports: ~40 oneshot channels, or ~5 MPSC channels with N=64, or a mix
- The remaining **44KB** is available for user data (`scope.alloc()`)
- The watermark allocator in sc-design.2 tracks this naturally — channels are just another allocation

**Important:** The shared memory allocator must account for channel metadata alignment. `MpscSlot<T>` requires 8-byte alignment for the `MaybeUninit<T>` field. The watermark allocator's `alloc_raw` already handles this via the `align` parameter.

### 5.2 Can oneshot channels fit in registers?

**For same-warp communication: yes, conceptually, but no practical benefit.**

A warp-internal "oneshot" where lane A sends to lane B is just:
```rust
let val_from_lane_a = shfl_sync_idx_u32(FULL_MASK, my_val, lane_a_id);
```
This is 1 cycle, zero memory, zero buffering. But it is NOT a channel — it is a synchronous exchange that requires both lanes to be at the same instruction.

For asynchronous oneshot (send now, receive later): **registers cannot store pending state across yield points.** When a GPU future yields (returns `Poll::Pending`), the executor context-switches to another task. Register values are not preserved across task switches — they are part of the warp's execution state, not the task's state. The task's state must be in memory (shared or global).

**Conclusion:** Oneshot channels require memory-backed storage. The smallest possible is a shared-memory `BlockOneshotSlot<u32>` at 12 bytes.

### 5.3 Fairness under contention

When K warps contend on a shared-memory MPSC's head pointer:

**CAS fairness analysis:**
- Each warp attempts `atom.cas.cta.shared.b32` on the head pointer
- On SM75, CAS operations to the same shared memory address are serialized by the shared memory unit
- The hardware does NOT guarantee FIFO ordering — the warp scheduler picks which warp's CAS succeeds
- Under sustained contention, a warp can be **starved** indefinitely (though unlikely in practice due to round-robin warp scheduling)

**Practical fairness properties:**
- SM75's warp scheduler uses a **round-robin** policy among eligible warps (warps not stalled on memory)
- Since CAS retry immediately re-reads and retries, all contending warps remain eligible
- In practice, with K=8 contending warps, each warp gets ~1/K throughput (approximately fair)
- Worst case: K CAS retries = K * ~4 cycles = ~32 cycles for K=8 — still far better than global memory

**Mitigation for high contention (>8 warps):**
1. **Backoff:** After failed CAS, execute `nanosleep.u32 N` where N doubles on each retry (exponential backoff). Reduces CAS contention.
2. **Warp-cooperative send:** If multiple lanes in the same warp want to send, use ballot + count-leading-zeros to serialize within the warp before attempting the MPSC CAS. Reduces CAS operations from 32 (one per lane) to 1 (per warp).
3. **Partitioned channels:** Instead of one MPSC with K producers, use K SPSCs (one per warp) — eliminates contention entirely. The consumer round-robins across SPSCs.

## 6. Recommended Channel Architecture

### 6.1 Three-tier architecture

```
Tier        Backing           Atomics         Latency     Scope
----------- ----------------- --------------- ----------- -----------
Warp        (not a channel)   shfl.sync       ~1 cycle    warp-internal
Block       shared memory     atom.cta.shared ~2-6 cycles BlockScope
Grid/Global global memory     atom.gpu.global ~50-100 cy  GridScope
Host-vis    global memory     atom.sys.global ~80-200 cy  standalone
```

### 6.2 API surface

```rust
// === Block-scoped channels (shared memory) ===
impl<'scope> BlockScope<'scope> {
    /// Create a oneshot channel in shared memory.
    /// Returns sender + receiver bounded by 'scope.
    pub fn oneshot<T: Copy>(&self) -> (BlockSender<'scope, T>, BlockReceiver<'scope, T>);

    /// Create an MPSC channel in shared memory.
    /// N must be a power of 2. Consumes N * (8 + sizeof(T)) + 36 bytes of shared memory.
    pub fn mpsc<T: Copy, const N: usize>(&self) -> (BlockMpscSender<'scope, T, N>, BlockMpscReceiver<'scope, T, N>);
}

// === Grid-scoped channels (global memory, device-scope) ===
impl<'scope> GridScope<'scope> {
    /// Create a oneshot channel in global memory (device-scope atomics).
    pub fn oneshot<T: Copy>(&self) -> (GridSender<'scope, T>, GridReceiver<'scope, T>);

    /// Create an MPSC channel in global memory.
    pub fn mpsc<T: Copy, const N: usize>(&self) -> (GridMpscSender<'scope, T, N>, GridMpscReceiver<'scope, T, N>);
}

// === Standalone channels (global memory, system-scope) — existing API, unchanged ===
pub fn oneshot<T: Copy>(slot: &mut OneshotSlot<T>) -> (OneshotSender<T>, OneshotReceiver<T>);
pub fn mpsc<T: Copy, const N: usize>(ch: &MpscChannel<T, N>) -> (MpscSender<T, N>, MpscReceiver<T, N>);
```

### 6.3 Implementation plan

**Phase 1: CTA-scope atomic primitives** (prerequisite)
- Add to `gpu-atomics`: `cta_store_release_shared_u32`, `cta_load_acquire_shared_u32`, `cta_cas_shared_u32`
- These emit `.cta.shared` scoped PTX instructions
- Test on GTX 1660 to verify SM75 supports these qualifiers

**Phase 2: Block-scoped oneshot** (simplest channel)
- `BlockOneshotSlot<T>` in shared memory, allocated by `BlockScope::alloc`
- Same state machine as `OneshotSlot<T>`, swapped to CTA-scope atomics
- `BlockSender<'scope, T>` and `BlockReceiver<'scope, T>` with `'scope` lifetime
- `BlockReceiver` implements `Future` with CTA-scope polling

**Phase 3: Block-scoped MPSC** (multi-producer)
- `BlockMpscChannel<T, N>` in shared memory
- CAS contention mitigation: warp-cooperative send (ballot + elect)
- Waker integration: store waker in shared memory (CTA-scope)

**Phase 4: Grid-scoped channels** (device-scope optimization)
- New device-scope atomics: `gpu_store_release_u32`, `gpu_load_acquire_u32`, `gpu_cas_u32`
- `GridOneshotSlot<T>` and `GridMpscChannel<T, N>` using device-scope (not system-scope)
- Allocated from `GridScope`'s global memory pool

**Phase 5: Unified Future integration**
- Ensure all channel receivers (`BlockReceiver`, `GridReceiver`, `OneshotReceiver`) implement the same `Future` trait
- The executor's `WarpFuture` polling loop works identically regardless of channel transport — the `poll()` method abstracts the atomic scope

### 6.4 What the implementer needs to know

1. **SM75 CTA-scope shared-memory atomics are supported.** PTX ISA 7.0+ (Turing) supports `atom.cas.cta.shared.b32`, `ld.acquire.cta.shared.u32`, and `st.release.cta.shared.u32`. The GTX 1660 (SM75) supports PTX ISA 6.4, which includes `.cta` scope for shared memory atomics. **Verify this experimentally before committing** — the PTX documentation is sometimes ambiguous about which ISA version introduced which scope+space combination.

2. **Shared memory pointer conversion.** `block::shared_mem_at::<T>(offset)` returns a generic pointer. For CTA-scope atomics, the PTX address space must be `.shared`. The `cvta.shared.u64` instruction (already used in `block.rs`) converts a shared-memory offset to a generic pointer. For inline PTX, use the `.shared` qualifier directly on the atomic instruction, which forces shared-memory addressing.

3. **Waker storage in shared memory.** The existing `MpscChannel::store_waker` copies 16 bytes of waker data. For `BlockMpscChannel`, this waker must be stored in shared memory. Since the executor's waker is a packed `(work_queue_ptr, task_id)` in global memory, the waker pointer itself lives in global memory — only the cached copy in the channel needs to be in shared memory. This is fine because `wake()` on the reconstructed waker enqueues to the global work queue.

4. **No `Drop` in shared memory.** `T: Copy` is already required, so there are no destructors. The watermark allocator reclaims all shared memory at scope exit without per-element cleanup.

5. **Thread safety model.** Block-scoped channels are `!Send` across block boundaries but `Send` within the block (across warps). This is enforced by the `'scope` lifetime bound — not by `Send`/`Sync` trait implementations. The `BlockSender`/`BlockReceiver` types can be `Send + Sync` because all intra-block access is correct; the `'scope` lifetime prevents cross-block misuse.

## Files Changed: none
