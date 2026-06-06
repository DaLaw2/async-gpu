# coro-design.2: GpuGenerator Trait + MIR Pass Extension Design

## Summary

This design specifies the `GpuGenerator<Y, R>` trait, `WarpCoroutineState<Y, R>` enum, `WarpBroadcast<T>` trait for yield-value propagation, executor integration via `GeneratorTask`, and the zero-buffering streaming pipeline API. The MIR pass (`WarpCooperativeTransform`) requires **no changes** for basic generator support — the existing discriminant broadcast and return barrier already handle generator coroutine bodies correctly. The streaming pipeline is best expressed as a direct `for_each_yield` combinator on the `GpuGenerator` trait itself.

## 1. GpuGenerator Trait

### Core Types

```rust
/// Result of resuming a warp-cooperative generator.
/// Mirror of `core::ops::CoroutineState` but warp-safe:
/// all 32 lanes observe the same variant after resume_warp().
pub enum WarpCoroutineState<Y, R> {
    /// Generator yielded a value. All lanes see the same Y.
    Yielded(Y),
    /// Generator completed with a return value. All lanes see the same R.
    Complete(R),
}
```

### Trait Definition

```rust
/// A generator representing an entire warp (32 lanes) in SIMT lockstep.
///
/// # Contract
/// - All active lanes must call `resume_warp()` simultaneously.
/// - The coroutine state discriminant is broadcast from lane 0 to all lanes
///   via `shfl.sync.idx.b32` (handled by the MIR pass automatically).
/// - Yielded values are broadcast from lane 0 to all lanes via WarpBroadcast<Y>.
/// - The resume argument `arg` is only consumed by lane 0; all lanes must
///   pass the same value (or a dummy — only lane 0's value is used).
///
/// # Safety
/// Implementing this trait requires maintaining warp convergence.
/// Breaking convergence causes deadlock or incorrect results.
pub unsafe trait GpuGenerator<R = ()> {
    /// Type of values yielded at each suspension point.
    type Yield: WarpBroadcast + Copy;

    /// Type of the final return value when the generator completes.
    type Return: WarpBroadcast + Copy;

    /// Resume the generator. Called by all active lanes simultaneously.
    ///
    /// Lane 0 calls the inner `Coroutine::resume(arg)` and obtains a
    /// `CoroutineState<Y, R>`. The discriminant (Yielded vs Complete)
    /// is broadcast via shfl.sync. The payload (Y or R) is broadcast
    /// via `WarpBroadcast::broadcast()`. All lanes return the same
    /// `WarpCoroutineState<Y, R>`.
    fn resume_warp(
        &mut self,
        arg: R,
        wcx: &mut WarpContext,
    ) -> WarpCoroutineState<Self::Yield, Self::Return>;
}
```

### Design Rationale

1. **Models after `WarpFuture`**: The trait mirrors `WarpFuture::poll_warp(&mut self, wcx) -> WarpPoll<T>` — same pattern of lane-0 execution + broadcast. The key difference is that `resume_warp` returns a two-variant enum (Yielded/Complete) instead of (Ready/Pending).

2. **Resume argument `R` defaults to `()`**: For the streaming pipeline use case (epic success criteria 4), the producer generates values without needing input from the consumer. Resume args with actual data are a T2+ extension.

3. **`Copy` bound on `Yield` and `Return`**: Required because values are broadcast via `shfl.sync` or shared memory, which are bitwise copy operations. This matches the existing `Copy` requirement on channel value types (`OneshotSlot<T: Copy>`).

4. **No `Pin`**: Unlike `Future::poll` which takes `Pin<&mut Self>`, the generator takes `&mut self`. On GPU, generators live in global memory task slots with fixed addresses — they are never moved. `Pin` adds complexity with no safety benefit in this context.

5. **`unsafe trait` (not `unsafe fn`)**: The trait itself is unsafe to implement (like `WarpFuture`) because implementors must maintain warp convergence. Individual calls to `resume_warp` don't need additional `unsafe` — the convergence contract is on the implementation.

## 2. WarpBroadcast Trait — Yield Value Propagation

### Trait Definition

```rust
/// Broadcast a value from lane 0 to all lanes in a warp.
///
/// Implementations use the most efficient hardware path:
/// - Scalars (<=32 bits): single `shfl.sync.idx.b32`
/// - Multi-word (33-128 bits): multiple `shfl.sync.idx.b32` calls
/// - Large types (>128 bits): shared memory write + barrier + read
///
/// # Safety
/// All active lanes must call `broadcast()` simultaneously with the
/// same `mask`. Only lane 0's `value` is used; other lanes' values
/// are overwritten with lane 0's.
pub unsafe trait WarpBroadcast: Copy {
    /// Broadcast this value from lane 0 to all lanes.
    /// Returns the broadcast value (same on all lanes).
    fn broadcast(value: Self, mask: u32) -> Self;
}
```

### Built-in Implementations

```rust
// Single shfl.sync — 1 cycle
unsafe impl WarpBroadcast for u8  { ... } // widen to u32, shfl, truncate
unsafe impl WarpBroadcast for u16 { ... } // widen to u32, shfl, truncate
unsafe impl WarpBroadcast for u32 { ... } // direct shfl.sync.idx.b32
unsafe impl WarpBroadcast for i32 { ... } // transmute to u32, shfl, transmute back
unsafe impl WarpBroadcast for f32 { ... } // transmute to u32, shfl, transmute back
unsafe impl WarpBroadcast for bool { ... } // u32 encoding (0/1)

// Two shfl.sync calls — 2 cycles
unsafe impl WarpBroadcast for u64 { ... } // lo/hi halves (existing pattern from warp_hostcall_wait_u64)
unsafe impl WarpBroadcast for i64 { ... }
unsafe impl WarpBroadcast for f64 { ... }

// Four shfl.sync calls — 4 cycles
unsafe impl WarpBroadcast for u128 { ... } // four 32-bit words
unsafe impl WarpBroadcast for [u32; 2] { ... }
unsafe impl WarpBroadcast for [u32; 3] { ... }
unsafe impl WarpBroadcast for [u32; 4] { ... }
unsafe impl WarpBroadcast for (u32, u32) { ... }

// Unit type — no broadcast needed (0 bytes)
unsafe impl WarpBroadcast for () {
    fn broadcast(_value: (), _mask: u32) -> () { () }
}
```

### Shared Memory Fallback

For types > 128 bits (rare but possible), the broadcast goes through shared memory:

```rust
// Generic fallback for types > 128 bits
// Lane 0 writes to __shared__ temp buffer
// bar.warp.sync
// All lanes read from __shared__ temp buffer
//
// Not implemented as a blanket impl (would conflict with scalar impls).
// Users with large yield types implement WarpBroadcast manually using
// the `warp_broadcast_via_smem` helper function.
pub unsafe fn warp_broadcast_via_smem<T: Copy>(
    value: T,
    mask: u32,
    smem_slot: *mut T,  // caller provides shared memory slot
) -> T;
```

### Design Rationale

1. **Separate trait, not baked into GpuGenerator**: `WarpBroadcast` is useful beyond generators — any warp-cooperative code that needs to broadcast a value benefits. The existing `broadcast_u32` function in `warp_future.rs` is a special case of this trait for `u32`.

2. **No shared memory in the default path**: The GTX 1660 (sm_75) has 48KB shared memory, but accessing it costs ~5 cycles vs ~1 cycle for `shfl.sync`. Since most yield types will be small scalars or tuples, the register-based path is the default. The shared memory path is an opt-in for large types.

3. **`Copy` bound is sufficient**: We don't need `Send`/`Sync` bounds on the broadcast trait. The value is being broadcast within a single warp — all lanes are in the same thread block and share the same memory space.

## 3. MIR Pass Assessment

### Verdict: NO CHANGES NEEDED

The existing `WarpCooperativeTransform` already handles generator coroutine bodies correctly. Here is the detailed analysis:

**What happens when a `#[coroutine]` closure is compiled for nvptx64:**

1. `StateTransform` runs first: rewrites `yield val` to `_0 = CoroutineState::Yielded(val); discriminant = N; return;`. This erases all `Yield` terminators.

2. `WarpCooperativeTransform` runs second: sees `body.coroutine.is_some()`, processes the body.
   - **Discriminant broadcast** (Phase 2): The dispatch `SwitchInt` on the discriminant gets `activemask` + `shfl.sync.idx.b32` inserted before it. This ensures all 32 lanes agree on which state the generator is in (Unresumed, Yielded at point K, Returned, etc.). **Works unchanged for generators.**
   - **Return barrier** (Phase 4): Every `Return` terminator gets `activemask` + `bar.warp.sync` inserted before it. After `StateTransform`, `yield` points become `Return` terminators, so they get the barrier too. **Works unchanged for generators.**

3. The `CoroutineAnalysis` already tracks `yield_points` (pre-StateTransform) and `return_blocks` (post-StateTransform). It emits diagnostics noting the counts. **Works unchanged.**

**Why no MIR pass changes for yield-value broadcast:**

The yield value (`CoroutineState::Yielded(val)`) is written to the return place (`_0`) by `StateTransform`. This is a regular `Assign` statement in the MIR. The broadcast of this value is a **runtime** concern, not a compiler concern:

- The `GpuGenerator::resume_warp()` method inspects the return value on lane 0 and broadcasts it using `WarpBroadcast<Y>::broadcast()`.
- This is exactly analogous to how `warp_poll_future()` encodes `Poll<bool>` as a `u32` result code and broadcasts it via `shfl.sync`.
- No compiler magic needed — the trait implementation handles it.

**Edge cases considered:**

| Edge Case | Handled? | How |
|-----------|----------|-----|
| Multiple yield points | Yes | StateTransform creates distinct discriminant values (3, 4, 5, ...) for each yield. Discriminant broadcast ensures all lanes agree. |
| Generator in async context | Yes | `async { gen_block }` creates nested coroutines. StateTransform processes each independently. WarpCooperativeTransform processes each body independently. |
| Nested generators | Yes | Each generator is a separate coroutine body with its own state machine. No interaction at the MIR level. |
| Generator with non-trivial state | Yes | StateTransform captures all locals live across yield points into the coroutine state struct. This is transparent to WarpCooperativeTransform. |
| Panic in generator | Yes | Existing panic handler sends message via hostcall. The discriminant state 2 (Panicked) is broadcast like any other. |

### One Possible Future Enhancement (NOT required for this epic)

If profiling shows that broadcasting large yield values is a bottleneck, the MIR pass could be extended to insert `shfl.sync` instructions directly after the discriminant broadcast, targeting the yield value locals. This would bypass the runtime trait dispatch. However:
- The runtime approach is simpler and more maintainable
- For <=128-bit types, `shfl.sync` is 1-4 cycles — negligible vs the generator's compute work
- This optimization can be added later without API changes

## 4. Executor Integration

### Approach: GeneratorTask Wrapper

The existing `GpuExecutor` works with `Future<Output = ()>`. Rather than adding a second task kind, generators are wrapped in a `Future` adapter that drives the generator and processes yielded values:

```rust
/// Wraps a GpuGenerator + consumer closure into a Future<Output = ()>
/// that the GpuExecutor can schedule.
///
/// Each poll drives one resume_warp + consumer call. When the generator
/// yields, the consumer processes the value and the task returns Pending
/// (to be re-polled). When the generator completes, the task returns Ready.
pub struct GeneratorTask<G, F>
where
    G: GpuGenerator,
    F: FnMut(G::Yield),
{
    generator: G,
    consumer: F,
    completed: bool,
}

impl<G, F> Future for GeneratorTask<G, F>
where
    G: GpuGenerator,
    F: FnMut(G::Yield),
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.completed {
            return Poll::Ready(());
        }
        let this = unsafe { self.get_unchecked_mut() };
        let mut wcx = unsafe { WarpContext::new() };

        match this.generator.resume_warp((), &mut wcx) {
            WarpCoroutineState::Yielded(value) => {
                (this.consumer)(value);
                Poll::Pending  // come back for more
            }
            WarpCoroutineState::Complete(_) => {
                this.completed = true;
                Poll::Ready(())
            }
        }
    }
}
```

### Why Not a Separate Generator Queue?

1. **Reuse**: The existing `WorkQueue`, `TaskSlot`, and `FreeSlotStack` infrastructure works unchanged. No new lock-free data structures needed.
2. **Coexistence**: Generators and futures can coexist in the same executor. A warp might poll a future task, then a generator task, then another future — all from the same work queue.
3. **Wakers**: Generator tasks that yield return `Poll::Pending`, which parks them. The executor's sweep re-enqueues parked tasks, so generators naturally get re-polled.

### Spawn Helper

```rust
impl GpuExecutor {
    /// Spawn a generator task that feeds yielded values to a consumer.
    ///
    /// The generator is resumed repeatedly. Each yielded value is passed
    /// to `consumer`. When the generator completes, the task finishes.
    ///
    /// # Safety
    /// Same as `spawn()` — must be called from lane 0, executor must
    /// be in global memory.
    pub unsafe fn spawn_generator<G, F>(
        &self,
        generator: G,
        consumer: F,
    ) -> Result<TaskId, ExecutorError>
    where
        G: GpuGenerator + 'static,
        F: FnMut(G::Yield) + 'static,
    {
        self.spawn(GeneratorTask {
            generator,
            consumer,
            completed: false,
        })
    }
}
```

## 5. Streaming Pipeline API — Zero-Buffering Producer-Consumer

### Design: `for_each_yield` Combinator

The simplest and most efficient streaming pipeline is a direct inline loop — the producer yields, the consumer processes, all within one warp. No channels, no buffering, no inter-task coordination.

```rust
/// Drive a generator to completion, calling `consumer` on each yielded value.
/// All 32 lanes participate — the yielded value is broadcast to all lanes
/// before the consumer runs, enabling data-parallel consumption.
///
/// # Zero Buffering
/// There is no intermediate buffer. The producer yields one value,
/// the consumer processes it, then the producer yields the next.
/// At any point in time, at most ONE yielded value exists.
///
/// # Safety
/// All active lanes must call this simultaneously.
pub unsafe fn for_each_yield<G, F>(
    generator: &mut G,
    mut consumer: F,
    wcx: &mut WarpContext,
) -> G::Return
where
    G: GpuGenerator,
    F: FnMut(G::Yield, &mut WarpContext),
{
    loop {
        match generator.resume_warp((), wcx) {
            WarpCoroutineState::Yielded(value) => {
                consumer(value, wcx);
            }
            WarpCoroutineState::Complete(ret) => {
                return ret;
            }
        }
        syncwarp(wcx.active_mask);
    }
}
```

### User-Facing API Example

```rust
// ---- Producer: yields fibonacci numbers ----
#[coroutine]
fn fibonacci_gen() -> impl Coroutine<(), Yield = u32, Return = ()> {
    let (mut a, mut b) = (0u32, 1u32);
    for _ in 0..20 {
        yield a;
        let next = a + b;
        a = b;
        b = next;
    }
}

// ---- Consumer: processes each value with all 32 lanes ----
// In a kernel:
unsafe {
    let mut wcx = WarpContext::new();
    let mut gen = FibonacciWarpGenerator::new(fibonacci_gen());

    for_each_yield(&mut gen, |value, wcx| {
        // All 32 lanes see the same `value`
        // Each lane can do different work on it (SIMD)
        let lane_result = value + wcx.lane_id;
        output[wcx.lane_id as usize] = lane_result;
    }, &mut wcx);
}
```

### Pipe Combinator (Convenience, Not Required for MVP)

```rust
/// Chain a producer generator into a consumer generator.
/// The consumer receives each yielded value from the producer
/// as its resume argument.
///
/// This is sugar over for_each_yield where the consumer is itself
/// a generator that yields transformed values.
pub unsafe fn pipe<P, C>(
    producer: &mut P,
    consumer: &mut C,
    wcx: &mut WarpContext,
) -> C::Return
where
    P: GpuGenerator<(), Yield = C::Resume>,
    C: GpuGenerator<P::Yield>,
{
    // ...
}
```

This pipe combinator is T2+ — requires non-`()` resume arguments. For the MVP demo, `for_each_yield` with an inline closure is sufficient and clearer.

## Trade-offs and Alternatives

### Alternative A: Per-Lane Yield Values

Instead of broadcasting the same value to all lanes, each lane could yield a different value. This would be useful for data-parallel generators where lane K produces element K.

**Rejected for MVP because:**
- The epic's North Star says "producers yield values to consumers with zero buffering" — singular yield, not per-lane
- Per-lane yield changes the semantics fundamentally: the generator state machine must run on all lanes, not just lane 0
- This would require MIR pass changes (no longer broadcast discriminant from lane 0)
- Can be added as a separate `DataParallelGenerator` trait in T2+

### Alternative B: Generator as a Separate Executor Task Kind

Instead of wrapping generators in `Future`, add a `GeneratorSlot` variant to the task slot with its own `resume_fn` pointer and yield value storage.

**Rejected because:**
- Doubles the executor complexity (two task kinds, two poll paths)
- The Future adapter (`GeneratorTask`) achieves the same result with zero new infrastructure
- Adding a generator-specific task kind can be done later as an optimization if needed

### Alternative C: Yield via Channel

Use the existing `MpscChannel` for producer-consumer streaming.

**Rejected because:**
- Adds 64-slot ring buffer (not zero-buffering)
- Requires two task slots (producer + consumer) instead of one
- Higher latency: CAS on head pointer + sequence number store + waker wake
- The direct `for_each_yield` loop is simpler, faster, and truly zero-buffered

## Open Questions

1. **`gen fn` vs `#[coroutine]` syntax**: Rust's `gen fn` uses `Iterator`/`Option<Y>` semantics, while `#[coroutine]` uses `CoroutineState<Y, R>`. Should `GpuGenerator` support both? For now, use raw `#[coroutine]` — it's more expressive (has `Return` type) and maps directly to `CoroutineState`.

2. **Generator lifetime and `Drop`**: Generators captured across yield points may hold references. On GPU, all memory is either global or shared with known lifetimes. The `Copy` bound on `Yield`/`Return` prevents yield-value lifetime issues, but the generator state itself may contain non-Copy types. This needs investigation if we support complex generator bodies.

3. **Warp divergence on early return**: If the generator returns `Complete` while a consumer expected more values, all lanes must agree (guaranteed by discriminant broadcast). But the consumer's cleanup path must also be convergent. The `for_each_yield` function handles this naturally (all lanes see `Complete` and exit the loop together).

4. **Interaction with async generators**: Rust supports `async gen` blocks that can both `yield` and `.await`. A `WarpAsyncGenerator` that combines `GpuGenerator` yield semantics with `WarpFuture` await semantics is feasible but T2+ scope.

## Impact on Downstream Tasks

- **coro-design.3** (MIR pass extension): **No MIR pass changes needed** for basic generators. Task can focus on verification — compile a `#[coroutine]` closure for nvptx64, inspect the MIR, confirm discriminant broadcast and return barrier are present. If the task is still scoped for "extension", it can add optional diagnostics (e.g., "generator has N yield points with type Y").

- **coro-design.4** (streaming pipeline demo): Implement `GpuGenerator` trait, `WarpBroadcast` for `u32`/`u64`, `WarpCoroutineState`, and `for_each_yield`. Write a kernel with a `#[coroutine]` producer that yields values consumed by an inline closure. The fibonacci example above is a concrete starting point. The demo should produce visible output (print each yielded value) to prove zero-buffering.

- **Multi-generator theme**: `GeneratorTask` adapter enables multiple generators in the same `GpuExecutor` run — each generator is just another `Future<Output=()>` in the work queue. No additional infrastructure needed.
