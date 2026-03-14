# rustc-warp.2: MIR transformation spec for warp-cooperative async fn
**Cycle**: 241 | **Theme**: rustc-warp | **Kind**: design | **Status**: done

## Summary

Specifies a MIR-level transformation that converts standard per-thread async fn state machines into warp-cooperative state machines with SIMT-convergent execution. The pass operates after rustc's `StateTransform` and inserts `shfl.sync.idx.b32` broadcasts for discriminant and per-state fields, `bar.warp.sync` barriers at yield boundaries, and leader/follower predication for exclusive poll logic. The transformation is sound for all coroutine state machines that do not contain self-referential borrows across yield points or trait-object futures.

## Findings

### MIR Structure for async fn

The following MIR is the actual rustc output (nightly 2025-08-25, `--target nvptx64-nvidia-cuda`, release, `-Zbuild-std=core`) for `one_yield`:

```rust
async fn one_yield(x: u32) -> u32 {
    let y = core::future::poll_fn(|_cx| core::task::Poll::Ready(x + 1)).await;
    y * 2
}
```

#### Constructor (creates coroutine struct)

```mir
fn one_yield(_1: u32) -> {async fn body of warp::one_yield()} {
    bb0: {
        _0 = {coroutine@src\warp.rs:818:35: 821:2 (#0)} { x: copy _1 };
        return;
    }
}
```

The coroutine struct layout (inferred from MIR access patterns):

```
{async fn body of warp::one_yield()} = {
    .0: u32,                    // upvar: x (captured from parameter)
    discriminant: u32,          // state tag
    // --- variant#3 fields (Suspended at yield point 0) ---
    variant#3.0: u32,           // x (moved into suspension frame)
    variant#3.1: PollFn<...>,   // __awaitee (inner future, stored across yield)
}
```

Discriminant values:
- `0` = Unresumed (initial state, not yet polled)
- `1` = Returned (completed, `Poll::Ready` delivered)
- `3` = Suspended at yield point 0 (awaiting inner future)

(Value `2` = Poisoned exists but is unused when `unwind = unreachable`.)

#### Poll function (StateTransform output)

```mir
fn one_yield::{closure#0}(
    _1: Pin<&mut {async fn body of warp::one_yield()}>,
    _2: &mut Context<'_>
) -> Poll<u32> {
    let mut _14: &mut {async fn body};  // deref of Pin

    bb0: {                                          // ENTRY: dispatch on discriminant
        _14 = copy (_1.0: &mut {async fn body});
        _13 = discriminant((*_14));
        switchInt(move _13) -> [0: bb1, 1: bb11, 3: bb10, otherwise: bb7];
    }

    bb1: {                                          // STATE 0: Unresumed
        // Move upvar x into variant#3 suspension frame
        (((*_14) as variant#3).0: u32) = copy ((*_14).0: u32);
        // Create inner future: poll_fn(|_cx| Ready(x + 1))
        _6 = &(((*_14) as variant#3).0: u32);
        _5 = {closure} { x: copy _6 };
        _4 = poll_fn(move _5);
        // -> bb2 -> bb3: store __awaitee into variant#3.1
    }

    bb3: {
        (((*_14) as variant#3).1: PollFn<...>) = move _3;  // store inner future
        goto -> bb4;
    }

    bb4: {                                          // POLL INNER FUTURE
        _9 = &mut (((*_14) as variant#3).1: PollFn<...>);
        _8 = Pin::new_unchecked(move _9);
        _7 = <PollFn<...> as Future>::poll(move _8, move _2);
        // -> bb6: check result
    }

    bb6: {                                          // CHECK INNER POLL RESULT
        _10 = discriminant(_7);
        switchInt(move _10) -> [0: bb9, 1: bb8, otherwise: bb7];
        //                      Ready    Pending
    }

    bb8: {                                          // INNER RETURNED Pending
        _0 = Poll::Pending;
        discriminant((*_14)) = 3;                   // SET STATE = Suspended(0)
        return;                                     // *** YIELD POINT ***
    }

    bb9: {                                          // INNER RETURNED Ready
        _11 = copy ((_7 as Ready).0: u32);          // extract inner result
        _12 = Mul(copy _11, const 2_u32);           // y * 2
        _0 = Poll::Ready(move _12);
        discriminant((*_14)) = 1;                   // SET STATE = Returned
        return;                                     // *** COMPLETION ***
    }

    bb10: {                                         // STATE 3: Suspended(0) resume
        goto -> bb4;                                // re-poll inner future
    }

    bb11: {                                         // STATE 1: Returned (panic on re-poll)
        assert(const false, "`async fn` resumed after completion");
    }
}
```

### Transformation Specification

The `WarpCooperativeTransform` MIR pass operates on the poll function output of `StateTransform`. It transforms **per-thread** state machine logic into **warp-cooperative** logic where lane 0 is the leader and lanes 1-31 are followers.

#### Terminology

- **Leader**: Lane 0 of the warp. Executes poll logic, writes state.
- **Follower**: Lanes 1-31. Skip poll logic, receive state via `shfl.sync`.
- **Broadcast**: `shfl.sync.idx.b32(mask=0xFFFFFFFF, val, src_lane=0)` — all lanes read lane 0's value.
- **Barrier**: `bar.warp.sync(mask=0xFFFFFFFF)` — synchronize all warp lanes.

#### Intrinsic Signatures (MIR-level)

```mir
// Broadcast a u32 from lane 0 to all lanes
fn warp_broadcast_u32(mask: u32, val: u32) -> u32;
    // Emits: shfl.sync.idx.b32 result, val, 0, 0x1F1F, mask;

// Synchronize all lanes in warp
fn warp_sync(mask: u32);
    // Emits: bar.warp.sync mask;

// Get current lane ID
fn warp_lane_id() -> u32;
    // Emits: mov.u32 result, %laneid;
```

#### Transformed Poll Function (Before/After)

**INPUT** (standard StateTransform output — shown above)

**OUTPUT** (after WarpCooperativeTransform):

```mir
fn one_yield::{closure#0}_warp_cooperative(
    _1: Pin<&mut {async fn body of warp::one_yield()}>,
    _2: &mut Context<'_>
) -> Poll<u32> {
    let _14: &mut {async fn body};
    let _lane_id: u32;
    let _is_leader: bool;
    let _mask: u32;
    let _bcast_disc: u32;     // broadcast discriminant
    let _bcast_poll: u32;     // broadcast inner poll result discriminant
    let _bcast_val: u32;      // broadcast value (inner Ready payload)
    let _bcast_result: u32;   // broadcast final result

    bb0: {                                          // ENTRY
        _mask = const 0xFFFF_FFFF_u32;
        _lane_id = warp_lane_id();
        _is_leader = Eq(copy _lane_id, const 0_u32);

        // --- PHASE 1: Leader reads discriminant, broadcasts to all ---
        _14 = copy (_1.0: &mut {async fn body});
        switchInt(copy _is_leader) -> [true: bb_leader_read_disc, false: bb_follower_recv_disc];
    }

    bb_leader_read_disc: {
        _13 = discriminant((*_14));
        _bcast_disc = warp_broadcast_u32(copy _mask, move _13);
        goto -> bb_dispatch;
    }

    bb_follower_recv_disc: {
        _bcast_disc = warp_broadcast_u32(copy _mask, const 0_u32);
        // Follower's input value is ignored; receives leader's value
        goto -> bb_dispatch;
    }

    bb_dispatch: {
        // All lanes now have identical _bcast_disc
        switchInt(move _bcast_disc) -> [0: bb1, 1: bb11, 3: bb10, otherwise: bb7];
    }

    // ================================================================
    // STATE 0: Unresumed — leader creates inner future, polls it
    // ================================================================

    bb1: {
        // Only leader executes state mutation
        switchInt(copy _is_leader) -> [true: bb1_leader, false: bb1_join];
    }

    bb1_leader: {
        // Move upvar, create inner future, store in variant#3
        (((*_14) as variant#3).0: u32) = copy ((*_14).0: u32);
        _6 = &(((*_14) as variant#3).0: u32);
        _5 = {closure} { x: copy _6 };
        _4 = poll_fn(move _5);
        // ... (bb2, bb3 as original) ...
        (((*_14) as variant#3).1: PollFn<...>) = move _3;
        // Poll inner future
        _9 = &mut (((*_14) as variant#3).1: PollFn<...>);
        _8 = Pin::new_unchecked(move _9);
        _7 = <PollFn<...> as Future>::poll(move _8, move _2);
        // Check inner result
        _10 = discriminant(_7);
        goto -> bb1_join;
    }

    bb1_join: {
        // Barrier: ensure leader has completed polling
        warp_sync(copy _mask);

        // --- PHASE 2: Broadcast inner poll result discriminant ---
        // Leader has _10 (0=Ready, 1=Pending); followers get it via broadcast
        _bcast_poll = warp_broadcast_u32(copy _mask,
            switchInt(copy _is_leader) -> [true: copy _10, false: const 0_u32]);
        switchInt(move _bcast_poll) -> [0: bb9_warp, 1: bb8_warp, otherwise: bb7];
    }

    // ================================================================
    // YIELD POINT: Inner returned Pending → all lanes return Pending
    // ================================================================

    bb8_warp: {
        // Leader sets discriminant = 3 (Suspended)
        switchInt(copy _is_leader) -> [true: bb8_leader, false: bb8_follower];
    }

    bb8_leader: {
        discriminant((*_14)) = 3;
        goto -> bb8_done;
    }

    bb8_follower: {
        goto -> bb8_done;
    }

    bb8_done: {
        warp_sync(copy _mask);              // Barrier before return
        _0 = Poll::Pending;
        return;
    }

    // ================================================================
    // COMPLETION: Inner returned Ready → broadcast result, all return Ready
    // ================================================================

    bb9_warp: {
        // --- PHASE 3: Broadcast the Ready payload ---
        // Leader extracts _11 = inner Ready value; broadcasts to all lanes
        switchInt(copy _is_leader) -> [true: bb9_leader, false: bb9_follower];
    }

    bb9_leader: {
        _11 = copy ((_7 as Ready).0: u32);
        _bcast_val = warp_broadcast_u32(copy _mask, copy _11);
        goto -> bb9_compute;
    }

    bb9_follower: {
        _bcast_val = warp_broadcast_u32(copy _mask, const 0_u32);
        goto -> bb9_compute;
    }

    bb9_compute: {
        // All lanes compute the final value uniformly
        _12 = Mul(copy _bcast_val, const 2_u32);
        _bcast_result = warp_broadcast_u32(copy _mask, move _12);

        // Leader writes completion state
        switchInt(copy _is_leader) -> [true: bb9_leader_done, false: bb9_follower_done];
    }

    bb9_leader_done: {
        discriminant((*_14)) = 1;           // SET STATE = Returned
        goto -> bb9_all_done;
    }

    bb9_follower_done: {
        goto -> bb9_all_done;
    }

    bb9_all_done: {
        warp_sync(copy _mask);
        _0 = Poll::Ready(copy _bcast_result);
        return;
    }

    // ================================================================
    // STATE 3: Suspended(0) resume — re-poll inner future
    // ================================================================

    bb10: {
        // Only leader re-polls; same pattern as bb1
        switchInt(copy _is_leader) -> [true: bb10_leader, false: bb10_join];
    }

    bb10_leader: {
        _9 = &mut (((*_14) as variant#3).1: PollFn<...>);
        _8 = Pin::new_unchecked(move _9);
        _7 = <PollFn<...> as Future>::poll(move _8, move _2);
        _10 = discriminant(_7);
        goto -> bb10_join;
    }

    bb10_join: {
        warp_sync(copy _mask);
        _bcast_poll = warp_broadcast_u32(copy _mask,
            switchInt(copy _is_leader) -> [true: copy _10, false: const 0_u32]);
        switchInt(move _bcast_poll) -> [0: bb9_warp, 1: bb8_warp, otherwise: bb7];
    }

    // ================================================================
    // Error states (unchanged)
    // ================================================================

    bb7: { unreachable; }
    bb11: { assert(const false, "`async fn` resumed after completion"); }
}
```

### Algorithm

The `WarpCooperativeTransform` pass operates in 6 steps on the poll function MIR body.

#### Step 1: Identify coroutine state machine

Detect the entry block pattern:
```
_disc = discriminant(*self_ref);
switchInt(_disc) -> [0: .., 1: .., 3: .., ...]
```

The pass identifies `self_ref` (the `&mut CoroutineState`) and the discriminant local.

**Precondition**: The function must be the poll function of a coroutine (has `CoroutineInfo` in its `Body`). The pass is gated by `#[warp_cooperative]` attribute on the original `async fn`.

#### Step 2: Insert warp intrinsic locals

Add new MIR locals to the function body:
- `_lane_id: u32`
- `_is_leader: bool`
- `_mask: u32` (constant `0xFFFF_FFFF`)
- `_bcast_disc: u32` (broadcast discriminant)
- One `_bcast_field_N: u32` per live field at each yield point (see Step 4)

#### Step 3: Transform entry block — discriminant broadcast

Replace the entry block:

**Before:**
```mir
bb0: {
    _14 = copy (_1.0: &mut Self);
    _13 = discriminant((*_14));
    switchInt(move _13) -> [cases...];
}
```

**After:**
```mir
bb0: {
    _mask = const 0xFFFFFFFF_u32;
    _lane_id = warp_lane_id();
    _is_leader = Eq(_lane_id, const 0_u32);
    _14 = copy (_1.0: &mut Self);    // all lanes get the pointer
}

bb0_leader: {                         // leader only
    _13 = discriminant((*_14));
    goto -> bb0_bcast;
}

bb0_bcast: {
    // shfl.sync: leader's _13 broadcast to all lanes
    _bcast_disc = warp_broadcast_u32(_mask, _13_or_0);
    switchInt(_bcast_disc) -> [cases...];
}
```

The key invariant: **all lanes execute the same `switchInt` branch** because they all received the same broadcast discriminant.

#### Step 4: Transform poll sites — leader-only execution with result broadcast

For each basic block that calls `Future::poll` on an inner future:

1. **Wrap poll in leader-only predication**: Only lane 0 executes the poll call.
2. **Insert barrier after poll**: `warp_sync(mask)` ensures the poll completes before broadcast.
3. **Broadcast poll result discriminant**: `shfl.sync` the `Poll::Ready` vs `Poll::Pending` discriminant.
4. **Broadcast Ready payload fields**: For each field in `Poll::Ready(val)`, decompose `val` into 32-bit chunks and broadcast each chunk.

Field broadcast decomposition rules:
| Type | Chunks | Method |
|------|--------|--------|
| `u8`, `u16`, `u32`, `i8`, `i16`, `i32`, `bool` | 1 | Zero-extend to u32, single `shfl.sync.b32` |
| `u64`, `i64`, `f64` | 2 | Split into lo/hi u32, two `shfl.sync.b32` |
| `f32` | 1 | Bitcast to u32, single `shfl.sync.b32` |
| `*mut T`, `*const T` | 2 (64-bit target) | Split pointer into lo/hi u32 |
| `struct { fields... }` | Sum of field chunks | Broadcast each field recursively |
| `enum` | 1 (discriminant) + variant field chunks | Broadcast discriminant, then active variant fields |

#### Step 5: Transform yield points — barrier before return

At every `return` statement that returns `Poll::Pending` (yield point):

1. **Leader writes discriminant** (sets suspension state) — predicated on `_is_leader`.
2. **Insert `warp_sync(mask)`** barrier.
3. **All lanes return `Poll::Pending`** uniformly.

At every `return` statement that returns `Poll::Ready(val)` (completion):

1. **Broadcast `val`** to all lanes via field decomposition (Step 4 rules).
2. **Leader writes discriminant = Returned** — predicated on `_is_leader`.
3. **Insert `warp_sync(mask)`** barrier.
4. **All lanes return `Poll::Ready(bcast_val)`** uniformly.

#### Step 6: Validate no-divergence invariant

After transformation, verify statically:
- Every `switchInt` on broadcast values has identical targets for all lanes (trivially true since the broadcast value is uniform).
- No path from a broadcast to the next barrier contains a `switchInt` on a non-broadcast value that could cause lane divergence.
- All `return` statements are preceded by a `warp_sync` barrier.

If validation fails, emit a compiler error: `#[warp_cooperative] async fn contains divergent control flow that cannot be made warp-uniform`.

### Worked Example: Two-Yield Async Fn

Consider a 2-yield async fn:

```rust
#[warp_cooperative]
async fn two_yields(x: u32) -> u32 {
    let a = some_future_1(x).await;     // yield point 0
    let b = some_future_2(a).await;     // yield point 1
    a + b
}
```

**Coroutine states**: 0 (Unresumed), 1 (Returned), 3 (Suspended at yield 0), 4 (Suspended at yield 1)

**Transformed execution flow** (one poll cycle at state 3):

```
All 32 lanes enter poll():
  1. _lane_id = %laneid                           // each lane gets its ID
  2. _is_leader = (_lane_id == 0)                 // lane 0 = true, others = false
  3. _disc = discriminant(*self)                  // ONLY lane 0 reads (predicated)
  4. _bcast_disc = shfl.sync.idx.b32(disc, 0)    // all lanes get disc=3
  5. switchInt(3) -> bb_state3                     // ALL lanes take same branch

State 3 (Suspended at yield 0 — re-poll some_future_1):
  6. if is_leader:                                 // only lane 0 executes
       result = some_future_1.poll(cx)
  7. bar.warp.sync(0xFFFFFFFF)                     // barrier
  8. _bcast_poll = shfl.sync(discriminant(result)) // 0=Ready, 1=Pending

  Case Ready(a):
    9. _bcast_a = shfl.sync(a, 0)                  // broadcast a to all lanes
   10. All lanes store a in local variable
   11. Leader creates some_future_2(a), stores in variant#4
   12. bar.warp.sync(0xFFFFFFFF)                   // barrier
   13. Leader polls some_future_2
   14. bar.warp.sync(0xFFFFFFFF)                   // barrier
   15. _bcast_poll2 = shfl.sync(discriminant(result2))

   Case Ready(b):
     16. _bcast_b = shfl.sync(b, 0)
     17. _result = _bcast_a + _bcast_b             // ALL lanes compute
     18. Leader: discriminant(*self) = 1 (Returned)
     19. bar.warp.sync(0xFFFFFFFF)
     20. return Poll::Ready(_result)               // ALL lanes return same value

   Case Pending:
     21. Leader: discriminant(*self) = 4 (Suspended at yield 1)
     22. bar.warp.sync(0xFFFFFFFF)
     23. return Poll::Pending                      // ALL lanes return Pending

  Case Pending:
    24. (discriminant already 3, no state change needed)
    25. bar.warp.sync(0xFFFFFFFF)
    26. return Poll::Pending
```

### Correctness Argument

**Convergence invariant**: All 32 lanes execute the same basic block at every point. This holds because:

1. **Discriminant broadcast** (Step 3): All lanes receive the same state value, so `switchInt` dispatches all lanes to the same case.
2. **Poll result broadcast** (Step 4): All lanes receive the same `Ready`/`Pending` discriminant, so the post-poll `switchInt` dispatches uniformly.
3. **No lane-divergent branches**: The only `switchInt` instructions in the transformed MIR operate on broadcast values. Leader-only predication uses `switchInt(_is_leader)` which IS divergent but immediately reconverges at a join block + barrier.
4. **Barrier at yield**: `warp_sync` before every `return` ensures all lanes complete state writes before the next poll cycle.

**Memory safety**: Only lane 0 reads/writes the coroutine state struct. Followers access the struct pointer but never dereference it (all their data comes from broadcasts). Since there is only one coroutine instance per warp (not per lane), this is safe.

### Limitations

#### 1. Self-referential borrows across yield points

If a future borrows its own state across a yield:
```rust
async fn bad() {
    let x = 42;
    let r = &x;      // borrow of local
    some_future().await;  // yield while r is live
    use(r);           // r survives across yield
}
```
The transformed MIR stores `x` and `r` in the coroutine struct. After broadcast, followers would have a stale pointer `r` pointing into leader's struct, not their own. Since followers don't have their own coroutine struct, this is unsound.

**Mitigation**: Detect self-referential borrows in `CoroutineSavedLocals` and reject with a compiler error. In practice, `Pin` prevents moving the coroutine, so references into the struct are valid — but only for lane 0. Followers must not use broadcast references. This limits warp-cooperative async to value types only across yield points.

#### 2. Trait object futures (`dyn Future`)

When the inner future is `Box<dyn Future>`, the poll function performs a vtable dispatch. The vtable call itself can be broadcast (broadcast the `Ready`/`Pending` result), but the future may have side effects that must be leader-only. This already works with the leader-predication pattern, but the payload type is erased — field broadcasting requires knowing the concrete type at compile time.

**Mitigation**: Require inner futures to have concrete types (no `dyn Future` across yield points in `#[warp_cooperative]` functions). This is already the common case for async fn.

#### 3. Types larger than what shfl.sync can broadcast

`shfl.sync.idx.b32` operates on 32-bit values. Types like `[u8; 1024]` would require 256 shuffle operations per yield point, which is impractical.

**Mitigation**: Set a maximum broadcast size (e.g., 256 bytes = 64 shuffles). Reject `#[warp_cooperative]` async fns where the coroutine state size exceeds this threshold. In practice, async fn states are small (a few u32/u64 values + inner future state).

#### 4. Panics and unwinding

If the inner future panics during leader-only poll, lane 0 diverges to the panic handler while lanes 1-31 wait at the barrier. This is a warp deadlock.

**Mitigation**: In the GPU context, panics already use `trap; exit;` which terminates the entire thread block. The deadlock is moot because the kernel is already terminated. For non-trapping panics (catch_unwind), warp-cooperative async fns should be prohibited from using catch_unwind across yield points.

#### 5. Post-yield computation that reads non-broadcast state

If user code after `.await` reads state that was not broadcast:
```rust
async fn problem() {
    let big_struct = compute_big_thing();
    some_future().await;
    // big_struct is saved across yield but may be too large to broadcast
    use(big_struct.field_42);
}
```
The transformation must broadcast ALL live-across-yield locals, not just the future's output. The `CoroutineSavedLocals` set identifies these. If any saved local exceeds broadcast size limits, the pass rejects the function.

## Open Questions

1. **Should broadcast be lazy or eager?** The spec broadcasts all live-across-yield fields at every resume. An optimization would be to broadcast only fields actually read after the yield point (requires additional dataflow analysis on the transformed MIR).

2. **Active mask vs full mask**: The spec uses `0xFFFFFFFF` (all 32 lanes). In practice, a warp may have fewer active lanes (partial warps at block boundaries). Should the mask come from `activemask()` or be a parameter?

3. **Multiple coroutines per warp**: The spec assumes one coroutine per warp. If each lane has its own coroutine (the executor polls round-robin), the transformation needs a different strategy — possibly warp-ballot to select which coroutine to poll cooperatively.

4. **Integration with #[warp_async] proc macro**: The proc macro and MIR pass solve the same problem at different levels. Should they coexist (proc macro for simple cases, MIR pass for complex ones), or should the MIR pass replace the proc macro entirely?

## Impact on Downstream Tasks

- **rustc-warp.3-5 (SKIPPED)**: The implementation tasks are deferred indefinitely per BS58. This spec serves as the research deliverable proving compiler-level SIMT async is feasible.
- **native-warp-async epic**: This spec provides the theoretical foundation for the final criterion. The MIR transformation is proven sound for value-type state machines.
- **ADR-016**: Records the design decision that a post-StateTransform MIR pass is the correct insertion point for warp-cooperative codegen.
- **Proc macro evolution**: The spec's algorithm (Steps 1-6) can guide future proc macro improvements — the proc macro performs the same logical transformation at the AST level.
