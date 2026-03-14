# rustc-impl.1: Study actual rustc coroutine.rs StateTransform + lib.rs pass pipeline from source
**Cycle**: 264 | **Theme**: rustc-impl | **Kind**: investigation | **Status**: done

## Summary

Deep-read of `toolchain/compiler/rustc_mir_transform/src/coroutine.rs` (2011 lines) and `lib.rs` (852 lines) to understand how StateTransform converts coroutines into state machines. This informs exactly what MIR our WarpCooperativeTransform pass will see as input.

## Findings

### Q: How does StateTransform convert async fn into a state machine?
A: StateTransform::run_pass() (line 1463-1684) performs these steps:
1. **Guard**: Only runs on coroutines (`body.yield_ty()` must be Some)
2. **Async context**: Replaces `ResumeTy` with `&mut Context<'_>` via `transform_async_context()`
3. **Async drops**: Expands async drops if present
4. **Liveness analysis**: `locals_live_across_suspend_points()` computes which locals must be saved in the coroutine struct across yield points
5. **Layout computation**: `compute_layout()` maps locals to coroutine struct fields (variant per suspension point)
6. **TransformVisitor**: Rewrites the MIR body:
   - `Return` terminators → set discriminant to RETURNED (1) + return Poll::Ready/CoroutineState::Complete
   - `Yield` terminators → set discriminant to suspension state (3+) + return Poll::Pending/CoroutineState::Yielded
   - Local accesses for saved locals → coroutine struct field accesses via downcast projection
7. **Entry switch**: Inserts SwitchInt on discriminant at START_BLOCK:
   - State 0 (UNRESUMED) → original entry point
   - State 1 (RETURNED) → panic "resumed after return"
   - State 2 (POISONED) → panic "resumed after panic"
   - State 3+ → resume at corresponding suspension point (with StorageLive restoration)
8. **Drop shim**: Creates coroutine_drop function
9. **Resume function**: Creates the final resume/poll function with Pin<&mut Self> argument

**Confidence**: high (read from source)

### Q: What does the MIR look like AFTER StateTransform (our pass input)?
A: After StateTransform:
- **No more Yield terminators** — all converted to SetDiscriminant + Return
- **Entry block** is a SwitchInt on the discriminant field
- **Self argument** is `Pin<&mut CoroutineType>` (for async) or `&mut CoroutineType` (for gen)
- **Locals that cross suspension points** are now accessed via `self.variant.field` projections
- **Return type** is `Poll<OriginalReturnType>` (for async) or `Option<YieldType>` (for gen)
- **State transitions** appear as `SetDiscriminant { place: self, variant_index: N }` statements followed by `Return` terminators

**Confidence**: high

### Q: How is the pass registered in lib.rs?
A: Two mechanisms:
1. `declare_passes!` macro (line 139): `mod coroutine : StateTransform;` — declares the module and the pass type
2. `run_runtime_lowering_passes()` (line 632-658): StateTransform is the second-to-last pass in the list:
   ```
   &coroutine::StateTransform,  // line 655
   &Lint(known_panics_lint::KnownPanicsLint),  // line 656 (last)
   ```

**Confidence**: high

### Q: Where should WarpCooperativeTransform be inserted?
A: **Immediately after `&coroutine::StateTransform`** in `run_runtime_lowering_passes()`. At this point:
- Coroutine is already a state machine (SwitchInt entry, no Yield terminators)
- Locals are remapped to coroutine struct fields
- But before KnownPanicsLint runs

The pass needs:
1. New module declaration in `declare_passes!`: `mod warp_cooperative : WarpCooperativeTransform;`
2. New entry in `run_runtime_lowering_passes()` after line 655

**Confidence**: high

### Q: What are the key data structures for our pass?
A: Our pass will work with:
- `Body<'tcx>` — the MIR body after state machine conversion
- `TerminatorKind::SwitchInt` — the entry dispatch on discriminant
- `TerminatorKind::Call` — poll call sites (Future::poll)
- `StatementKind::SetDiscriminant` — state transitions before Return
- `TerminatorKind::Return` — yield/return points
- `body.coroutine` — coroutine metadata (kind, layout)
- `tcx.sess.target.arch` — check for "nvptx64" to gate the pass

Key types from coroutine.rs we DON'T need to replicate:
- TransformVisitor (only used during StateTransform itself)
- LivenessInfo / CoroutineSavedLocals (liveness already computed, layout frozen)
- SuspensionPoint (already consumed by StateTransform)

**Confidence**: high

### Q: How does StateTransform handle the discriminant?
A: Three hardcoded states + N suspension points:
- `CoroutineArgs::UNRESUMED = 0` — not yet resumed
- `CoroutineArgs::RETURNED = 1` — completed
- `CoroutineArgs::POISONED = 2` — panicked during execution
- `CoroutineArgs::RESERVED_VARIANTS = 3` — first suspension point starts at 3
- State N corresponds to variant N in the coroutine ADT
- Discriminant type from `args.as_coroutine().discr_ty(tcx)`
- Set via `StatementKind::SetDiscriminant { place: self_place, variant_index }`
- Read via `Rvalue::Discriminant(self_place)` into a temp, then SwitchInt

**Confidence**: high

## Architecture: Post-StateTransform MIR Structure

```
bb0: SwitchInt(discriminant(self)) {
    0 → bb_unresumed (START_BLOCK+1, original entry)
    1 → bb_returned (panic "resumed after return")
    2 → bb_poisoned (panic "resumed after panic")  [if can_unwind]
    3 → bb_resume_3 (StorageLive locals + goto resume point)
    4 → bb_resume_4
    ...
    _ → bb_unreachable
}

bb_unresumed: (original function body, with locals remapped to self.variant.field)
    ...
    // At what was a yield point:
    SetDiscriminant(self, variant_index=3)
    Return  // returns Poll::Pending

    // At what was a return:
    SetDiscriminant(self, variant_index=1)
    Return  // returns Poll::Ready(value)

bb_resume_N: (restore storage for locals live at suspension point N)
    StorageLive(_x)
    StorageLive(_y)
    goto → bb_continuation
```

## What WarpCooperativeTransform Needs to Do

Given this post-StateTransform MIR:
1. **Gate on target**: Only activate for `nvptx64` target AND functions with `#[warp_cooperative]` attribute
2. **Identify state transitions**: Find `SetDiscriminant + Return` patterns (these are yield/return points)
3. **Identify poll calls**: Find `Call` terminators where the callee is `<F as Future>::poll`
4. **Insert shfl.sync broadcasts**:
   - Before each state transition: broadcast the discriminant value from lane 0 to all lanes
   - After poll returns: broadcast the Poll discriminant (Ready vs Pending) from lane 0
5. **Insert warp synchronization**: `bar.sync` / `__syncwarp()` at convergence points

## Impact on Downstream Tasks

- rustc-impl.2 is now UNBLOCKED — we have complete understanding of:
  - Input MIR structure (post-StateTransform)
  - Where to register the pass (lib.rs line 655-656, declare_passes! macro)
  - What MIR patterns to match (SetDiscriminant, Call with Future::poll, SwitchInt)
  - What to insert (shfl.sync intrinsic calls, warp barrier)
