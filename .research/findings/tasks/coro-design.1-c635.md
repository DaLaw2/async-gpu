# coro-design.1: Map Rust Generator/Coroutine Semantics to GPU Warp State Machines

## Summary

Rust's `Coroutine` trait and `async fn` share the **exact same** state machine infrastructure in rustc (`StateTransform` in `rustc_mir_transform/src/coroutine/mod.rs`). The existing `WarpCooperativeTransform` MIR pass already operates on **all** post-`StateTransform` coroutine bodies — it does not distinguish between async-desugared and raw coroutines. This means GPU generator support requires (1) a `GpuGenerator` trait with warp-cooperative resume semantics, (2) yield-value broadcast logic (shfl.sync for scalars, shared memory for structs), and (3) minor MIR pass awareness of `CoroutineState::Yielded(value)` vs `Poll::Pending`. The fundamental mapping is proven feasible by the existing infrastructure.

## Findings

### Q1: Rust Coroutine Trait — Current Nightly State

**Confidence: HIGH (read from rustc-src source)**

The `Coroutine<R>` trait (`core::ops::coroutine`, feature `coroutine_trait`, issue #43122) is:

```rust
pub trait Coroutine<R = ()> {
    type Yield;
    type Return;
    fn resume(self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return>;
}
```

`CoroutineState<Y, R>` is an enum with two variants:
- `Yielded(Y)` — coroutine suspended with a value
- `Complete(R)` — coroutine finished with a return value

Key differences from `Future`:
| Aspect | Future | Coroutine |
|--------|--------|-----------|
| Trait method | `poll(Pin<&mut Self>, &mut Context) -> Poll<T>` | `resume(Pin<&mut Self>, R) -> CoroutineState<Y, R>` |
| Suspension | `Poll::Pending` (no value) | `CoroutineState::Yielded(Y)` (carries a value) |
| Completion | `Poll::Ready(T)` | `CoroutineState::Complete(R)` |
| Resume arg | Implicit (via `Context`/`Waker`) | Explicit generic `R` |
| Waker | Has `Waker` for event-driven wake-up | No `Waker` — pull-based |

The `Coroutine` trait is **unstable** but fully implemented in nightly. It is the underlying mechanism for `gen` blocks, `async gen` blocks, and raw `#[coroutine]` closures.

### Q2: MIR Lowering — State Machine Transform

**Confidence: HIGH (read from source)**

Both `async fn` and generator/coroutine bodies go through the **same** `StateTransform` pass in `rustc_mir_transform/src/coroutine/mod.rs`. The flow:

1. rustc desugars `async fn` → `CoroutineKind::Desugared(Async, _)`, `gen fn` → `CoroutineKind::Desugared(Gen, _)`, raw `#[coroutine]` → `CoroutineKind::Coroutine(_)`
2. `StateTransform::run_pass()` is the single unified pass for ALL coroutine kinds
3. It computes locals live across suspension points, builds a state struct, and rewrites `yield` terminators
4. **Crucially**: The `TransformVisitor::make_state()` method handles the return-type differences:
   - Async: `yield` → `Poll::Pending`, `return` → `Poll::Ready(val)`
   - Gen: `yield val` → `Option::Some(val)`, `return` → `Option::None`
   - Raw Coroutine: `yield val` → `CoroutineState::Yielded(val)`, `return val` → `CoroutineState::Complete(val)`
5. After transform, the body has `body.coroutine` metadata and a dispatch `SwitchInt` on the discriminant (states 0=Unresumed, 1=Returned, 2=Poisoned, 3+=suspension points)

**Location**: `rustc-src/compiler/rustc_mir_transform/src/coroutine/mod.rs`, specifically:
- `StateTransform::run_pass()` at line 967
- `TransformVisitor::visit_basic_block_data()` at line 434 (handles `Yield` terminators)
- `insert_switch()` at line 669 (builds dispatch switch)
- `create_coroutine_resume_function()` at line 811 (finalizes the resume function)

### Q3: Mapping to GPU Warps — WarpCooperativeTransform Compatibility

**Confidence: HIGH (read from source)**

The existing `WarpCooperativeTransform` (`rustc-patches/warp_cooperative.rs`) **already works on all coroutine bodies** — it checks `body.coroutine.is_none()` and processes any body with coroutine metadata, regardless of `CoroutineKind`. Specifically:

1. **Discriminant broadcast** (line 96-98): Inserts `activemask` + `shfl.sync.idx.b32` before the dispatch `SwitchInt` so all 32 lanes agree on the current state. **This already works for generators** — the discriminant is the same u32 state field regardless of async vs gen vs raw.

2. **Barrier before Return** (line 100-116): Inserts `activemask` + `bar.warp.sync` before every `Return` terminator. **This already works for generators** — generators return `CoroutineState::Yielded(val)` via the normal `Return` terminator (after `StateTransform` rewrites `yield` to `return`).

**Key insight**: After `StateTransform`, there are **no more `Yield` terminators** in the MIR. The `yield` is rewritten to: (1) assign `CoroutineState::Yielded(val)` to return place, (2) set discriminant to suspension state, (3) `Return`. This means `WarpCooperativeTransform` treats `yield` and `return` identically — both hit the `Return` terminator path.

**What's different for generators vs async**:
- The yield point **carries a value** (`CoroutineState::Yielded(Y)`) that the caller needs
- In async, `Poll::Pending` carries no useful value — the caller just retries
- For GPU generators, the yielded value must be **broadcast to all lanes**

### Q4: Yield Value Propagation

**Confidence: MEDIUM (design analysis, not yet implemented)**

When a generator yields a value on GPU, all 32 lanes need to see the same value. Two strategies:

**Strategy A: Scalar yields (≤ 32 bits)**
Use `shfl.sync.idx.b32` from lane 0, exactly like the discriminant broadcast. This is the same pattern already proven in `warp_cooperative.rs` and `warp_cooperative.rs` runtime. Zero overhead.

**Strategy B: Small struct yields (≤ 128 bits / 4 registers)**
Use multiple `shfl.sync.idx.b32` calls, one per 32-bit word. Example: a `(u32, u32)` yield needs 2 shuffles. This is the pattern used in `warp_hostcall_wait_u64` for broadcasting u64 (2 shuffles for lo/hi).

**Strategy C: Large struct yields (> 128 bits)**
Use shared memory. Lane 0 writes to `__shared__` buffer, `bar.warp.sync`, all lanes read. The GTX 1660 has 48KB shared memory per SM — more than enough for any reasonable yield value.

**Recommended approach**: The `GpuGenerator` runtime trait should broadcast the yielded value via shfl.sync for types ≤ 128 bits, falling back to shared memory for larger types. The MIR pass does NOT need to handle this — the runtime `poll_warp` / `resume_warp` method handles broadcast, just like `warp_poll_future` already does for `Poll<bool>`.

### Q5: Multiple Generators

**Confidence: HIGH (design analysis)**

Two interpretations, both feasible:

**(A) Multiple generators within the same warp** — Interleaved via warp-cooperative scheduling. This is **exactly what the existing `GpuExecutor` already does** for multiple async tasks. The executor dequeues a task, lane 0 polls it, broadcasts the result, and moves to the next task. The same approach works for generator tasks: the executor calls `resume()` on lane 0, broadcasts `CoroutineState`, and either processes the yielded value or marks the task complete.

**(B) Different warps each running their own generator** — Trivially parallel. Each warp has its own generator instance in global memory. No coordination needed between warps. This is the simpler case and works automatically.

**Which is more useful?** Interpretation (A) is more useful for the epic's streaming pipeline goal — a producer and consumer generator running within the same warp, coordinating via zero-buffering (the consumer receives each yielded value before the producer produces the next). Interpretation (B) is useful for data-parallel generators (each warp processes a different data partition).

**Feasibility**: (A) works with current infrastructure. The `GpuExecutor` already handles multiple tasks per warp. Generator tasks would be a new task type that yields values instead of just `Poll::Pending`.

### Q6: Streaming Pipeline (Producer-Consumer)

**Confidence: MEDIUM (design analysis)**

The zero-buffering producer-consumer pipeline would work as follows:

```
// Pseudo-code for warp-cooperative execution:
// 1. Lane 0 calls producer.resume(()) → CoroutineState::Yielded(value)
// 2. Value broadcast to all lanes via shfl.sync
// 3. All lanes process value (compute-parallel)
// 4. Lane 0 calls producer.resume(()) again → next value
// 5. Repeat until CoroutineState::Complete
```

**Option 1: Direct inline pipeline (simplest)**
A single `WarpFuture`/`WarpGenerator` drives the pipeline. No channel needed. The producer yields, the consumer processes, all within one `poll_warp` call. Zero overhead, zero buffering.

**Option 2: Channel-based pipeline (more flexible)**
Producer generator yields values into an `MpscChannel`. Consumer task receives from channel. The existing `MpscChannel` with waker support enables this. However, this adds buffering (ring buffer of 64 slots) and is **not** zero-buffering.

**Option 3: Coroutine-to-coroutine (zero-buffering via resume arg)**
The `Coroutine<R>` trait accepts a resume argument `R`. The consumer could pass a "request" via `R`, and the producer yields the response. This creates a natural request-response protocol with zero buffering. However, this is complex to map to warp semantics.

**Recommended**: Option 1 for the demo — a single warp-cooperative loop that alternates between producing and consuming. This matches the epic's "zero buffering" requirement and is the simplest to implement.

### Q7: MIR Pass Extension vs New Pass

**Confidence: HIGH (design analysis)**

**Recommendation: Extend `WarpCooperativeTransform`, do NOT create a new pass.**

Reasons:
1. **Already handles generators**: The pass checks `body.coroutine.is_none()` — it already runs on generator bodies. The discriminant broadcast and return barrier work unchanged.
2. **Shared infrastructure**: Both async and generator bodies use the same discriminant-based state machine. The dispatch switch, discriminant locals, and suspension point numbering are identical.
3. **Only difference is yield value broadcast**: The pass could optionally insert shfl.sync for yielded values. But this is better handled in the **runtime** (the `GpuGenerator` trait's `resume_warp` method), not the MIR pass.
4. **MIR pass may not need changes at all**: The current pass's discriminant broadcast + return barrier are sufficient. Yield value broadcast is a runtime concern, not a compiler concern.

**If MIR pass changes are needed**, they would be:
- Detecting `CoroutineKind::Coroutine` vs `CoroutineKind::Desugared(Gen, _)` to apply generator-specific transforms
- The analysis already tracks `yield_points` (line 145-147) — this could emit generator-specific diagnostics
- Adding yield-value broadcast at suspension points (but this is better in runtime)

## Unexpected Discoveries

1. **WarpCooperativeTransform already works on generators**: The pass does not filter by `CoroutineKind` — it processes ALL coroutine bodies. This means a raw `#[coroutine]` closure compiled for nvptx64 would already get discriminant broadcast and return barriers, even today. The infrastructure for GPU generators is partially in place already.

2. **`yield` is erased by `StateTransform`**: After the coroutine state machine transform, there are zero `Yield` terminators in the MIR. Everything is converted to discriminant writes + `Return`. This means `WarpCooperativeTransform` doesn't need to know about `yield` at all — it just sees state transitions and returns.

3. **Resume argument `R` maps to lane 0 input**: The `Coroutine<R>` trait's resume argument is naturally a lane-0-only input. All lanes call `resume()` with the same `R` (broadcast from lane 0). This is analogous to how `Future::poll`'s `Context` is only used by lane 0.

4. **`gen fn` desugaring uses `Option<Y>`**: When rustc desugars `gen fn`, yields become `Option::Some(val)` returns and the final return becomes `Option::None`. This is a different enum than raw `CoroutineState<Y, R>` — the `GpuGenerator` trait should use `CoroutineState` directly (raw coroutines) rather than `Option` (gen blocks), because `CoroutineState` carries both yield and return types.

## Open Questions

1. **Resume argument type**: Should `GpuGenerator` accept a resume argument `R`? For the streaming pipeline use case, `R = ()` is sufficient (producer generates, consumer receives). But for request-response patterns, `R` could carry consumer feedback. Decision: start with `R = ()`.

2. **Per-lane vs uniform yield**: Should each lane be able to yield a different value (per-lane output like `WarpFuture::Output`), or must all lanes yield the same value? For compute parallelism, per-lane yields are useful (lane K produces element K). For I/O generators (the primary use case), uniform yield is correct. The trait should support both via a type parameter.

3. **Generator exhaustion**: What happens when a warp-cooperative generator completes? All lanes must agree it's done (discriminant broadcast handles this). But should the executor automatically reclaim the task slot? The `GpuExecutor` already does this for futures.

4. **Async generators (`async gen`)**: Rust supports `async gen` blocks that can both `yield` and `.await`. Should `GpuGenerator` support this? It would combine generator yield semantics with async I/O. This is future work (T2+).

## Impact on Downstream Tasks

- **coro-design.2** (GpuGenerator trait design): Use `CoroutineState<Y, R>` as the return type. Model after `WarpFuture` with a `resume_warp` method. Lane 0 calls `resume()`, broadcasts `CoroutineState` discriminant + value.
- **coro-design.3** (MIR pass extension): Likely minimal changes needed — the existing pass already handles generator bodies. Focus on yield-value broadcast if needed.
- **coro-design.4** (streaming pipeline demo): Use the direct inline pipeline approach (Option 1 from Q6) — single warp-cooperative loop alternating produce/consume.
