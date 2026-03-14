# rustc-warp-async.1: Rustc Async Transform Pipeline Analysis
**Cycle**: 260 | **Theme**: rustc-warp-async | **Kind**: investigation | **Status**: done

## Summary
Comprehensive analysis of rustc's async/await transformation pipeline, identifying the exact compiler stages, source files, MIR constructs, and viable insertion points for warp-cooperative SIMT synchronization (shfl.sync). The most promising approach is a custom MIR pass that intercepts `Yield` terminators and wraps resume paths with shfl.sync broadcast logic.

---

## 1. Async Desugaring Pipeline

The transformation from `async fn` to executable code occurs in four major stages:

### Stage 1: AST → HIR — Async desugaring (syntax level)

**What happens**: `async fn` is desugared into a regular function that returns an `impl Future`. The function body becomes a coroutine (formerly "generator"). Each `.await` becomes a yield point with polling logic.

**Key file**: `compiler/rustc_ast_lowering/src/expr.rs`

**Key functions**:
- `lower_expr_await()` (line ~1356) — Entry point for `.await` desugaring
- `make_lowered_await()` (line ~1365) — Core desugaring that generates:
  1. `IntoFuture::into_future(<expr>)` call
  2. A loop that calls `Future::poll()` on the inner future
  3. Match arms for `Poll::Ready(val)` (break with value) and `Poll::Pending` (yield)
  4. Task context (`&mut Context<'_>`) threading
- `make_desugared_coroutine_expr()` (line ~1099) — Wraps the function body in a coroutine closure, handling capture semantics and resume arguments

**HIR output**: The `.await` expression becomes:
```
match IntoFuture::into_future(expr) {
    mut __awaitee => loop {
        match unsafe { Future::poll(Pin::new_unchecked(&mut __awaitee), cx) } {
            Poll::Ready(result) => break result,
            Poll::Pending => yield (),  // <-- suspension point
        }
    }
}
```

**Coroutine kinds tracked by the compiler**:
- `CoroutineDesugaring::Async` — standard async fn/block
- `CoroutineDesugaring::AsyncGen` — async generators
- `CoroutineDesugaring::Gen` — sync generators

### Stage 2: HIR → MIR — Coroutine body generation

**What happens**: The coroutine's HIR is lowered to MIR. Each `yield` becomes a `Yield` terminator in the MIR control-flow graph. Locals are tracked with `StorageLive`/`StorageDead` statements for liveness analysis.

**Key constructs in MIR**:
- `TerminatorKind::Yield { value, resume, resume_arg, drop }` — The yield point. `resume` is the basic block to jump to when the coroutine is resumed. `resume_arg` receives the value passed to `resume()` (for async, this is `()`).
- `TerminatorKind::Return` — Coroutine completion (maps to `Poll::Ready`)
- `StorageLive(local)` / `StorageDead(local)` — Scope boundaries for locals, critical for the optimizer to determine which variables are live across yield points

### Stage 3: MIR Transform — `StateTransform` pass (coroutine → state machine)

**What happens**: This is the most important stage. The `StateTransform` MIR pass rewrites the coroutine MIR into a resumable state machine. This is where the coroutine struct layout is computed, the discriminant is managed, and yield/resume paths are generated.

**Key file**: `compiler/rustc_mir_transform/src/coroutine.rs`

**Key struct**: `StateTransform` implements `MirPass<'tcx>`

**Transformation pipeline (in order)**:
1. **Liveness analysis** — `locals_live_across_suspend_points()`: Uses dataflow analysis to determine which locals are alive across each `Yield` terminator
2. **Layout computation** — `compute_layout()`: Assigns locals to variants of a multi-variant enum. Storage conflicts are computed so non-overlapping locals can share memory.
3. **Local remapping** — `TransformVisitor`: Rewrites all references to saved locals as field projections into the coroutine struct (e.g., `_3` becomes `(*_1).field[2]`)
4. **Drop shim generation** — Creates the `drop()` implementation for each state variant
5. **Resume function creation** — Generates a dispatch switch at the entry point that reads the discriminant and jumps to the correct resume block
6. **Dereference fixing** — `deref_finder()`: Ensures all self-references go through the correct indirection

**State discriminant constants**:
```rust
CoroutineArgs::UNRESUMED = 0   // Initial state, not yet polled
CoroutineArgs::RETURNED = 1    // Completed (Poll::Ready)
CoroutineArgs::POISONED = 2    // Panicked
// RESERVED_VARIANTS = 3
// Suspension point N → discriminant = RESERVED_VARIANTS + N
```

**Generated resume function structure**:
```
fn resume(&mut self, arg) {
    match self.discriminant {
        0 (UNRESUMED) → goto START_BLOCK
        1 (RETURNED)  → panic!("resumed after completion")
        2 (POISONED)  → panic!("resumed after panic")
        3             → { restore StorageLive for suspend point 0; goto resume_block_0 }
        4             → { restore StorageLive for suspend point 1; goto resume_block_1 }
        ...
    }
}
```

**Key supporting types**:
- `SuspensionPoint<'tcx>`: Records each yield's state index, resume block, and set of live locals
- `CoroutineSavedLocals`: BitSet of locals that need to be stored in the coroutine struct
- `LivenessInfo`: Aggregates saved locals, per-suspension-point liveness, and storage conflict info
- `CoroutineLayout<'tcx>`: Final struct layout — field types, field names, variant assignments

**Supporting files**:
- `compiler/rustc_mir_transform/src/coroutine/by_move_body.rs` — Synthesizes by-move MIR bodies for `AsyncFnOnce`
- `compiler/rustc_mir_transform/src/coroutine/drop.rs` — Drop glue generation for coroutine variants

### Stage 4: MIR → LLVM IR → Machine code

**What happens**: The state machine MIR is lowered to LLVM IR like any other function. The coroutine struct becomes a regular Rust enum in LLVM IR. The resume function becomes a normal function with a switch on the discriminant.

**Key files**:
- `compiler/rustc_codegen_ssa/src/mir/` — MIR-to-LLVM codegen
- `compiler/rustc_codegen_llvm/src/` — LLVM-specific codegen

For nvptx64 targets, the LLVM IR is emitted as PTX assembly via LLVM's NVPTX backend.

---

## 2. Key Source Files in rustc (Summary Table)

| Stage | File | Key Functions/Types |
|-------|------|-------------------|
| Async→Coroutine (HIR) | `rustc_ast_lowering/src/expr.rs` | `lower_expr_await`, `make_lowered_await`, `make_desugared_coroutine_expr` |
| Async fn item lowering | `rustc_ast_lowering/src/item.rs` | `lower_fn_decl`, async fn return type handling |
| Coroutine type repr | `rustc_middle/src/ty/sty.rs` | `TyKind::Coroutine`, `CoroutineArgs` |
| Coroutine→State machine | `rustc_mir_transform/src/coroutine.rs` | `StateTransform`, `TransformVisitor`, `locals_live_across_suspend_points`, `compute_layout`, `create_cases` |
| By-move body | `rustc_mir_transform/src/coroutine/by_move_body.rs` | By-move body synthesis for `AsyncFnOnce` |
| Drop shim | `rustc_mir_transform/src/coroutine/drop.rs` | Drop implementation per variant |
| Async closure desugaring | `rustc_ast_lowering/src/expr.rs` | `lower_expr_coroutine_closure` |
| Async closure types | `rustc_middle/src/ty/` | `TyKind::CoroutineClosure`, `CoroutineClosureArgsParts` |
| MIR terminators | `rustc_middle/src/mir/terminator.rs` | `TerminatorKind::Yield`, `TerminatorKind::Return` |
| Upvar analysis | `rustc_hir_typeck/src/upvar.rs` | By-ref vs by-move capture decisions |

---

## 3. Insertion Points for Warp-Cooperative Logic

Three viable approaches, ordered from most to least promising:

### Approach A: Custom MIR Pass (after `StateTransform`) — RECOMMENDED

**Where**: New MIR pass registered after `StateTransform` in the MIR pass pipeline.

**What it does**: After the coroutine is already transformed into a state machine with a dispatch switch, insert shfl.sync broadcast logic at each resume point:

1. **Intercept the dispatch switch**: The generated resume function has a `SwitchInt` on the discriminant. After the switch, each case restores `StorageLive` and jumps to a resume block.
2. **Wrap each resume block entry**: Before executing the resumed computation, insert:
   - `syncwarp(0xFFFFFFFF)` — ensure all lanes are converged
   - After the `Future::poll()` call in the awaited-future loop, intercept the result:
     - Lane 0: write poll result to a shared location
     - All lanes: `shfl.sync.idx.b32` to broadcast the poll discriminant
     - All lanes: decode the broadcast and proceed uniformly

**Advantages**:
- Works on the already-flattened state machine — no need to understand the full coroutine transformation
- The state discriminant is already a u32, perfect for `shfl.sync.idx.b32`
- Can be applied selectively via an attribute (`#[warp_cooperative]`)
- Does not break non-GPU code (pass only activates for nvptx64 target)

**Challenges**:
- Must identify which `Call` terminators are `Future::poll()` calls in the MIR
- Must handle the `Poll::Ready(T)` → `Poll::Pending` branch uniformly across lanes
- Complex types `T` in `Poll::Ready(T)` need multi-word broadcast (multiple shfl.sync calls)

**Implementation sketch**:
```
// Pseudo-MIR for a single await point after warp-cooperative transformation:

bb_dispatch_state_N:
    StorageLive(_saved_locals...)
    // NEW: warp convergence barrier
    _mask = Call(activemask)
    Call(syncwarp, _mask)
    goto → bb_resume_N

bb_resume_N:
    // Original: poll inner future
    _poll_result = Call(Future::poll, pinned_future, context)
    // NEW: broadcast poll discriminant
    _poll_discr = Discriminant(_poll_result)  // 0=Ready, 1=Pending
    _broadcast_discr = Call(shfl_sync_idx_u32, _mask, _poll_discr, 0)
    SwitchInt(_broadcast_discr) → [0: bb_ready_N, 1: bb_pending_N]

bb_ready_N:
    // NEW: broadcast the Ready value
    _ready_val = Field(_poll_result, 0)
    _broadcast_val = Call(shfl_sync_idx_u32, _mask, _ready_val, 0)
    // Continue with _broadcast_val instead of _ready_val
    ...

bb_pending_N:
    // Set discriminant to state N, return Pending
    _self.discriminant = N + RESERVED_VARIANTS
    Return(Poll::Pending)
```

### Approach B: LLVM IR Custom Pass

**Where**: As an LLVM `FunctionPass` or `ModulePass` running after rustc emits LLVM IR but before LLVM's backend lowers to PTX.

**What it does**: Pattern-match on the LLVM IR for the resume function's switch, find call instructions to poll, and insert shfl.sync intrinsics.

**Advantages**:
- No rustc fork needed — can use LLVM plugin infrastructure
- Works at a lower level where PTX intrinsics are native

**Disadvantages**:
- LLVM IR has lost most semantic information — hard to reliably identify poll calls vs other calls
- Fragile: LLVM optimizations may restructure the switch or inline the poll call
- Harder to debug than MIR

### Approach C: Enhanced Proc Macro (current approach)

**Where**: Before the compiler — rewrite the source code.

**What it does**: `#[warp_async]` generates a WarpFuture impl with explicit state machine, shfl.sync broadcasts at each state transition, and lane 0 polling logic.

**Advantages**:
- Already implemented and working in this project
- No compiler fork needed
- Full control over generated code

**Disadvantages**:
- Cannot use standard `async fn` / `.await` syntax (must use custom macros or restricted patterns)
- Type inference is limited at the proc macro level (must know concrete future types)
- Maintenance burden: every new hostcall pattern requires macro updates

---

## 4. State Machine Layout Details

### Coroutine struct layout

The compiler generates a multi-variant enum-like struct:

```
struct CoroutineState {
    discriminant: u32,     // Which state we're in
    // Variant 0 (UNRESUMED): captured upvars only
    // Variant 1 (RETURNED): empty
    // Variant 2 (POISONED): empty
    // Variant 3 (Suspend0): locals live across yield point 0
    // Variant 4 (Suspend1): locals live across yield point 1
    // ...
}
```

Non-overlapping locals across different suspension points **share memory** (union-like). The compiler uses storage conflict analysis to determine which locals can overlap.

### Yield/resume point representation

Each `.await` in the original async fn produces one `Yield` terminator in MIR:

```
bb_N:
    // ... compute poll result ...
    SwitchInt(poll_discriminant) → [Ready: bb_break, Pending: bb_yield]

bb_yield:
    Yield(value=(), resume=bb_resume_N, drop=bb_drop_N)
    // MIR after this point is unreachable until resumed
```

The `resume` field of the `Yield` terminator points to the basic block that will execute when the coroutine is next called with `resume()`.

### Discriminant management

- Written by `set_discr()` in `TransformVisitor` — generates `SetDiscriminant` statements
- Read by `get_discr()` — copies discriminant to a temporary local for the dispatch switch
- The dispatch switch is a `SwitchInt` terminator at the entry block of the transformed function
- For async, the discriminant directly corresponds to which `.await` was last suspended at

### Poll → Pending/Ready decision in MIR

The `.await` desugaring generates a match on the `Poll` enum:

```
_poll = Call(Future::poll, pin, cx)
_discr = Discriminant(_poll)
SwitchInt(_discr) → [0: bb_ready, 1: bb_pending]

bb_ready:
    _result = Field(_poll, 0)   // Extract the Ready value
    break → bb_after_await

bb_pending:
    yield ()                     // Suspend here
    // On resume, re-enter the poll loop
    goto → bb_poll_loop_head
```

This pattern is the exact target for warp-cooperative interception. The `SwitchInt` on the poll discriminant is where lane 0's result must be broadcast to all lanes.

---

## 5. Implications for This Project

### Current state (proc macro approach)

The project already has a working `#[warp_async]` proc macro that:
- Transforms sequential code with `warp_*!()` calls into a WarpFuture state machine
- Supports `.await` on inner futures via `warp_poll_future()` (broadcasts poll result)
- Supports `?` operator via broadcast of Ok/Err discriminant
- Generates explicit state machine with `broadcast_u32` at each transition

See: `crates/warp-macro/src/lib.rs`, `crates/gpu-runtime/src/lib.rs` (warp_cooperative module)

### Path to compiler-level support

**Phase 1 (minimal rustc patch)**: Add a MIR pass that, for functions annotated with `#[warp_cooperative]`:
1. Runs after `StateTransform`
2. Identifies `Call` terminators targeting `Future::poll`
3. Inserts `shfl_sync_idx_u32` intrinsic calls to broadcast the poll discriminant
4. Wraps the Ready value extraction with broadcast logic

**Phase 2 (full integration)**:
- Handle complex `Output` types (multi-word broadcast)
- Handle `?` operator (Result discriminant broadcast)
- Integrate with the nvptx64 target's intrinsic definitions
- Add convergence barriers (`syncwarp`) at state transitions

**Phase 3 (upstream RFC)**:
- Propose a `#[warp_cooperative]` attribute for SIMT targets
- Define the semantics: "all lanes in a warp must call this function together"
- Extend to other SIMT targets (AMD RDNA wave64, Intel Xe subgroup)

### Key constraint

The MIR pass approach requires that the coroutine state discriminant fits in 32 bits (it does — rustc uses u32) and that `Poll<T>::Ready(T)` values can be broadcast via shfl.sync. For `bool` and integer types this is trivial. For complex types, either:
- Require the output type to implement a `WarpBroadcast` trait
- Store the result in shared memory and have all lanes read from there

---

## Open Questions

1. **MIR pass registration**: Where in `rustc_mir_transform/src/lib.rs` should the new pass be inserted relative to `StateTransform`? It must run after state machine generation but before codegen.

2. **Intrinsic availability**: Does rustc's nvptx64 target expose `shfl.sync` as a recognized LLVM intrinsic, or must we use inline asm? (Current project uses inline PTX asm — see `crates/gpu-atomics/src/lib.rs`)

3. **Identifying poll calls in MIR**: After inlining and optimization, can we reliably identify `Future::poll()` call sites? The `Call` terminator includes the callee as an `Operand` — need to check if it resolves to the `Future::poll` trait method.

4. **Multi-variant state broadcast**: When resuming from different states, do all lanes need to agree on the discriminant? Yes — this means the discriminant itself must be broadcast before the dispatch switch, not just the poll result.

5. **Nested futures**: When an async fn awaits another async fn, the inner future's state machine is embedded. Does the MIR pass need to handle this recursively, or does the outer state machine's poll loop already encapsulate it?

## Impact on Downstream Tasks

- **rustc-warp-async.2**: Can proceed with prototyping a minimal MIR pass that inserts a no-op marker at `Yield` terminators for nvptx64 target
- **rustc-warp-async.3**: Requires answers to open questions 2 and 3 before implementing actual shfl.sync insertion
- Validates the native-warp-async epic's third criterion: compiler-level support is architecturally feasible
