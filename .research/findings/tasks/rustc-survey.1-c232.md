# rustc-survey.1: Map rustc async fn → coroutine → state machine pipeline
**Cycle**: 232 | **Theme**: rustc-survey | **Kind**: investigation | **Status**: done

## Summary

Mapped the complete rustc transformation pipeline that converts `async fn` through coroutine representation into a state machine. The pipeline spans three major compiler phases (AST→HIR lowering, HIR→MIR building, MIR transforms) with the critical state machine generation happening in the `coroutine::StateTransform` MIR pass during the Analysis→Runtime phase transition. The entire state machine transform is **target-independent** — it operates purely at MIR level before any backend-specific codegen. This means warp-cooperative modifications would need to intercept at the MIR level or add a parallel codegen path.

## Findings

### Q: What is the exact transform pipeline?

The pipeline has 5 distinct stages:

#### Stage 1: AST → HIR Lowering (syntactic desugaring)

**Files:**
- `compiler/rustc_ast_lowering/src/expr.rs` — `.await` and async block desugaring
- `compiler/rustc_ast_lowering/src/item.rs` — `async fn` declaration desugaring
- `compiler/rustc_ast_lowering/src/lib.rs` — `LoweringContext` with `coroutine_kind` tracking

**Key functions:**
- `lower_maybe_async_body()` — converts `async fn foo() -> T { body }` into `fn foo() -> impl Future<Output=T> { async { body } }`
- `make_async_expr()` — wraps async block body into a coroutine via `std::future::from_generator(<static move generator>)`
- `lower_expr_await()` — desugars `.await` into a poll loop with yield:
  ```
  match ::std::future::IntoFuture::into_future(expr) {
      mut __awaitee => loop {
          match unsafe { ::std::future::Future::poll(
              Pin::new_unchecked(&mut __awaitee),
              ::std::future::get_context(task_context)
          )} {
              Poll::Ready(result) => break result,
              Poll::Pending => {
                  task_context = yield ();  // <-- becomes a yield point
              }
          }
      }
  }
  ```
- `lower_coroutine_body_with_moved_arguments()` — moves all fn arguments into the coroutine body to ensure the future owns them (stable drop order)

**Lang items referenced:** `from_generator`, `into_future`, `poll`, `Ready`, `Pending`, `get_context`

#### Stage 2: HIR Type Checking

**Files:**
- `compiler/rustc_hir_typeck/src/check.rs` — `check_fn()` returns `CoroutineTypes` for coroutine bodies
- `compiler/rustc_hir_typeck/src/fn_ctxt/` — `FnCtxt` tracks coroutine context

**What happens:** Type inference determines `resume_ty`, `yield_ty`, and return type. For async fns, `resume_ty` = `&mut Context<'_>`, `yield_ty` = `()`. The "sugared" return type (user-specified) is preserved while the actual return type is `impl Future<Output=T>`.

#### Stage 3: HIR → MIR Building

**Files:**
- `compiler/rustc_mir_build/src/build/` — MIR construction from THIR
- `compiler/rustc_middle/src/mir/mod.rs` — `CoroutineInfo` struct definition

**What happens:** The coroutine body is built as regular MIR with `Yield` terminators at each yield point. `CoroutineInfo` is attached to the `Body` with fields: `coroutine_kind`, `yield_ty`, `resume_ty`, `coroutine_drop`, `coroutine_layout`. The self argument is a coroutine type containing only upvars at this stage.

#### Stage 4: MIR Analysis Passes (borrowck, promotion)

**Queries in order:**
1. `mir_built(D)` — initial MIR, `MirPhase::Built`
2. `mir_const(D)` — constant preparation
3. `mir_promoted(D)` — constant promotion extracted, `MirPhase::Analysis`

**What happens:** Borrow checking validates the coroutine body. Promotion extracts constants. The MIR still has `Yield` terminators and coroutine-specific semantics.

#### Stage 5: MIR Runtime Lowering (THE STATE MACHINE TRANSFORM)

**Query:** `mir_drops_elaborated_and_const_checked(D)` transitions to `MirPhase::Runtime`

**Pass ordering (relevant subset):**
1. `ElaborateDrops` — creates drop flags, handles conditional drops
2. `AddMovesForPackedDrops` — handles packed struct alignment
3. `AddRetag` — inserts retag operations
4. `ElaborateBoxDerefs` — rewrites Box derefs
5. **`coroutine::StateTransform`** — THE KEY PASS (see below)
6. `KnownPanicsLint` — post-transform linting

After this, `optimized_mir(D)` runs optimization passes (inlining, const prop, etc.) on the already-transformed state machine.

### Q: Where is the state machine struct generated?

In `compiler/rustc_mir_transform/src/coroutine.rs`, specifically:

**Layout computation:**
- `locals_live_across_suspend_points()` — dataflow analysis determines which MIR locals must survive across yield points
- `compute_storage_conflicts()` — builds bitsets of locals that are never simultaneously `StorageLive` (can share space)
- `compute_layout()` — determines the final coroutine struct layout

**Struct layout:**
```
struct CoroutineStateMachine {
    upvars: (captured_var_1, captured_var_2, ...),  // captured variables
    discriminant: u32,                                // state field
    // MIR locals live across suspension points, laid out
    // with storage-conflict-aware overlap optimization
    local_3: MaybeUninit<T3>,
    local_7: MaybeUninit<T7>,
    ...
}
```

**Hardcoded discriminant states:**
- `0` — Unresumed (coroutine has not been resumed yet)
- `1` — Returned (coroutine completed normally)
- `2` — Poisoned (coroutine panicked during execution)
- `3+` — Suspended at yield point N (one per suspension point)

### Q: Where are yield points identified and state transitions emitted?

All in `compiler/rustc_mir_transform/src/coroutine.rs`:

**Yield point identification:**
- `SuspensionPoint` struct represents each yield point
- `CoroutineSavedLocals` tracks which locals are live at each suspension
- `LivenessInfo` provides dataflow-based liveness information
- Yield points come from `TerminatorKind::Yield` in the MIR (generated from the `.await` desugaring's `yield ()`)

**State transition emission:**
- `insert_switch()` — replaces the entry basic block (bb0) with a switch on the discriminant field that dispatches to the correct resume point
- `create_cases()` — generates one switch case per coroutine state (each mapping to the basic block after the corresponding yield)
- `TransformVisitor` — MIR visitor that rewrites:
  - `return x` → set discriminant to `1` (Returned), return `Poll::Ready(x)` or `CoroutineState::Complete(x)`
  - `yield y` → set discriminant to `3+N` (suspended at point N), return `Poll::Pending` or `CoroutineState::Yielded(y)`
- `return_poll_ready_assign()` — specifically converts `return x` into `Poll::Ready(x)` for async coroutines

**Panic/poison handling:**
- `insert_panic_block()` — generates a panic for resuming a completed (1) or poisoned (2) coroutine
- `generate_poison_block_and_redirect_unwinds_there()` — redirects all unwind paths to set discriminant to `2` (Poisoned)

### Q: Where is the Future::poll() wrapper generated?

**Two functions are generated by the StateTransform pass:**

1. **Resume/Poll function** — `create_coroutine_resume_function()`
   - For async coroutines: becomes the `Future::poll(&mut self, cx: &mut Context)` implementation
   - For sync coroutines: becomes `Coroutine::resume(&mut self, arg: R)`
   - Entry block switches on discriminant → dispatches to appropriate resume point
   - Self argument conversion: `make_coroutine_state_argument_indirect()` changes from pass-by-value to `&mut Self`, inserting MIR derefs
   - For async: `make_coroutine_state_argument_pinned()` applies `Pin<&mut Self>`
   - `eliminate_get_context_call()` removes the `get_context` intrinsic call, directly using the context argument

2. **Drop glue** — coroutine drop shim
   - Switches on discriminant to know which locals need dropping
   - At each suspension point, drops only the locals that are live at that point
   - Handles async drop expansion (async drop → additional yield points for poll loop)
   - Stored in `CoroutineInfo::coroutine_drop`

**The `GenFuture<T>` / `from_generator` wrapper:**
- At the library level, `std::future::from_generator()` wraps the coroutine in a `GenFuture<T>` struct
- `GenFuture`'s `Future` impl calls `gen.resume(ResumeTy(...))` and maps:
  - `GeneratorState::Yielded(())` → `Poll::Pending`
  - `GeneratorState::Complete(x)` → `Poll::Ready(x)`
- The `ResumeTy` wraps `&mut Context<'_>` behind a `NonNull` raw pointer for type-system reasons

### Q: What parts are target-independent vs target-specific?

**Entirely target-independent (MIR level):**
- ALL of the async→coroutine→state machine transformation
- AST→HIR lowering of async/await
- MIR building of coroutine bodies
- `coroutine::StateTransform` pass (layout computation, switch insertion, state transitions)
- Drop elaboration for coroutines
- The state machine struct layout computation

**Target-independent but backend-aware (codegen interface):**
- `compiler/rustc_codegen_ssa/` — abstract codegen traits all backends implement
- The state machine is just a regular struct + switch statement in MIR by this point
- Any backend (LLVM, Cranelift, GCC) can codegen it identically

**Target-specific (backend):**
- `compiler/rustc_codegen_llvm/` — LLVM IR generation from MIR
- Jump table generation for the state switch (LLVM decides branch vs jump table)
- Register allocation, calling conventions
- `PtxLinker` in `rustc_codegen_ssa::back::linker` — PTX-specific linker

**Key insight:** By the time MIR reaches codegen, the coroutine is already a plain struct with a switch-based poll function. There is NO coroutine-specific codegen logic — the backend sees ordinary MIR (struct access, switch, function calls).

### Q: Where could warp-cooperative codegen be inserted?

Four potential insertion points, ordered by increasing invasiveness:

#### 1. Library-level replacement (LEAST invasive)
**Where:** Replace `std::future::from_generator` / `GenFuture` with a GPU-aware wrapper
**How:** Custom `GpuFuture<T>` that wraps the coroutine but implements warp-cooperative polling (all threads in warp vote on state, execute same branch)
**Limitation:** Cannot change the state machine structure itself; can only wrap the poll dispatch

#### 2. Custom MIR pass AFTER StateTransform (MODERATE)
**Where:** Add a new pass after `coroutine::StateTransform` in the `mir_drops_elaborated_and_const_checked` pipeline
**How:** Post-process the generated switch statement to:
- Insert `__syncwarp()` barriers at state transition points
- Replace the switch dispatch with warp-uniform branching (all threads agree on branch)
- Add warp-ballot instructions before yield points
**Files to modify:** `compiler/rustc_mir_transform/src/lib.rs` (pass scheduling), new pass file
**Advantage:** Works on the already-lowered state machine MIR; no need to understand coroutine semantics

#### 3. Modified StateTransform pass (SIGNIFICANT)
**Where:** Fork/modify `compiler/rustc_mir_transform/src/coroutine.rs`
**How:** Generate a warp-cooperative state machine instead of per-thread:
- Discriminant stored in shared memory (one per warp)
- State transitions use warp-level voting (`__ballot_sync`)
- Yield points emit `__syncwarp()` instead of just returning
- Resume function polls all warp threads' futures together
**Key functions to modify:** `create_coroutine_resume_function()`, `insert_switch()`, `create_cases()`
**Advantage:** Full control over the state machine structure

#### 4. Custom codegen backend (MOST invasive)
**Where:** Fork `rustc_codegen_llvm` or create new backend
**How:** When emitting LLVM IR for coroutine state machine patterns, emit PTX-specific instructions:
- `bar.warp.sync` for synchronization
- Warp-shuffle for state broadcasting
- Predicated execution for divergent states
**Files:** `compiler/rustc_codegen_llvm/src/builder.rs`, `compiler/rustc_codegen_llvm/src/intrinsic.rs`
**Advantage:** Most precise control; **Disadvantage:** Massive maintenance burden

#### Recommended approach for async_gpu:
Option 2 (custom MIR pass after StateTransform) offers the best tradeoff. The state machine is already lowered to plain MIR (struct + switch), so a post-processing pass can:
1. Detect coroutine state machine patterns (switch on discriminant field)
2. Insert warp synchronization intrinsics at transitions
3. Optionally rewrite the switch to use warp-uniform dispatch
4. Remain as a separate, optional pass that can be toggled per-target

This avoids forking core rustc code while getting full access to the state machine structure.

## Key Source Files Summary

| File | Role |
|------|------|
| `compiler/rustc_ast_lowering/src/expr.rs` | `.await` → poll loop + yield; async block → coroutine |
| `compiler/rustc_ast_lowering/src/item.rs` | `async fn` → `fn` returning `impl Future` |
| `compiler/rustc_ast_lowering/src/lib.rs` | `LoweringContext`, `coroutine_kind` tracking |
| `compiler/rustc_hir_typeck/src/check.rs` | `check_fn()` extracts `CoroutineTypes` |
| `compiler/rustc_mir_build/src/build/` | HIR→MIR with `Yield` terminators |
| `compiler/rustc_middle/src/mir/mod.rs` | `CoroutineInfo`, `Body` definition |
| `compiler/rustc_middle/src/mir/syntax.rs` | `MirPhase`, `TerminatorKind::Yield` |
| **`compiler/rustc_mir_transform/src/coroutine.rs`** | **THE state machine transform** |
| `compiler/rustc_mir_transform/src/coroutine/by_move_body.rs` | Async closure FnOnce body |
| `compiler/rustc_mir_transform/src/coroutine/drop.rs` | Coroutine drop shim generation |
| `compiler/rustc_mir_transform/src/lib.rs` | Pass pipeline scheduling |
| `compiler/rustc_codegen_ssa/` | Backend-agnostic codegen traits |
| `compiler/rustc_codegen_llvm/` | LLVM backend (incl. NVPTX) |
| `library/core/src/future/mod.rs` | `from_generator`, `GenFuture`, `poll` |

## Open Questions

1. **Has `from_generator`/`GenFuture` been removed in recent nightly?** Some compiler changes suggest the wrapping layer may have been simplified or inlined. Need to verify against latest nightly source.
2. **Async drop expansion:** The `StateTransform` pass can expand async drops into additional yield points. How does this interact with warp-cooperative scheduling?
3. **Coroutine layout optimization:** The storage-conflict analysis overlaps locals that are never simultaneously live. Does this optimization interact poorly with warp-level shared state?
4. **`ResumeTy` elimination:** PR #105977 replaced `ResumeTy` locals with direct `&mut Context<'_>`. Need to verify this is the current state in nightly 2025-08-25+.

## Impact on Downstream Tasks

- **Warp-cooperative executor design:** The state machine is a plain struct + switch by the time it reaches codegen. A warp-cooperative executor can wrap the poll function without compiler modifications.
- **MIR pass approach:** A custom MIR pass after `StateTransform` is viable for inserting sync barriers. This could be implemented as a `#[cfg(target_arch = "nvptx64")]`-conditional pass.
- **No codegen fork needed:** Since the transform is entirely at MIR level, the existing LLVM NVPTX backend can codegen the modified MIR without changes.
- **proc-macro alternative:** The existing `#[warp_async]` proc macro approach (from the warp-future epic) operates at the source/AST level, which is complementary to a MIR-level approach. Both could coexist.
