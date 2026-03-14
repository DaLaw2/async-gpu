# bs57 — Skeptic: Native `async fn` → Warp-Cooperative State Machine on GPU

**Role**: Skeptic | **Date**: 2026-03-14

## Framing

I accept the premise: upstream acceptance is irrelevant, fork maintenance is an acceptable cost if the proof is compelling. My job is to find the **real technical risks** — the places where this approach might be fundamentally impossible, silently incorrect, or so much harder than it appears that the effort dwarfs the benefit over the working proc-macro system.

---

## Challenge 1: MIR Cannot Emit Inline Assembly

**The proposal's central assumption is wrong.** A post-StateTransform MIR pass cannot directly insert `shfl.sync`, `syncwarp`, or `activemask` instructions.

MIR operates on Rust-level constructs: assignments, function calls, switches, drops. There is no `TerminatorKind::InlineAsm` or `StatementKind::InlineAsm` that lets you emit arbitrary PTX at MIR level... actually, wait — `InlineAsm` *does* exist in MIR as `TerminatorKind::InlineAsm`. But here's the real problem:

**MIR inline asm is target-independent syntax.** You write `asm!("shfl.sync.idx.b32 ...")` in Rust source, and it becomes a `TerminatorKind::InlineAsm` node. But to *construct* this node programmatically in a MIR pass, you need to:

1. Create the inline asm template string with correct PTX syntax
2. Set up operand bindings (input/output registers) using MIR `Operand` and `Place`
3. Handle the PTX register class constraints (`reg32`, `reg64`)
4. Ensure the asm constraints match the LLVM NVPTX backend expectations

**Risk**: The MIR `InlineAsm` representation uses `InlineAsmRegOrRegClass` which maps to LLVM register classes. The MIR pass must know which register class to use for NVPTX (`reg32` for shfl.sync outputs). This means the "target-independent" MIR pass is actually deeply target-coupled. You're building a pass that only makes sense for one target and must know LLVM NVPTX register class names. This isn't conceptually fatal, but it eliminates the claimed elegance of "just a MIR pass."

**Alternative that actually works**: Call an `extern "C"` function (`__gpu_syncwarp()`, `__gpu_shfl_sync()`) from MIR, which is trivial — just insert a `TerminatorKind::Call`. These functions are defined in `gpu-atomics` with inline PTX. The MIR pass inserts calls to these functions, and `#[inline(always)]` + LTO ensures they compile to single instructions.

**But this creates a new problem**: The MIR pass must reference these functions by `DefId`. How does a compiler pass find a function defined in an external crate? It needs to resolve `gpu_atomics::shfl_sync_idx_u32` at MIR-transform time, which requires the crate to be loaded and the symbol resolved. This is doable (rustc does it for lang items), but requires registering these functions as lang items or diagnostic items in the fork, adding ~50-100 lines of plumbing per intrinsic.

**Verdict**: Technically feasible but significantly more complex than "insert a MIR node." Estimate: 2-3 weeks just for the intrinsic plumbing.

---

## Challenge 2: State Struct Broadcasting is an Unsolved Problem

The StateTransform pass generates a coroutine struct like:

```
struct Coroutine {
    discriminant: u32,
    field_a: SomeType,      // live across yield point 0
    field_b: AnotherType,   // live across yield point 1
    upvar_buf: *mut u8,     // captured variable
}
```

For warp-cooperative execution, lane 0 owns the state and other lanes need to see it. The existing `WarpFuture` system broadcasts individual `u32` values via `shfl.sync.idx.b32`. But `shfl.sync` operates on exactly 32-bit values. To broadcast arbitrary state:

**Problem A: Multi-word fields.** A `u64` field requires two shuffles. A `*mut u8` on nvptx64 is 64 bits — two shuffles. A `[u8; 44]` message buffer? 11 shuffles. The MIR pass must decompose every cross-yield-point local into 32-bit chunks and reconstruct them. This requires knowing the size and layout of every type at MIR level — which is available via `TyCtxt::layout_of()`, but the decomposition logic is non-trivial.

**Problem B: Not all fields need broadcasting.** The discriminant must be broadcast (all lanes need to agree on which state). But data fields that are only accessed by lane 0 (e.g., packet index, file descriptor) don't need broadcasting — they're leadership-only state. The MIR pass cannot know which fields are "leader-only" without semantic annotation. The current proc macro handles this because the programmer explicitly writes `warp_print!(buf, ...)` where the macro knows buf is uniform and the packet index is leader-only.

**Problem C: The compiler's state machine packs fields with overlap.** The `compute_storage_conflicts()` analysis allows fields that are never simultaneously live to share the same memory. After StateTransform, the struct layout uses `MaybeUninit` with overlapping storage. Broadcasting the "active" interpretation of overlapping bytes requires knowing which state you're in *before* you interpret the bytes — which requires the discriminant to be broadcast first, then conditionally broadcast only the fields valid in that state.

**This is the approach the current hand-written WarpFuture already takes**: broadcast discriminant, then in each match arm, broadcast only the relevant fields. But a generic MIR pass would need to do this automatically, which requires a per-state field liveness analysis on the *transformed* MIR (not the original coroutine's liveness, which is already consumed).

**Verdict**: This is the hardest technical problem. It requires re-doing liveness analysis on the transformed state machine, decomposing types into 32-bit chunks, and generating conditional broadcast sequences per state. Estimate: 4-8 weeks of compiler engineering.

---

## Challenge 3: Pin<&mut Self> and Self-Referential State

Rust's `async fn` futures can be self-referential. Consider:

```rust
async fn example(data: Vec<u8>) {
    let slice = &data[..];  // borrows data
    some_io().await;         // yield point — slice and data both live across suspension
    use_slice(slice);
}
```

After StateTransform, the coroutine struct contains both `data: Vec<u8>` and `slice: &[u8]` where `slice` points into `data`'s heap allocation. `Pin<&mut Self>` guarantees the struct won't move, so the self-reference remains valid.

**For warp-cooperative execution**: If lane 0 holds the pinned state and broadcasts fields via `shfl.sync`, the `slice` pointer broadcast to other lanes still points into **lane 0's** `data` allocation. Other lanes cannot dereference it (the pointer is in lane 0's local registers or memory).

**But wait — is this actually a problem?** In the current WarpFuture model, non-leader lanes don't access the state fields directly. They receive broadcast values (discriminant, specific u32/u64 values) and use those for cooperative work. The self-referential pointer would only be dereferenced by lane 0, which is the owner.

**The real risk**: A generic MIR-level transformation doesn't know which pointer dereferences are "lane 0 only" vs "all lanes." If the MIR pass blindly inserts broadcasts for all cross-yield locals, it would broadcast stale pointers that non-leader lanes might try to dereference. The pass would need to understand the leader/follower distinction — which is a semantic concept that doesn't exist in standard MIR.

**Mitigation**: If the approach only broadcasts the discriminant and requires explicit `shfl` calls for data sharing (like today's model), self-referential state stays in lane 0 and is never broadcast. But then the MIR pass is doing very little: inserting `syncwarp` and a discriminant broadcast. Most of the warp-cooperative logic still lives in user code or library code.

**Verdict**: Not a fundamental blocker, but constrains the MIR pass to minimal intervention (discriminant broadcast + sync barriers only). The heavy lifting of data sharing stays in user/library code, reducing the value of the rustc approach.

---

## Challenge 4: Drop Glue — Who Runs the Destructor?

When a future is dropped mid-execution (e.g., `select!` cancels one branch), the compiler generates drop glue that switches on the discriminant to drop only the locals live at the current suspension point.

**On GPU with warp-cooperative execution**:

- If lane 0 is the state owner, only lane 0 should run the drop logic (it holds the real state).
- But `syncwarp` barriers in the drop path would deadlock if only lane 0 enters the drop while other lanes are doing something else.
- If all lanes must enter the drop path together, they must all agree on the discriminant — requiring a broadcast in the drop glue too.

**The fundamental issue**: Drop is *implicit*. The compiler inserts it, and a MIR pass would need to find and modify every drop path for the coroutine type. The coroutine drop shim is generated by `create_coroutine_drop_shim()` — a separate function stored in `CoroutineInfo::coroutine_drop`. The MIR pass would need to modify this shim too, not just the poll function.

**Practical consideration**: On GPU, `select!`-style cancellation doesn't exist yet. All existing WarpFutures run to completion. If the project commits to "no mid-flight cancellation for warp-cooperative futures," this problem disappears entirely. But that's a semantic restriction that native `async fn` does *not* have, and users will eventually hit it.

**Verdict**: Not a blocker if you accept the restriction "warp-cooperative futures always run to completion." Adding this restriction defeats part of the goal of supporting native `async fn` semantics.

---

## Challenge 5: Nested `.await` Does NOT Compose Trivially

```rust
async fn outer() {
    inner().await;  // inner is also async fn
}
```

The compiler inlines `inner`'s state machine into `outer`'s coroutine struct. After StateTransform, `outer`'s struct contains `inner`'s state as a nested enum, and `outer`'s poll function contains `inner`'s poll logic inlined.

**For warp-cooperative lowering**: If both `outer` and `inner` need syncwarp/shfl, the MIR pass would insert warp sync into both the outer and inner poll logic. But inner's poll logic is now *inlined into* outer's body. This means:

1. **The pass must recognize nested state machines.** The inner future's discriminant is a field of the outer struct. The pass must find the inner switch-on-discriminant pattern and insert warp sync there too — not just at the top-level switch.

2. **Double-broadcast problem.** Outer broadcasts its discriminant to enter the state that polls inner. Then inner's discriminant must also be broadcast. This is two levels of broadcast per poll. With 3 levels of nesting, it's 3 broadcasts. Each broadcast is a `shfl.sync` instruction (cheap but not free).

3. **Partial completion semantics.** If inner completes (returns `Ready`), outer must transition to its next state. But in warp-cooperative mode, all lanes must agree that inner completed. This requires broadcasting inner's return value (via `shfl`) before outer can transition. The MIR pass must insert this broadcast at every point where a nested `.await` resolves.

**The proc macro doesn't have this problem** because `warp_print!`, `warp_open!`, etc. are leaf operations — they don't nest. The macro generates a flat state machine. If you wanted nested warp-async calls with the macro, you'd need to manually flatten them.

**But this is precisely where the rustc approach should shine** — handling nesting automatically. If it can't, the value proposition is severely weakened.

**Verdict**: This is solvable but requires the MIR pass to be recursive/pattern-matching on nested state machines. Adds significant complexity (estimate: 2-3 weeks beyond the base pass).

---

## Challenge 6: What Does the Rustc Route Buy Over the Proc Macro?

Let me be specific. The current `#[warp_async]` proc macro:

**What it does well:**
- Transforms sequential code with `warp_print!`, `warp_open!`, `warp_read!`, etc. into flat state machines
- Handles `if/else`, `match`, `loop/break`, nested control flow
- Generates correct leader/follower patterns with `broadcast_u32`, `syncwarp`
- Works TODAY on stable nightly, no fork needed
- ~1400 lines of proc macro code (manageable)

**What it cannot do (and the rustc route theoretically can):**

1. **Support arbitrary `.await` expressions.** The macro only supports `warp_*!()` built-in calls. You can't `.await` an arbitrary future. With the rustc route, `some_future.await` would generate warp-cooperative code automatically.

2. **Compose with the ecosystem.** Library futures (tokio, embassy, custom) could theoretically become warp-cooperative automatically. The macro requires wrapping every async operation in a `warp_*!()` call.

3. **Handle complex data flow.** The macro operates on AST patterns. It doesn't understand types, borrowing, or lifetimes. The rustc route has full type information via `TyCtxt`.

4. **Support closures and async closures.** The macro can't transform closures. Rustc handles them natively.

**But consider the realistic scope:**

- On GPU, you don't have tokio. You have hostcalls (print, open, read, write, close). That's 5 operations.
- The "ecosystem" on nvptx64 is this project. There are no third-party async GPU crates to compose with.
- Complex data flow on GPU means shared memory, registers, and global memory — all of which require explicit management regardless of the async model.
- Closures containing warp operations are an edge case; the primary pattern is sequential pipelines.

**The honest assessment**: For the current and foreseeable use cases (hostcall-based I/O), the proc macro covers ~95% of the functionality. The rustc route buys:
- **Aesthetics**: `some_io().await` instead of `warp_open!(buf, ...)`
- **Nested composition**: Genuinely useful if you build a library of async GPU primitives
- **Proof of concept**: Demonstrates that warp-cooperative semantics can be first-class in the language

The third point — proof of concept — is the strongest argument, and I acknowledge it's explicitly the project goal.

**Verdict**: The rustc route is a research investment with high novelty value but marginal practical benefit over the proc macro for the current GPU I/O use cases. This is fine if the goal is research/proof, but the cost must be weighed honestly.

---

## Challenge 7: Testing and Correctness — The Convergence Oracle Problem

The hardest class of bugs in GPU warp programming is **silent divergence**: lanes that should be executing the same code path are actually in different states. The warp doesn't crash — it just computes wrong results.

**Testing the MIR pass requires**:

1. **A correctness specification.** What does "correct warp-cooperative state machine" mean formally? The current WarpFuture has an informal invariant: "all active lanes are in the same match arm at every poll." But this invariant is maintained by construction in the hand-written code. A compiler pass must maintain it by transformation, which requires:
   - Proving that the inserted `syncwarp` barriers are in the right places
   - Proving that the discriminant broadcast happens before any state-dependent access
   - Proving that no lane can "skip ahead" past a sync barrier

2. **An oracle for differential testing.** You could run the same async fn as (a) a regular per-thread future and (b) a warp-cooperative future, then compare results. But the semantics differ: the warp version has shared state while the per-thread version has independent state. They compute different things. You'd need a semantic oracle that understands the expected mapping.

3. **Detection of divergence.** If lane 17 thinks it's in state 3 while lane 0 thinks it's in state 2, the program doesn't crash — it reads the wrong fields, computes garbage, and writes it out. To detect this, you'd need runtime assertions: after every syncwarp + broadcast, verify all lanes see the same discriminant. This adds overhead but is essential for testing.

4. **Stress testing nondeterminism.** Warp-cooperative bugs are often timing-dependent. A hostcall that completes faster than usual might cause the state machine to skip a sync point. You need adversarial host response timing.

**What the project currently does**: Each WarpFuture kernel has a corresponding host test that launches it and checks output values. This is end-to-end testing, not unit testing of the state machine itself. For a compiler pass, you'd want MIR-level tests (check the generated MIR against expected patterns), PTX-level tests (check the emitted assembly), and end-to-end GPU tests.

**Verdict**: Testing is feasible but requires building significant infrastructure. The proc macro can be tested by examining generated code (it's Rust source). The MIR pass output is MIR, which is less inspectable. Estimate: 2-3 weeks for testing infrastructure alone.

---

## Challenge 8: The Real Cost of Forking Rustc

Even without upstream concerns, let me quantify the engineering cost:

### Initial implementation

| Task | Estimate |
|------|----------|
| Fork rustc, set up build system | 1 week |
| Implement intrinsic plumbing (lang items for shfl/sync/activemask) | 2 weeks |
| Implement the MIR pass (basic: discriminant broadcast + syncwarp insertion) | 3-4 weeks |
| Handle state field decomposition and broadcasting | 4-6 weeks |
| Handle nested state machines | 2-3 weeks |
| Handle drop glue | 1-2 weeks |
| Testing infrastructure | 2-3 weeks |
| Integration with existing runtime | 1-2 weeks |
| **Total initial implementation** | **16-23 weeks (~4-6 months)** |

### Ongoing maintenance

- **Nightly pin**: The fork must track a specific nightly. Every developer must build the custom compiler. Build time: ~30-60 minutes for a full rustc build, ~5-10 minutes for incremental.
- **LLVM updates**: Rust updates LLVM roughly every 6 months. Each update can change NVPTX code generation, register allocation, and inline asm handling. Historical example: the NVVM atomics broke on a recent nightly (documented in MEMORY.md).
- **MIR representation changes**: The MIR data structures change periodically. `TerminatorKind`, `StatementKind`, `Body` — any change requires updating the pass. Estimate: 1-2 weeks per quarter.
- **Coroutine transform changes**: Rust is actively evolving async (async closures, async drops, generators). Each change to `coroutine.rs` may require updating the warp pass. Estimate: 1-3 weeks per breaking change, ~2-4 per year.

**Annual maintenance estimate**: 8-16 weeks/year (2-4 months/year)

### Comparison with proc macro maintenance

The proc macro requires updating when:
- Syn/quote/proc-macro2 update (rare, usually backward compatible)
- New warp operations are added (incremental: ~1 day each)
- New control flow patterns are added (moderate: ~1 week each)

**Annual proc macro maintenance**: ~2-4 weeks/year

### Developer experience cost

Every contributor must:
1. Clone the rustc fork
2. Build the custom compiler (30-60 min first time)
3. Configure rust-analyzer to use the custom compiler
4. Debug with a non-standard compiler (GDB/LLDB breakpoints, MIR dumps)
5. Keep the fork updated when upstream changes

This is non-trivial friction for contributors, though acceptable for a research project with a small team.

**Verdict**: The fork costs ~5 months initial + ~3 months/year ongoing, versus ~0 months initial + ~0.5 months/year for the proc macro. The fork is ~10x more expensive. Whether this is "acceptable" depends on how much the proof-of-concept matters to the project's goals.

---

## Summary: Risk Matrix

| Challenge | Severity | Solvable? | Effort |
|-----------|----------|-----------|--------|
| 1. MIR cannot directly emit PTX (needs intrinsic plumbing) | Medium | Yes | 2-3 weeks |
| 2. State struct broadcasting (multi-word, overlapping, conditional) | **High** | Probably | 4-8 weeks |
| 3. Pin + self-referential state | Low | Yes (by constraint) | 1 week |
| 4. Drop glue for warp state | Medium | Yes (by restricting cancel) | 1-2 weeks |
| 5. Nested .await composition | **High** | Yes (but complex) | 2-3 weeks |
| 6. Marginal benefit over proc macro | Strategic | N/A (research goal) | N/A |
| 7. Convergence testing oracle | Medium | Yes (with infrastructure) | 2-3 weeks |
| 8. Fork cost (5mo + 3mo/yr) | Strategic | N/A (cost decision) | 16-23 weeks |

## Bottom Line

The proposal is **technically feasible but substantially harder than it appears.** The two killer problems are (2) state struct broadcasting and (5) nested .await composition. These require deep compiler engineering — not because they're theoretically impossible, but because they require re-implementing dataflow analyses that the compiler already ran (but whose results are consumed and discarded by the time the warp pass runs).

The proc macro already solves the practical problem. The rustc route is a research statement: "warp-cooperative execution can be a first-class compiler concern." If that statement is worth 5 months of engineering + ongoing maintenance, the approach is viable. If the project has higher-priority work, the proc macro is a very good solution that should not be undervalued.

**My recommendation**: Before committing to the full fork, do a **minimal spike** (2-3 weeks): implement just the discriminant-broadcast + syncwarp MIR pass for a trivial single-yield-point async fn. If that works, you have evidence the approach is tractable. If it reveals unexpected obstacles in MIR construction or LLVM codegen, you've saved months.
