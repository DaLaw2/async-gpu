# bs21-skeptic: Challenges to GPU-Native Async Pipeline proposal

**Date**: 2026-03-12
**Role**: Skeptic
**Level**: deep (proposer + skeptic)

---

## Challenge 1: The proc macro expansion is a dead end — don't invest more

**Claim challenged**: Expanding `#[warp_async]` to support variable bindings, control flow, and arbitrary hostcall types will make GPU async ergonomic.

**Counterargument**: The current `#[warp_async]` is 430 lines of proc macro code that can ONLY handle `warp_print!()`. Adding support for `warp_read!()`, `warp_write!()`, `warp_open!()`, variable bindings between yield points, conditional logic, and loops would require essentially writing a mini-compiler in proc_macro land — without access to type information, MIR, or any of the tools that make this tractable. Every new hostcall type doubles the match arm generation complexity. Control flow (if/else around a yield point) requires splitting into exponentially many states. Loops with yields inside are fundamentally impossible in a proc macro without reimplementing the generator transform.

The honest assessment: `#[warp_async]` will always be a thin wrapper for linear sequences of macro calls. That is fine. The hand-written WarpFuture state machine (which already works and is proven) is the real API. The proc macro is syntactic sugar for the trivial case.

**Risk if ignored**: Months of proc macro engineering that produces a fragile, hard-to-debug macro supporting 60% of use cases poorly, when hand-written state machines cover 100% of use cases well. The proc macro becomes the maintenance bottleneck.

**Recommendation**: Cap proc macro investment at supporting `warp_print!` + `warp_write!` + `warp_read!` in linear sequence. No control flow. No variable bindings across yield points. Document the hand-written pattern as the "advanced" (actually: normal) API. Stop.

---

## Challenge 2: "GPU self-coordinating multi-step pipelines" is not a real use case

**Claim challenged**: The vision of GPU as an autonomous compute environment running multi-step async pipelines justifies significant infrastructure investment.

**Counterargument**: Name one concrete, performance-justified workload where a GPU kernel needs to:
1. Open a file
2. Read data
3. Process it per-lane
4. Write results
5. Read more data
6. Process again
7. Write final output

...all within a single kernel launch, via hostcall round-trips at ~13us each.

A CPU can do steps 1-2 and 5-6 in bulk, launch a GPU kernel for steps 3 and 6, and do step 4 and 7 on the CPU — all with zero hostcall overhead. The only scenario where GPU-initiated I/O wins is when the GPU needs to make data-dependent I/O decisions (e.g., "read block X only if hash matches"). But this is a niche use case, not a pipeline.

The VectorWare blog demonstrates the concept. Demonstrating the concept is the vision. A 3-step pipeline (print, compute, print) already demonstrates it. A 7-step pipeline with file I/O is not qualitatively different — it's quantitatively more of the same.

**Risk if ignored**: Building infrastructure for a use case nobody has. The project becomes an architecture astronaut exercise.

---

## Challenge 3: The "per-thread blocks can't yield" limitation is load-bearing and being hand-waved

**Claim challenged**: Hybrid executor (ADR-10) cleanly handles mixed warp-cooperative I/O and per-thread computation by forbidding yields in per-thread blocks.

**Counterargument**: The constraint "per-thread blocks MUST NOT yield" is the fundamental limitation of the entire WarpFuture model, and it's being treated as a minor footnote. Consider real workloads:

- **GPU-side parsing**: Lane 3 needs to read more data because its chunk contained an escape sequence. It MUST yield (hostcall read). But it can't — it's in a per-thread block. The entire warp must exit the compute block, re-enter a warp-cooperative I/O phase, all 32 lanes request data (31 of them unnecessarily), then re-enter compute. This is exactly the 1/32 SIMT waste that WarpFuture was supposed to avoid, just at a different level.

- **Iterative algorithms**: Lane 17 converges in 10 iterations, lane 3 needs 10,000. syncwarp() handles this correctly, but lane 17 burns cycles spinning in a loop doing nothing for 9,990 iterations. This is acceptable for short compute blocks but problematic for anything substantial.

- **Error handling**: Lane 5 encounters an error mid-computation. What does it do? It can't yield with an error. It can't return early. It must either store the error and continue computing garbage, or set a flag that ALL lanes check after the compute block. Neither is clean.

The honest truth: WarpFuture works beautifully for the pattern "all lanes do the same I/O, all lanes do similar-length computation." It degrades to per-thread-level efficiency for anything else. The proposal needs to be honest about this boundary.

**Risk if ignored**: Users try to build real workloads, hit the yield restriction, discover that WarpFuture doesn't work for their use case, conclude the entire system is a toy.

---

## Challenge 4: Priority inflation — not everything is P0

**Claim challenged**: The proposed task list likely marks pipeline infrastructure, new hostcall types, and proc macro expansion as P0.

**Counterargument**: The user said "stop when the vision is achieved." The vision from the VectorWare blog is:
1. Rust std on GPU (done)
2. Async/await on GPU (done — per-thread Embassy executor works)
3. Warp-level async (done — WarpFuture + `#[warp_async]` with print works)
4. Hybrid execution (done — per-thread compute blocks in WarpFuture proven)

What's actually missing? A convincing DEMO that ties it all together. Not more infrastructure. Not more hostcall types. Not a more powerful proc macro. A single example that does: warp_print → per-thread compute → warp_print, showing real data flowing through, with measured SIMT utilization.

That's ONE task, not an epic.

**Risk if ignored**: Scope creep. The project never ships because there's always one more feature to add before the "vision" is complete.

---

## Challenge 5: 25 completed themes and 88 tasks — is MORE the answer?

**Claim challenged**: The project needs a new epic with multiple new themes and tasks to achieve the vision.

**Counterargument**: The project has been in research mode for 88 tasks across 25 themes. At some point, more research has negative returns. The codebase has:
- Working GPU compilation pipeline
- Working hostcall protocol with file I/O
- Working Embassy async executor on GPU
- Working WarpFuture with proc macro
- Working hybrid executor with per-thread compute blocks
- Working panic handler, sideband buffer, per-block sharding
- Benchmarks, CI, documentation, examples

What is a "GPU-Native Async Pipeline" epic going to prove that isn't already proven? The components exist. The architecture is validated. The measurements are taken. Adding more tasks is avoiding the conclusion: the research is done.

**Risk if ignored**: The project loops indefinitely, adding features that don't advance the research conclusion.

---

## Challenge 6: The sideband buffer and large-payload work are premature for the pipeline vision

**Claim challenged**: Bulk data transfer via sideband buffer is needed for real async pipelines.

**Counterargument**: The sideband buffer (ADR-7) already exists and works. `SERVICE_BULK_WRITE` and `SERVICE_BULK_READ` are implemented. What more is needed? If the argument is "we need a warp-cooperative version of bulk read/write," that's a 2-hour task: write a WarpFuture that calls the existing bulk services, same as the print WarpFuture but with different service IDs. This does not justify an epic.

**Risk if ignored**: Building infrastructure for infrastructure's sake.

---

## Challenge 7: The proc macro hides complexity users need to understand

**Claim challenged**: `#[warp_async]` improves ergonomics and should be the primary API.

**Counterargument**: Anyone writing GPU async code needs to understand:
- Warp convergence and when it breaks
- The syncwarp() barrier and why it's needed
- Lane 0 leadership pattern
- The "no yield in per-thread blocks" invariant
- Packet payload layout and coalesced writes

The proc macro hides ALL of this. A user who writes:
```rust
#[warp_async]
unsafe fn my_pipeline(buf: *mut u8) -> bool {
    warp_print!(buf, b"hello");
    warp_print!(buf, b"world");
}
```
...has no understanding of what happens when they want to add a per-thread compute block between the prints. The macro can't help them. They must learn the hand-written pattern anyway.

The proc macro serves one purpose: making the initial demo look clean. That purpose is already achieved. Expanding it further is complexity without proportional value.

**Risk if ignored**: Two-tier API where the "easy" tier (proc macro) covers 10% of use cases and the "real" tier (hand-written state machines) covers 100%. Users learn the wrong API first and must unlearn it.

---

## What I Agree With

1. **WarpFuture is architecturally correct**: The insight that per-thread Futures cause 1/32 SIMT utilization for divergent states is real and important. WarpFuture is the right answer for I/O-heavy patterns.

2. **The hostcall packet layout was prescient**: The 32-lane x 8-slot layout fits WarpFuture perfectly. One packet per warp instead of 32 packets per warp is a genuine architectural win.

3. **Hybrid execution is a good design**: ADR-10's per-thread compute blocks are the pragmatic way to mix SIMT-convergent I/O with divergent computation. The limitations are real but the design is honest about them.

4. **The project has achieved something genuinely novel**: Rust async/await on GPU, with warp-level convergence, hostcall RPC, file I/O, panic handling — this is real systems research with real results.

5. **The demo/example approach is right**: Proving the vision through working examples is better than building abstract infrastructure.

---

## Minimum Viable Vision

The absolute minimum to demonstrate "GPU-native async pipeline" is:

### One example kernel that does all three phases:
1. **Warp-cooperative I/O**: Read input data via hostcall (or print a status message)
2. **Per-thread compute**: Each lane processes its portion independently
3. **Warp-cooperative I/O**: Write results via hostcall (or print computed results)

### This already exists as hybrid-executor.1.

The question is: does hybrid-executor.1 constitute "vision achieved"? I argue YES, with one addition:

### The one thing actually missing:
A **clean, documented example** (in `examples/`) that a human can read and understand in 5 minutes. Not a test kernel buried in a test crate. An example with comments explaining:
- What WarpFuture is and why it exists
- How the state machine works
- Where the per-thread compute block is
- What the output means

That's it. One file. One afternoon. Vision achieved. Stop.

### What to cut:
- New proc macro features (warp_read!, warp_write!, control flow) — defer indefinitely
- New hostcall service types for warp-cooperative mode — existing services work fine
- Pipeline orchestration infrastructure — YAGNI
- Additional stress tests beyond hybrid-executor.2 — diminishing returns
- Any new themes — the project has 25 completed themes, that's enough
