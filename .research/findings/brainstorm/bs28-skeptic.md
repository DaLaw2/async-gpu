# BS28 Skeptic — Warp-Cooperative Async with Full Control Flow via Rustc Modifications

**Date**: 2026-03-12
**Role**: Skeptic
**Topic**: Challenging the proposer's analysis of rustc modification for warp-cooperative async

---

## 1. Feasibility Challenges

### 1.1 Forking Rustc Is a Research Project Killer, Not an Enabler

The proposer acknowledges the maintenance burden but drastically underestimates it. "1-2 days per nightly update" is best-case fiction. Here is reality:

- **Rustc's async/generator infrastructure has been actively refactored throughout 2025-2026.** The `generator.rs` file alone has seen the generator → coroutine rename, changes to `CoroutineLayout`, and reworked `StateTransform`. A patch touching these structures will break on approximately every third nightly.
- **Rebase conflict resolution requires deep compiler knowledge.** When upstream changes a data structure your patch touches, you don't just resolve textual conflicts — you must understand the semantic change and update your transform accordingly. This is not "1-2 days" work; it is "1-2 days if you're lucky, 1-2 weeks if a MIR refactor happened."
- **The nvptx64 backend in LLVM is barely maintained.** It has a single-digit number of active contributors. Bugs filed against it sit open for months. The proposer's design assumes LLVM's NVPTX backend will correctly lower the warp intrinsics and guard patterns. If it doesn't, you're now debugging LLVM too — a second compiler fork.
- **Bus factor = 1.** If the sole maintainer is unavailable for two months, the fork falls behind 60+ nightlies. Catching up requires replaying every conflict across those nightlies or rebasing onto a drastically different codebase. In practice, the fork dies.

### 1.2 The "4-8 Weeks for a Minimal Prototype" Estimate Is Unrealistic

Building rustc from source takes 30-60 minutes per iteration. A minimal MIR pass touching generator internals will require dozens of build-test cycles. Factor in:
- Learning the MIR representation (undocumented internal API)
- Understanding how `StateTransform` actually works (the code is ~1500 lines of dense, uncommented MIR manipulation)
- Debugging MIR passes (no debugger support, printf-debugging via `-Zdump-mir`)
- Getting the generated PTX to actually run on GPU (LLVM NVPTX backend quirks)

A more realistic estimate: 12-16 weeks for a minimal linear-pipeline prototype via the fork route. The proposer's Option B (custom driver) estimate of "6-10 weeks, harder than a fork" is actually more honest.

### 1.3 The Project Already Pins to a Specific Nightly

The project uses `nightly-2025-08-25` because NVVM intrinsics broke on later nightlies (per project memory). A rustc fork pins you even harder — you can't move to ANY other nightly without re-verifying the entire patch set. If a critical bug is fixed in a later nightly, you're stuck choosing between "fix our fork" and "live with the bug."

---

## 2. Technical Holes

### 2.1 Nested Async Calls Are Unaddressed

The proposer's examples are all single-level: one `async fn` with `warp_*` calls. Real code will have:

```rust
#[warp_async]
async fn outer(buf: &WarpBuf) {
    let fd = warp_open(buf, "file.txt").await;
    process_file(buf, fd).await;  // calls another warp_async fn
    warp_close(buf, fd).await;
}

#[warp_async]
async fn process_file(buf: &WarpBuf, fd: u64) {
    // ...more awaits...
}
```

How does the state machine compose? Standard Rust async nests generators — the outer generator's state includes the inner generator as a variant field. For warp futures, the nested state machine must ALSO broadcast from lane 0. The proposer's MIR pass would need to recognize that an inner `.await` is on a `WarpFuture`, not a regular `Future`, and ensure the broadcast/barrier invariants hold transitively. This is a significant complexity not even mentioned.

### 2.2 Drop and Destructors Are a Landmine

Consider:

```rust
#[warp_async]
async fn example(buf: &WarpBuf) {
    let guard = SomeGuard::new();  // has Drop impl
    if condition().await {
        return;  // Drop runs here for guard
    }
    do_something(buf).await;
    // Drop runs here for guard
}
```

When a `WarpFuture` is dropped (cancelled), destructors for live-across-yield locals run. But which lane runs the destructor? Lane 0 only? All lanes? If the destructor has side effects (releases a resource, writes to memory), running it on all 32 lanes is wrong. Running it on lane 0 only requires the destructor to know it's in a warp context. The proposer's invariant "field mutations are lane-0 only" implies Drop should only run on lane 0, but standard Rust Drop doesn't know about lanes.

This is not a theoretical concern — the standard generator transform generates Drop glue for each state variant. The warp MIR pass would need to wrap ALL drop calls in `if lane_id == 0` guards, which means understanding the drop elaboration pass and how it interacts with generator state.

### 2.3 Panics in Warp-Async Code

If lane 0 panics during a branch condition evaluation, what happens to lanes 1-31? They're waiting for a broadcast that never comes. The proposer says "Lane 0 is the decision maker" but doesn't address lane 0 failure. The existing panic handler (ADR-5) sends a hostcall and traps — but only for the panicking thread. The other 31 lanes are now in undefined behavior territory (reading uninitialized broadcast values or spinning forever).

### 2.4 Memory Layout and Broadcast Friendliness

The proposer claims "the struct is identical to a standard generator struct." This is dangerously wrong for GPU performance. A standard generator struct has:

- Discriminant (u32)
- Saved locals of arbitrary size and alignment (u8, u16, u32, u64, arrays, nested structs)

The `shfl.sync.idx.b32` instruction broadcasts exactly 32 bits. Broadcasting a u64 requires two shuffles. Broadcasting a `[u8; 56]` (a path string) requires 14 shuffles. If the state machine saves a large local across a yield point, the broadcast cost scales linearly with the local's size.

The proposer's design broadcasts ALL fields from lane 0. For a state machine with 5 saved locals totaling 64 bytes, that's 16 shuffle instructions PER poll. Compare to the current hand-written WarpFuture, which broadcasts only the discriminant and reads shared data from the buffer pointer (which all lanes can access directly).

**The hand-written approach is actually more efficient because the programmer knows which values need broadcasting and which can be read from shared memory.**

### 2.5 Register Pressure

GPU register files are precious. Each SM has 65,536 registers shared across all threads. The generator state machine struct lives in registers (no heap on GPU). A state machine with 5 yield points and 3 saved locals already consumes:
- 1 discriminant (1 reg)
- 3 saved u64 locals (6 regs)
- Temporary registers for shuffle instructions (2-4 regs per broadcast)
- Branch condition registers
- State dispatch switch (jump table or cascaded branches)

For comparison, the hand-written WarpFuture in async-pipeline.3 uses a compact struct with known register footprint. A compiler-generated state machine cannot optimize for register pressure because it doesn't know the GPU's constraints. It will save any local that might be needed across a yield, even if the programmer could trivially restructure the code to avoid it.

### 2.6 The "All Branches Must Be Warp-Uniform" Constraint Is Extremely Limiting

The proposer's invariant #2 states: "When any lane executes a yield, ALL lanes must yield." This means:

```rust
#[warp_async]
async fn example(buf: &WarpBuf) {
    let data = warp_read(buf, fd, 1024).await;
    // Can only branch on warp-uniform values
    if data == 0 {  // OK: data came from warp-uniform hostcall
        // ...
    }

    // CANNOT DO THIS:
    let my_data = compute_per_lane(lane_id);
    if my_data > threshold {  // ERROR: per-lane condition
        warp_write(buf, fd, my_data).await;
    }
}
```

Per-lane branching to an `.await` is forbidden. The proposer frames this as a safety feature, but it's actually a severe expressiveness limitation. Many GPU workloads are data-dependent — "process this element only if it meets a condition." The warp-cooperative model forces all lanes to do the same I/O operations regardless of per-lane data. This is fine for "open file, read, process, write, close" pipelines but useless for data-dependent I/O patterns.

**The compiler cannot statically prove that a branch condition is warp-uniform.** The proposer handwaves this: "since fd came from a hostcall response (warp-uniform — one response per packet), the condition is deterministic across all lanes." But what if the user computes a derived value that's also uniform? The compiler would need a warp-uniformity analysis pass — which is an active research problem in GPU compilation with no general solution.

---

## 3. Alternative Approaches the Proposer Undervalues

### 3.1 The Proc Macro Approach (Option C) Is Sufficient and the Proposer Admits It

The proposer recommends Option C and says "start with Option C... pivot to Option A [if limitations hit]." This is reasonable, but the skeptic's point is stronger: **Option A will likely never be needed.**

The `warp-cfg` theme (proc macro with CFG support) is estimated at 2-3 weeks. The proc macro already handles linear pipelines (736 lines of code). Adding `if`/`loop`/`match` to a state machine generator is well-understood compiler construction — it's a sophomore-level assignment, not a research problem. The proc macro approach:

- Handles 100% of the warp-async use cases that are actually expressible (warp-uniform control flow only)
- Does NOT require a compiler fork
- Does NOT require tracking upstream nightly changes
- Does NOT require users to install a custom toolchain
- Is ALREADY partially implemented

The proposer's "trigger conditions" for pivoting to Option A are:
1. "Proc macro cannot handle nested control flow without exponential state explosion" — Linear state numbering with a CFG does not cause exponential explosion. This trigger condition is imaginary.
2. "Generated error messages are unusable" — Proc macros support `Span`-based error messages since Rust 1.45. They're not great, but they're workable.
3. "Variable scoping across branches becomes intractable at the token level" — Variable scoping is straightforward: track a scope stack, emit struct fields for cross-yield locals. This is exactly what the current macro already does for linear pipelines.

### 3.2 An LLVM Pass Would Be Strictly Better Than a Rustc Fork

The proposer dismisses Option C (LLVM pass) as "most complex and least Rust-specific... needs heuristics to recognize state machines (fragile)." This is unfair.

If the generator state machine reaches LLVM as a function with a switch on a discriminant field (which it does — the proposer says so in Section 2), then identifying it is not "heuristic" — it's pattern matching on a known structure. An LLVM pass could:
- Recognize the switch-on-discriminant pattern (generated by rustc's generator transform)
- Insert `shfl.sync.idx.b32` before discriminant reads
- Wrap store instructions in lane-0 guards
- Insert `bar.warp.sync` at state transitions

This is MORE maintainable than a rustc fork because LLVM IR is more stable than rustc MIR. LLVM's pass manager is a well-documented, stable API. An LLVM pass doesn't need to track rustc internals at all — it operates on LLVM IR regardless of which rustc version generated it.

The downside is less precise information (no trait awareness, no attribute propagation). But the proposer's design already assumes Option A's MIR pass operates on post-generator MIR, where trait information is largely erased anyway.

### 3.3 Hand-Written WarpFuture Is Actually Fine

The proposer treats hand-written WarpFuture as a problem to be solved. Let's challenge this premise.

The async-pipeline.3 task demonstrated a hand-written branching WarpFuture that works correctly. The "verbosity" argument is real but overstated — a WarpFuture with 7 states is ~150 lines of code. Most of that is payload fill logic, which would be identical in compiler-generated code.

For a research project aiming to reproduce VectorWare's demo, the number of distinct warp-async pipelines needed is small (5-10). Writing them by hand is boring but not a bottleneck. The proc macro covers linear pipelines; hand-written code covers branching. Together, they cover every use case in the project's scope.

**The effort to build a compiler extension for warp-async would exceed the effort of hand-writing every pipeline the project will ever need.** This is the definition of over-engineering.

---

## 4. Project Organization Risks

### 4.1 CI with Custom Rustc Is a Resource Sinkhole

Building rustc takes 30-60 minutes on a beefy machine. GitHub Actions runners have 2 vCPUs and 7GB RAM. A rustc build on a GitHub runner takes **90-120 minutes** and will frequently OOM if the runner is under memory pressure.

The proposer suggests caching the built toolchain. Caching is not free:
- Cache invalidation: any change to the fork invalidates the cache
- Cache size: a built rustc stage-2 toolchain is ~2-3 GB
- Cache upload/download time: significant on Actions runners
- GPU testing still requires a self-hosted runner with a GPU

The total CI pipeline for a single PR would be: 90min (build rustc) + 10min (build PTX) + 5min (GPU test) = ~105 minutes. Compare to the current setup: 2min (build PTX with stock nightly) + 5min (GPU test) = 7 minutes. That's a 15x increase in CI time.

### 4.2 Custom Toolchain Distribution Limits Adoption to Zero

Nobody outside this project will install a custom Rust toolchain to use warp-async. The Docker image approach helps for CI but not for development. Every developer would need to:
1. Download a ~2GB custom toolchain
2. Link it via `rustup toolchain link`
3. Remember to use `+warp-rustc` on every cargo command
4. Rebuild when the fork updates
5. Accept that their IDE (rust-analyzer) won't work correctly with the custom toolchain

For a research project with a single developer, this is manageable. For an open-source project hoping for community adoption, it's a non-starter.

### 4.3 Separate Repo Coordination Is Worse Than the Proposer Admits

The proposer recommends separate repos (`async-gpu` + `warp-rustc`) linked by toolchain version. This means:
- A breaking change in the fork requires synchronized updates to both repos
- CI must cross-reference specific fork commits with main project commits
- Bisecting a GPU runtime bug may require bisecting across two repos simultaneously
- Contributors must clone and build both repos to make changes

Monorepo avoids these problems but adds 2.5GB to clone size and makes the project intimidating to newcomers.

---

## 5. Scope Concerns

### 5.1 "Full Control Flow in Warp Async" Is Not a Real Requirement

The proposer never identifies a concrete workload that REQUIRES conditional `.await` in a warp context and cannot be solved by restructuring the pipeline. Every example given is a variation of "if condition, do IO-A; else do IO-B." These can all be expressed as:

```rust
// Instead of conditional await:
let action = if condition { ACTION_A } else { ACTION_B };
warp_dispatch(buf, action).await;
```

Or as separate linear pipelines selected at launch time. The "dispatch command" pattern (match with different awaits per arm) is actually an argument for **multiple linear pipelines** selected by a runtime parameter, not a single pipeline with internal branching.

### 5.2 The Proposer's Own Recommendation Contradicts the Premise

The title of BS28 is "Warp-Cooperative Async with Full Control Flow via Rustc Modifications." The proposer's actual recommendation is: "Start with Option C (proc macro)." This means the proposer agrees that a rustc fork is not justified today. The skeptic's position is stronger: **the fork is not justified ever**, because the proc macro approach will prove sufficient, and the remaining gaps are better addressed by hand-written WarpFutures or pipeline restructuring.

### 5.3 The Project Has More Urgent Priorities

The `last_summary` in state.toml says "async-pipeline EPIC COMPLETE." The project memory mentions a "Product Ready Epic" with 6 directions. Time spent on a rustc fork is time NOT spent on:
- Performance measurement and optimization
- Host listener stability
- CI/CD pipeline
- Clean public API and documentation
- Real-world demo applications

These are all higher-impact, lower-risk activities. A rustc fork is the highest-risk, most speculative direction possible.

---

## 6. Summary of Counterarguments

| Proposer Claim | Skeptic Counter |
|---|---|
| Fork maintenance is "1-2 days per nightly" | Realistically 1-2 weeks per nightly when MIR/generator internals change |
| Minimal prototype in "4-8 weeks" | More like 12-16 weeks given MIR learning curve and NVPTX backend quirks |
| MIR pass is "least invasive" | Still requires intimate knowledge of generator internals that change frequently |
| "The struct is identical to a standard generator" | Broadcasting arbitrary-sized saved locals is far more expensive than hand-written selective broadcast |
| LLVM pass is "fragile heuristics" | Switch-on-discriminant is a recognizable pattern; LLVM pass API is more stable than rustc MIR |
| Full control flow is needed | No concrete workload identified that requires it and cannot be restructured |
| Proc macro will hit "intractable limitations" | The identified trigger conditions are either imaginary or solvable |
| Option A is the "only path to a clean solution" | Clean solutions that nobody can use are worse than messy solutions that work |

**Bottom line**: The proposer's own analysis argues against a rustc fork (recommends Option C first, parks the fork theme). The skeptic goes further: **the fork should not even be a parked theme — it should be rejected outright.** The proc macro approach plus hand-written WarpFutures for edge cases covers every realistic use case. The project's time is better spent on product readiness than on compiler hacking.
