# BS28 Proposer — Warp-Cooperative Async with Full Control Flow via Rustc Modifications

**Date**: 2026-03-12
**Role**: Proposer
**Topic**: Modifying rustc to support warp-cooperative async with full control flow on GPU

---

## 1. Active Epics Assessment

### What This Epic Means

The user is proposing a fundamentally new epic: **"Warp-cooperative async with full control flow via rustc modifications."** This goes beyond anything the project has attempted. Every prior epic operated within the constraints of stable (or nightly) rustc — proc macros, inline asm, `-Zbuild-std`, `extern "ptx-kernel"`. This epic proposes modifying the compiler itself.

### Current State of Affairs

The project has completed:
- **async-pipeline EPIC** (all 5 tasks): `#[warp_async]` proc macro generates WarpFuture state machines for linear pipelines. Supports `warp_open!`, `warp_close!`, `warp_read!`, `warp_write!`, `warp_bulk_read!`, `warp_bulk_write!`, `warp_print!`. Variable bindings work. All 32 warp lanes run in lockstep.
- **WarpFuture infrastructure** (ADR-9, ADR-10): `WarpFuture` trait, `WarpExecutor`, `broadcast_u32`, `warp_hostcall_submit`, `warp_hostcall_wait_u64` — all in `gpu-runtime::warp_future`.
- **Hand-written branching** (async-pipeline.3): Conditional state transitions work by manually setting `self.state = BRANCH_X` based on hostcall responses. Proven but verbose.

The critical limitation from bs27: the proc macro ONLY supports linear pipelines. Adding `warp_if!` was explicitly parked with the assessment "transforms [the proc macro] from a linear code generator into a control flow graph compiler... the road to building a custom compiler inside a proc macro."

**This epic acknowledges that assessment and says: "If we need a compiler, let's use the actual compiler."**

### Strategic Significance

This is a Phase 5+ direction. It requires:
1. Deep understanding of rustc internals (MIR, generators, async desugaring)
2. A fork or plugin mechanism for the Rust compiler
3. A testing and distribution strategy for a custom toolchain
4. Long-term maintenance commitment (tracking upstream nightly changes)

This is the most ambitious direction the project could take. It would transform async-gpu from "research prototype with a clever proc macro" into "a compiler extension that natively supports warp-cooperative async." If successful, it would be the first compiler-level support for SIMT-cooperative async in any language.

---

## 2. Rustc Async Transform Analysis

### How Rustc's Current Async Transform Works

Rust's async transform happens in several stages:

**Stage 1: AST → HIR (Desugaring)**
`async fn foo() -> T { body }` becomes `fn foo() -> impl Future<Output = T> { async move { body } }`. The `async` block is represented in HIR as a `GeneratorKind::Async` closure.

**Stage 2: HIR → MIR (Generator lowering)**
The MIR pass `StateTransform` (`compiler/rustc_mir_transform/src/generator.rs`) transforms the generator body into a state machine:
1. Each `.await` point is a **yield point** (`Yield` terminator in MIR)
2. The compiler computes which locals are **live across yield points** — these become fields of the generator struct
3. A **discriminant** field tracks which state the generator is in
4. The original function body is split into **resumption points** — one basic block per state
5. A `match` on the discriminant dispatches to the correct resumption point

Key data structures:
- **`GeneratorLayout`** (`rustc_middle::ty::layout`): Describes the fields of the generated state machine struct (discriminant + live-across-yield locals)
- **`GeneratorSavedLocal`**: A local variable that must be saved in the generator struct because it's live across a yield point
- **`GeneratorSavedTy`**: The type of a saved local, with its source info
- **`VariantIdx`**: The state discriminant value (0 = unresumed, 1 = returned/panicked, 2+ = suspended states)

**Stage 3: MIR → MIR (Optimization)**
Standard MIR optimizations run on the generated state machine. Importantly, `SimplifyCfg` and `SimplifyBranches` clean up the generated code.

**Stage 4: MIR → LLVM IR → PTX**
For `nvptx64-nvidia-cuda`, LLVM's NVPTX backend lowers to PTX. The state machine becomes a regular function with a switch on the discriminant field. No special GPU awareness exists at this point.

### Where Would Warp-Cooperative Changes Go?

There are several candidate insertion points:

**Option A: Between Stage 2 and Stage 3 (MIR-to-MIR pass)**
After the standard generator transform creates a single-threaded state machine, a new MIR pass could transform it into a warp-cooperative version:
- Replace loads of the discriminant with `shfl.sync.idx.b32` broadcast from lane 0
- Wrap all state-machine field mutations in `if lane_id == 0 { ... }` guards
- Insert `syncwarp()` barriers at state transitions
- Replace `Waker::wake()` with no-ops (warp futures use spin-poll)

This is the least invasive option. The standard async transform does all the hard work (computing live variables, assigning states, splitting the CFG). The warp-cooperative pass just adds SIMT decorations.

**Option B: Modify Stage 2 directly (Generator lowering)**
Change the `StateTransform` pass itself to emit warp-cooperative code when targeting `nvptx64-nvidia-cuda` and the function has a `#[warp_async]` attribute. This allows the generator lowering to make globally-informed decisions (e.g., "this branch is warp-uniform, this branch needs broadcast").

This is more invasive but potentially produces better code. The standard transform doesn't know about warp semantics, so Option A must conservatively add broadcasts everywhere.

**Option C: Custom codegen pass (LLVM level)**
Add an LLVM pass (or NVPTX backend modification) that recognizes "generator state machine" patterns and transforms them for warp-cooperative execution. This operates at the LLVM IR level, after Rust MIR has been lowered.

This is the most complex option and least Rust-specific. It would need heuristics to recognize state machines (fragile). Not recommended.

**Recommendation: Option A** — a post-generator MIR pass. It has the best complexity/impact ratio. The standard generator transform handles all the hard problems (live variable analysis, state assignment). The warp pass adds the SIMT-specific transformations.

### Key Data Structures to Modify or Interface With

1. **`GenFuture<T>`** (`core::future::from_generator`): The wrapper that implements `Future` for a generator. For warp-cooperative, we'd need a `WarpGenFuture<T>` that implements `WarpFuture` instead.

2. **`Pin<&mut Self>` / `Context` / `Waker`**: The standard `Future::poll(self: Pin<&mut Self>, cx: &mut Context<'_>)` signature doesn't make sense for warp futures. WarpFuture uses `poll_warp(&mut self, wcx: &mut WarpContext)`. The compiler would need to generate code for the `WarpFuture` trait instead of `Future`.

3. **`Poll<T>`**: Becomes `WarpPoll<T>`. No change in semantics (Ready/Pending), but the type is different.

4. **Generator discriminant**: Currently a simple integer stored in `self`. For warp futures, the discriminant must be broadcast from lane 0 via `shfl.sync.idx.b32` on every read.

### Could This Be Done as a MIR Pass?

**Yes**, with some caveats:

A MIR pass can:
- Rewrite loads of the generator discriminant to call an intrinsic (`warp_broadcast_state`)
- Guard field stores behind `lane_id == 0` checks
- Insert `syncwarp()` calls at state transitions
- Replace `Future`-specific operations with `WarpFuture` equivalents

A MIR pass **cannot**:
- Change the trait being implemented (the generator already implements `Future`, not `WarpFuture`)
- Change the function signature (it's already `poll(Pin<&mut Self>, &mut Context)`)

This means the MIR pass approach would need one of:
- **Trait aliasing**: Make `WarpFuture` and `Future` structurally compatible, and use the MIR pass to transform the body. The executor would call `poll()` but the body would execute warp-cooperative code. Semantically dirty but functional.
- **Separate trait lowering**: Modify the desugaring (Stage 1) so that `#[warp_async] async fn` desugars to a generator that implements `WarpFuture` instead of `Future`. Then the MIR pass decorates the body with SIMT operations.

The second approach is cleaner. It requires a small change in HIR desugaring (detect `#[warp_async]` attribute, use `WarpFuture` trait) plus a MIR pass (add broadcasts and barriers).

---

## 3. Warp-Cooperative State Machine Design

### Conditional `.await` — How It Works

Consider:
```rust
#[warp_async]
async fn example(buf: &WarpBuf) -> bool {
    let fd = warp_open(buf, "file.txt").await;
    if check_error(fd) {
        warp_close(buf, fd).await;   // Branch A
    } else {
        warp_print(buf, "ok").await;  // Branch B
    }
    true
}
```

The compiler generates states:
```
State 0: Submit warp_open
State 1: Wait for warp_open response → fd
State 2: Branch decision
  Lane 0 evaluates check_error(fd)
  Lane 0 broadcasts result via shfl.sync
  All lanes read the broadcast → if true, goto State 3; else goto State 5
State 3: Submit warp_close (Branch A)
State 4: Wait for warp_close → goto State 7
State 5: Submit warp_print (Branch B)
State 6: Wait for warp_print → goto State 7
State 7: Done, return true
```

**Critical invariant**: The branch condition `check_error(fd)` is evaluated ONLY on lane 0, and the result is broadcast to all lanes. This ensures all 32 lanes take the same branch. Since `fd` came from a hostcall response (warp-uniform — one response per packet), the condition is deterministic across all lanes.

**What the compiler must do**:
1. Identify branch conditions in the original async fn
2. For each branch that contains an `.await` in any arm, insert a "broadcast decision" operation
3. All lanes read the broadcast and follow the same path

**What the compiler must NOT do**:
- Allow per-lane evaluation of branch conditions that lead to different `.await` paths (this would cause warp divergence at yield points, which is a hard error)

### Loops with `.await`

```rust
#[warp_async]
async fn example(buf: &WarpBuf) {
    loop {
        let data = warp_read(buf, fd, 1024).await;
        if data == 0 { break; }
        warp_write(buf, fd2, data).await;
    }
}
```

States:
```
State 0: Submit warp_read
State 1: Wait for warp_read → data
State 2: Loop condition check
  Lane 0 evaluates (data == 0)
  Broadcast result
  If true → goto State 5 (after loop)
  If false → goto State 3
State 3: Submit warp_write
State 4: Wait for warp_write → goto State 0 (loop back)
State 5: Done
```

The loop becomes a cycle in the state machine CFG. The compiler's standard generator transform already handles this — loops with `.await` create cycles in the state graph. The warp-cooperative pass just needs to ensure the loop condition is broadcast.

### `match` with `.await` in Arms

```rust
#[warp_async]
async fn example(buf: &WarpBuf, cmd: u32) {
    match cmd {
        1 => { warp_open(buf, "a.txt").await; }
        2 => { warp_open(buf, "b.txt").await; warp_close(buf, fd).await; }
        _ => { warp_print(buf, "unknown").await; }
    }
}
```

States:
```
State 0: Match dispatch
  Lane 0 evaluates cmd
  Broadcast cmd value (or branch index)
  Goto State 1/3/6 based on broadcast value
State 1: Submit warp_open("a.txt") → State 2 → goto State 7
State 3: Submit warp_open("b.txt") → State 4
State 4: Wait → State 5: Submit warp_close → State 5b: Wait → goto State 7
State 6: Submit warp_print → State 6b: Wait → goto State 7
State 7: Done
```

The match discriminant is broadcast from lane 0, ensuring all lanes enter the same arm. Arms of different lengths converge at the post-match state.

### Invariants the Generated Code Must Maintain

1. **State uniformity**: All 32 lanes must agree on the current state at every yield/resume point. Enforced by broadcasting the discriminant from lane 0 via `shfl.sync.idx.b32`.

2. **No warp divergence at yield points**: When any lane executes a yield (returning `WarpPoll::Pending`), ALL lanes must yield. A lane cannot independently decide to yield or continue.

3. **Lane 0 is the decision maker**: All branch conditions, match discriminants, and loop conditions that affect which `.await` is reached must be evaluated on lane 0 and broadcast.

4. **Field mutations are lane-0 only**: The state machine struct (discriminant, saved locals) is mutated only by lane 0. Other lanes read via broadcast.

5. **Syncwarp at state transitions**: `syncwarp(active_mask)` must be called after lane 0 mutates state and before any lane reads the new state. This ensures all lanes see the same state.

6. **Hostcall responses are warp-uniform**: Each hostcall uses one packet per warp. The response is read by lane 0 and broadcast. All lanes see the same response value.

### State Storage

The generated struct stores:
- `state: u32` — the discriminant, accessed via broadcast from lane 0
- `pkt_idx: u16` — current in-flight packet index, broadcast from lane 0
- Saved locals from the async fn body (live across yield points) — stored by lane 0, broadcast on access
- Function parameters — immutable, same across all lanes (passed as kernel arguments)

The struct is identical to a standard generator struct. The difference is in how fields are accessed: always through lane-0 broadcast.

---

## 4. Implementation Strategy

### Option A: Fork Rustc — Modify the Async Transform for nvptx64

**Approach**: Fork `rust-lang/rust`, modify the compiler to:
1. Recognize `#[warp_async]` attribute on `async fn`
2. Desugar to a generator implementing `WarpFuture` instead of `Future`
3. Run a post-generator MIR pass that adds broadcasts and barriers
4. Compile to PTX with the warp-cooperative state machine

**Modifications needed**:
- `compiler/rustc_ast_lowering/src/item.rs`: Detect `#[warp_async]` and set a flag on the generator
- `compiler/rustc_mir_transform/src/generator.rs`: When the flag is set, generate `WarpFuture`-compatible dispatch code
- New file `compiler/rustc_mir_transform/src/warp_cooperative.rs`: MIR pass that:
  - Wraps discriminant reads with `warp_broadcast_state()` intrinsic calls
  - Guards field mutations behind `lane_id == 0` checks
  - Inserts `syncwarp()` at state transitions
  - Wraps branch conditions in broadcast operations
- `library/core/src/future/`: Add `WarpFuture`, `WarpPoll`, `WarpContext` to core (or as a separate crate that the compiler knows about)

**Pros**:
- Full control over generated code
- Can make globally-informed decisions (the compiler sees the entire function)
- Can optimize: e.g., merge consecutive non-yielding states, elide unnecessary broadcasts
- Natural Rust syntax: users write standard `async fn` with `#[warp_async]`
- Full control flow support (if, match, loop, for, while) comes "for free" from the standard generator transform

**Cons**:
- Massive maintenance burden: must track upstream nightly changes
- Nightly Rust changes the compiler internals frequently (MIR, generator lowering have changed multiple times in 2025)
- Distribution: users must install a custom rustc toolchain
- Bus factor: if the maintainer stops updating, the fork dies
- Build time: building rustc from source takes 30-60 minutes
- CI complexity: must build custom rustc, then build GPU kernels with it
- Upstream compatibility: can never merge back (Rust team will not accept GPU-specific compiler changes without a proper RFC process that could take years)

**Estimated effort**: 4-8 weeks for a minimal working prototype. Ongoing maintenance: 1-2 days per nightly update.

### Option B: Custom Compiler Pass/Plugin

**Approach**: Use rustc's (unstable) plugin infrastructure or a custom driver to intercept MIR after the standard generator transform and rewrite it.

**Mechanisms**:
- `rustc_driver`: Custom compiler driver that invokes standard rustc passes, then runs a custom MIR pass
- `-Z mir-opt-level` + custom pass: Register a custom MIR optimization pass
- `rustc_plugin` (deprecated/removed as of 2024): Not viable

**Current state of rustc plugin API**: As of nightly 2025, there is no stable plugin API for custom MIR passes. The `rustc_interface` crate provides `Callbacks` that can intercept compilation after various stages, but modifying MIR requires accessing internal data structures.

**Practical approach**: A custom `rustc_driver` wrapper that:
1. Calls standard rustc to compile to MIR
2. Deserializes/intercepts the MIR
3. Runs the warp-cooperative transform
4. Continues compilation to LLVM IR → PTX

**Pros**:
- Potentially less coupled to rustc internals than a fork
- Could be distributed as a separate tool (`warp-rustc`)
- Doesn't modify the compiler source, just wraps it

**Cons**:
- Accessing MIR programmatically outside rustc is extremely difficult
- `rustc_interface` is unstable and changes every nightly
- No official API for MIR manipulation
- In practice, this approach ends up requiring the same internal rustc knowledge as a fork
- Less control than a fork (can only transform MIR, can't change desugaring)
- Still requires a specific nightly version (same as fork)

**Estimated effort**: 6-10 weeks (harder than a fork due to fighting the lack of API). Same maintenance burden.

### Option C: Hybrid — Proc Macro for Parsing + Compiler Intrinsics for Warp Primitives

**Approach**: Extend the existing `#[warp_async]` proc macro to handle control flow, using a combination of:
1. **Proc macro**: Parse the async fn body at the token level, including `if`, `loop`, `match` statements
2. **Compiler intrinsics**: Define `warp_broadcast()`, `warp_syncwarp()`, etc. as `extern "rust-intrinsic"` functions that the NVPTX backend knows how to lower
3. **The proc macro generates a state machine** with explicit broadcast/barrier calls

**How control flow would work**:
The proc macro becomes a full control-flow-graph compiler operating on Rust tokens:
1. Parse the function body into an AST
2. Identify `.await` points (or `warp_*!()` calls)
3. Build a CFG with yield points
4. Assign state numbers to each yield point
5. Generate a `match` dispatch with the correct state transitions
6. For branches containing yields, generate broadcast-based decision code
7. For loops containing yields, generate cycle-back state transitions

**Pros**:
- No rustc fork required
- Works with any nightly that supports `nvptx64-nvidia-cuda`
- Distributed as a crate (`warp-macro`), not a custom toolchain
- Users only need `cargo` to build
- The project already has a working proc macro (700 lines) as a foundation
- Full control over generated code (the macro IS the compiler for warp async)

**Cons**:
- Building a compiler inside a proc macro is genuinely hard:
  - Must handle Rust's full expression syntax (closures, method chains, operators)
  - Must track variable scoping and lifetimes manually
  - Error messages from proc macros are notoriously bad
  - Testing/debugging generated code is difficult
- Limited to the expressiveness of the proc macro:
  - Can't perform type inference (proc macros see tokens, not types)
  - Can't do borrow checking on the generated code (rustc does this, but errors point to generated code)
  - Can't optimize across state boundaries (no MIR-level analysis)
- The proc macro would grow to 3000-5000 lines
- Doesn't compose with standard Rust async (a `#[warp_async]` function can't use standard `.await`)

**Estimated effort**: 4-6 weeks for full control flow support. Ongoing maintenance: low (proc macros are stable).

### Comparison Matrix

| Criterion | A: Fork Rustc | B: Custom Driver | C: Hybrid Proc Macro |
|-----------|---------------|-------------------|----------------------|
| **Full control flow** | Yes (free from generator transform) | Yes (MIR rewrite) | Yes (manual CFG builder) |
| **Code quality** | Best (MIR-level optimization) | Good (MIR rewrite) | Adequate (token-level) |
| **Error messages** | Standard rustc errors | Poor (MIR rewrite artifacts) | Poor (proc macro spans) |
| **Distribution** | Custom toolchain install | Custom tool + specific nightly | `cargo` dependency |
| **Maintenance** | High (track nightly) | High (track nightly) | Low (stable proc macro API) |
| **Initial effort** | 4-8 weeks | 6-10 weeks | 4-6 weeks |
| **User friction** | High (custom toolchain) | High (custom tool) | Low (just add a dependency) |
| **Composability** | High (standard async syntax) | High (standard async syntax) | Low (custom macro syntax) |

### Minimum Viable Change

**Option C (Hybrid Proc Macro)** is the minimum viable change. It requires no compiler modifications, works with existing toolchains, and builds on the existing `warp-macro` crate.

However, if the goal is truly natural Rust async syntax with full language support, **Option A (Fork Rustc)** is the only path to a clean solution. Option C will always be a "compiler inside a macro" with rough edges.

**My recommendation**: Start with Option C (extend the proc macro to handle `if`/`loop`/`match`). If the proc macro approach hits intractable limitations (which I predict will happen when users want nested control flow with variable scoping across branches), pivot to Option A.

---

## 5. Project Organization

### Rustc Fork Organization

**Recommended: Separate repository, git submodule link.**

```
github.com/DaLaw2/async-gpu          # Main project (existing)
github.com/DaLaw2/warp-rustc         # Forked rustc (if Option A)
```

The main project references the fork via:
- `rust-toolchain.toml` pointing to a custom toolchain build
- Build scripts that invoke the custom `rustc`
- CI downloads pre-built toolchain artifacts

**Why not monorepo**: Rustc is ~2.5GB of source. Embedding it in async-gpu would make the repo unwieldy. Also, the rustc fork has its own release cadence (tracking upstream nightlies).

### Tracking Upstream Rustc Changes

**Strategy: Rebase-on-release**

1. Fork from a specific nightly commit (e.g., `nightly-2025-08-25`, the currently pinned version)
2. Apply warp-cooperative patches as commits on top
3. When a new nightly is needed, rebase the patch series onto the new nightly
4. If conflicts arise, resolve them (this is the maintenance cost)

**Tooling**:
- Tag each working combination: `warp-rustc-nightly-2025-08-25-v1`
- CI tests: build the custom rustc + build all async-gpu kernels + run GPU tests
- Automatic upstream tracking: GitHub Action that periodically attempts rebase and reports conflicts

### CI/CD Strategy

**Phase 1 (no fork — Option C)**:
```yaml
# .github/workflows/ci.yml
jobs:
  build-ptx:
    runs-on: ubuntu-latest
    steps:
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: nightly-2025-08-25
          target: nvptx64-nvidia-cuda
      - run: cargo build --target nvptx64-nvidia-cuda -Zbuild-std=core

  test-gpu:
    runs-on: [self-hosted, gpu]  # Requires a GPU runner
    needs: build-ptx
    steps:
      - run: cargo test --package gpu-host
```

**Phase 2 (with fork — Option A)**:
```yaml
jobs:
  build-rustc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: DaLaw2/warp-rustc
      - run: python x.py build --stage 2
      - uses: actions/cache/save@v4  # Cache the built toolchain

  build-ptx:
    needs: build-rustc
    steps:
      - # Download cached toolchain
      - run: RUSTC=./warp-rustc/build/host/stage2/bin/rustc cargo build ...
```

**Key insight**: Building rustc in CI takes 30-60 minutes. Cache the built toolchain aggressively. Only rebuild when the fork changes.

### Distribution Strategy

**For Option C (proc macro)**: No special distribution needed. `warp-macro` is a regular crate. Users add it to their `Cargo.toml`.

**For Option A (rustc fork)**:
1. **Pre-built binaries**: GitHub Releases with `warp-rustc` binaries for Linux x86_64
2. **Rustup-compatible**: Build a custom toolchain that can be installed via `rustup toolchain link warp-rustc /path/to/build`
3. **Docker image**: `ghcr.io/dalaw2/warp-rustc:nightly-2025-08-25` with the custom toolchain pre-installed
4. **Documentation**: "Install the warp-rustc toolchain, then build with `cargo +warp-rustc build --target nvptx64-nvidia-cuda`"

---

## 6. Concrete Recommendations

### Proposed Epic

**Epic: rustc-warp-async** — Warp-cooperative async with full control flow

**Goal**: Write natural Rust async functions with `if`, `loop`, `match` and have the compiler generate correct warp-cooperative state machines for GPU execution.

**Exit criteria**:
1. `#[warp_async] async fn` supports `if`/`else` with `.await` in both arms
2. `#[warp_async] async fn` supports `loop` with `.await` in the body
3. `#[warp_async] async fn` supports `match` with `.await` in arms
4. Generated code maintains warp convergence invariant (verified by GPU test)
5. At least one complex example (conditional file I/O pipeline with branching and loops) compiles and runs correctly

### Proposed Themes and Tasks

#### Theme: `warp-cfg` (Control Flow Graph for Warp Async)

**Status**: active
**Goal**: Extend `#[warp_async]` proc macro to support full control flow (if/loop/match with `.await`)
**Success criteria**: Conditional, looping, and matching pipelines compile and run correctly on GPU

| Task | Kind | Depends | Description | Effort |
|------|------|---------|-------------|--------|
| `warp-cfg.1` | investigation | — | Analyze proc-macro-based CFG building: study how to parse Rust `if`/`loop`/`match` at the token level, assign state numbers to a DAG (not linear list), handle variable scoping across branches. Produce a design document with concrete examples of generated code for each control flow construct. | 2-3 days |
| `warp-cfg.2` | experiment | warp-cfg.1 | Implement `if`/`else` support in `#[warp_async]`: parse `if` statements containing `warp_*!()` calls, generate forked state ranges with broadcast-based branch decisions, merge states after the if/else. Test with "if file exists, read it; else create it" pipeline. | 3-4 days |
| `warp-cfg.3` | experiment | warp-cfg.2 | Implement `loop`/`while`/`break` support: parse loop bodies containing `warp_*!()` calls, generate cycle-back state transitions, handle `break` as a jump to post-loop state. Test with "read until EOF" pipeline. | 2-3 days |
| `warp-cfg.4` | experiment | warp-cfg.2 | Implement `match` support: parse match expressions where arms contain `warp_*!()` calls, broadcast match discriminant from lane 0, generate per-arm state ranges. Test with "dispatch command" pipeline. | 2-3 days |
| `warp-cfg.5` | experiment | warp-cfg.3, warp-cfg.4 | Nested control flow stress test: write a pipeline with nested `if` inside `loop` with `match`, all containing `.await` calls. Verify state numbering correctness and warp convergence. | 1-2 days |

#### Theme: `rustc-warp` (Compiler-Level Warp Async — Future Phase)

**Status**: parked (activate only if `warp-cfg` hits proc-macro limitations)
**Goal**: Modify rustc to natively support warp-cooperative async desugaring
**Success criteria**: Standard `async fn` with `#[warp_async]` attribute compiles to warp-cooperative state machine without proc macro

| Task | Kind | Depends | Description | Effort |
|------|------|---------|-------------|--------|
| `rustc-warp.1` | investigation | — | Deep dive into rustc's generator/async lowering: read and document `compiler/rustc_mir_transform/src/generator.rs`, identify exact insertion points for warp-cooperative transforms, assess MIR representation of branch/loop/match after generator transform | 3-4 days |
| `rustc-warp.2` | design | rustc-warp.1 | Design the MIR warp-cooperative pass: specify exactly which MIR operations are added/modified, define `warp_broadcast` and `warp_syncwarp` intrinsics, design the `WarpFuture` trait integration with desugaring | 3-4 days |
| `rustc-warp.3` | experiment | rustc-warp.2 | Fork rustc, implement the minimal warp-cooperative MIR pass: discriminant broadcast only (no branch handling). Test with a linear pipeline that the fork produces correct PTX. | 2-3 weeks |
| `rustc-warp.4` | experiment | rustc-warp.3 | Add branch/loop/match support to the MIR pass: broadcast branch conditions, handle cycle-back, handle match discriminants. Test with the same examples as warp-cfg.5. | 2-3 weeks |
| `rustc-warp.5` | experiment | rustc-warp.4 | CI and distribution: build the custom rustc in CI, publish pre-built binaries, document installation process. | 1 week |

### Dependencies Between Themes

```
warp-cfg (proc macro approach)
    ├── warp-cfg.1 → warp-cfg.2 → warp-cfg.3
    │                            → warp-cfg.4
    │                 warp-cfg.3 + warp-cfg.4 → warp-cfg.5
    │
    └── [If limitations hit] → rustc-warp (compiler approach)
                                 ├── rustc-warp.1 → rustc-warp.2 → rustc-warp.3 → rustc-warp.4 → rustc-warp.5
```

### Priority Ordering with Rationale

1. **`warp-cfg.1` (investigation)** — FIRST. Cannot write code without a design. The proc macro approach is cheaper to explore and may be sufficient.

2. **`warp-cfg.2` (if/else)** — SECOND. `if`/`else` is the most requested control flow construct (bs27 explicitly identified "if file exists" as the canonical use case). It also proves out the CFG builder infrastructure.

3. **`warp-cfg.3` (loop) and `warp-cfg.4` (match)** — THIRD, parallel. Both build on the CFG infrastructure from warp-cfg.2. Loops enable "read until EOF" patterns. Match enables "dispatch command" patterns.

4. **`warp-cfg.5` (stress test)** — FOURTH. Validates the combined system.

5. **`rustc-warp.*`** — PARKED. Only activate if the proc macro approach proves insufficient. Specific trigger conditions:
   - Proc macro cannot handle nested control flow without exponential state explosion
   - Generated error messages are unusable
   - Users consistently request standard async syntax instead of macro DSL
   - Variable scoping across branches becomes intractable at the token level

### Each Task's Epic Reference

All `warp-cfg.*` tasks serve the **rustc-warp-async** epic's Phase 1 (proc macro approach).
All `rustc-warp.*` tasks serve the **rustc-warp-async** epic's Phase 2 (compiler approach).

The epic is structured as a two-phase effort where Phase 1 is the minimum viable product and Phase 2 is activated only on evidence of Phase 1's limitations.

---

## Summary

The proc macro approach (Option C / `warp-cfg` theme) should be pursued first. It is cheaper, requires no compiler fork, and may be sufficient for the 80% case. The compiler fork approach (Option A / `rustc-warp` theme) is kept in reserve as a parked theme, activated only if the proc macro hits concrete, documented limitations.

The key technical insight is that Rust's standard async transform already does the hard work of CFG analysis and state machine generation. For the proc macro approach, we must replicate that work at the token level. For the compiler approach, we can leverage it directly and add a thin SIMT-specific layer on top.

Total estimated effort: 2-3 weeks for the proc macro approach (warp-cfg). 8-12 weeks additional if the compiler fork is needed (rustc-warp). The recommended path is to complete warp-cfg first and evaluate.
