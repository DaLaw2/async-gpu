# BS56 — Proposer Analysis: Aligning `#[warp_async]` with Rust Native `async/await`

**Epic**: gpu-autonomous v3 — Native async/await on GPU
**Date**: 2026-03-14
**Cycle**: 232

---

## 1. Active Epics Assessment

### All Active Epics

| Epic ID | Title | Status | Progress |
|---------|-------|--------|----------|
| gpu-perf | GPU Inference Performance Optimization | active | Criterion 1 DONE. Criteria 2-4 PARKED |
| real-std | Real std on GPU | active | 5/5 surface, single-thread only |
| codebase-health | Codebase Health | active (evergreen) | Ongoing |
| public-api | Public API | active (evergreen) | 3-4/4 criteria |
| gpu-autonomous | GPU Autonomous Compute v2 | active | All v2 criteria MET (4/4). Ready for v3 |
| gpu-error | GPU Error Handling | completed | 4/4, closed |
| gpu-debug | GPU Debugging & Observability | completed | 4/4, closed |

### gpu-autonomous v3 — The Case for Native async/await

The gpu-autonomous v2 epic proved that the GPU can drive multi-step workflows autonomously
using `#[warp_async]` + `WarpFuture`. The showcased `autonomous_pipeline` function
demonstrates file I/O, branching, loops, and match — all in a readable sequential style.

However, this system is fundamentally disconnected from Rust's async ecosystem:

1. **WarpFuture ≠ Future**: A parallel universe of async. No `.await`, no `?` operator in async
   context, no composition with other futures.
2. **Custom macros**: `warp_print!`, `warp_open!`, `warp_write!` etc. — users must learn a
   bespoke API instead of using `File::create()`, `println!()`, etc.
3. **Two worlds**: `gpu-kernel-std` already has real `println!()` and `File::create()` working
   synchronously. `#[warp_async]` has its own parallel set of macros doing the same thing.

The dream: **unify these worlds** so GPU code looks like normal Rust async code.

---

## 2. Three Approaches

### Approach A: Wrapper Layer — Async Hostcall I/O with Standard Future

**Idea**: Make hostcall I/O operations return `impl Future`. Write a `no_std`-compatible
executor (like Embassy). GPU code uses `async fn` + `.await` natively. No rustc changes.

**What the user writes**:
```rust
async fn gpu_pipeline() -> Result<(), GpuIoError> {
    gpu_println!("start").await;
    let mut f = GpuFile::create("data.txt").await?;
    f.write_all(b"GPU output").await?;
    f.close().await?;
    Ok(())
}
```

**What gets generated** (by the Rust compiler, not our macro):
```rust
// Standard Rust async state machine — Pin<&mut Self>, Context<'_>, Poll<T>
enum GpuPipelineStateMachine {
    Start,
    AwaitPrint(GpuPrintFuture),
    AwaitCreate(GpuCreateFuture),
    AwaitWrite(GpuWriteFuture),
    AwaitClose(GpuCloseFuture),
    Done,
}

impl Future for GpuPipelineStateMachine {
    type Output = Result<(), GpuIoError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Standard state machine — one state per .await point
    }
}
```

**GpuFile::create() implementation sketch**:
```rust
pub struct GpuCreateFuture {
    buf: *mut u8,
    path: &'static [u8],
    state: CreateState, // Submitted | Waiting(pkt_idx)
}

impl Future for GpuCreateFuture {
    type Output = Result<GpuFile, GpuIoError>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            CreateState::Init => {
                // Submit hostcall packet
                let pkt_idx = hostcall_submit(this.buf, SERVICE_OPEN, ...);
                this.state = CreateState::Waiting(pkt_idx);
                Poll::Pending
            }
            CreateState::Waiting(idx) => {
                // Check if host responded
                if hostcall_check_ready(this.buf, idx) {
                    let fd = hostcall_read_result(this.buf, idx);
                    hostcall_release(this.buf, idx);
                    Poll::Ready(Ok(GpuFile { fd, buf: this.buf }))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
```

**Executor**: Spin-poll loop (Embassy-style, already proven to compile for nvptx64).

**Feasibility**:
- PROVEN: Embassy executor already compiles and runs on GPU (embassy-test crate)
- PROVEN: `core::future::Future` works on nvptx64
- PROVEN: Waker can be a no-op (GPU has no wakeup mechanism anyway)
- Each hostcall becomes a 2-state Future (submit → wait)
- Composition via standard `.await` chains

**Fundamental Blockers**:
- **No warp cooperation**: Each GPU thread runs its own Future independently.
  With 32 threads in a warp, all 32 run the same state machine redundantly — no
  `shfl.sync` to share state, no lane-0-only packet management. This is 32x waste
  for hostcall I/O.
- **Thread divergence**: If threads progress at different rates (e.g., different
  file sizes), warps diverge. `Future::poll()` has no mechanism to keep lanes
  synchronized.
- **Single-thread limitation**: Current std on GPU only works with (1,1,1) launch.
  This approach inherits that limitation unless multi-thread std is fixed first.
- **No `println!()` or `File::create()`**: These would need to be `GpuFile::create()`
  and `gpu_println!()` — new async wrappers, not actual std. So the "dream" syntax
  isn't fully achieved.

**Effort**: 2-3 weeks for basic async hostcall + executor. Does NOT solve warp cooperation.

---

### Approach B: WarpFuture ↔ Future Bridge

**Idea**: Make `WarpFuture` work with `.await` syntax. Two sub-approaches:

#### B1: WarpFuture implements core::future::Future

```rust
// Adapter: wraps a WarpFuture to implement core::future::Future
struct WarpFutureAdapter<F: WarpFuture> {
    inner: F,
    wcx: WarpContext,
}

impl<F: WarpFuture> Future for WarpFutureAdapter<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.inner.poll_warp(&mut this.wcx) {
            WarpPoll::Ready(v) => Poll::Ready(v),
            WarpPoll::Pending => Poll::Pending,
        }
    }
}
```

**Problem**: This compiles, but `.await` on a WarpFutureAdapter doesn't get us
`async fn` syntax for writing new WarpFutures. The user still writes `poll_warp`
manually or uses `#[warp_async]`.

#### B2: Custom `.await` desugaring via proc macro

```rust
#[gpu_async]  // our proc macro
async fn gpu_pipeline(buf: *mut u8) -> bool {
    gpu_print(buf, b"start").await;
    let fd = gpu_open(buf, b"data.txt", 1).await;
    gpu_write(buf, fd, b"output", 6).await;
    gpu_close(buf, fd).await;
    true
}
```

**What the proc macro does**: Parses the `async fn`, finds `.await` points,
generates a `WarpFuture` state machine (same as `#[warp_async]` does today, but
using `.await` syntax instead of `warp_*!()` macros).

**What gets generated**:
```rust
struct GpuPipeline {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
}

unsafe impl WarpFuture for GpuPipeline {
    type Output = bool;
    fn poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<bool> {
        // Same state machine as #[warp_async] generates today
    }
}
```

**Feasibility**:
- B1 is trivial but useless for ergonomics
- B2 is essentially a syntax reskin of `#[warp_async]`: replace `warp_print!(buf, ...)` with
  `gpu_print(buf, ...).await`. The `.await` is cosmetic — the proc macro strips it and
  generates WarpFuture code.
- Functions like `gpu_print()`, `gpu_open()` would return marker types (not real Futures)
  that the proc macro recognizes

**Fundamental Blockers**:
- **`.await` is a lie**: The proc macro intercepts it before the compiler sees it. Rust's
  type system will fight you — `async fn` returns `impl Future`, but the proc macro wants
  to generate `WarpFuture`. You'd need `#[gpu_async]` to consume the `async` keyword
  and NOT generate a real async fn.
- **No real composition**: You can't mix standard Futures with WarpFuture `.await`s.
  `select!`, `join!`, `FuturesUnordered` — none of these work.
- **Still custom API**: `gpu_open()` is not `File::create()`. The gap narrowed but not
  eliminated.

**Effort**: 2-3 weeks. Marginal improvement over current `#[warp_async]`.

---

### Approach C: rustc Modification — Warp-Cooperative Async Codegen

**Idea**: Modify the Rust compiler so that `async fn` targeting `nvptx64` generates
warp-cooperative state machines instead of per-thread state machines. The compiler
would emit `shfl.sync` instructions for state broadcasting, lane-0-only side effects,
and warp barriers at yield points.

**What the user writes**:
```rust
// Exact same code as normal Rust async!
async fn gpu_pipeline() -> io::Result<()> {
    println!("start");
    let mut f = File::create("data.txt").await?;
    f.write_all(b"GPU output").await?;
    Ok(())
}
```

**What the compiler generates** (for nvptx64 target):
```rust
// Conceptual — actually emitted as MIR/LLVM IR, not Rust source
enum GpuPipelineState {
    Print,
    WaitPrint { pkt: u16 },
    Create,
    WaitCreate { pkt: u16 },
    Write { fd: u64 },
    WaitWrite { pkt: u16 },
    Done,
}

// The state is stored ONCE per warp, not per thread.
// Lane 0 owns the state; other lanes receive it via shfl.sync.
fn poll_warp(state: &mut GpuPipelineState) -> WarpPoll<io::Result<()>> {
    let current = shfl_sync_idx(FULL_MASK, *state as u32, 0);
    match current {
        // ... warp-cooperative state machine
    }
}
```

**What this requires in rustc**:

1. **New generator lowering pass**: When `target = nvptx64`, the MIR generator
   transform emits warp-cooperative code instead of per-thread code:
   - State variable stored once per warp (in lane 0's registers)
   - State broadcast via `shfl.sync.idx.b32` at each poll
   - Side effects guarded by `if lane_id == 0`
   - `syncwarp()` barriers at state transitions

2. **Waker suppression**: For nvptx64 targets, the `Context`/`Waker` machinery is
   replaced with `WarpContext` (active_mask, lane_id). The `RawWaker` vtable
   becomes no-ops.

3. **async trait interception**: `File::create()` is synchronous in std. To make it
   `.await`-able, we need either:
   - A GPU-specific async I/O trait system (like `tokio::fs` vs `std::fs`)
   - Or modify std's CUDA PAL to return Futures when compiled for nvptx64

4. **LLVM backend changes**: The nvptx64 LLVM backend would need to understand
   warp-cooperative patterns and emit appropriate PTX (shfl.sync, bar.warp.sync).

**Feasibility**:
- This is the only approach that truly achieves the dream syntax
- Precedent: CUDA C++ has cooperative groups, but not at the language level
- The Rust compiler's async transform is in `rustc_mir_transform::coroutine`
  (coroutine_layout.rs, coroutine.rs) — well-isolated code

**Fundamental Blockers**:
- **Massive scope**: Modifying rustc's coroutine transform is non-trivial. The
  existing code handles general-purpose Futures, generators, async generators,
  and coroutines. Adding a target-specific code path is unprecedented.
- **Upstream acceptance**: A GPU-specific code generation path would face extreme
  scrutiny from the Rust compiler team. "No target-specific semantics in the
  language" is a core principle.
- **std::fs is synchronous**: `File::create()` in std blocks. Making it async
  requires either a separate async API crate or deep std modifications.
  `File::create("x").await` doesn't make sense unless `File::create` returns a
  Future — which it doesn't.
- **LLVM nvptx limitations**: The nvptx backend has known issues (no TLS, no
  dynamic linking, limited atomics). Adding warp-level intrinsics to the
  coroutine lowering would interact with these limitations.
- **Fork maintenance**: A rustc fork diverges from upstream nightly. Every
  nightly update requires merge conflict resolution in the coroutine transform.
- **Testing**: No existing test infrastructure for target-specific async semantics.

**Effort**: 3-6 months for a proof-of-concept fork. Indefinite for upstream acceptance.

---

## 3. Detailed Comparison

| Dimension | A: Wrapper Layer | B: WarpFuture Bridge | C: rustc Modification |
|-----------|-----------------|---------------------|----------------------|
| User syntax | `GpuFile::create().await` | `gpu_open().await` | `File::create().await` |
| Uses `.await` | Yes (real) | Cosmetic (proc macro) | Yes (real) |
| Warp cooperation | NO — per-thread | YES (via WarpFuture) | YES (compiler-generated) |
| Standard Future | YES | NO (looks like Future) | YES |
| Composable | YES (select!, join!) | NO | YES |
| Works with std | Needs async wrappers | NO | Needs std changes |
| rustc changes | None | None | Major |
| Effort | 2-3 weeks | 2-3 weeks | 3-6 months |
| Multi-thread GPU | Needs multi-thread fix | YES (warp-native) | YES (warp-native) |

### The Fundamental Tension

There is an irreconcilable tension between two goals:

1. **Warp cooperation**: 32 lanes sharing one state machine, communicating via
   `shfl.sync`, with lane 0 managing side effects. This is what makes GPU compute
   efficient. `WarpFuture` achieves this.

2. **Standard `core::future::Future`**: Each task is independent. No concept of
   32 copies running in lockstep. No `shfl.sync`. No lane 0 leader election.

You cannot have both without compiler support. Approach A gives you standard
Futures without warp cooperation. Approach B gives you warp cooperation without
standard Futures. Only Approach C gives you both — at enormous cost.

---

## 4. Concrete Recommendations

### The Pragmatic Path: Approach A + B Hybrid (Recommended)

Instead of choosing one approach, combine them for different use cases:

**Layer 1: Async Hostcall Primitives** (Approach A, 2-3 weeks)
- Create `gpu-async-io` crate with `GpuFile`, `GpuPrint` etc. that implement
  `core::future::Future`
- Use Embassy executor (already proven)
- Target: single-thread kernels, prototyping, correctness testing
- This gives real `.await` with real Futures — composable, debuggable, standard

**Layer 2: Keep `#[warp_async]` for Performance** (Already exists)
- When you need warp cooperation (performance-critical paths), use `#[warp_async]`
- This is the "CUDA kernel" path — maximum GPU efficiency
- Already proven, already working, already has if/else/match/loop

**Layer 3: Syntax Sugar for `#[warp_async]`** (Approach B2, optional, 1-2 weeks)
- Rename `warp_print!` → `warp::print().await` (cosmetic, proc macro intercepts)
- Purely ergonomic — no semantic change
- Low priority, can be deferred

**Layer 4: rustc Investigation** (Approach C, research only)
- Create a focused investigation task: study `rustc_mir_transform::coroutine`
- Understand the exact code path from `async fn` → generator → state machine
- Write a design document for a warp-cooperative generator transform
- Do NOT fork rustc yet — just understand what would be needed
- This is a 6-12 month horizon, inform but don't commit

### Proposed Themes and Tasks

**Theme: `async-io` (gpu-autonomous v3)**
- Goal: `core::future::Future`-based hostcall I/O for single-thread GPU kernels
- Success criteria:
  1. `GpuFile::create("x").await?` works with Embassy executor on GPU
  2. `gpu_println!("msg").await` works
  3. Multi-step pipeline using only `.await` (no `warp_*!()` macros)

**Tasks**:
- `async-io.1` (design): Define `GpuFile`, `GpuPrint` Future types + error types
- `async-io.2` (experiment): Implement `GpuCreateFuture` + `GpuWriteFuture` using
  existing hostcall protocol
- `async-io.3` (experiment): Embassy executor integration — spawn async GPU task,
  poll to completion
- `async-io.4` (experiment): Multi-step async pipeline test
  (`create → write → read → verify` using `.await`)

**Theme: `rustc-research` (gpu-autonomous v3)**
- Goal: Understand feasibility of warp-cooperative coroutine lowering in rustc
- Success criteria:
  1. Document the exact MIR transform pipeline for `async fn`
  2. Identify insertion points for warp-cooperative codegen
  3. Write a design ADR: "Warp-Cooperative Async for nvptx64"

**Tasks**:
- `rustc-research.1` (investigation): Map the `async fn` → coroutine → state machine
  pipeline in rustc source. Identify key files and transform passes.
- `rustc-research.2` (design): Draft ADR for warp-cooperative generator lowering.
  Define what changes, what's preserved, what breaks.

### Priority Ordering

1. **async-io.1** → **async-io.2** → **async-io.3** → **async-io.4** (immediate value)
2. **rustc-research.1** → **rustc-research.2** (inform future direction)

The async-io theme delivers immediate value: users get `.await` syntax for GPU I/O
today, using proven infrastructure (Embassy + existing hostcall). The rustc-research
theme is strategic — it tells us whether the "dream" is achievable and at what cost.

### What About the "Dream" Syntax?

```rust
async fn gpu_pipeline() {
    println!("start");
    let mut f = File::create("data.txt").await?;
    f.write_all(b"GPU output").await?;
}
```

This specific syntax requires:
- `println!` → already works synchronously. Making it async would mean changing std.
- `File::create().await?` → `File::create()` is synchronous in std. Would need
  `AsyncFile::create()` or a GPU-specific prelude that shadows std.
- `.await` on I/O → requires async I/O primitives (what async-io theme provides)

**Honest assessment**: The dream syntax is achievable **if** we accept `GpuFile`
instead of `std::fs::File`, and `gpu_println!` instead of `println!`. The fully
transparent version (where `std::fs::File` itself is async on GPU) would require
either:
- A GPU-specific std fork where all I/O is async (huge maintenance burden)
- Or rustc changes to make synchronous calls async when targeting nvptx64 (Approach C)

The pragmatic middle ground — `GpuFile` with `.await` — is achievable in 2-3 weeks
and gives 90% of the ergonomic benefit.
