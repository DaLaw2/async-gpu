# BS57 — Proposer Analysis: Warp-Cooperative Async via rustc Modification

**Role**: Proposer
**Epic**: gpu-autonomous v3 — Native async/await on GPU
**Date**: 2026-03-14
**Cycle**: 232

**Framing**: Upstream acceptance is NOT a constraint. We will fork rustc if needed.
The goal is to prove that warp-cooperative async/await can work natively in Rust.

> "The reasonable man adapts himself to the world; the unreasonable one persists in
> trying to adapt the world to himself. Therefore all progress depends on the
> unreasonable man." — George Bernard Shaw

---

## Section 1: The Minimal rustc Change

### The Four Options

#### Option A: Custom MIR Pass After StateTransform

**Files to create/modify:**
- CREATE: `compiler/rustc_mir_transform/src/warp_cooperative.rs` — the new pass
- MODIFY: `compiler/rustc_mir_transform/src/lib.rs` — register the pass after `StateTransform`

**How it works:**

After `StateTransform` runs, the coroutine is already a plain struct with a
switch-based `poll()` function. The MIR looks like:

```
// Post-StateTransform MIR (simplified)
fn poll(_1: Pin<&mut CoroutineState>, _2: &mut Context) -> Poll<T> {
    bb0: {
        _disc = (_1.0).0;           // read discriminant
        switchInt(_disc) -> [0: bb1, 3: bb2, 4: bb3, ...];
    }
    bb1 (state 0 - Unresumed): {
        // ... initial code ...
        (_1.0).0 = 3;               // set state to suspended-at-yield-1
        return Poll::Pending;
    }
    bb2 (state 3 - resume after yield 1): {
        // ... code between yield 1 and yield 2 ...
        (_1.0).0 = 4;               // set state to suspended-at-yield-2
        return Poll::Pending;
    }
    // ...
}
```

The `warp_cooperative` pass transforms this into:

```
fn poll(_1: Pin<&mut CoroutineState>, _2: &mut Context) -> Poll<T> {
    bb0: {
        _disc = (_1.0).0;
        // INSERT: broadcast discriminant from lane 0 to all lanes
        _disc_uniform = shfl_sync_idx(0xFFFFFFFF, _disc, 0);
        switchInt(_disc_uniform) -> [0: bb1, 3: bb2, 4: bb3, ...];
    }
    bb1 (state 0 - Unresumed): {
        // INSERT: guard side effects with lane_id == 0 check
        _lane = lane_id();
        if _lane == 0 {
            // ... side-effecting code (hostcall submit, etc.) ...
        }
        // INSERT: syncwarp barrier before state transition
        syncwarp(0xFFFFFFFF);
        if _lane == 0 {
            (_1.0).0 = 3;
        }
        syncwarp(0xFFFFFFFF);
        return Poll::Pending;
    }
    // ... same pattern for all basic blocks ...
}
```

**Warp convergence**: Maintained because:
1. All lanes read the same discriminant (broadcast from lane 0)
2. All lanes take the same switch branch
3. Side effects are guarded but all lanes execute the same control flow
4. `syncwarp` barriers enforce reconvergence at state transitions

**State broadcast**: `shfl.sync.idx.b32` from lane 0 at the top of each poll.

**Side effect guarding**: The pass must identify which MIR statements are
"side effects" (stores to shared memory, hostcall submissions, etc.) and
wrap them in `if lane_id == 0 { ... }` guards. This is the hardest part —
the pass needs a heuristic or annotation to distinguish data-parallel writes
(all lanes write different positions) from leader-only writes (one lane
submits a packet).

**Advantages:**
- Does NOT modify `coroutine.rs` — clean separation
- Can be toggled per-target: `#[cfg(target_arch = "nvptx64")]`
- Works on the already-lowered MIR — no coroutine-specific knowledge needed
- Smallest diff: ~500-1000 lines in a new file + 5-line registration

**Disadvantages:**
- Side effect classification is imprecise without annotations
- Cannot change the state machine layout (still per-thread struct)
- Post-hoc transformation may miss optimization opportunities

---

#### Option B: Modified StateTransform

**Files to modify:**
- FORK: `compiler/rustc_mir_transform/src/coroutine.rs`
- Key functions: `create_coroutine_resume_function()`, `insert_switch()`,
  `create_cases()`, `TransformVisitor`

**How it works:**

Instead of post-processing, modify the state machine generation itself.
When `target == nvptx64` (or when `#[warp_cooperative]` attribute is present):

1. `insert_switch()` emits broadcast + switch instead of plain switch:
   ```
   bb0: {
       _disc = self.discriminant;
       _disc = shfl_sync_idx(FULL_MASK, _disc, 0);  // broadcast
       switchInt(_disc) -> [...]
   }
   ```

2. `TransformVisitor` modifies yield point emission:
   ```
   // Standard: set discriminant, return Pending
   // Warp-cooperative: syncwarp, lane-0-only set discriminant, syncwarp, return Pending
   ```

3. `create_coroutine_resume_function()` changes the self argument handling:
   - Standard: `Pin<&mut Self>` with each task owning its state
   - Warp-cooperative: state is still `Pin<&mut Self>` but only lane 0's
     copy is authoritative; other lanes' copies are stale/ignored

4. Layout computation stays the same — the struct is identical, only the
   generated code around state transitions changes.

**Generated code (conceptual):**
```rust
struct MyAsyncState {
    discriminant: u32,
    // ... saved locals across yield points ...
}

fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<T> {
    let disc = unsafe { shfl_sync_idx(FULL_MASK, self.discriminant, 0) };
    match disc {
        0 => {
            // User code: let result = hostcall_submit(...)
            // Only lane 0 actually submits:
            let mut result = 0u64;
            if lane_id() == 0 {
                result = hostcall_submit(self.buf, ...);
            }
            // Broadcast result to all lanes:
            result = shfl_sync_idx_u64(FULL_MASK, result, 0);

            // Save state
            syncwarp(FULL_MASK);
            if lane_id() == 0 {
                self.discriminant = 3;
                self.saved_result = result;
            }
            syncwarp(FULL_MASK);
            Poll::Pending
        }
        3 => {
            // Resume: broadcast saved state
            let result = shfl_sync_idx_u64(FULL_MASK, self.saved_result, 0);
            // Check completion
            let ready = hostcall_check_ready(result);
            let ready = shfl_sync_idx(FULL_MASK, ready as u32, 0);
            if ready != 0 {
                syncwarp(FULL_MASK);
                if lane_id() == 0 { self.discriminant = 1; }
                syncwarp(FULL_MASK);
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }
        _ => panic!("invalid state"),
    }
}
```

**Advantages:**
- Full control over the state machine — can optimize layout for warp access
- Can emit data-parallel constructs (all lanes write different offsets)
- The "right" place architecturally — the transform understands coroutine semantics

**Disadvantages:**
- Forks `coroutine.rs` — must track upstream changes
- `coroutine.rs` is ~2500 lines of dense MIR manipulation; understanding it deeply
  takes significant effort
- Target-specific code in a target-independent pass is architecturally questionable
  (though we don't care about upstream acceptance)

---

#### Option C: Custom Codegen Backend

**Files to modify:**
- `compiler/rustc_codegen_llvm/src/builder.rs` — intercept codegen for coroutine patterns
- `compiler/rustc_codegen_llvm/src/intrinsic.rs` — add warp intrinsics

**How it works:**

Leave MIR completely unchanged. At LLVM IR generation time, detect the
coroutine state machine pattern (struct with discriminant + switch) and
emit warp-cooperative LLVM IR:

1. Replace `load discriminant` with `load + shfl.sync` in the LLVM IR
2. Wrap state-transition stores in predicated blocks (lane 0 only)
3. Insert `bar.warp.sync` barriers at yield boundaries

**Advantages:**
- MIR stays standard — everything before codegen is vanilla Rust
- Can leverage LLVM's understanding of the target

**Disadvantages:**
- Pattern detection at LLVM IR level is fragile — optimizations may have
  already transformed the switch beyond recognition
- LLVM's nvptx backend has NO concept of warps — we'd be fighting the backend
- Cannot use MIR-level information (which locals are across yield points)
- Most invasive option; highest maintenance burden

**Verdict: REJECTED.** The MIR is the right abstraction level. By the time we
reach LLVM IR, too much semantic information is lost.

---

#### Option D: Attribute-Driven

**Files to create/modify:**
- CREATE: `compiler/rustc_mir_transform/src/warp_cooperative.rs` (same as Option A)
- MODIFY: `compiler/rustc_ast_lowering/src/expr.rs` — recognize `#[warp_cooperative]`
- MODIFY: `compiler/rustc_middle/src/mir/mod.rs` — add flag to `CoroutineInfo`

**How it works:**

The user writes:
```rust
#[warp_cooperative]
async fn gpu_pipeline(buf: *mut u8) -> Result<(), GpuIoError> {
    let fd = gpu_open(buf, b"data.txt", 1).await?;
    gpu_write(buf, fd, b"output", 6).await?;
    gpu_close(buf, fd).await?;
    Ok(())
}
```

The `#[warp_cooperative]` attribute sets a flag on the `CoroutineInfo` during
AST→HIR lowering. The standard `StateTransform` runs normally. Then, a
post-transform pass (like Option A) checks the flag and applies warp-cooperative
modifications only to flagged coroutines.

**This is Option A + an attribute for opt-in semantics.**

**Advantages:**
- Explicit opt-in — no accidental warp-cooperative code
- Works alongside normal async code (non-warp futures compile normally)
- Clear intent for both the compiler and the programmer
- Smallest semantic change to Rust: "this async fn is warp-cooperative"

**Disadvantages:**
- Requires touching AST lowering (to propagate the attribute)
- New attribute in the language — but since we're forking, this is fine

---

### RECOMMENDATION: Option D (Attribute-Driven) = Option A + explicit opt-in

Option D is Option A with an attribute trigger. It is the smallest change that
achieves the goal:

1. Standard `StateTransform` generates the normal state machine (unchanged)
2. A new `WarpCooperativeTransform` pass runs after `StateTransform`
3. It only activates for coroutines with `#[warp_cooperative]` on the `CoroutineInfo`
4. It inserts: broadcast at dispatch, syncwarp at transitions, lane-0 guards for effects

Total diff: ~1000 lines new code + ~20 lines touching existing files.

---

## Section 2: The Dream Syntax — What's Actually Possible?

### Level 1: What we can achieve TODAY (no rustc changes)

```rust
#[warp_async]
unsafe fn gpu_pipeline(buf: *mut u8) -> bool {
    warp_print!(buf, b"start");
    let fd = warp_open!(buf, b"data.txt", 1);
    warp_write!(buf, fd, b"GPU output", 10);
    warp_close!(buf, fd);
    warp_print!(buf, b"done");
}
```

This already works. The proc macro generates a `WarpFuture` state machine.
It is NOT `async fn` — the `#[warp_async]` macro consumes the function and
emits a completely different struct.

### Level 2: Proc-macro-intercepted async syntax (no rustc changes)

```rust
#[warp_async]
async fn gpu_pipeline(buf: *mut u8) -> bool {
    gpu_print(buf, b"start").await;
    let fd = gpu_open(buf, b"data.txt", 1).await;
    gpu_write(buf, fd, b"GPU output", 10).await;
    gpu_close(buf, fd).await;
    gpu_print(buf, b"done").await;
}
```

The proc macro strips `async`, recognizes `.await` expressions, and generates
the same `WarpFuture` state machine. The `.await` is cosmetic — the compiler
never sees an async fn. `gpu_open()`, `gpu_print()` etc. return marker types
that the macro recognizes.

**What this buys**: familiar syntax. `.await` signals "this is an async operation"
to anyone reading the code, even if the underlying mechanism is WarpFuture.

**What this costs**: the `.await` is a lie. IDE support breaks (rust-analyzer
thinks it's a real async fn). `?` after `.await` doesn't work naturally
(the macro must handle it). Composition with `join!`/`select!` is impossible.

**Verdict**: Marginal improvement. Not worth the confusion.

### Level 3: Real async fn with warp-cooperative codegen (requires rustc fork)

```rust
#[warp_cooperative]
async fn gpu_pipeline(buf: *mut u8) -> Result<(), GpuIoError> {
    gpu_println(buf, b"start").await;
    let fd = GpuFile::create(buf, "data.txt").await?;
    fd.write_all(buf, b"GPU output").await?;
    fd.close(buf).await?;
    gpu_println(buf, b"done").await;
    Ok(())
}
```

Here:
- `async fn` is REAL — the compiler generates a real `impl Future` state machine
- `#[warp_cooperative]` triggers the post-StateTransform MIR pass
- The pass inserts warp synchronization into the state machine
- `.await` is REAL — standard Rust desugaring into poll loop + yield
- `?` operator works naturally (it's just normal Rust)
- `GpuFile::create()` returns a real `impl Future` that submits a hostcall
  and yields until the host responds
- The warp-cooperative pass ensures all 32 lanes poll together

**What `GpuFile::create()` looks like:**
```rust
pub struct GpuCreateFuture {
    buf: *mut u8,
    path: &'static [u8],
    state: GpuCreateState,
}

enum GpuCreateState {
    Init,
    Waiting { pkt_idx: u16 },
}

impl Future for GpuCreateFuture {
    type Output = Result<GpuFile, GpuIoError>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            GpuCreateState::Init => {
                let pkt_idx = hostcall_submit(this.buf, SERVICE_OPEN, ...);
                this.state = GpuCreateState::Waiting { pkt_idx };
                Poll::Pending
            }
            GpuCreateState::Waiting { pkt_idx } => {
                if hostcall_ready(this.buf, pkt_idx) {
                    let fd = hostcall_read_result(this.buf, pkt_idx);
                    hostcall_release(this.buf, pkt_idx);
                    Poll::Ready(Ok(GpuFile { fd, buf: this.buf }))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
```

**Key insight**: The `GpuCreateFuture` itself is a standard per-thread Future.
It doesn't know about warps. The OUTER state machine (generated by `async fn`
+ StateTransform + WarpCooperativeTransform) handles warp cooperation. This
means:

1. Library authors write normal `impl Future` — no warp knowledge needed
2. The compiler's warp-cooperative pass handles synchronization
3. The inner future's `poll()` is called only by lane 0 (side effect guarding)
4. The result is broadcast to all lanes via `shfl.sync`

**This is the key architectural insight**: warp cooperation is a PROPERTY OF
THE CALLER'S STATE MACHINE, not of the individual futures being awaited.

### Level 4: The Ultimate Dream — std integration

```rust
#[warp_cooperative]
async fn gpu_pipeline() -> io::Result<()> {
    println!("start");
    let mut f = File::create("data.txt").await?;
    f.write_all(b"GPU output").await?;
    println!("done");
    Ok(())
}
```

This requires `std::fs::File::create()` to return `impl Future` on GPU.
This means changing the return type of std functions based on target — which
violates Rust's type system invariants.

**Verdict**: NOT achievable without fundamentally changing Rust's type system.
The right answer is `GpuFile`, not `std::fs::File`. Level 3 is the realistic
dream.

### The "Why .await?" Question — Answered

The BS56 skeptic asked: "what does `.await` buy you over `warp_print!()`?"

With a rustc fork (Level 3), the answer is substantial:

1. **`?` operator**: `let fd = GpuFile::create(buf, "x").await?;` — error
   propagation works naturally. No proc-macro tricks needed. The `?` desugars
   to `match result { Ok(v) => v, Err(e) => return Err(e.into()) }` which is
   standard MIR that the warp-cooperative pass handles correctly.

2. **Real composition**: `futures::join!(op_a, op_b).await` — both operations
   proceed concurrently (in a warp-cooperative manner). The joined future's
   state machine polls both sub-futures, and the warp-cooperative pass ensures
   all 32 lanes are synchronized.

3. **Type safety**: The compiler checks that `.await` is used in async context,
   that the Future trait is implemented, that lifetimes are correct. None of
   this works with proc-macro interception.

4. **IDE support**: rust-analyzer understands real async fn. Autocompletion,
   type inference, error highlighting all work.

5. **Debuggability**: Standard debugger support for async state machines.
   The coroutine state is a real struct with named fields.

6. **No learning curve**: Any Rust developer can read and write
   `#[warp_cooperative] async fn` code. The only new concept is the attribute.

---

## Section 3: Proof of Concept Plan

### Phase 1: Manual Proof (no compiler changes, 1-2 weeks)

**Goal**: Write by hand the EXACT code that a warp-cooperative rustc would generate,
and prove it works on GPU.

**The simplest async fn to prove:**
```rust
async fn two_prints(buf: *mut u8) {
    gpu_println(buf, b"hello").await;
    gpu_println(buf, b"world").await;
}
```

**What rustc's StateTransform generates (standard):**
```rust
struct TwoPrintsState {
    discriminant: u32,
    buf: *mut u8,
    __awaitee: GpuPrintFuture,  // live across yield point
}

impl Future for TwoPrintsState {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.discriminant {
            0 => {  // Unresumed — start first print
                this.__awaitee = GpuPrintFuture::new(this.buf, b"hello");
                this.discriminant = 3;  // suspended at yield 1
                // Fall through to poll the sub-future
                match this.__awaitee.poll(cx) {
                    Poll::Ready(()) => {
                        this.discriminant = 4;  // move to yield 2
                        // ... start second print ...
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            3 => {  // Resume after yield 1
                match this.__awaitee.poll(cx) {
                    Poll::Ready(()) => {
                        this.__awaitee = GpuPrintFuture::new(this.buf, b"world");
                        this.discriminant = 4;
                        match this.__awaitee.poll(cx) {
                            Poll::Ready(()) => {
                                this.discriminant = 1;  // Returned
                                Poll::Ready(())
                            }
                            Poll::Pending => Poll::Pending,
                        }
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            4 => {  // Resume after yield 2
                match this.__awaitee.poll(cx) {
                    Poll::Ready(()) => {
                        this.discriminant = 1;
                        Poll::Ready(())
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            _ => panic!("resumed completed coroutine"),
        }
    }
}
```

**What the warp-cooperative version looks like** (what we write BY HAND):
```rust
struct WarpTwoPrintsState {
    discriminant: u32,
    buf: *mut u8,
    pkt_idx: u16,       // packet index for current hostcall
    sub_state: u32,     // sub-future state (Init / Waiting)
}

impl WarpTwoPrintsState {
    /// Warp-cooperative poll. All 32 lanes must call simultaneously.
    unsafe fn poll_warp(&mut self) -> WarpPoll<()> {
        // Broadcast discriminant from lane 0
        let disc = shfl_sync_idx(FULL_MASK, self.discriminant, 0);

        match disc {
            0 => {
                // Start first print: lane 0 submits, all lanes proceed
                let mut pkt = NULL_INDEX as u32;
                if lane_id() == 0 {
                    pkt = hostcall_submit_print(self.buf, b"hello") as u32;
                }
                pkt = shfl_sync_idx(FULL_MASK, pkt, 0);

                if pkt == NULL_INDEX as u32 {
                    return WarpPoll::Pending;  // backpressure
                }

                syncwarp(FULL_MASK);
                if lane_id() == 0 {
                    self.pkt_idx = pkt as u16;
                    self.discriminant = 3;
                }
                syncwarp(FULL_MASK);
                WarpPoll::Pending
            }
            3 => {
                // Wait for first print to complete
                let idx = shfl_sync_idx(FULL_MASK, self.pkt_idx as u32, 0) as u16;
                let ready = hostcall_check_ready(self.buf, idx);
                let ready = shfl_sync_idx(FULL_MASK, ready as u32, 0);

                if ready != 0 {
                    if lane_id() == 0 {
                        hostcall_release(self.buf, idx);
                    }
                    // Start second print
                    let mut pkt = NULL_INDEX as u32;
                    if lane_id() == 0 {
                        pkt = hostcall_submit_print(self.buf, b"world") as u32;
                    }
                    pkt = shfl_sync_idx(FULL_MASK, pkt, 0);

                    if pkt == NULL_INDEX as u32 {
                        // Released old packet but can't allocate new one —
                        // need intermediate state. Transition to state 4 (init second).
                        syncwarp(FULL_MASK);
                        if lane_id() == 0 { self.discriminant = 4; }
                        syncwarp(FULL_MASK);
                        return WarpPoll::Pending;
                    }

                    syncwarp(FULL_MASK);
                    if lane_id() == 0 {
                        self.pkt_idx = pkt as u16;
                        self.discriminant = 5;
                    }
                    syncwarp(FULL_MASK);
                    WarpPoll::Pending
                } else {
                    WarpPoll::Pending
                }
            }
            4 => {
                // Init second print (retry after backpressure)
                let mut pkt = NULL_INDEX as u32;
                if lane_id() == 0 {
                    pkt = hostcall_submit_print(self.buf, b"world") as u32;
                }
                pkt = shfl_sync_idx(FULL_MASK, pkt, 0);

                if pkt == NULL_INDEX as u32 {
                    return WarpPoll::Pending;
                }

                syncwarp(FULL_MASK);
                if lane_id() == 0 {
                    self.pkt_idx = pkt as u16;
                    self.discriminant = 5;
                }
                syncwarp(FULL_MASK);
                WarpPoll::Pending
            }
            5 => {
                // Wait for second print
                let idx = shfl_sync_idx(FULL_MASK, self.pkt_idx as u32, 0) as u16;
                let ready = hostcall_check_ready(self.buf, idx);
                let ready = shfl_sync_idx(FULL_MASK, ready as u32, 0);

                if ready != 0 {
                    if lane_id() == 0 {
                        hostcall_release(self.buf, idx);
                        self.discriminant = 1;  // Done
                    }
                    syncwarp(FULL_MASK);
                    WarpPoll::Ready(())
                } else {
                    WarpPoll::Pending
                }
            }
            _ => WarpPoll::Ready(()),
        }
    }
}
```

**Wait — this is exactly what `#[warp_async]` already generates!**

Yes. And that's the point. Phase 1 is about proving that the STRUCTURE of a
compiler-generated warp-cooperative state machine is isomorphic to what
`#[warp_async]` already produces. The difference:

- `#[warp_async]` generates this from `warp_print!(buf, b"hello")` syntax
- A rustc fork would generate this from `gpu_println(buf, b"hello").await` syntax

The state machine is the same. The input syntax is different.

**But there IS a real structural difference**: With the rustc approach, the inner
futures (`GpuPrintFuture`) are standard `impl Future`. They don't need to know
about warps. The warp-cooperative pass wraps the poll calls:

```
// Standard codegen:
match inner_future.poll(cx) { ... }

// Warp-cooperative codegen:
let poll_result;
if lane_id() == 0 {
    poll_result = inner_future.poll(cx);
} else {
    poll_result = Poll::Pending;  // doesn't matter, will be broadcast
}
let is_ready = shfl_sync_idx(FULL_MASK, poll_result.is_ready() as u32, 0);
if is_ready != 0 {
    // broadcast the Ready value from lane 0
    // ... state transition ...
}
```

This is the key difference from `#[warp_async]`:
- `#[warp_async]` requires the INNER operations to be warp-aware (`warp_print!`)
- The rustc approach makes the OUTER state machine warp-aware; inner futures are standard

**Phase 1 experiment:**

1. Write `GpuPrintFuture` as a standard `impl Future` (per-thread, no warp knowledge)
2. Write the warp-cooperative state machine BY HAND that wraps it (as shown above)
3. Run it on GPU with 32 lanes
4. Verify: lane 0 does the actual polling, all lanes get the result via broadcast

This proves that standard `impl Future` can be wrapped in a warp-cooperative
executor without modifying the inner futures.

### Phase 2: Enhanced Proc Macro (no rustc changes, 2-3 weeks)

**Goal**: Extend `#[warp_async]` to accept `async fn` syntax with `.await`.

The macro would:
1. Accept `async fn` (strip the `async` keyword)
2. Recognize `.await` expressions on known hostcall future types
3. Generate the same WarpFuture state machine as today
4. Support `?` operator on `.await` expressions (new capability!)

```rust
#[warp_async]
async fn pipeline(buf: *mut u8) -> Result<bool, GpuIoError> {
    gpu_println(buf, b"start").await;
    let fd = GpuFile::create(buf, "data.txt").await?;  // ? works!
    fd.write_all(buf, b"output").await?;
    fd.close(buf).await?;
    Ok(true)
}
```

The macro generates a WarpFuture with error propagation. The `.await` signals
yield points; `?` after `.await` generates an error-check state that broadcasts
the error status and short-circuits if any error occurred.

**This is the pragmatic middle ground**: better syntax than `warp_print!`, real
`?` support, and no rustc fork required.

### Phase 3: rustc MIR Pass (requires rustc fork, 2-4 months)

**Goal**: Implement the `WarpCooperativeTransform` pass (Option D from Section 1).

Steps:
1. Fork rust-lang/rust at a specific nightly (pin the version)
2. Add `#[warp_cooperative]` attribute support
3. Implement the MIR pass
4. Build the forked rustc targeting nvptx64
5. Compile `#[warp_cooperative] async fn` and verify the generated PTX

The pass implementation:

```
WarpCooperativeTransform:
  for each body in MIR:
    if body.coroutine_info.is_some() && body.has_warp_cooperative_attr():
      // Step 1: Find the discriminant switch (always in bb0)
      let switch_bb = find_discriminant_switch(body);

      // Step 2: Insert broadcast before the switch
      insert_shfl_broadcast(body, switch_bb, discriminant_local);

      // Step 3: For each resume point (case in the switch):
      for (state, target_bb) in switch_cases:
        // Insert syncwarp at the beginning
        insert_syncwarp(body, target_bb, ENTRY);

        // Find sub-future poll calls and wrap in lane-0 guard
        for stmt in target_bb.statements:
          if is_future_poll_call(stmt):
            wrap_in_lane0_guard(body, target_bb, stmt);
            insert_result_broadcast(body, target_bb, stmt.result);

        // Find state transition (discriminant write) and wrap
        for terminator in target_bb.terminator:
          if writes_discriminant(terminator):
            wrap_in_lane0_guard(body, target_bb, terminator);
            insert_syncwarp(body, target_bb, AFTER_TERMINATOR);
```

**Estimated diff size**: ~1500 lines total
- 1000 lines: `warp_cooperative.rs` (the pass)
- 200 lines: attribute plumbing (AST → HIR → MIR)
- 100 lines: intrinsic declarations for `shfl.sync`, `syncwarp`, `lane_id`
- 200 lines: tests

---

## Section 4: The LLVM Problem — Is It Real?

### Claim: "LLVM could break warp-cooperative code"

The BS56 skeptic raised three specific concerns:
1. LLVM could hoist `shfl.sync` out of a loop
2. LLVM could sink `syncwarp` past a conditional
3. LLVM could merge lane-guarded regions

### Analysis: Does LLVM reorder inline asm on nvptx?

**No, it does not.**

The `asm!` macro in Rust generates LLVM inline assembly. LLVM treats inline
assembly as a black box with declared side effects. The key constraints:

1. **`options(nostack)`**: Tells LLVM the asm doesn't use the stack. Does NOT
   remove ordering constraints. LLVM will not move an `asm!` block past another
   `asm!` block or past a memory operation unless it can prove they don't
   interact — and with inline asm, it can't prove this.

2. **Memory clobber**: Rust's `asm!` with `options()` but without `pure` or
   `readonly` implies the asm may read and write arbitrary memory. LLVM treats
   this as a full compiler barrier — nothing is reordered across it.

3. **`shfl.sync`**: The inline asm:
   ```
   asm!("shfl.sync.idx.b32 {result}, {val}, {src}, 0x1f, {mask};",
        result = out(reg32) result,
        val = in(reg32) val,
        src = in(reg32) src_lane,
        mask = in(reg32) mask,
        options(nostack));
   ```
   LLVM sees: "I have an instruction with 3 inputs and 1 output. I don't
   know what it does. I cannot move it past anything that might affect its
   inputs." This is correct behavior.

4. **`syncwarp`**: The inline asm:
   ```
   asm!("bar.warp.sync {mask};",
        mask = in(reg32) mask,
        options(nostack));
   ```
   LLVM sees: "I have an instruction with 1 input and no output. It might
   have side effects (no `pure` option). I cannot move it."

### Does LLVM optimize across asm barriers on nvptx?

**Empirically: no.** This project has been running inline PTX asm for warp
intrinsics, system-scope atomics, and memory barriers since its inception.
All of the following work correctly:

- `shfl.sync.idx.b32` broadcasts (used in every WarpFuture poll)
- `bar.warp.sync` barriers (used at every state transition)
- `st.release.sys.global` stores (used for hostcall protocol)
- `ld.acquire.sys.global` loads (used for hostcall polling)
- `activemask.b32` queries (used for mask tracking)
- `atom.global.sys.cas.b64` CAS operations (used for lock-free stacks)

None of these have ever been broken by LLVM optimization. The test suite
includes 40+ GPU tests exercising these paths, run on real hardware.

### What about the `volatile` semantics of `shfl.sync`?

`shfl.sync` is not `volatile` in the C/C++ sense. It is a cross-lane data
movement instruction. LLVM cannot understand its semantics because it's inline
asm. This is actually an advantage — LLVM won't try to optimize it because it
doesn't know what it does.

The PTX specification states that `shfl.sync` has **implicit barrier semantics**
for the threads specified in the mask. All participating threads must reach the
`shfl.sync` before any can proceed. This provides hardware-level convergence
guarantee that LLVM cannot break because LLVM doesn't generate the `shfl.sync`
— we do, via inline asm.

### What about LLVM optimizations that DON'T involve inline asm?

This is a valid concern. If the warp-cooperative MIR pass generates code like:

```rust
let lane = lane_id();  // inline asm
if lane == 0 {
    *ptr = value;       // normal store — LLVM CAN optimize this
}
syncwarp(mask);         // inline asm
```

LLVM could theoretically:
- Prove that `lane` is always non-zero on 31 of 32 threads and eliminate the
  store for those threads → **This is fine! That's exactly what we want.**
- Hoist the store above the `if` check → **Cannot happen.** The condition
  depends on `lane_id()` which is inline asm. LLVM cannot evaluate it at
  compile time. The branch is genuinely conditional.
- Merge the store with a later store → **Possible but benign.** If two
  stores to the same address are both inside `if lane == 0`, LLVM could
  merge them. This is semantically correct — lane 0 would do both stores,
  and merging them produces the same result.

### Verdict: The LLVM Problem Is Not Real

LLVM cannot break warp-cooperative code generated from inline PTX asm because:

1. LLVM treats inline asm as an opaque barrier
2. LLVM's nvptx backend does not understand warp semantics, so it cannot
   "optimize" warp-level constructs — it simply passes them through
3. The implicit barrier semantics of `shfl.sync` and `bar.warp.sync` are
   enforced by the GPU hardware, not by LLVM
4. 40+ tests on real hardware confirm correct behavior

The skeptic's concern was theoretically sound but empirically unfounded.
LLVM's nvptx backend is conservative precisely because it lacks warp awareness
— it cannot make optimizations that assume single-thread execution because
the entire PTX execution model is multi-thread.

---

## Section 5: Concrete Research Tasks

### Epic: `native-warp-async` — Prove warp-cooperative async/await works in Rust

**Goal**: Demonstrate that Rust's `async fn` can generate warp-cooperative
state machines for GPU execution, either via proc macro or rustc modification.

### Theme 1: `warp-future-bridge` — Standard Future wrapped in warp cooperation

**Goal**: Prove that standard `impl Future` types can be executed
warp-cooperatively by wrapping them in a warp-aware outer state machine.

**Success criteria**:
1. A standard `impl Future` (per-thread, no warp knowledge) is polled by a
   warp-cooperative executor where lane 0 polls and broadcasts results
2. Two sequential `.await` points work with warp convergence maintained
3. Error propagation (`?`) works across warp-cooperative `.await` boundaries

**Tasks**:

- **`warp-future-bridge.1`** (experiment): Write `GpuPrintFuture` as standard
  `impl Future`. Verify it compiles for nvptx64 and works with a simple
  per-thread executor (single lane).

- **`warp-future-bridge.2`** (experiment): Write a hand-coded warp-cooperative
  wrapper that polls `GpuPrintFuture` from lane 0 only, broadcasts the
  `Poll` result via `shfl.sync`, and maintains convergence. This is the
  "manual proof" that the compiler could generate this code.

- **`warp-future-bridge.3`** (experiment): Chain two `GpuPrintFuture`s in a
  hand-coded warp-cooperative state machine. This proves multi-yield-point
  warp cooperation works with standard futures.

- **`warp-future-bridge.4`** (experiment): Add `Result<T, E>` to the chain.
  Hand-code a warp-cooperative state machine where `.await?` broadcasts
  the error status and all lanes either continue or early-return together.

### Theme 2: `warp-async-v2` — Enhanced proc macro with async syntax

**Goal**: Extend `#[warp_async]` to accept `async fn` syntax with `.await`
and `?` operator.

**Success criteria**:
1. `#[warp_async] async fn` compiles and generates correct WarpFuture
2. `.await` on known hostcall types works
3. `?` after `.await` propagates errors correctly
4. Backward compatible — old `warp_print!` syntax still works

**Tasks**:

- **`warp-async-v2.1`** (design): Design the macro's `.await` recognition.
  Define which types are recognized as "hostcall futures" and how `.await`
  maps to INIT/WAIT state pairs.

- **`warp-async-v2.2`** (experiment): Implement `?` operator support in the
  existing `#[warp_async]` macro (even without `.await` syntax). This is
  the highest-value standalone improvement.

- **`warp-async-v2.3`** (experiment): Implement `.await` recognition in the
  macro. The macro parses `expr.await` and generates the same INIT/WAIT states
  as `warp_print!(...)`.

- **`warp-async-v2.4`** (experiment): End-to-end test: `async fn` with
  multiple `.await?` calls, error handling, if/else branching.

### Theme 3: `rustc-warp` — rustc MIR pass for warp-cooperative codegen

**Goal**: Implement a proof-of-concept `WarpCooperativeTransform` MIR pass
in a rustc fork.

**Success criteria**:
1. Forked rustc compiles a `#[warp_cooperative] async fn` to PTX
2. The generated PTX contains `shfl.sync` and `bar.warp.sync` instructions
3. The generated kernel executes correctly on GPU with 32 lanes

**Tasks**:

- **`rustc-warp.1`** (investigation): Set up rustc fork build environment.
  Build rustc from source targeting nvptx64. Verify baseline: compile a
  normal `async fn` for nvptx64 and inspect the generated PTX.

- **`rustc-warp.2`** (design): Write detailed MIR transformation specification.
  For a specific 2-yield-point async fn, show the exact MIR before and after
  the `WarpCooperativeTransform` pass.

- **`rustc-warp.3`** (experiment): Implement the `WarpCooperativeTransform`
  pass. Start with the simplest case: a single `.await` point with no
  error handling. The pass inserts broadcast + syncwarp + lane-0 guard.

- **`rustc-warp.4`** (experiment): Extend the pass to handle multiple yield
  points, `?` operator (which becomes additional MIR blocks for error
  checking), and verify the generated PTX.

- **`rustc-warp.5`** (experiment): Integration test — compile a
  `#[warp_cooperative] async fn` that does two hostcall prints, run it on
  GPU hardware, verify output.

### Priority Ordering

1. **`warp-future-bridge.1-4`** (1-2 weeks) — Prove the concept WITHOUT any
   tooling changes. If standard `impl Future` can be wrapped in warp cooperation,
   the rest follows.

2. **`warp-async-v2.2`** (1 week) — Add `?` to `#[warp_async]`. Immediate
   ergonomic win regardless of future direction.

3. **`warp-async-v2.1,3,4`** (2-3 weeks) — Full async syntax in proc macro.
   Pragmatic solution that doesn't require a rustc fork.

4. **`rustc-warp.1-5`** (2-4 months) — The long game. Only pursue after
   phases 1-3 prove the concept works.

### Why This Ordering?

The `warp-future-bridge` theme answers the FUNDAMENTAL question: can standard
`impl Future` types work inside a warp-cooperative state machine? If yes,
then both the proc macro path (Theme 2) and the rustc path (Theme 3) are
viable. If no, we learn why and can adjust.

The `warp-async-v2.2` task (`?` operator) is a standalone win that improves
the existing system regardless of future direction. It should be done early.

The rustc fork (Theme 3) is the most ambitious but also the most transformative.
It should only be attempted after the concept is proven in Themes 1-2.

---

## Summary

### The Key Insight

The warp-cooperative state machine that `#[warp_async]` generates TODAY is
structurally isomorphic to what a rustc warp-cooperative MIR pass WOULD
generate. The difference is:

| Aspect | `#[warp_async]` (today) | rustc MIR pass (proposed) |
|--------|------------------------|--------------------------|
| Input syntax | `warp_print!(buf, msg)` | `gpu_println(buf, msg).await` |
| State machine generator | Proc macro (AST-level) | Compiler (MIR-level) |
| Inner operations | Must be warp-aware macros | Standard `impl Future` |
| Error handling | None (planned in v2) | `?` works naturally |
| Composition | None | `join!`, `select!` possible |
| Type safety | Limited (proc macro) | Full (compiler-checked) |
| Maintenance | Proc macro code (~1500 LOC) | rustc fork (~1500 LOC diff) |

### The Minimal Change

A `#[warp_cooperative]` attribute + a ~1000 line MIR pass that runs AFTER
the standard `StateTransform`. No existing compiler code is modified. The
pass inserts:

1. `shfl.sync.idx` broadcast at discriminant dispatch
2. `lane_id() == 0` guard around side effects
3. `bar.warp.sync` barrier at state transitions
4. Result broadcast after inner future poll

### The LLVM Problem Is Not Real

40+ tests on real hardware prove that LLVM's nvptx backend correctly handles
inline PTX asm for `shfl.sync`, `bar.warp.sync`, and system-scope atomics.
LLVM cannot break warp-cooperative code because it treats inline asm as an
opaque barrier and has no warp-level optimization passes.

### The Path Forward

1. Prove the concept manually (standard Future wrapped in warp cooperation)
2. Improve the proc macro (add `?`, optionally `.await` syntax)
3. Fork rustc and implement the MIR pass

Each phase is independently valuable. Phase 1 proves the concept. Phase 2
improves today's system. Phase 3 achieves the dream.
