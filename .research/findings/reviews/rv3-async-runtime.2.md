# Review: async-runtime.2 — GPU Executor Architecture Design
**Reviewer**: single-agent | **Seq**: rv3

## Verdict: rework

## Summary
The design is well-researched, clearly written, and makes sound top-level decisions (per-thread executor, bitmask run queue, no-op critical section). However, there are several issues that need resolution before implementation: (1) the HostcallFuture conflates per-lane async semantics with the warp-granular packet format from ADR-3, creating an unaddressed mismatch; (2) the self-waking pattern makes every pending future effectively busy-poll, which undermines the purpose of async; and (3) the Phase 1-3 plan should be reordered given that LTO has been confirmed to work, making Embassy the primary path. None of these are fatal — they are addressable with targeted rework.

## Correctness Issues

### Issue 1 — Warp-granular packets vs per-lane async hostcall: major
The hostcall protocol (ADR-3, hostcall.3) defines **warp-granular packets**: one packet serves all 32 lanes in a warp simultaneously. The `active_mask` field indicates which lanes participate, and the payload has `slots[32][8]` — one slot-row per lane. The protocol assumes all active lanes in the warp coordinate to fill a single packet together.

The HostcallFuture in this design treats the hostcall as a **per-lane operation**: each lane independently pops a free packet, fills it, pushes it to ready, and waits for the response. This means:
- If all 32 lanes in a warp independently call `hostcall_async`, they each pop a separate packet from the free pool (32 packets consumed instead of 1).
- The `active_mask` field is never set (the design shows `service` and `args` but no `__activemask()` call).
- The payload format is wasted: each packet has 32 lane slots but only 1 is used.

This is a fundamental mismatch. The synchronous hostcall in ADR-3 uses `__activemask()` and warp-cooperative filling. The async version needs to either:
- (a) Retain warp-cooperative semantics via a warp-level barrier before submitting (complex with async, since lanes may reach the hostcall at different times), or
- (b) Explicitly redesign the async path to use per-lane packets (smaller, 1 lane worth of payload), or
- (c) Document that async hostcall uses 1 packet per lane and the pool must be sized accordingly (32x more packets needed).

**Recommendation**: Address this in the design. Option (c) is simplest for Phase 1 but the pool sizing implications are significant (32x pressure). Add a note that warp-cooperative async hostcall is a Phase 4 optimization.

### Issue 2 — `hc_pop_free` blocking in Init state: major
In the `HostcallState::Init` branch, the code calls `hc_pop_free(this.buffer)?` with a `?` operator. Looking at the ADR-3 protocol, popping from the free stack is a CAS loop that spins until a free packet is available (or returns `PoolExhausted`).

If the pool is exhausted, the `?` propagates the error immediately, turning the future into `Poll::Ready(Err(...))`. But the caller probably wants to retry later, not fail permanently. The design text in section D4 point 5 mentions "back-pressure via `Poll::Pending` when the free stack is empty (retry on next poll cycle)" but the code does not implement this — it returns an error.

**Recommendation**: Add a `HostcallState::WaitingForPacket` state (or handle pool exhaustion by returning `Poll::Pending` and re-waking, rather than erroring). Alternatively, document that `PoolExhausted` is a fatal error and callers must handle it.

### Issue 3 — No-op critical section correctness boundary: minor
The no-op critical section analysis is correct for the per-thread executor case. The document properly identifies the boundary where it breaks (inter-thread Embassy primitives like `Mutex<CriticalSection, T>`). However, this boundary is not enforced at compile time — nothing prevents a user from using `embassy_sync::Mutex` across threads with the no-op CS, silently producing data races.

**Recommendation**: Add a note that a lint, documentation warning, or wrapper type should prevent accidental misuse of Embassy's CS-based sync primitives across GPU threads until a proper spinlock CS is implemented.

### Issue 4 — Waker data encoding assumes single executor: minor
The waker encodes `executor pointer + task index` in a single 64-bit word (bits 63..8 for pointer, bits 7..0 for index). The design says "or unused if single executor per thread." If there is ever more than one executor per thread (unlikely but possible), the pointer bits become necessary. More importantly, the `wake()` function accesses `CURRENT_EXECUTOR` as a global — this is thread-local state that does not exist on GPU.

The design hand-waves this as "a register variable or function parameter" but does not specify the actual mechanism. On nvptx64, there is no thread-local storage. The executor pointer must be passed through registers (function arguments) or stored in a known location.

**Recommendation**: Specify the concrete mechanism for `CURRENT_EXECUTOR` access. Options: (a) pass executor pointer as a kernel argument and propagate it, (b) store in a `static` variable indexed by `threadIdx` (expensive — global memory), (c) since waker only fires within the same thread's poll loop, the executor reference is always on the call stack — just use the local variable directly.

## Architecture Issues

### Issue 5 — Phase ordering should prioritize Embassy: major
The design presents Phase 1 (custom GpuExecutor) before Phase 3 (Embassy integration). However, async-runtime.1.2 has confirmed that fat LTO resolves all Embassy cross-crate calls with zero modifications needed. This means Embassy works today — no fork, no vendoring.

Building a custom GpuExecutor first and then evaluating Embassy afterward is backwards. The custom executor duplicates work that Embassy already provides. The implementation plan should be:
- **Phase 1**: Embassy integration (provide `__pender` no-op + CS no-op + `lto = "fat"`, done).
- **Phase 2**: Async HostcallFuture on top of Embassy's executor.
- **Phase 3**: Measure register pressure. If unacceptable, build the stripped-down custom executor as a fallback.

**Recommendation**: Reorder the phases. Embassy-first is now the lower-risk, lower-effort path.

### Issue 6 — Type erasure for task array is unresolved: minor
Open Question 5 lists several options for holding heterogeneous future types in the inline task array but defers the decision. This is a core design question — the answer significantly affects the implementation. If Embassy is used (per the recommendation above), this question is moot because Embassy handles type erasure via `TaskStorage<F>` and `TaskPool<F, N>`.

**Recommendation**: If the custom GpuExecutor is retained as a fallback, resolve this during the design phase, not during implementation. Embassy's approach (separate `TaskPool<F, N>` per future type) is proven and should be the default choice.

## Performance Issues

### Issue 7 — Self-waking creates busy-poll for all pending futures: major
The `HostcallFuture` calls `cx.waker().wake_by_ref()` on every `Poll::Pending` return. This means every pending hostcall future is re-enqueued for polling on the next executor cycle. With N pending futures, the executor polls all N every cycle, even if none are ready.

This is functionally equivalent to busy-polling all futures in a loop — the async machinery (waker vtable dispatch, bitmask operations, state matching) adds overhead compared to a simple `for` loop over pending requests. The design acknowledges this in R4 but underestimates the impact: the self-waking pattern means the executor **never** enters the `nanosleep` idle path (there is always at least one task in the run queue if any hostcall is pending).

**Recommendation**: Consider an alternative where the executor itself performs a scan of pending futures on each cycle (without requiring explicit self-waking). This is what Embassy's `arch-spin` effectively does — it polls all tasks unconditionally in a tight loop. Alternatively, accept busy-poll as the model and remove the waker machinery entirely for the GPU case, using a simpler "poll all tasks every cycle" approach. The nanosleep optimization should only trigger when there are genuinely no pending futures (all tasks completed or no tasks spawned).

### Issue 8 — Register pressure estimate may be optimistic: minor
The estimate of "20-30 registers for 2 tasks" counts executor metadata but may undercount:
- The `HostcallFuture` state machine includes a raw pointer (`*const HostcallBuffer`), a `u16` packet index, and a 3-variant enum — that alone is ~3-4 registers.
- The `match` arms in `poll()` generate branch code that may require additional temporaries.
- The `RawWaker` vtable dispatch (`wake_by_ref`) is an indirect call that requires loading the vtable pointer and function pointer (2 loads + call setup).
- If LTO inlines Embassy's executor, the combined code may use more registers than the custom executor estimate.

This is not a design flaw — just a note that the estimates should be verified empirically (as the design itself recommends in Phase 4). The 32-register and 64-register occupancy calculations are useful baselines.

**Recommendation**: No design change needed, but flag that empirical measurement (Phase 4 / `ptxas -v`) should happen early, not late. If Embassy + HostcallFuture exceeds 64 registers, the fallback plan must be ready.

## Recommendations

1. **Resolve the warp-granular vs per-lane packet mismatch** (Issue 1). This is the most important architectural question. Document the chosen approach and its pool-sizing implications.

2. **Reorder phases**: Embassy-first (Issue 5). The custom GpuExecutor becomes the fallback, not the starting point.

3. **Handle pool exhaustion gracefully in HostcallFuture** (Issue 2). Either add a `WaitingForPacket` state or document that exhaustion is a hard error.

4. **Reconsider the self-waking pattern** (Issue 7). For a per-thread executor with no external wake source, a simple "poll all pending tasks" loop may be more efficient than the waker machinery. This aligns with Embassy's `arch-spin` model.

5. **Specify the CURRENT_EXECUTOR mechanism** (Issue 4). The waker needs a concrete way to access the executor on GPU.

6. **Add a compile-time or doc-level guard** against misusing the no-op critical section for inter-thread sync (Issue 3).

7. **Measure register pressure early** (Issue 8). Do not wait for Phase 4 — a quick `ptxas -v` after Phase 1 (Embassy integration) will validate or invalidate the occupancy estimates.
