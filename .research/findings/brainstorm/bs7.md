# BS7 — VectorWare Parity Analysis & Integration Planning
**Date**: 2026-03-12
**Brainstorm seq**: 7
**Trigger**: interval (completed_tasks 23 - last_brainstorm 21 = 2, forced by integration.1 readiness)
**Level**: standard (single document, structured analysis)

## Context

All prerequisites for integration.1 are met. 23 tasks completed across 3 completed themes
(toolchain, hostcall, atomics) and 3 active themes (gpu-std, async-runtime, integration).
The project has demonstrated every major component individually. The question is now: how to
combine them for maximum VectorWare parity, and what gaps remain unbridgeable without rustc
patches.

---

## Section 1: Technical Analysis

### 1.1 Feasible NOW vs Requires Std Patching

**Feasible today (no std source patching):**
- Async hostcall (HostcallFuture) — Embassy executor + hostcall protocol are both working;
  combining them requires wrapping the spin-wait in a Future that returns Pending while
  the host hasn't responded. The poll function checks the packet's control word; if
  READY bit is set, extract response and return Ready. Otherwise return Pending.
- stdin hostcall — add SERVICE_STDIN opcode to gpu-protocol, implement host-side handler
  that reads from stdin, GPU-side wrapper. Straightforward copy of the write/read pattern.
- SystemTime/Instant — Instant can use PTX `%globaltimer` (64-bit nanosecond counter,
  available since SM 3.0). SystemTime needs a hostcall to get wall-clock time from host.
  Both can be implemented as standalone functions without std.
- futures_util on GPU — since Fat LTO resolves all cross-crate calls, any no_std-compatible
  futures crate should compile. futures_util with `no-std` feature + alloc is the candidate.
  The key test is whether the async combinators (join!, select!) produce valid PTX.
- Custom `println!` / `print!` macros — use `core::fmt::Write` + hostcall. No std needed.

**Requires std source patching (~5 lines in 3 files):**
- Full `-Zbuild-std=std` compilation — blocked by 5 missing `cfg_select!` cases in
  `sys/alloc`, `sys/thread_local`, `sys/random`. The fix is mechanical:
  - `sys/alloc/mod.rs`: add `target_os = "cuda"` case pointing to a new `gpu.rs` that
    delegates to our bump allocator
  - `sys/alloc/mod.rs`: add `nvptx64` to the 16-byte MIN_ALIGN tier
  - `sys/thread_local/mod.rs`: add `target_os = "cuda"` to the no_threads case
  - `sys/random/mod.rs`: add `target_os = "cuda"` to the unsupported case
- This gives us `std::io::stdin().read_line()`, `std::time::Instant`, `std::fs::File`
  with native Rust types (not our custom wrappers). VectorWare specifically demonstrated
  `stdin().read_line()` — this is a signature capability.

**Requires unknown rustc patches (likely infeasible without VectorWare's fork):**
- The `extern "gpu-kernel"` ABI (functionally identical to ptx-kernel on nvptx64, so
  not actually needed)
- Any deeper std integration that VectorWare's unpublished patches enable (unknown scope)
- Compiler-level support for GPU-aware thread_local (we use no-op, which is correct for
  single-thread-per-block model)

### 1.2 Impact Ranking of Remaining Gaps

| Gap | Impact for VectorWare parity | Effort | Priority |
|-----|------------------------------|--------|----------|
| Async hostcall (HostcallFuture) | **Critical** — VectorWare's key demo | Medium | P0 |
| futures_util on GPU | **High** — proves third-party async crates work | Low-Medium | P1 |
| Std source patching | **High** — enables `-Zbuild-std=std` | Low (5 lines) | P1 |
| stdin hostcall | **Medium** — completes the I/O story | Low | P2 |
| SystemTime/Instant | **Medium** — VectorWare showed this | Low | P2 |
| Benchmarks vs native CUDA | **Medium** — quantifies overhead | Medium | P2 |

### 1.3 Std Patching: Vendor vs Fork

**Option A: Vendor std source into the repo**
- Copy the affected files (~3 files from `library/std/src/sys/`) into a local `std-patch/`
  directory. Apply the ~5 line changes. Use `-Zbuild-std` with `CARGO_ENCODED_RUSTFLAGS`
  pointing to patched sysroot.
- Pros: Self-contained, reproducible, no external dependency.
- Cons: Must track upstream std changes (but only 3 files, changes are stable cfg_select blocks).

**Option B: Fork rust-lang/rust on GitHub**
- Create a fork with a `nvptx64-std` branch. Apply patches. Use `rustup toolchain link`.
- Pros: Clean separation, easy to rebase.
- Cons: Heavyweight for 5 lines, requires building rustc, user must install custom toolchain
  (violates HOST ENVIRONMENT POLICY — requires user action).

**Option C: Use `-Zbuild-std` with patched std source overlay**
- Cargo's `-Zbuild-std` rebuilds std from source. We can use `[patch.crates-io]` or
  `__CARGO_TESTS_ONLY_SRC_ROOT` to redirect std source to a local patched copy.
- Pros: Cleanest approach if it works.
- Cons: Undocumented, fragile, may break across nightly versions.

**Recommendation: Option A (vendor)** — minimal, self-contained, and the 3 files are
extremely stable (cfg_select dispatch tables that haven't changed structure in years).
The actual patch is ~5 lines, so maintenance burden is negligible.

### 1.4 Async Hostcall (HostcallFuture) Feasibility

**Architecture:**
```
struct HostcallFuture {
    packet_ptr: *mut u64,  // Pointer to allocated hostcall packet
    state: HostcallState,  // Submitted | WaitingResponse | Done
}

impl Future for HostcallFuture {
    type Output = HostcallResponse;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.state {
            Submitted => {
                // Fill packet, push to ready stack, ring doorbell
                self.state = WaitingResponse;
                cx.waker().wake_by_ref(); // re-enqueue for next poll round
                Poll::Pending
            }
            WaitingResponse => {
                let ctrl = sys_load_acquire_u64(self.packet_ptr);
                if ctrl & READY_BIT != 0 {
                    let response = extract_response(self.packet_ptr);
                    release_packet(self.packet_ptr);
                    self.state = Done;
                    Poll::Ready(response)
                } else {
                    cx.waker().wake_by_ref(); // poll again next round
                    Poll::Pending
                }
            }
            Done => panic!("polled after completion"),
        }
    }
}
```

**Feasibility assessment: HIGH.** All building blocks exist:
1. Embassy executor polls tasks in a loop ✓ (async-runtime.3)
2. Hostcall packet allocation + submission ✓ (hostcall.4, gpu-std.3)
3. System-scope atomic loads for checking response ✓ (gpu-atomics)
4. The spin-wait in current hostcall (`sys_spin_load_acquire_u64`) simply becomes
   a non-spinning check that returns Pending if not ready

**Key advantage over spin-wait:** While one task waits for a hostcall response, the
executor can poll other tasks. This enables true concurrent I/O — e.g., two async tasks
each doing file I/O simultaneously, with the executor interleaving their polls.

**Risk:** Per ADR-4, each lane acquires its own packet (per-lane model). With the
Embassy executor running per-thread, each thread's executor manages its own tasks
independently. There is no cross-thread coordination needed — this simplifies the design.

---

## Section 2: Skeptic Challenges

### 2.1 Untested Assumptions

**A1: futures_util compiles for nvptx64.**
While Fat LTO handles cross-crate calls, futures_util uses complex trait machinery
(FuturesUnordered, stream combinators) that may generate PTX-incompatible code patterns.
Specifically:
- `FuturesUnordered` uses `Arc<Task>` internally — needs heap allocation (we have bump alloc,
  but no deallocation). This will leak memory.
- `select!` macro generates match arms with complex control flow — may hit LLVM PTX backend
  limitations.
- Test: compile `futures::future::join(a, b)` first (simplest combinator) before attempting
  complex combinators.

**A2: Multiple HostcallFutures won't exhaust the packet pool.**
Current pool has 4 packets. If 3 async tasks each hold a packet while waiting for response,
only 1 remains free. A 4th task requesting a hostcall would get Pending (back-pressure per
ADR-4), but this means the executor spin-polls that task repeatedly without progress until
a packet is freed. This is correct but wasteful.
- Mitigation: size the pool to match max concurrent async tasks per thread.
- For MVP: 2-4 tasks per thread with 4 packets is fine.

**A3: Std source patching actually works end-to-end.**
The 5-line patch fixes compilation errors, but runtime behavior of std on GPU is untested.
Specifically:
- `std::io::Write` for `Stdout` goes through `sys::stdio` which uses the PAL `unsupported`
  module — it would return "unsupported" errors, NOT use our hostcall path.
- To make `println!` work through std (not our custom macro), we'd need ALSO to patch
  `sys/pal/unsupported/stdio.rs` to route through hostcall. This is more than 5 lines.
- **The honest assessment**: patching std for compilation is easy; making std I/O actually
  work requires routing through hostcall at the PAL level, which is a larger effort.

**A4: Register pressure stays under 64 with real async workloads.**
The 56-register measurement was for trivial CountdownFuture. A HostcallFuture with packet
pointer, state enum, response buffer, plus the executor overhead, could push past 64 regs.
This would trigger PTXAS spilling to local memory, reducing occupancy.
- Mitigation: measure immediately after implementing HostcallFuture.
- If over 64: consider `maxrregcount` pragma or accept reduced occupancy.

### 2.2 Is Std Patching Worth the Maintenance Burden?

**Arguments FOR:**
- VectorWare specifically showed `-Zbuild-std=std`. Matching this is important for parity.
- The fix is genuinely tiny (~5 lines in 3 cfg_select blocks).
- It enables the entire std API surface (even if most functions return "unsupported").
- Future rustc versions may add native nvptx64 support, eliminating the patch.

**Arguments AGAINST:**
- Compilation alone is not parity. VectorWare's `stdin().read_line()` works because they
  patched the PAL layer too (or have a different mechanism). Our patch only fixes cfg_select.
- Every nightly update could break the patch (though cfg_select blocks are stable).
- The real value (working I/O) comes from our hostcall + gpu-libc, not from std itself.
- A custom `gpu_println!` macro gives the same demo capability without std.

**Verdict:** Pursue std patching as a SEPARATE task from the main integration work. It's a
nice-to-have for demo completeness but NOT required for proving the core technology. The
custom macro path (gpu_println! + gpu-libc + hostcall) is the reliable path.

### 2.3 Can We Achieve VectorWare Parity Without Their Rustc Patches?

**Honest assessment: YES, for the demonstrated features.** VectorWare's two blog posts show:
1. `println!("Hello from GPU")` — we have this via hostcall ✓
2. `stdin().read_line()` — achievable with new hostcall opcode + either std patch or custom fn
3. File I/O — we have this ✓
4. Embassy async/await — we have this ✓
5. `futures_util` combinators — achievable (needs testing)
6. `-Zbuild-std=std` — achievable with ~5 line patch

What we CANNOT know: whether VectorWare's rustc patches enable deeper capabilities they
haven't published about. Their blog posts are the only public reference.

**The gap that matters most is not technical — it's polish.** VectorWare's demo shows
clean Rust syntax (`stdin().read_line(&mut buffer)`) vs our approach which requires custom
functions (`gpu_hostcall_read(...)`). Std patching closes this ergonomic gap.

### 2.4 Warp Divergence with Mixed Async Hostcall Patterns

**Scenario:** 32 threads in a warp, each running its own Embassy executor with different
async tasks. Thread 0 does file I/O (hostcall), thread 1 does computation, thread 2 does
println (hostcall).

**Risk:** The threads diverge at every branch point:
- Thread 0 enters `HostcallFuture::poll` → packet allocation → CAS loop
- Thread 1 enters `ComputeFuture::poll` → arithmetic
- Thread 2 enters `HostcallFuture::poll` → different CAS loop timing

This creates massive warp divergence. Each thread's execution path is serialized by the
SIMT scheduler. With 32 divergent paths, theoretical throughput drops to 1/32.

**Mitigation strategies:**
1. **Homogeneous tasks per warp**: All threads in a warp run the same async task type.
   The executor polls the same future type, keeping threads converged.
2. **Warp-cooperative hostcall**: Return to warp-granular packets (ADR-3 original design)
   where lane 0 does CAS and broadcasts to all lanes. This was deferred in ADR-4 but
   becomes important for performance.
3. **Accept divergence for MVP**: Single-thread kernels (1 block × 1 thread) avoid the
   issue entirely. Multi-thread scaling is a later optimization.

**Recommendation for integration.1:** Use single-thread kernels. Document warp divergence
as a known limitation and future optimization target.

---

## Section 3: Recommendations

### 3.1 Split integration.1 into Focused Tasks

The current integration.1 is too broad ("async GPU kernel + hostcall + std" with 4 research
questions spanning different concerns). Split into:

**integration.1** (NARROW — rename): "Async hostcall: HostcallFuture with Embassy executor"
- Kind: experiment
- Scope: Implement HostcallFuture that wraps hostcall in a Future. Run 2 concurrent async
  tasks doing hostcalls (e.g., task A prints, task B writes file) on the same executor.
  Measure register pressure. Verify correctness.
- Research questions:
  1. Does HostcallFuture correctly yield and resume across poll rounds?
  2. Can two async hostcall tasks run concurrently on one executor?
  3. What is the register pressure for async hostcall vs sync hostcall?
- Depends: gpu-std.3, async-runtime.3

**integration.2** (NEW): "Third-party async crate: futures_util on GPU"
- Kind: experiment
- Scope: Add futures-util (no-std + alloc) as dependency. Test `futures::future::join(a, b)`
  where a and b are HostcallFutures. If join works, try `select!`.
- Research questions:
  1. Does futures_util compile for nvptx64 with Fat LTO?
  2. Does `join(hostcall_a, hostcall_b)` produce correct concurrent execution?
  3. What is the register pressure impact of futures combinators?
- Depends: integration.1

**integration.3** (NEW): "Vendor and patch std source for -Zbuild-std=std"
- Kind: experiment
- Scope: Vendor the 3 affected std source files. Apply cfg_select patches. Verify compilation.
  Test whether std::io, std::fs types are usable (even if they return unsupported errors).
- Research questions:
  1. Does vendored std compile with patches?
  2. Can we route std::io::stdout() through hostcall at the PAL level?
  3. What additional patches are needed beyond cfg_select?
- Depends: gpu-std.2

**integration.4** (NEW): "End-to-end benchmark and VectorWare feature comparison"
- Kind: investigation
- Scope: Benchmark async hostcall latency vs sync. Compare feature checklist against
  VectorWare's published demos. Document what we match and what we don't.
- Research questions:
  1. Async hostcall latency vs sync hostcall latency?
  2. Feature-by-feature comparison with VectorWare blog posts?
  3. What would be needed to close remaining gaps?
- Depends: integration.1, integration.2 (optional)

### 3.2 New Standalone Tasks

**gpu-std.4** (NEW): "Implement stdin hostcall + SystemTime/Instant"
- Kind: experiment
- Theme: gpu-std
- Scope: Add SERVICE_STDIN opcode. Implement host-side stdin reader. GPU-side wrapper.
  Also implement Instant using PTX `%globaltimer` and SystemTime via hostcall.
- Depends: gpu-std.3

### 3.3 Priority Ordering

| Priority | Task | Rationale |
|----------|------|-----------|
| P0 | integration.1 (HostcallFuture) | Core VectorWare parity feature; combines the two key innovations |
| P1 | integration.2 (futures_util) | High impact demo; proves ecosystem compatibility |
| P1 | gpu-std.4 (stdin + time) | Completes I/O story; low effort |
| P2 | integration.3 (std patching) | Nice-to-have; higher risk due to PAL routing |
| P3 | integration.4 (benchmarks) | Final polish; do last |

### 3.4 Tasks to Skip/Defer

- **Warp-cooperative async hostcall**: Deferred. Per-lane model works for MVP. Warp
  optimization is a performance concern, not a correctness concern.
- **Multi-thread executor scaling**: Deferred. Single-thread-per-block is sufficient for
  all VectorWare demo features.
- **CUDA device malloc integration**: Deferred. Bump allocator is sufficient; device malloc
  adds complexity with no demo benefit.
- **rustc fork investigation**: Skip. VectorWare's patches are unpublished and likely
  unnecessary for the features they've publicly demonstrated.

### 3.5 Theme Status Assessment

| Theme | Current Status | Recommendation |
|-------|----------------|----------------|
| toolchain | completed | No change |
| hostcall | completed | No change |
| atomics | completed | No change |
| gpu-std | active | Keep active; gpu-std.4 (stdin + time) remaining |
| async-runtime | active | **COMPLETE** — all 3 success criteria met (poll ✓, concurrent ✓, register pressure ✓) |
| integration | active | Keep active; split integration.1 into 4 focused tasks |

**async-runtime should be marked COMPLETED.** The findings from async-runtime.3 confirm
all three success criteria are satisfied. Remaining async work (HostcallFuture, futures_util)
is integration work, not async-runtime work.

---

## Summary of Proposed Changes

### State Changes
- Mark `async-runtime` theme as **completed**
- Split `integration.1` into 4 tasks (integration.1/2/3/4)
- Add `gpu-std.4` task

### New Tasks (6 total)
1. `integration.1` — HostcallFuture + Embassy (narrowed scope)
2. `integration.2` — futures_util on GPU
3. `integration.3` — Vendor and patch std source
4. `integration.4` — Benchmarks + VectorWare comparison
5. `gpu-std.4` — stdin hostcall + SystemTime/Instant

### Execution Plan (next batch)
1. **integration.1** (HostcallFuture) — highest priority, do first
2. **gpu-std.4** (stdin + time) — independent, can parallel with integration.1
3. **integration.2** (futures_util) — depends on integration.1
4. **integration.3** (std patch) — independent but lower priority
5. **integration.4** (benchmarks) — do last

## Key Insight

The project is at an inflection point: all foundational components work individually.
The remaining work is purely integration and polish. The most impactful single task is
HostcallFuture — it transforms the hostcall spin-wait into an async operation, which is
the exact capability VectorWare demonstrated as their headline feature. Everything else
(futures_util, std patching, stdin) builds on top of that foundation. The skeptic's most
important warning is that std patching for compilation is NOT the same as making std I/O
work — the PAL layer routing is the real challenge, and we should not underestimate it.
