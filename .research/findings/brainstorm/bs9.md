# BS9 — Product Ready: PAL Routing, Dynamic Alloc, Multi-Warp Scaling & Showcase
**Date**: 2026-03-12
**Brainstorm seq**: 9
**Trigger**: user-directed (Product Ready epic)
**Level**: standard (structured analysis)

## Context

All 6 research themes completed (29 tasks). The project has proven every individual
component: GPU compilation, lock-free hostcall, libc shim, Embassy executor, async
HostcallFuture, futures_util, `-Zbuild-std=std`, stdin, file I/O, timing. The user
has defined 6 directions for a "Product Ready" push that transitions from proving
components work to making them work together cleanly and at scale.

The 6 directions span three tiers:
- **Tier 1 — std integration polish**: PAL stdout routing (#1), stdin via std (#4)
- **Tier 2 — robustness testing**: Dynamic allocation (#2), multi-warp scaling (#5)
- **Tier 3 — demonstration**: Multi-step async workflow (#3), showcase demo (#6)

---

## Section 1: Technical Analysis

### 1.1 Feasibility Assessment

#### Direction 1: PAL stdout routing

**Goal**: Route `println!` through `std::io::stdout()` at the PAL level so that
`println!("{}", x)` works natively without `gpu_hostcall_print()`.

**Feasibility: MEDIUM-HIGH.** The path is architecturally clear but requires more
std patching than the compilation fixes (integration.3). Specifically:

1. Create `sys/pal/cuda/mod.rs` (or modify `sys/pal/unsupported/stdio.rs`) to
   implement `Stdout::write()` via hostcall.
2. The main challenge: `Stdout::write()` needs access to the hostcall buffer pointer.
   In the current design, `buf: *mut u8` is a kernel argument passed at launch. The
   PAL layer has no access to kernel arguments.
3. **Solution**: A global static `HOSTCALL_BUF: AtomicPtr<u8>` set by an init function
   at kernel entry. The PAL's `Stdout::write()` reads this global. This is the same
   pattern used by gpu-libc — a well-proven approach.
4. The `Write` trait impl needs to split the message into 56-byte chunks (the hostcall
   payload limit) for messages longer than one packet.

**Build-on**: std-patches/, gpu-kernel's `gpu_hostcall_print()`, gpu-protocol.

**Effort**: ~50-80 lines of PAL code + ~10 lines of global init.

#### Direction 2: Dynamic allocation testing

**Goal**: Verify the bump allocator works with truly dynamic data that LLVM cannot
constant-fold.

**Feasibility: HIGH.** This is a testing task, not a design task. The bump allocator
(cuda.rs) already exists and is correct (CAS-based, thread-safe). The gap is that
all current tests (`vec![1,2,3,4,5]`, `format!("value = {}", 42)`) use compile-time
constants that LLVM optimizes away.

**Approach**:
1. Pass dynamic values as kernel arguments (e.g., `fn kernel(input: *const u32, len: u32)`)
2. Create a `Vec` from runtime data: read from kernel args, push to Vec, iterate
3. Build a `String` from runtime data: `format!("result = {}", *input)`
4. The allocator's `GPU_HEAP_POS` atomic counter should advance (unlike current tests
   where it stays at 0)
5. Verify with PTX inspection: the allocator's CAS loop should appear in active code,
   not dead code

**Build-on**: crates/std-build-test, std-patches/cuda.rs.

**Effort**: ~30-50 lines of kernel code + host test harness.

#### Direction 3: Multi-step async workflow

**Goal**: Demonstrate read file → process → write result → print summary, all async.

**Feasibility: HIGH.** All individual operations work:
- Async file read: HostcallFuture + SERVICE_READ (proven)
- Async file write: HostcallFuture + SERVICE_WRITE (proven)
- Async print: HostcallFuture + SERVICE_PRINT (proven)
- `futures::future::join` for concurrent ops (proven)

The new element is chaining these sequentially with `.await` and processing data
between steps. This is standard Rust async — the GPU executor handles it identically
to CPU async.

**Design sketch**:
```
async fn pipeline(buf: *mut u8) {
    // Step 1: Read input file
    let data = async_read_file(buf, "input.txt").await;
    // Step 2: Process (e.g., sum bytes, transform)
    let result = process(&data);
    // Step 3: Write result file
    async_write_file(buf, "output.txt", &result).await;
    // Step 4: Print summary
    async_print(buf, "Pipeline complete!").await;
}
```

**Risk**: Each `.await` point holds the HostcallFuture state. With 4 sequential
awaits, the future state machine has 5 states. Register pressure could increase
significantly. The two-task async test used ~82 virtual regs; a 4-step pipeline
could push past 100.

**Build-on**: crates/async-hostcall-test, gpu-kernel's hostcall helpers.

**Effort**: ~100-150 lines.

#### Direction 4: stdin via std

**Goal**: Route `std::io::stdin().read_line()` through the PAL level.

**Feasibility: MEDIUM-HIGH.** Architecturally identical to stdout routing (#1).
The SERVICE_STDIN opcode already exists (gpu-std.4). The PAL needs:

1. `Stdin::read()` implementation that calls the hostcall stdin service
2. Same global `HOSTCALL_BUF` pointer as stdout routing
3. `BufReader` wrapping for `read_line()` — std provides this automatically if
   `Stdin::read()` works

**Dependency**: Should be implemented alongside or after stdout routing (#1), since
they share the same PAL infrastructure.

**Build-on**: gpu-kernel's `gpu_hostcall_stdin_read()`, std-patches/.

**Effort**: ~30-40 lines (after stdout PAL infrastructure exists).

#### Direction 5: Multi-warp async scaling

**Goal**: 32 GPU threads (1 block × 32 threads), each running its own Embassy
executor with async hostcall tasks.

**Feasibility: MEDIUM.** This is the highest-risk direction. Several unknowns:

1. **Packet pool sizing**: 32 threads each needing 1+ packets = 32+ packets minimum.
   Current default is 4 packets. Need to scale to 32-64 packets. The hostcall buffer
   structure supports up to 65534 packets (u16 index), so no protocol change needed.
2. **Host listener throughput**: The host polls the ready stack in a loop. With 32
   concurrent GPU threads pushing packets, the host must process 32 requests per
   kernel cycle. Current host uses a single-threaded listener — may need to batch
   process or add worker threads.
3. **Warp divergence**: All 32 threads in a warp running *different* async tasks
   will diverge at every branch. The SIMT scheduler serializes divergent paths,
   reducing throughput to ~1/32 in the worst case. However, if all threads run the
   *same* async task type (e.g., all do async_print), they may reconverge.
4. **Embassy per-thread**: Each thread needs its own `Executor` + `TaskStorage`.
   With 32 threads, that's 32× the static storage. On GPU, statics are in global
   memory — this is fine for space but increases global memory traffic.
5. **Critical section**: The no-op critical section (gpu-critical-section) is safe
   for single-thread. With 32 threads on the same block, inter-thread races on
   Embassy's internal state are possible IF any state is shared. Since each thread
   has its own executor, this should be safe — but needs verification.

**Build-on**: crates/async-hostcall-test, crates/gpu-critical-section.

**Effort**: ~150-200 lines + significant debugging.

#### Direction 6: Showcase demo kernel

**Goal**: VectorWare-style clean demo combining all capabilities.

**Feasibility: HIGH** (once directions 1-3 are complete). This is a composition
task — no new technical challenges, just combining proven pieces.

**Ideal demo script**:
```rust
#[gpu_kernel]
async fn showcase(buf: *mut u8) {
    println!("GPU Showcase: Rust std + async/await on GPU");

    // std types
    let mut data = Vec::new();
    data.push(42u32);
    let msg = format!("Vec contents: {:?}", data);
    println!("{}", msg);

    // Async file I/O
    let content = async_read_file(buf, "input.txt").await;
    println!("Read {} bytes from file", content.len());

    // Concurrent async operations
    let (a, b) = futures::future::join(
        async_write_file(buf, "out.txt", &content),
        async_print(buf, "Writing file concurrently..."),
    ).await;

    println!("Showcase complete!");
}
```

**Dependency**: Directions 1 (PAL stdout) and 2 (dynamic alloc) should land first.
Without PAL routing, the demo uses `gpu_hostcall_print()` instead of `println!` —
still impressive but less clean.

**Build-on**: All existing crates.

**Effort**: ~100 lines of kernel code + host harness.

### 1.2 Dependencies Between Directions

```
Direction 1 (PAL stdout) ─────┐
                               ├──→ Direction 6 (showcase demo)
Direction 4 (stdin via std) ───┘         ↑
        ↑                                │
        └── depends on Direction 1       │
            (shared PAL infra)           │
                                         │
Direction 2 (dynamic alloc) ─────────────┘
                                         │
Direction 3 (multi-step async) ──────────┘

Direction 5 (multi-warp) ── independent (can parallel with everything)
```

Key ordering constraints:
1. Direction 1 before Direction 4 (shared PAL infrastructure)
2. Directions 1, 2, 3 before Direction 6 (showcase needs all features working)
3. Direction 5 is independent — can run in parallel with any other direction

### 1.3 Risk Assessment

| Direction | Risk Level | Primary Risk |
|-----------|-----------|--------------|
| 1. PAL stdout | Medium | Global hostcall pointer initialization ordering |
| 2. Dynamic alloc | Low | Bump allocator OOM on larger workloads |
| 3. Multi-step async | Medium | Register pressure with 4+ await points |
| 4. stdin via std | Low | Blocking behavior in host listener |
| 5. Multi-warp scaling | **High** | Warp divergence + packet pool exhaustion |
| 6. Showcase demo | Low | Composition only — no new risks |

### 1.4 Existing Infrastructure to Build On

| Direction | Primary Crates | Key Files |
|-----------|---------------|-----------|
| 1. PAL stdout | std-patches/, gpu-protocol | std-patches/cuda.rs, gpu-kernel/src/lib.rs |
| 2. Dynamic alloc | std-build-test | std-patches/cuda.rs |
| 3. Multi-step async | async-hostcall-test | async-hostcall-test/src/lib.rs |
| 4. stdin via std | std-patches/, gpu-protocol | gpu-kernel's gpu_hostcall_stdin_read |
| 5. Multi-warp | async-hostcall-test, gpu-host | gpu-host's listener loop |
| 6. Showcase | all crates | new crate or extend std-build-test |

---

## Section 2: Skeptic Challenges

### 2.1 Untested Assumptions

**A1: PAL routing is worth the effort vs. keeping gpu_hostcall_print.**

Arguments AGAINST PAL routing:
- `gpu_hostcall_print()` already works. The demo impact of `println!("x = {}", x)`
  vs `gpu_hostcall_print(buf, msg, len)` is ergonomic, not functional.
- PAL routing adds ~100 lines of code INSIDE the vendored std source, increasing
  maintenance burden on every nightly update.
- The global `HOSTCALL_BUF` pointer pattern is fragile — if the kernel forgets to
  call `init_hostcall(buf)` at entry, the PAL silently crashes or returns errors.
- VectorWare likely has deeper rustc integration (their own PAL target) that we
  cannot replicate without forking rustc.

Arguments FOR PAL routing:
- VectorWare specifically demonstrated `println!` and `stdin().read_line()` working
  through std. This is the signature "it just works" moment.
- The global pointer pattern is well-proven (gpu-libc already does this).
- Once the PAL infrastructure exists, ALL std I/O operations route through hostcall
  automatically — no need for custom wrappers for each new service.
- The maintenance burden is real but small — the PAL files are structurally stable.

**Verdict**: Worth doing. The ergonomic value IS the demo value. "It just works"
is the whole point of the project. But implement it AFTER dynamic alloc testing (#2)
to ensure the allocator foundation is solid.

**A2: The bump allocator will survive real dynamic allocation.**

The bump allocator has three fundamental limitations:
1. **No deallocation.** Every `Vec::push`, `String::push_str`, `format!()` that
   triggers a reallocation leaks the old buffer. A Vec that grows from capacity
   4 → 8 → 16 → 32 allocates 60 elements worth of memory but only uses 32.
2. **1 MB heap limit.** With leaked memory from reallocations, the effective usable
   heap is much less than 1 MB. A Vec growing to 10,000 u32s (40 KB) could consume
   ~80 KB of heap due to geometric growth + leaked old buffers.
3. **No heap reset between kernel invocations.** If the bump allocator uses a static
   global, the heap position persists across kernel launches. Multiple kernel launches
   will exhaust the heap. (Mitigation: reset `GPU_HEAP_POS` to 0 from host between
   launches, or use the atomic to detect OOM and handle gracefully.)

**Real risk**: The current LLVM constant-folding hides these issues entirely. The
moment we use runtime data, allocator OOM is a real possibility for non-trivial
workloads. Dynamic alloc testing (#2) MUST exercise:
- Vec growing beyond initial capacity (multiple reallocations)
- Multiple Vecs alive simultaneously
- format!() with runtime values (allocates String internally)
- OOM behavior: what happens when the bump allocator returns null?

**Mitigation**: For the showcase demo, pre-size collections (`Vec::with_capacity`)
and keep total allocations under ~100 KB. Document the bump allocator limitation
as a known constraint.

**A3: 32-thread async scaling will work without warp divergence catastrophe.**

The Embassy executor's poll loop is:
```
loop {
    for task in tasks {
        if !task.done { task.poll(); }
    }
}
```

With 32 threads, each running this loop with different tasks at different completion
stages, every branch diverges. The worst case:
- Thread 0: task A at state Init (allocating packet)
- Thread 1: task A at state WaitingResponse (checking control word)
- Thread 2: task B at state Init
- ...

Each thread takes a different branch. SIMT serializes them. Throughput drops to
~1/32 of peak.

**However**: if all 32 threads run the same kernel code and start simultaneously,
they may stay roughly synchronized through the first few poll rounds. The divergence
grows over time as host responses arrive at different times.

**Practical mitigation**: Accept that per-thread executors on a single warp will
diverge. The demo value is "it works correctly at 32 threads" not "it's efficient
at 32 threads." Performance optimization (warp-cooperative hostcall) is a separate
future effort.

**Quantifiable risk**: If 32 threads each submit 1 hostcall, the host must process
32 packets. With 100-200μs per hostcall round-trip, 32 sequential packets take
3.2-6.4ms. If threads block waiting for responses, the kernel runtime is dominated
by host processing time, not GPU compute or divergence.

**A4: Multi-step async pipeline won't blow the register budget.**

The async hostcall two-task test used ~82 virtual PTX registers. A 4-step sequential
pipeline (`read → process → write → print`) creates a state machine with 5 states.
Each state potentially holds different local variables:
- State 0 (pre-read): kernel args
- State 1 (reading): file descriptor, buffer pointer
- State 2 (processing): data buffer, result buffer
- State 3 (writing): output fd, result buffer
- State 4 (printing): message buffer

The compiler may keep ALL state across all await points, since the state machine
struct contains a union of all per-state fields. If LLVM doesn't optimize this well,
register pressure could exceed 128 virtual regs, forcing spills to local memory.

**Mitigation**: Use `drop()` explicitly to release resources before the next await.
Keep per-state data small (pointers, not buffers). If register pressure is too high,
split the pipeline into multiple sequential kernels.

### 2.2 Which Directions Are Actually Harder Than They Seem?

**Direction 1 (PAL stdout) is harder than it seems.** The analysis in integration.3
called it "the path is clear" but glossed over:
- `std::io::Stdout` holds a `ReentrantLock<BufWriter<StdoutRaw>>`. On the unsupported
  PAL, this is a no-op lock, but the `BufWriter` adds buffering that may interfere
  with our hostcall model (which expects complete messages per packet).
- `write_all()` may call `write()` multiple times if the buffer is larger than the
  internal buffer. Each `write()` call would trigger a separate hostcall.
- The `\n` flushing behavior of `println!` depends on line-buffered stdout, which
  the PAL needs to implement correctly.
- Error handling: if a hostcall fails (pool exhaustion), `Stdout::write()` must
  return `io::Error`. Translating hostcall failure to `io::Error` requires the
  `unsupported` PAL's error infrastructure.

**Direction 5 (multi-warp scaling) is MUCH harder than it seems.** Beyond warp
divergence:
- The host listener is single-threaded. Processing 32 concurrent requests serially
  creates a bottleneck. The host may need a thread pool for handlers.
- Packet pool fairness: with 32 threads competing for packets via CAS, some threads
  may starve while others succeed repeatedly (CAS ABA is prevented by tags, but
  fairness is not guaranteed).
- Embassy's `TaskStorage` is a static. With 32 threads, we need either 32 separate
  statics (code explosion) or a way to dynamically allocate TaskStorage (requires
  the bump allocator + Embassy modifications).
- `gpu-critical-section` uses a no-op. If Embassy's internal state has any
  shared-mutable access pattern that assumes the critical section is real, 32
  threads will corrupt state. This needs careful audit.

### 2.3 What Could Be Cut Without Losing Demo Value?

If time is constrained, the minimum viable Product Ready is:
1. **Direction 2 (dynamic alloc)** — proves the allocator actually works (essential)
2. **Direction 3 (multi-step async)** — proves async pipelines work (high demo value)
3. **Direction 6 (showcase demo)** — the deliverable (essential)

Directions that can be deferred:
- Direction 1 (PAL stdout) — nice ergonomics but `gpu_hostcall_print` suffices
- Direction 4 (stdin via std) — nice but stdin via direct hostcall already works
- Direction 5 (multi-warp scaling) — impressive but high risk, single-thread demos
  are sufficient

---

## Section 3: Concrete Recommendations

### 3.1 Theme Organization

Propose **2 new themes** to organize the Product Ready work:

**Theme: `std-pal`** — "Route std I/O through hostcall at the PAL level"
- Goal: Make std::io::stdout, std::io::stdin, and potentially std::fs work
  natively through the vendored std's PAL layer, so that `println!` and
  `stdin().read_line()` use hostcall transparently.
- Success criteria:
  1. `println!("{}", runtime_value)` works from GPU kernel via std
  2. `std::io::stdin().read_line(&mut buf)` works from GPU kernel via std
  3. No custom macros or wrapper functions needed for basic I/O

**Theme: `product`** — "Product-ready demonstrations and stress testing"
- Goal: Validate the complete stack under realistic conditions and produce a
  polished showcase demo.
- Success criteria:
  1. Bump allocator verified with truly dynamic (non-constant-folded) data
  2. Multi-step async workflow demonstrated end-to-end
  3. Multi-warp (32-thread) async execution tested
  4. Showcase demo combining std types + async + I/O + concurrent operations

### 3.2 Task Breakdown

#### Theme: std-pal

**`std-pal.1`** — "Implement CUDA PAL for stdout with hostcall routing"
- Kind: experiment
- Depends on: [] (integration.3 already done)
- Research questions:
  1. Can we add a `cuda` PAL variant (or modify `unsupported`) to route `Stdout::write()` through hostcall?
  2. How to pass the hostcall buffer pointer to the PAL layer? (global static vs. TLS)
  3. Does `BufWriter` wrapping in std's Stdout interfere with per-packet message semantics?
  4. Does `println!("{}", runtime_value)` produce a correct hostcall with the formatted output?
- Notes: Creates new file `sys/pal/cuda/stdio.rs` in vendored std. Requires a kernel init function to set the global hostcall buffer pointer.

**`std-pal.2`** — "Route stdin through CUDA PAL"
- Kind: experiment
- Depends on: [std-pal.1]
- Research questions:
  1. Does `std::io::stdin().read_line(&mut buf)` correctly invoke SERVICE_STDIN hostcall?
  2. Can `BufReader` wrapping handle the 56-byte packet limit for stdin data?
  3. Does blocking stdin behavior work correctly (GPU thread waits for host input)?

#### Theme: product

**`product.1`** — "Dynamic allocation stress test with runtime data"
- Kind: experiment
- Depends on: []
- Research questions:
  1. Does Vec::push with runtime (kernel argument) data trigger the bump allocator?
  2. Can Vec grow beyond its initial capacity (multiple reallocations)?
  3. How much heap does a Vec growing to N elements consume (including leaked realloc buffers)?
  4. What happens when the bump allocator hits OOM (does std panic or return null)?
  5. Does format!() with runtime values correctly use the allocator?
- Notes: Pass data as kernel arguments to prevent LLVM constant-folding. Inspect PTX to verify allocator code is NOT dead code.

**`product.2`** — "Multi-step async I/O pipeline"
- Kind: experiment
- Depends on: [product.1]
- Research questions:
  1. Can a single async fn chain 4+ sequential hostcall awaits (read → process → write → print)?
  2. What is the register pressure of a 4-state async state machine?
  3. Does the executor correctly advance through all states?
  4. What is the end-to-end latency for the full pipeline?
- Notes: Uses HostcallFuture pattern from async-hostcall-test. Processing step should use Vec/String (validates dynamic alloc under async).

**`product.3`** — "Multi-warp async scaling (32 threads)"
- Kind: experiment
- Depends on: [product.1]
- Research questions:
  1. Can 32 threads each run their own Embassy executor with async hostcall tasks?
  2. How many hostcall packets are needed for 32 concurrent threads? (pool sizing)
  3. Does the host listener handle 32 concurrent requests without dropping any?
  4. What is the observable warp divergence? (measure with %globaltimer per-thread)
  5. Does the no-op critical section remain safe with 32 independent executors?
- Notes: Start with all 32 threads doing the SAME task (async print) to minimize divergence. Then try heterogeneous tasks.

**`product.4`** — "Showcase demo kernel"
- Kind: experiment
- Depends on: [std-pal.1, product.1, product.2]
- Research questions:
  1. Can a single kernel combine Vec, String, format!, async file I/O, concurrent operations, and println! via std?
  2. Is the total register pressure acceptable for a single kernel with all features?
  3. Does the demo run reliably on repeated invocations (heap reset between launches)?
- Notes: This is the deliverable. Should be a clean, well-commented crate that serves as a reference implementation. If PAL routing (std-pal.1) is not ready, fall back to gpu_hostcall_print.

### 3.3 Priority Ordering

| Priority | Task | Rationale |
|----------|------|-----------|
| P0 | product.1 (dynamic alloc) | Foundation — must verify allocator before anything else |
| P0 | std-pal.1 (PAL stdout) | Highest demo value — `println!` via std is the signature feature |
| P1 | product.2 (async pipeline) | Core demo scenario — chains all capabilities |
| P1 | std-pal.2 (PAL stdin) | Completes the I/O story — trivial after std-pal.1 |
| P2 | product.3 (multi-warp) | High risk, high reward — but single-thread demos suffice |
| P3 | product.4 (showcase) | Final deliverable — depends on everything else |

### 3.4 Suggested Batch Groupings

**Batch A (foundation + PAL infrastructure)**:
- product.1 (dynamic alloc testing) — independent
- std-pal.1 (PAL stdout routing) — independent
- These two are fully independent and can execute in parallel within a single session.

**Batch B (pipeline + stdin)**:
- product.2 (multi-step async pipeline) — depends on product.1
- std-pal.2 (PAL stdin routing) — depends on std-pal.1
- Both depend on Batch A outputs. Can execute in parallel.

**Batch C (scaling)**:
- product.3 (multi-warp 32-thread) — depends on product.1
- Can execute independently after Batch A.

**Batch D (showcase)**:
- product.4 (showcase demo) — depends on Batch A + B outputs
- Final session. Composes all proven components.

Optimal ordering: A → B+C (parallel) → D. Total: 3-4 sessions.

### 3.5 Review Triage for New Tasks

| Task | Review Level | Rationale |
|------|-------------|-----------|
| product.1 | Skip | Extends proven pattern (std-build-test with new inputs) |
| std-pal.1 | Full | New PAL layer in vendored std — architectural change |
| product.2 | Light | Extends proven async pattern to longer pipeline |
| std-pal.2 | Skip | Identical pattern to std-pal.1 for stdin |
| product.3 | Full | Multi-thread scaling — new failure modes possible |
| product.4 | Light | Composition — verify correctness, no new architecture |

---

## Section 4: Risks and Mitigations Summary

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Bump allocator OOM with real data | High | Medium | Pre-size collections, keep total < 100KB, document limitation |
| PAL BufWriter interferes with hostcall messages | Medium | Medium | Flush after every write, or bypass BufWriter |
| Multi-step async blows register budget (>128 regs) | Medium | Medium | Explicit drop(), split into smaller async fns |
| 32-thread warp divergence kills throughput | Medium | High | Accept for correctness demo; optimize later |
| Host listener bottleneck at 32 concurrent packets | Medium | Medium | Batch process or add thread pool on host |
| Global HOSTCALL_BUF init ordering race | Low | Low | Init before any I/O; single-thread kernel entry guarantees ordering |
| Nightly rustc update breaks vendored std patches | Low | Low | Pin nightly version; patches touch stable cfg_select blocks |

---

## Key Insight

The Product Ready phase has two distinct value streams:

1. **Ergonomic parity with VectorWare** (PAL routing): Makes the code look like normal
   Rust. `println!` instead of `gpu_hostcall_print()`, `stdin().read_line()` instead of
   `gpu_hostcall_stdin_read()`. This is the "wow factor" for demos.

2. **Robustness validation** (dynamic alloc, multi-warp): Proves the stack works beyond
   trivial constant-folded examples. This is the "is it real?" answer.

Both are necessary for a credible demo. The showcase kernel (product.4) is meaningless
if the allocator only works because LLVM optimizes it away, and less impressive if it
uses custom wrapper functions instead of std APIs.

**Critical path**: product.1 (dynamic alloc) is the riskiest P0 task. If the bump
allocator fails under real workloads, the showcase demo must be redesigned around
stack-allocated data only. This should be the first task executed.

**Proposed state.toml changes**:
- Add theme `std-pal` (status: active)
- Add theme `product` (status: active)
- Add tasks: std-pal.1, std-pal.2, product.1, product.2, product.3, product.4
- Increment brainstorm_seq to 9
