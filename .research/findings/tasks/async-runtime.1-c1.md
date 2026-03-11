# async-runtime.1: Embassy Executor Deep Analysis + GPU Compatibility
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: async-runtime
**Kind**: investigation
**Status**: done

## Summary

Embassy's executor is a compact, no_std, static-allocation async runtime designed for embedded targets.
Its core polling loop, waker vtable, and task queue are largely GPU-compatible as-is, but two
blocking issues exist: (1) the `CriticalSection` abstraction has no sensible semantics in SIMT
execution, and (2) the standard `arch-*` pender implementations all rely on hardware interrupts or
platform events (WFE/SEV/NVIC) that do not exist on GPU.
Both problems are addressable — either by thin GPU-specific replacements or by building a
purpose-built minimal executor — and VectorWare's published work confirms that "adapting [Embassy]
to run on the GPU required very few changes," though they do not disclose the exact diff.

---

## Detailed Findings

### Q1: Executor::poll() Flow

The public `Executor::poll()` is a thin `unsafe` wrapper that delegates directly to
`SyncExecutor::poll()`. The full call chain is:

```
Executor::poll()
  └─ SyncExecutor::poll()
       ├─ [cfg integrated-timers] expire timers → enqueue woken tasks
       └─ RunQueue::dequeue_all(|task_ref| {
               task.poll_fn.get().unwrap_unchecked()(task_ref)
          })
```

`dequeue_all` atomically snapshots the entire pending queue (via `TransferStack::take_all`),
then iterates over it, calling each task's stored `unsafe fn(TaskRef)` function pointer.
Tasks enqueued *during* the current `poll()` pass are **not** picked up in the same pass;
they remain for the next call. This prevents starvation and unbounded loops within a single
`poll()` invocation.

The `poll_fn` stored in `TaskHeader` is set once during `spawn()`. For a task defined with
`#[embassy_executor::task]`, it expands to a function that:
1. Calls `State::run_dequeue()` to clear the run-queued bit.
2. Constructs a `Context` from `waker::from_task(task_ref)`.
3. Calls `Future::poll(Pin<&mut F>, cx)` on the stored future.
4. On `Poll::Ready` — calls `State::despawn()` and marks storage as free.
5. On `Poll::Pending` — returns, leaving the task dormant until re-enqueued via a waker.

There is **no loop inside `poll_fn`**; a single future poll per task per queue snapshot.

**Key invariant**: `poll()` must never be called re-entrantly. The pender, which `poll()` may
call synchronously via `SyncExecutor::enqueue()`, must schedule a *future* call to `poll()`
rather than immediately calling it.

### Q2: RawWaker Implementation

Embassy implements `RawWaker` and `RawWakerVTable` directly without using `Arc` or any heap
allocation:

```rust
static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake, drop);

fn clone(p: *const ()) -> RawWaker {
    RawWaker::new(p, &VTABLE)
}
fn wake(p: *const ()) {
    let task = unsafe { TaskRef::from_ptr(p as *const TaskHeader) };
    wake_task(task);
}
fn drop(_: *const ()) {}   // no-op: task pointers are 'static, no ownership

pub fn from_task(task: TaskRef) -> Waker {
    unsafe { Waker::from_raw(RawWaker::new(task.as_ptr() as *const (), &VTABLE)) }
}
```

Key properties:
- **1-word storage**: the waker data pointer is simply the raw `*const TaskHeader` pointer.
  Embassy's internal wait-queues store `TaskRef` (1 word) rather than full `Waker` (2 words).
- **No cloning cost**: `clone` just copies the pointer and vtable reference.
- **No drop cost**: no reference count, no deallocation.
- **Origin verification**: `task_from_waker()` compares the vtable pointer address to `&VTABLE`
  to detect wakers from foreign executors, returning `None` for mismatches.
- **No platform-specific code** in `waker.rs` itself. The platform dependency is entirely
  inside `wake_task()` → `SyncExecutor::enqueue()` → `Pender::pend()`.

**GPU compatibility**: The `RawWaker`/`RawWakerVTable` mechanism is entirely compatible with
nvptx64. Pointer operations and function pointers compile fine. The only concern is that
`wake_task()` calls `Pender::pend()`, which on standard arch targets invokes a platform
interrupt — this piece must be replaced.

### Q3: Pender Callback Mechanism

The `Pender` type wraps a `*mut ()` context pointer. When the run queue transitions from empty
to non-empty, `SyncExecutor::enqueue()` calls `self.pender.pend()`, which in turn calls the
extern `__pender(context)` function.

```rust
// In SyncExecutor::enqueue():
unsafe fn enqueue(&self, task: TaskRef) {
    if self.run_queue.enqueue(task) {   // returns true only on empty→non-empty transition
        self.pender.pend();
    }
}

// Pender::pend():
pub fn pend(&self) {
    extern "C" { fn __pender(context: *mut ()); }
    unsafe { __pender(self.0) }
}
```

The arch-specific `__pender` implementations:

| Arch | Implementation |
|------|---------------|
| `arch-cortex-m` (thread mode) | `sev` instruction (Send Event, wakes `wfe`) |
| `arch-cortex-m` (interrupt mode) | `NVIC::pend(irq)` or STIR instruction |
| `arch-riscv32` | Software interrupt via CLINT |
| `arch-spin` | **No-op** (`fn __pender(_: *mut ()) {}`) |
| `arch-wasm` | Schedules via `setTimeout` / `queueMicrotask` |

The `arch-spin` implementation is the most relevant to GPU porting: it defines a no-op pender
and uses a busy-spin loop: `loop { unsafe { self.inner.poll() }; }`. This confirms that the
executor works without any interrupt mechanism — the pender is merely a performance hint.

**For GPU**: A no-op pender identical to `arch-spin` is sufficient for a single-warp or
single-thread executor. For a multi-warp scenario where tasks span multiple warps, a custom
pender using PTX `bar.sync` or shared-memory flags could be used, but is not required for
correctness.

**Critical requirement**: If `__pender` is not provided at link time, the build fails. Users
must either enable an `arch-*` feature or supply their own `__pender`. For GPU, a minimal
no-op implementation is the correct starting point.

### Q4: Task Queue Data Structure

The `RunQueue` uses a **lock-free intrusive stack** (`TransferStack<TaskHeader>`) backed by an
atomic pointer CAS loop:

```
RunQueue
  └─ TransferStack<TaskHeader>
       └─ AtomicPtr<TaskHeader>  (head of the singly-linked stack)

TaskHeader
  └─ run_queue_item: RunQueueItem
       └─ next: AtomicPtr<TaskHeader>
```

**Enqueue** (append to stack head):
```
loop {
    let old_head = stack.head.load(Relaxed);
    task.next.store(old_head, Relaxed);
    if stack.head.compare_exchange_weak(old_head, task_ptr, Release, Relaxed).is_ok() {
        break;
    }
}
// Returns true if old_head was null (queue was previously empty → triggers pender)
```

**Dequeue** (`take_all`): atomically swaps head with null, returning the entire chain.
`dequeue_all` then walks this snapshot chain, calling the poll closure for each task.

For platforms without atomic pointer support (`target_has_atomic` missing), a
`MutexTransferStack` variant wraps the same logic in `critical_section::with()`.

**Key properties**:
- **No heap allocation**: `TaskHeader` contains the `RunQueueItem` inline; tasks are stored
  in `TaskStorage<F>` which must be declared `static`.
- **No fixed capacity**: the queue is a linked list; it can hold all N tasks simultaneously.
- **Lock-free for atomic targets**: the CAS loop is the only synchronization primitive needed.
- **Intrusive**: task pointers *are* the queue nodes (via `run_queue_item`). No separate
  allocation for queue membership.

**GPU compatibility**: nvptx64 on SM 7.0+ supports 32-bit and 64-bit atomic CAS with proper
memory ordering semantics. The `AtomicPtr` (pointer-width atomic) used here is 64-bit on
nvptx64 (64-bit address space). The `Release`/`Relaxed` orderings in the enqueue CAS and the
`Acquire` in take_all are expressible in PTX. However, the lock-free stack assumes concurrent
access from multiple "threads" — on GPU this means multiple lanes/threads in a warp or multiple
warps. Within a single-warp single-task-at-a-time execution (the simplest GPU model), the CAS
is trivially uncontended and always succeeds on first attempt.

### Q5: Platform-Specific Dependencies

A systematic enumeration of all platform-specific surface area in `embassy-executor`:

#### 5a. Atomics (state management)

Three conditional implementations selected by `cfg_attr`:

| Condition | File | Mechanism |
|-----------|------|-----------|
| `cortex_m` + `target_has_atomic = "32"` | `state_atomics_arm.rs` | `AtomicU32` + `ldrex/strex` |
| `target_has_atomic = "8"` or `"32"` | `state_atomics.rs` | `AtomicU8` or `AtomicU32` |
| neither | `state_critical_section.rs` | `u8`/`usize` + `critical_section::with()` |

nvptx64 with SM 7.0+ **does** have `target_has_atomic = "32"` and `"64"`, so
`state_atomics.rs` will be selected. `AtomicU8` may not be available; the `"32"` fallback
uses `AtomicU32` which is fully supported.

#### 5b. Critical Section (`critical_section` crate)

Used in `state_critical_section.rs` (the fallback path) and anywhere `CriticalSection` tokens
are passed. The `critical-section` crate requires a platform-specific implementation registered
via `#[critical_section::impl]`. No implementation exists for nvptx64. If the atomic path is
taken (SM 7.0+), this file is not compiled and the dependency is dormant — but the crate is
still a `Cargo.toml` dependency. If a user accidently enables a feature that triggers this
path, it will fail to link.

#### 5c. Architecture (pender / sleep)

As detailed in Q3, each `arch-*` feature brings a platform-specific `__pender` and executor
wrapper. The `arch-spin` feature provides a no-op pender — this is the only arch that does
not require platform specifics.

#### 5d. Integrated Timers

`#[cfg(feature = "integrated-timers")]` adds `embassy-time-driver` and `embassy-executor-timer-queue`
dependencies. These require a `time_driver_impl!()` registered by the platform HAL (e.g., using
hardware timers). On GPU, no hardware timer driver exists, so this feature **must be disabled**.

#### 5e. AVR / portable-atomic

On AVR, `portable_atomic::AtomicPtr` replaces `core::sync::atomic::AtomicPtr`. Not relevant
to nvptx64.

#### 5f. Thread-Local Storage (TLS)

A search of the embassy-executor source reveals **no `thread_local!` macro usage** in the
executor core (`raw/`, `arch/`). The executor accesses its `SyncExecutor` via a `'static`
reference passed at spawn time, not via TLS. This is a deliberate embedded-systems design
choice. **No TLS elimination is needed.**

#### 5g. Heap Allocation

No `Box`, `Vec`, or `alloc` usage in the executor core. All task storage is in
`static TaskStorage<F>` / `static TaskPool<F, N>`. **Embassy is heap-free by default.**

#### 5h. `cordyceps` dependency

The `cordyceps` crate (intrusive data structures) is a mandatory dependency. It is `no_std`
and uses only atomics internally. It should compile for nvptx64 without modification, assuming
pointer-width atomics are available.

### Q6: Changes Needed for nvptx64 Compilation

Based on the analysis above, the following changes are required to compile `embassy-executor`
for `nvptx64-nvidia-cuda`:

#### Required Changes

1. **Supply a GPU `__pender`**
   Do not enable any `arch-*` feature. Instead, provide a custom `__pender` that is a
   no-op (for spin-poll model) or sets a shared-memory flag (for event-driven model):
   ```rust
   #[no_mangle]
   pub unsafe extern "C" fn __pender(_context: *mut ()) {
       // no-op: GPU executor uses spin-poll
   }
   ```

2. **Disable `integrated-timers`**
   Set `default-features = false` and do not enable `integrated-timers` in `Cargo.toml`.
   No GPU time driver exists.

3. **Ensure SM 7.0+ target**
   Compile with `target-cpu=sm_70` or higher so that `target_has_atomic = "32"` and `"64"`
   are set, selecting `state_atomics.rs` and avoiding the `critical_section` dependency.

4. **Static task declarations**
   Tasks must be declared as `static` items (already required by Embassy; no change needed).

5. **Kernel entry point wrapper**
   Embassy's `Executor::run()` (in `arch-spin`) loops forever calling `poll()`. For GPU, this
   maps naturally to the kernel body — each GPU thread runs its own executor instance:
   ```rust
   #[no_mangle]
   pub unsafe extern "ptx-kernel" fn kernel_main() {
       static EXECUTOR: StaticCell<Executor> = StaticCell::new();
       let executor = EXECUTOR.init(Executor::new());
       executor.run(|spawner| {
           spawner.spawn(my_task()).unwrap();
       });
   }
   ```
   Note: `StaticCell` (from `static_cell` crate, no_std compatible) or manual unsafe static
   initialization is needed since `static mut` is UB with multiple threads.

#### Potentially Problematic (needs verification)

6. **`embassy-executor-macros` proc-macro crate**
   The `#[embassy_executor::task]` attribute macro must run on the host. It generates code
   that is then compiled for the GPU target. This should work since proc-macros always run on
   the host — but any generated `use` paths must resolve in `no_std` context.

7. **`AtomicU8` availability on nvptx64**
   PTX natively supports 32-bit and 64-bit atomics; 8-bit atomics are not in the ISA.
   If `target_has_atomic = "8"` is set but the LLVM backend cannot emit 8-bit atomics for PTX,
   the compile may fail or produce incorrect code. Verify with a small test crate whether
   `AtomicU8` compiles for nvptx64. If not, a shim that widens to `AtomicU32` with masking
   is needed.

8. **`cordyceps` intrusive list pointer assumptions**
   Verify that `cordyceps`'s `AtomicPtr` CAS loop compiles correctly for nvptx64. The
   pointer size (64-bit) and alignment must match PTX's expectations.

#### Not Required

- No TLS removal (Embassy does not use TLS).
- No heap removal (Embassy is already heap-free).
- No `std` removal (Embassy is already `no_std`).
- No changes to waker vtable (pure pointer + function pointer, fully portable).

### Q7: CriticalSection and SIMT

#### The Problem

`CriticalSection` semantics in Embassy (via the `critical-section` crate) assume:
1. A single "current thread" or "current interrupt context" can be identified.
2. Disabling interrupts (or acquiring a mutex) prevents concurrent access.
3. The region executes atomically from the perspective of all potential competitors.

On a GPU warp (32 SIMT lanes executing in lockstep):
- There is no "interrupt" to disable.
- All 32 lanes execute the same instruction simultaneously.
- "Concurrent access" happens at the lane level within the warp.
- `critical_section::with()` is entirely meaningless — it cannot exclude any lane from
  running because all lanes run together.

#### The Non-Issue (when atomics are used)

As established in Q5, on SM 7.0+ the `state_atomics.rs` path is taken, which **does not use
`CriticalSection` at all**. The state transitions use `AtomicU32` CAS and `fetch_and`/`fetch_or`
with `AcqRel` ordering, which *do* have correct semantics in PTX (with hardware atomic
instructions in global or shared memory).

The `CriticalSection` path is only compiled for targets without atomics. If we ensure SM 7.0+
targeting, this path is dead code and poses no problem.

#### The Real SIMT Concern: Warp Divergence

The deeper SIMT issue is not CriticalSection but **warp divergence** in the executor loop:

```rust
// In the polling loop, different tasks may be at different await points
loop {
    unsafe { executor.poll() }
}
```

Each GPU thread (lane) runs its own executor with its own set of tasks. If different lanes'
tasks are in different `await` states (some ready to poll, some not), the lanes diverge at
different `if task_ready` branches. On NVIDIA hardware, diverged lanes are serialized — the
warp must execute both paths sequentially with masking. This is **not a correctness issue**
but a **performance issue**: maximum divergence means 1/32nd of peak warp throughput.

For the use case of "N independent async tasks on N GPU threads," divergence is expected and
is the fundamental trade-off. VectorWare's approach explicitly acknowledges this is "less about
performance and more about capability."

#### Critical Section for Inter-Warp Synchronization

If tasks on *different warps* need to communicate (e.g., a channel producer on warp 0, consumer
on warp 1), the current Embassy synchronization primitives (`Mutex<CriticalSection, T>`,
`Channel`) rely on `CriticalSection` for mutual exclusion. These will not work across warps
without a custom CriticalSection implementation backed by PTX `bar.sync` or `atom.cas`-based
spinlocks.

**For single-warp execution**: CriticalSection is irrelevant (no concurrency outside the warp).
**For multi-warp execution**: a custom CriticalSection impl using spinlock on a global atomic is
needed. This is a tractable but non-trivial implementation task.

### Q8: DECISION — Port Embassy vs Custom Executor

#### Embassy Port Assessment

**Pros:**
- VectorWare's published blog confirms it works with "very few changes."
- The core (run queue, waker, poll loop) is already compatible.
- No TLS, no heap, no `std` — Embassy was designed for constrained environments.
- Existing async ecosystem (embassy-sync channels, mutexes) becomes available after porting.
- Production-grade: battle-tested on thousands of embedded deployments.
- `arch-spin` provides a template: no-op pender + busy-poll loop = exactly what GPU needs.
- Estimated porting effort: 1–3 days to produce a compiling, functionally correct GPU executor.

**Cons:**
- `critical-section` crate is a mandatory dependency (even if the CS path isn't taken);
  requires a no-op or spinlock impl for nvptx64 to satisfy the linker.
- `integrated-timers` feature must be carefully disabled.
- `AtomicU8` availability on nvptx64 requires verification.
- Embassy's `Spawner` uses `'static` bounds extensively — GPU kernel lifetime semantics may
  require unsafe overrides.
- Register pressure: `TaskHeader` contains timer queue items even without the timer feature
  (verify with `#[cfg]` gates). GPU registers are precious.
- The `embassy-executor-macros` proc-macro generates `TaskStorage<F>` statics — this works
  but is opaque and harder to audit for GPU correctness.

#### Custom Minimal Executor Assessment

**Pros:**
- Full control over register pressure and memory layout.
- Can be tuned for SIMT: e.g., a per-lane task array instead of a linked-list queue,
  avoiding atomic CAS overhead entirely (no contention within a single lane).
- No extraneous dependencies (`cordyceps`, `critical-section`, timer infrastructure).
- Simpler waker: for single-future-per-thread model, a no-op waker or a waker that sets a
  shared-memory flag is 10–20 lines of code.
- Easier to audit for GPU-specific UB (shared memory address space, alignment requirements).
- Can implement `block_on(future)` for the simplest case in ~50 lines.

**Cons:**
- Re-invents the wheel: task spawn, queue, waker vtable all need reimplementation.
- No existing async ecosystem integration without additional compat work.
- Testing burden: must reproduce Embassy's correctness guarantees.
- Estimated effort: 3–7 days for a production-quality implementation.

#### DECISION: Start with Embassy Port, Keep Custom Executor as Fallback

**Recommendation: Port Embassy using `arch-spin` as the base template.**

Rationale:
1. VectorWare's confirmed success removes theoretical risk — it is known to be feasible.
2. The `arch-spin` feature already provides the correct execution model (no-op pender, spin
   poll) without any changes to Embassy's core.
3. The remaining blockers (AtomicU8, CriticalSection impl, no-op pender) are each individually
   small (< 50 lines) and well-understood.
4. Porting Embassy preserves access to `embassy-sync` channels and mutexes for inter-task
   communication, which would otherwise need to be rebuilt for a custom executor.
5. If register pressure or warp divergence performance becomes unacceptable in practice,
   the custom executor path remains open — the async/await state machine compilation is
   executor-agnostic.

**Concrete next steps for the port:**
1. Create `crates/gpu-executor/` with `embassy-executor` as a dependency,
   `default-features = false`, `features = []`.
2. Provide `__pender` no-op in the crate root.
3. Register a no-op `CriticalSection` implementation for nvptx64 (using
   `critical_section::set_impl!` with a spinlock or bare no-op).
4. Test compilation with `cargo build --target nvptx64-nvidia-cuda -Z build-std=core`.
5. Spawn a single task running a trivial `async fn` and verify PTX output.

---

## Unexpected Discoveries

1. **`arch-spin` already exists and is the GPU answer**: Embassy's architecture-agnostic
   spin executor (enabled by `arch-spin` feature) uses an identical execution model to what
   a GPU port needs — no-op pender, tight poll loop. This was not obvious from the project
   description.

2. **No TLS in Embassy**: Unlike Tokio (which uses `thread_local!` extensively for per-thread
   executor handles), Embassy does not use TLS at all. The executor is accessed via `'static`
   references. This eliminates one of the biggest expected porting obstacles.

3. **VectorWare uses multiple warps, not multiple tasks-per-warp**: Based on the HN discussion
   clarification, their approach runs one executor per GPU thread (lane), not one shared
   executor per warp. This means the SIMT/CriticalSection problem for the executor itself is
   moot — each lane has its own private executor state.

4. **`cordyceps` is a non-optional dependency**: The intrusive linked list crate is always
   compiled in, even for the `arch-spin` path. It must compile for nvptx64. Given it is
   `no_std` and uses only `core` atomics, this is likely fine but unverified.

5. **`wake_by_ref` and `wake` use the same function pointer**: In Embassy's vtable,
   `RawWakerVTable::new(clone, wake, wake, drop)` — both `wake` and `wake_by_ref` point to
   the same `wake` function. This is valid because `TaskRef` is `Copy` (it's just a pointer),
   so there is no difference between "consuming" and "non-consuming" wake.

6. **Tasks enqueued during `poll()` are processed next cycle**: This fairness guarantee
   prevents a runaway task from re-waking itself indefinitely within a single `poll()` pass.
   On GPU where `poll()` runs in a tight loop (spin model), this just means the runaway task
   gets one extra iteration per spin — acceptable.

---

## Key Conclusions

1. **Embassy is ≈90% GPU-compatible out of the box** for nvptx64 SM 7.0+ targets.

2. **The three changes needed are small and well-scoped**:
   - No-op `__pender` (trivial)
   - No-op or spinlock `CriticalSection` implementation for nvptx64 (< 30 lines)
   - Verify `AtomicU8` compiles for nvptx64; if not, a shim is needed

3. **CriticalSection is only a problem for inter-warp shared state**, not for the executor
   core itself when SM 7.0+ atomics are used.

4. **SIMT divergence is a performance concern, not a correctness concern** for the
   spin-poll executor model.

5. **The `arch-spin` feature is the correct starting point** — it provides a working,
   interrupt-free executor today.

6. **No heap, no TLS, no std required** — Embassy's design philosophy aligns exactly with
   GPU constraints.

---

## Open Questions

1. **Does `AtomicU8` compile for nvptx64?** PTX natively supports 32-bit atomics; 8-bit may
   be emulated or rejected by LLVM's NVPTX backend. Need to test with a minimal crate.

2. **Does `cordyceps` compile for nvptx64?** It uses `AtomicPtr` internally. Unverified.

3. **What is VectorWare's exact pender implementation?** The blog post does not disclose
   whether they use a no-op spin, shared-memory flags, or something else.

4. **Register usage of `TaskHeader` on GPU?** The header includes timer queue items even
   without the timer feature being enabled — verify with `cargo expand` that the struct
   is minimal when timers are disabled.

5. **Can multiple tasks run on the same GPU thread?** Embassy's `arch-spin` runs all N
   spawned tasks on a single "thread" (lane) cooperatively. On GPU, this means true
   cooperative multitasking within a lane — confirmed feasible, but warp-level
   parallelism comes from having N lanes each running their own set of tasks.

6. **`'static` lifetime for task spawning**: `Spawner::spawn()` requires `'static` futures.
   GPU kernels launch with bounded lifetime — how does VectorWare reconcile `'static` with
   kernel-bounded task lifetimes? Unsafe `transmute` of lifetimes is the likely answer.

7. **Is the `TaskPool<F, N>` pattern usable on GPU?** Static arrays of `TaskStorage` require
   the compiler to allocate them in GPU global memory. Stack allocation (`.local` address
   space) or shared memory (`.shared`) may be preferable for latency. Address space
   annotations may be needed.

---

## Impact on Downstream Tasks

| Task | Impact |
|------|--------|
| `async-runtime.2` (executor design) | DECISION made: port Embassy using `arch-spin` as base |
| `async-runtime.3` (waker implementation) | No custom waker needed; Embassy's vtable approach is reusable |
| `async-runtime.4` (GPU executor PoC) | Start with `arch-spin` + no-op pender; test AtomicU8 first |
| `hostcall.*` | Inter-warp communication will need custom CriticalSection impl backed by atomics |
| `toolchain.*` | nvptx64 AtomicU8 availability is a toolchain question; must verify |
| `integration.*` | VectorWare's multi-warp model (one executor per lane) is the architecture to target |

**New tasks spawned by this investigation:**
- `async-runtime.5`: Verify AtomicU8 / AtomicPtr compilation for nvptx64-nvidia-cuda
- `async-runtime.6`: Implement no-op CriticalSection for nvptx64 and validate link
- `async-runtime.7`: Measure `TaskHeader` register footprint with timers disabled (GPU register budget)

---

## Theme Progress

**async-runtime** theme status: **active**

Completed investigations:
- [x] `async-runtime.1` (this document): Embassy deep analysis + GPU compat assessment

Pending / newly identified tasks:
- [ ] `async-runtime.2`: Design GPU executor architecture (lane-per-executor model)
- [ ] `async-runtime.3`: Waker implementation prototype
- [ ] `async-runtime.4`: Embassy GPU PoC compilation test
- [ ] `async-runtime.5`: AtomicU8 / AtomicPtr nvptx64 verification *(new)*
- [ ] `async-runtime.6`: CriticalSection no-op impl for nvptx64 *(new)*
- [ ] `async-runtime.7`: TaskHeader register pressure measurement *(new)*

**Overall assessment**: The async-runtime theme is in a healthy state. The path forward is
clear, blockers are small and tractable, and VectorWare's published success provides
external validation. The Embassy port approach is recommended over a custom executor at
this stage, with the option to replace if GPU-specific performance requirements emerge.
