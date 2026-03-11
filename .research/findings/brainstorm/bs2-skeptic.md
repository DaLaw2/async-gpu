# Brainstorm BS2 — Devil's Advocate / Skeptic Analysis
**Date:** 2026-03-11
**Role:** Skeptic
**Seq:** 2

---

## Preamble: The Stakes Have Changed

In BS1, the skeptic's job was to challenge unverified assumptions. In BS2, we are past the
assumption phase. We have empirical data. The question is no longer "what might go wrong" but
"given what has gone wrong, are we reasoning about it correctly?" The critical new finding —
that `core::sync::atomic` is silently broken on nvptx64 at the LLVM IR level — is not a
single problem to solve and move on from. It is a diagnostic signal about the class of problems
this toolchain will inflict on us. The skeptic's job here is to resist the committee's natural
instinct to declare a workaround and proceed.

---

## 1. Option A (LLVM NVPTX Intrinsics) Is Not a Reliable Workaround

The team has proposed using `llvm.nvvm.membar.sys` and `llvm.nvvm.atomic.add.gen.i.sys.*` as
a partial fix. There are multiple reasons to be skeptical of this path.

### 1a. LLVM optimizations can move code across the fence boundary

The `llvm.nvvm.membar.sys` intrinsic, when called via `extern "C"` with `#[link_name]`, is
treated by the LLVM optimizer as an opaque function call — not as a memory barrier. This is
the critical distinction. LLVM's optimizer respects `llvm.fence` and `memory::barrier` as
genuine barriers that prevent reordering of surrounding loads and stores. An `extern "C"`
function call is treated as having `anyregcc` calling convention implications — it may prevent
inlining, but it does not tell the LLVM alias analysis or the LLVM instruction scheduler that
it is a fence.

In other words: the data write may be reordered after the `nvvm_membar_sys()` call by LLVM's
own optimization passes before the instruction even reaches the PTX backend. The PTX fence
instruction would be emitted in the correct position in the PTX text, but the data write it is
supposed to precede may already have been hoisted above it in the LLVM IR. The bug is
invisible in the PTX output but present in the generated execution.

This is not a hypothetical. It is the standard hazard of calling barrier intrinsics through
`extern "C"` in LLVM. The correct way to call barrier intrinsics that have ordering semantics
is through LLVM's `llvm.fence` builtin or through the asm side-effect annotation (`"~{memory}"`
in inline asm). Neither is properly available on nvptx64.

### 1b. LLVM may remove the intrinsics in a future version

The `llvm.nvvm.atomic.*.gen.i.*` intrinsics are NVVM IR intrinsics, not LLVM core intrinsics.
They were added in D24943 (2016) for the OpenMP offloading and CUDA C++ toolchain that uses
libNVVM as its backend. The `nvptx64` Rust target does not go through libNVVM; it goes directly
to the LLVM NVPTX backend. The behavior of calling NVVM intrinsics via an `extern "C"` link
from the LLVM NVPTX path is not guaranteed. These intrinsics are documented for the NVVM IR
path. LLVM's NVPTX backend may handle them, or it may silently ignore them, or a future LLVM
version may remove or rename them as part of NVVM IR cleanup.

The atomics.1 finding itself notes: "Without the NVVM IR path (i.e., using raw nvptx64 target),
it may not be available or may link incorrectly. This needs empirical verification." This
means Option A is currently unverified. The team is proposing to build hostcall design decisions
on an unverified workaround. That is the same category of error that produced the original
atomics problem: assuming the standard path works without checking the PTX output.

### 1c. Option A does not work on all SM versions in the project's target range

The intrinsics require SM60+. The current nightly Rust default is `sm_30`. If any kernel is
compiled without an explicit `-C target-cpu=sm_70` override — which is easy to forget in a
multi-crate build with custom build scripts — the intrinsics will silently produce incorrect
behavior or fail. A toolchain that requires correct operation of an explicit opt-in flag that
is easy to omit is not a reliable toolchain.

More importantly, SM60+ system-scope atomics (the PTX 5.x form) provide scope without semantic
qualifiers. SM70+ is required for acquire-release semantics via the PTX 6.0 model. The project
targets a RTX 3060 (SM86, Ampere), which is SM70+, so the hardware is fine — but the
intrinsics from Option A (`llvm.nvvm.atomic.*.sys`) give you only relaxed scope, not
acquire-release. Any protocol that requires the CPU to observe GPU writes in order, or vice
versa, still does not work correctly with Option A even on SM86. Relaxed atomics with a
system-scope fence are not equivalent to acquire-release atomics with system scope.

---

## 2. Option C (Volatile) Is Based on a PTX ISA Claim That May Not Hold

The atomics.1 finding states: "ld.volatile.u32 → `.relaxed.sys` semantics per PTX ISA."
This is cited from "PTX ISA spec, v6.0+." Let us examine exactly what is being claimed and
whether it is reliable.

### 2a. The PTX ISA has changed this mapping across versions

In PTX ISA v6.0 (introduced with SM70 / Volta), NVIDIA redesigned the memory consistency model.
The relationship between `ld.volatile` / `st.volatile` and the new `.sem`/`.scope` model is
explicitly documented as:

> "An ld or st with qualifier `.volatile` is equivalent to ld or st with `.relaxed` memory
> ordering qualifier and scope `.sys`."

This is the statement the team is relying on. However, it applies to PTX ISA v6.0+. For PTX
ISA v5.x (targeting SM60–69), the volatile semantics are defined differently: volatile simply
prevents instruction reordering within the same thread (analogous to C's `volatile`, not C++'s
`volatile std::atomic`). The PTX spec v5.x does not guarantee cross-device visibility via
`ld.volatile`.

More importantly: NVIDIA's PTX ISA documentation is normative, but NVIDIA's hardware and
driver implementation is authoritative. If NVIDIA changes PTX semantics in a future driver
— which they have done before, particularly across major PTX ISA versions and SM generations
— code relying on "ld.volatile ≡ .relaxed.sys" could silently break. This is not a purely
theoretical concern; NVIDIA has historically changed PTX instruction semantics between major
driver generations, most notably in the SM70 memory model redesign itself.

### 2b. The compiler does not guarantee volatile maps to ld.volatile

The mapping `store(val, Relaxed)` → `st.volatile` is the result of LLVM patch D50391 (2019).
This patch was a pragmatic hack: it made monotonic atomics use volatile instructions because
the NVPTX backend could not handle proper atomic lowering. But D50391 could itself be
superseded by a future LLVM change that "correctly" implements monotonic lowering as
`st.relaxed.gpu` (without system scope). If a future LLVM version changes the D50391
behavior — either intentionally or as a side effect of the atomic lowering work that would
theoretically fix LLVM issue #173993 — all code relying on the D50391 volatile-implies-sys
mapping would be silently broken.

### 2c. Even if correct, Option C is insufficient for the hostcall protocol

The hostcall protocol requires, at minimum:
1. GPU writes payload data to shared buffer
2. GPU signals "request ready" flag
3. CPU observes flag, reads payload (in order, after the flag)
4. CPU writes response data
5. CPU signals "response ready" flag
6. GPU observes flag, reads response (in order, after the flag)

Step 3 requires that when the CPU observes the flag, the payload written in step 1 is visible.
This is an acquire-release pair: the GPU's flag write must "release" the payload writes, and the
CPU's flag read must "acquire" them. A relaxed `.sys` volatile load/store provides visibility
but no ordering. Without an acquire-release fence around the flag write and read respectively,
the CPU may observe the flag as set but read stale payload data (or the compiler/CPU may
reorder the payload read before the flag read). Option C, even if the `.relaxed.sys` claim holds,
is fundamentally insufficient for the hostcall protocol without additional fence infrastructure.
And the fences are either dropped (if using `core::sync::atomic::fence`) or also only relaxed
(if using `nvvm_membar_sys` with the optimizer concern from Section 1a).

---

## 3. We Are Massively Underestimating the Atomics Problem's Scope

The team is framing the atomics problem as: "the hostcall signaling protocol needs to be fixed."
This framing is too narrow. The correct framing is: "every single use of `core::sync::atomic`
in any code that runs on nvptx64 is silently wrong for cross-device use cases."

### 3a. `Arc<T>` is broken

`Arc<T>` uses `core::sync::atomic::AtomicUsize` for reference counting. On nvptx64, every
`Arc::clone()` and `Arc::drop()` call emits `atom.global.add.usize` with no scope and no
ordering. For GPU-only reference counting (all `Arc` clones on the same GPU device), this
might accidentally work — the GPU-internal atomics, though unscoped, are at least atomic
within the device. But any `Arc<T>` where one clone lives on the CPU (host-side Rust future
holding a clone while a GPU kernel holds another clone) is silently wrong.

This is particularly relevant for the async executor design. If the executor's task storage
uses `Arc<dyn Future>` — which is the natural design — and any part of that Arc's lifecycle
crosses the GPU-CPU boundary, the reference count is incorrect.

### 3b. `Mutex<T>` cannot be correctly implemented

A GPU-side `Mutex<T>` would need to implement a spinlock. A correct spinlock requires
compare-exchange with acquire semantics on lock acquisition and release semantics on unlock.
As established, neither can be expressed via `core::sync::atomic` on nvptx64. Any `Mutex<T>`
implementation using the standard atomic API will compile without error, appear to work in
low-contention tests, and produce data races under any concurrent access. There will be no
diagnostic.

### 3c. std internals use atomics we do not control

If we proceed with `-Zbuild-std=std` for nvptx64, the compiled `std` library will contain
dozens of uses of `core::sync::atomic` in its internals:
- The global allocator's free list management (if it uses a lock-free structure)
- `Once` (initialization guard, used extensively in std for lazy statics)
- `OnceLock<T>` and `LazyLock<T>`
- `mpsc` channel implementation
- `Condvar` notify counter
- `std::io::stdout()` locking mechanism

Every one of these will emit incorrect PTX. When we run `println!` from the GPU via the libc
shim, the write to `stdout` goes through `std::io::stdout().lock()`, which uses a `Mutex` that
uses `core::sync::atomic`. On a single-warp test this will appear to work (no contention, the
lock is effectively always "uncontended"). On a multi-warp test, data races in the `Mutex`
implementation will corrupt the output buffer.

The team is planning to port std to GPU. The std's own internal atomics are all broken for
cross-device use. We are not just fixing the hostcall protocol — we would need to audit and
replace every atomic in the std source tree, or compile std with a custom `atomic_*` intrinsic
implementation that redirects to correct scoped forms.

---

## 4. Should We Abandon nvptx64? The Case Is Stronger Than Acknowledged

ADR-1 committed to the nvptx64 upstream path. The atomics.1 finding should trigger a serious
re-examination of this decision. The team's current position appears to be "fix atomics with
workarounds, keep nvptx64." Let me argue the opposite.

### 4a. The combination of broken atomics + no inline asm is uniquely bad

On any other target with broken atomics, the fix is: write inline assembly with the correct
instruction. On nvptx64, that escape valve does not exist. The only paths are:
- LLVM intrinsics via `extern "C"` (with the optimizer safety concerns from Section 1a)
- Rust-CUDA (which is not nvptx64)

This makes nvptx64 uniquely fragile. Every other safety concern (missing libstd support,
limited intrinsics, LLVM backend bugs) has a known mitigation path. The atomics problem
combined with the absence of inline asm has no clean mitigation on nvptx64.

### 4b. ADR-1's rationale was toolchain stability. That rationale is now weaker.

ADR-1 chose nvptx64 because it is upstream Rust with official support. But "official support"
means "it compiles." It does not mean "it is correct." The LLVM issue #173993 is open with no
fix timeline. The NVPTX maintainer position is that this is an architectural gap requiring deep
rework. "Official support" for a target where the fundamental concurrency primitives are silently
broken is a false sense of security.

Rust-CUDA (codegen_nvvm) has the opposite problem: it is a fork, it tracks specific nightly
versions, and maintenance continuity is uncertain. But it provides:
- Inline PTX assembly (bypasses all LLVM NVPTX backend issues for atomics)
- `cuda_std::atomic` with correct scope-aware implementations
- Active community usage (by the people who actually write CUDA Rust code)

The stability tradeoff has shifted. nvptx64 is "stable" in the sense that it is in the upstream
repository and will not suddenly disappear. But it is "unstable" in the sense that its core
concurrency primitives are silently wrong with no fix. Rust-CUDA is "unstable" in the sense
that it is a fork with uncertain future, but "stable" in the sense that the code it generates
is actually correct.

### 4c. Rust-CUDA was ruled out too quickly in ADR-1

The ADR-1 decision to use nvptx64 should now be revisited. The question is not "is Rust-CUDA
in upstream Rust?" — it is not. The question is: "can we build VectorWare's technology on a
toolchain where the core concurrency primitive is silently incorrect?" The answer is almost
certainly no. The hostcall protocol, the async executor's waker mechanism, Arc-based task
storage, and every std Mutex — all of these require correct atomics.

If the project's goal is to reproduce VectorWare's technology, and VectorWare's technology
requires correct GPU-CPU memory ordering, then the toolchain question is settled by the
requirements: we need correct atomics. nvptx64 does not provide them. The upstream vs fork
discussion is secondary.

---

## 5. Toolchain.4 Success Is Misleading — The Risks Are Compounding

The team has reported "toolchain.4 success: minimal kernel runs on RTX 3060." This is
genuinely good progress. But it needs to be read with the following caveat: the kernel was
trivial. The success of a trivial kernel is not evidence that complex kernels will work.

### 5a. The trivial kernel has no atomics

A kernel that adds numbers and returns a result exercises: the nvptx64 target compilation
pipeline, cudarc launch infrastructure, PTX loading, argument passing, result retrieval. It
does not exercise: atomics, function pointers, dynamic dispatch, panic handling, stack frames
beyond a few hundred bytes, or any of the std infrastructure.

The first kernel that adds `Arc<T>` usage, or calls a `Mutex::lock()`, or invokes a `std::once`
guard, will silently produce incorrect results. The transition from "minimal kernel works" to
"real kernel works" will not be smooth. It will hit a series of subtle correctness failures that
are extremely difficult to diagnose without PTX inspection tooling.

### 5b. Async state machines are not trivial kernels

An async state machine compiles to a struct containing: the state discriminant, all local
variables from every suspend point (the largest set of locals across all suspend points is
stored simultaneously), and a vtable pointer for the waker. On a complex async function, this
struct can be hundreds or thousands of bytes. In CUDA's execution model, every thread's stack
frame is allocated from registers and local memory. Local memory spills (variables that don't
fit in registers) go to a per-thread local memory region in DRAM, which has 5-10x higher
latency than register access.

The toolchain.4 success says nothing about what register pressure and local memory spill
behavior looks like when the kernel is an async state machine awaiting a hostcall. The
nvcc profiler tools (`ncu`, `nvprof`) do not integrate with Rust's nvptx64 output cleanly.
Finding and diagnosing register spills will require manual PTX inspection. This is not
impossible, but it is not "the toolchain works" — it is a significant research task that is
currently not on the task list.

### 5c. Function pointers and dynamic dispatch may not work as expected

The waker mechanism in an async executor relies on dynamic dispatch (the `RawWaker` vtable,
or in Embassy's case, the `Pender` function pointer). LLVM's NVPTX backend has known issues
with function pointers: it can sometimes resolve them statically (monomorphization at PTX link
time), but dynamic function pointers that are not statically resolvable at compile time produce
PTX that either cannot be loaded by the CUDA driver or produces incorrect behavior.

The Embassy executor's `Pender` callback and the `RawWakerVTable` are both function pointer
dependent. async-runtime.1 claims "Embassy ~90% compatible" — but that 10% incompatibility
may contain precisely these dynamic dispatch cases. A 90% compatible async executor that fails
at waker invocation is not a 10% problem; it is a 100% failure because wakers are the core
mechanism of cooperative scheduling.

---

## 6. What Else Are We Missing? Compounding LLVM NVPTX Issues

The atomics bug is the most visible LLVM NVPTX backend issue, but it is not the only one.
Here are the categories of LLVM NVPTX backend problems that have been publicly reported and
that are likely to affect this project.

### 6a. Stack frame size limits and recursion

PTX kernels have a default stack size. Recursive functions and large stack frames can overflow.
The LLVM NVPTX backend's stack usage analysis is known to be imprecise; it sometimes fails to
detect stack overflows at compile time, leading to silent memory corruption at runtime. The
async state machine structs mentioned in Section 5b are exactly the kind of large stack
allocations that can trigger this.

### 6b. Alloca and local memory handling

LLVM's `alloca` instruction (stack allocation) maps to PTX local memory. The NVPTX backend's
handling of `alloca` in complex control flow (which includes async state machines with their
match-on-discriminant structure) has bugs. Specifically: address space inference for pointers
derived from `alloca` can be wrong, causing accesses to local memory to be misclassified as
global memory accesses, which either crashes or silently corrupts data.

### 6c. 128-bit integer operations

LLVM's NVPTX backend does not support `i128` operations natively. They are emulated via
software. In Rust, several standard library types use `u128` or `i128` (particularly in
formatting, hash functions, and UUID-like types). If std is compiled for nvptx64, any code
path that touches these types will invoke the software emulation — which may be incorrect
or may trigger backend errors.

### 6d. Exception handling and cleanup paths

Even with `panic = "abort"`, Rust generates landing pads and cleanup paths in LLVM IR for
`Result` and `Option` handling in some configurations. The NVPTX backend has historically had
issues with exception handling-related IR constructs. This is currently mitigated by Rust's
default `panic = "abort"` for no_std targets, but when compiling full std (which may require
`panic = "unwind"` for some features), this could surface.

### 6e. The CUDA driver API version dependency

The PTX that LLVM generates must be accepted by the CUDA driver's JIT compiler (`ptxas`).
The version of `ptxas` bundled with the CUDA toolkit has changed its behavior across versions.
PTX that is valid for `ptxas` 11.x may produce different results under `ptxas` 12.x, and
vice versa. The nightly Rust toolchain does not specify a minimum `ptxas` version. An
experiment that works on one machine (CUDA 11.8) may fail on another (CUDA 12.3) not because
of a Rust or LLVM change, but because of a `ptxas` change. This is a reproducibility hazard
that currently has no mitigation in the project plan.

### 6f. SM architecture-dependent correctness

The RTX 3060 is SM86 (Ampere). Some LLVM NVPTX backend behaviors differ across SM generations.
An experiment that "works" on SM86 may not generalize to SM70 (Volta) or SM75 (Turing). The
project has no stated requirement for multi-SM-generation support, but if the goal is
technology reproduction rather than a single-machine demo, the SM dependency needs to be
understood and documented.

---

## 7. Highest-Risk Conclusions

The skeptic's summary is that the project is at an inflection point. The atomics finding is
not just a task to assign and track — it is evidence that the fundamental toolchain choice may
need to be reconsidered. The team must resist two dangerous failure modes:

**Failure Mode 1: Workaround Accumulation.** Each broken component gets a workaround. LLVM
atomics → use intrinsics. Intrinsics have optimizer issue → add `#[no_inline]`. Fences dropped
→ use membar intrinsic. membar has ordering issue → restructure protocol. At some point the
workarounds compound to the point where the system is correct by coincidence under test
conditions but incorrect under load. We will not know which combination of workarounds is
actually safe because none of them individually have clear semantic guarantees.

**Failure Mode 2: False Progress via Trivial Success.** The minimal kernel runs. The hostcall
flag protocol passes a simple test. The async hello-world executes. Each of these is reported
as progress, but none of them tests the conditions under which the known bugs manifest (concurrent
access, complex state machines, cross-device ordering under load). The project could reach the
integration phase with a collection of components that each "work" in isolation and completely
fail when composed.

The correct response to the atomics finding is not to assign `atomics.2` and move on. It is to
conduct a deliberate toolchain decision — specifically: is the project viable on nvptx64 at all,
or does it require Rust-CUDA? Until that decision is made with clear eyes, all downstream work
(hostcall design, gpu-std porting, async executor design) is building on an uncertain foundation.

---

## Summary Table of Critical Open Questions

| Question | Consequence if Wrong | Current Status |
|----------|---------------------|----------------|
| Do LLVM NVVM intrinsics survive LLVM optimizer reordering? | Option A workaround is unsafe | Unverified |
| Is `ld.volatile ≡ .relaxed.sys` guaranteed across future PTX versions? | Option C becomes unreliable | Observed, not contractually guaranteed |
| Are all std internal atomics affected (Arc, Once, Mutex)? | std port is fundamentally broken | Yes — confirmed by atomics.1 |
| Does Rust-CUDA provide correct scoped atomics in practice? | Option B viability unknown | Claimed, not verified |
| Do async state machine structs cause register spill on nvptx64? | Occupancy unacceptable | Unknown, no experiment planned |
| Do Embassy's function pointer wakers compile correctly on nvptx64? | Async executor is broken at core | Unknown, not yet tested |
| Is toolchain.4 success reproducible with -C target-cpu=sm_70? | Current kernel may use sm_30 inadvertently | Not confirmed in findings |
