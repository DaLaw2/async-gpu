# Brainstorm BS2 — Systems Programmer Perspective
**Role**: Rust Systems Programmer
**Date**: 2026-03-11
**Focus**: Atomics breakage impact, workaround evaluation, hostcall protocol implications, waker mechanism, ADR-1 reconsideration

---

## 0. Executive Summary

The confirmation that `core::sync::atomic` emits unscoped PTX on `nvptx64` (LLVM bug #173993) is not a corner case — it is a foundational correctness failure. Every GPU-CPU synchronization primitive built on Rust atomics is silently undefined behavior. This finding forces a concrete decision at the lowest layer of the stack before any higher-level work (hostcall protocol, async executor waker) can be trusted. The good news: the problem is well-bounded and Option A (LLVM NVVM intrinsics) provides a viable path forward without abandoning ADR-1.

---

## 1. Impact on Hostcall Protocol Design

### What the protocol actually requires

A correct hostcall protocol requires three things from the memory model:

1. **A GPU-side release store**: Before signaling the host that a request is ready, all writes to the payload buffer must be visible to the CPU. This is `atom.sys.release` on the slot state.
2. **A CPU-side acquire load**: The host polling loop must see the request payload after it observes the ready signal. On x86 this is implicit (TSO), but the acquire semantics must also prevent CPU-side compiler reordering.
3. **A GPU-side acquire load**: When the GPU reads the host's response, it must observe the payload written by the host. This is `atom.sys.acquire` on the slot state from the GPU side.

Without `.sys` scope, the GPU operates with `.gpu` scope atomics at best, meaning the CPU is entirely invisible to the GPU's memory model. The protocol appears to work under low load (because the DRAM bus happens to flush) but fails intermittently under concurrent pressure. This is exactly the type of silent correctness bug that passes basic tests and breaks in production.

### What breaks today

- The slot state transitions (`FREE` → `PENDING` → `BUSY` → `DONE`) rely on CAS or swap operations that must be system-scoped.
- The payload write before a `PENDING` signal has no release semantics — the CPU may observe `PENDING` but read garbage payload.
- The GPU's read of the `DONE` state has no acquire semantics — the GPU may read the response before the CPU's writes are flushed.

**Conclusion**: The hostcall protocol cannot be implemented correctly using `core::sync::atomic` on `nvptx64` as it stands. This is a blocker for hostcall.3.

---

## 2. Option A: LLVM NVVM Intrinsics — Systems Evaluation

### Mechanism

LLVM exposes a set of `llvm.nvvm.*` intrinsics that map directly to PTX instructions. Relevant ones for our use case:

- `llvm.nvvm.membar.sys` → `membar.sys` (system-scope fence)
- `llvm.nvvm.membar.gl` → `membar.gl` (device-scope fence)
- `llvm.nvvm.atomic.load.add.f32.p0f32` → scoped atomic add (the `.sys` scope variants exist in the NVVM IR spec)

In Rust, these are accessible via:

```rust
extern "C" {
    #[link_name = "llvm.nvvm.membar.sys"]
    fn nvvm_membar_sys();
}
```

For scoped atomic operations, the pattern is more involved: LLVM's atomicrmw/cmpxchg instructions with `syncscope("system")` attribute produce the correct scoped PTX. The question is whether these are accessible from Rust `extern "C"` or require custom LLVM IR injection.

### Safety Assessment

**Correctness**: `llvm.nvvm.membar.sys` is a well-defined LLVM intrinsic that lowers to `membar.sys` unconditionally. It does not go through the broken path that `atomic::fence()` uses. It is safe in the sense that its PTX output is deterministic and correct.

**Safety concerns**:
- These intrinsics are `unsafe fn` by nature — no Rust safety wrapper exists.
- The intrinsic names are LLVM-internal and not part of any stable Rust API. An LLVM version bump could rename or remove them (low probability, but real).
- The `atomic.*.sys.*` variants require confirming the exact intrinsic name format for each operation (add, sub, CAS, etc.) against the LLVM NVVM IR specification. Getting the name wrong produces a link error, not a silent bug — which is recoverable.
- There is no guarantee that rustc will not optimize away the `extern "C"` call if the function is declared as returning `void` and has no observable output. This must be verified: the call may need `core::hint::black_box` wrapping or an `#[inline(never)]` guard on the wrapper.

**Portability**:
- Tied to LLVM NVPTX backend. Does not work with Rust-CUDA (codegen_nvvm uses NVVM IR, not LLVM IR intrinsics directly). This is acceptable given ADR-1, but must be documented.
- Does not work on AMD GPUs (different intrinsic names). If cross-GPU portability ever becomes a goal, this layer needs an abstraction.
- Works across SM architectures since `membar.sys` is not an SM-version-restricted instruction (unlike `nanosleep` which requires SM75+).

**Maintainability**:
- The `extern "C" { #[link_name = "llvm.nvvm.membar.sys"] fn ... }` pattern is fragile: if the intrinsic is renamed in a future LLVM version, the failure mode is a linker error ("undefined reference to llvm.nvvm.membar.sys"), not a runtime correctness bug. This is acceptable — it fails loudly.
- Should be encapsulated in a single `crates/gpu-atomics/` crate with a well-defined safe API surface. The unsafe intrinsic calls should be localized to one file.

**Verdict**: Option A is the correct choice for this project. It provides correctness guarantees, fails loudly if broken, and is scoped to LLVM/nvptx64 which is exactly ADR-1's target. The maintainability cost is low if properly encapsulated.

---

## 3. Volatile + System Fence: Can It Build a Correct Protocol?

### Theoretical analysis

PTX volatile load/store (`ld.volatile.global` / `st.volatile.global`) have the following semantics per PTX ISA 8.x:
- They bypass the L1 cache (treated as `.relaxed.sys` accesses in the PTX memory consistency model, CUDA 11+ unified memory model).
- They prevent compiler reordering around the access.
- They do NOT provide ordering with respect to other memory operations. A `st.volatile` followed by a `membar.sys` followed by a signal is ordered, but the volatile store itself carries no ordering.

So the combination is:
```
volatile_write(payload);   // st.volatile.global — bypass L1, no ordering
membar_sys();              // membar.sys via llvm.nvvm.membar.sys
volatile_write(signal);    // st.volatile.global — now visible to CPU
```

This is a valid release sequence for the signal/payload pattern. The `membar.sys` ensures all stores before it (including the volatile write to payload) are globally visible before the signal store proceeds.

On the read side:
```
let sig = volatile_read(signal);   // ld.volatile.global — bypass L1
membar_sys();                      // ensure all subsequent reads observe post-signal state
let data = volatile_read(payload); // ld.volatile.global
```

This is a valid acquire sequence.

### Is this equivalent to acquire/release atomics?

For the specific pattern of signal + payload (one writer, one reader, no CAS), volatile + system fence provides the same guarantees as `store(val, Release)` / `load(Acquire)` would if those worked correctly. The key insight: the ordering is provided by the `membar.sys`, not by the atomic operation itself.

However, this approach has serious limitations:

1. **No atomic CAS for multi-slot ring buffers**: Volatile load/store cannot implement compare-and-swap. A multi-warp ring buffer where multiple GPU threads compete for slots requires CAS. For that, Option A (LLVM atomic intrinsics with system scope) is mandatory.
2. **No ABA protection**: Without atomic CAS, the slot allocation in a concurrent ring buffer is racy.
3. **Single-producer single-consumer only**: The volatile + fence approach is only sufficient for SPSC communication. For the full hostcall protocol (multiple warps, concurrent requests), it is not enough.

**Conclusion**: Option C (volatile + separate system fence) is a valid fallback for the simplest case (one GPU thread polling a single flag), but cannot support the full hostcall protocol without Option A for the atomic CAS operations. The two options are complementary, not competing: use volatile stores for payload writes and Option A intrinsics for the CAS-based slot state machine.

---

## 4. Impact on the Async Executor Waker Mechanism

### What the Waker needs

The Embassy-derived executor's waker mechanism requires:
- A per-task `AtomicBool` or `AtomicU32` ready flag that the task's future can set when it wants to be polled again.
- The executor polls tasks whose ready flag is set, clears the flag, and runs `Future::poll`.
- The waker's `wake()` implementation sets the flag and signals the executor to run.

For an executor running within a single warp (lock-step SIMT), intra-warp communication between tasks does not require system-scope atomics — warp lanes are serialized naturally. But there are two cases where system scope matters:

**Case 1: Waker signaled by the CPU host thread**
When a hostcall completes, the host thread must signal the GPU task that issued the call. This signal goes CPU → GPU and requires a system-scope store on the CPU side and a system-scope acquire load on the GPU side (polling loop). Without correct GPU-side system-scope acquire, the GPU may spin forever or observe a stale ready flag.

**Case 2: Waker signaled across GPU blocks**
If the executor spans multiple blocks (e.g., a task in block 0 wakes a task in block 1), system-scope atomics are not strictly required (`.gl` device scope is sufficient), but `core::sync::atomic` emits unscoped PTX that may not even be `.gl`. This is a secondary concern compared to Case 1.

### Concrete impact

The `RawWakerVTable::wake` function pointer currently stored in the waker vtable assumes it can call `atomic.store(true, Release)` and have the executor on the GPU observe it after the CPU's response. This is broken.

**Fix**: The waker implementation for the "awaiting hostcall response" state must use system-scope atomics from Option A. Specifically, the `wake()` call (invoked by the host thread or the GPU's polling code after observing the host's response) must use an intrinsic-backed system-scope store for the ready flag. The GPU executor's polling loop that reads the ready flags must use the corresponding system-scope acquire load.

This means the `GpuWaker` type (whatever we implement) needs two variants:
- `IntraGpuWaker`: uses register atomics or no fence (for intra-block task handoff)
- `HostcallWaker`: uses system-scope intrinsics (for hostcall response signaling)

The `RawWakerVTable` dispatch mechanism selects between these based on the context. This adds complexity to the executor design but is unavoidable.

---

## 5. Should We Reconsider ADR-1 (nvptx64)?

### The case for reconsideration (Option B: Rust-CUDA / codegen_nvvm)

Rust-CUDA's `cuda_std::atomic::SystemAtomicU32` provides a working system-scope atomic backed by inline PTX. The NVVM IR path (libnvvm) has historically had better parity with CUDA's actual memory model. If system-scope atomics are the central problem, switching backends where they work out-of-the-box has appeal.

Arguments for switching:
- `cuda_std` already solved this problem; we would not be reinventing it.
- NVVM IR is NVIDIA's intended IR for GPU compilation, not a secondary target like LLVM NVPTX.
- The `SystemAtomicU32` API is safe Rust, not unsafe intrinsic wrangling.

Arguments against switching (maintaining ADR-1):

1. **Upstream is upstream**: `nvptx64-nvidia-cuda` is in upstream `rustc`. codegen_nvvm requires a custom build of rustc with the external codegen crate. The dependency chain is significantly more complex and fragile.

2. **Option A is implementable**: The LLVM intrinsic workaround for system-scope atomics is bounded in scope. It is a localized problem in one crate (`gpu-atomics`) with a clear implementation path. We are not proposing to use broken `core::sync::atomic` everywhere — we are replacing it at the layer that needs system scope.

3. **Rust-CUDA's maintenance status**: The rust-cuda project has had periods of reduced maintenance. Binding to it introduces supply-chain risk for a research project. If it falls behind a new rustc version, we cannot compile.

4. **Inline PTX option**: Beyond LLVM intrinsics, `nvptx64` supports inline PTX via `core::arch::asm!` (the `nvptx` asm dialect). This provides a direct escape hatch for any operation the LLVM backend fails to lower correctly: we can write the PTX instruction directly. This safety valve does not exist in codegen_nvvm's abstraction layer.

5. **Empirical verification still needed**: Even if we switch to Rust-CUDA, the claim that `SystemAtomicU32` is correct must be verified (atomics.2 stress test). The problem does not disappear; it moves to a different abstraction.

### Recommendation: Maintain ADR-1 with explicit mitigation

Do not switch to codegen_nvvm (Option B) as the primary path. The benefits do not outweigh the complexity and supply-chain risk. Instead, expand ADR-1 with a documented sub-policy:

> **ADR-1 Amendment**: `core::sync::atomic` is prohibited in any code path that crosses the GPU-CPU boundary. All GPU-CPU synchronization must use the `gpu-atomics` crate which wraps LLVM NVVM intrinsics or inline PTX for system-scope operations. Intra-GPU synchronization within a block may use `core::sync::atomic` only for operations that are proven to be device-scope-safe (non-shared-memory, no CPU visibility required).

This is a narrower prohibition than abandoning nvptx64. It allows the vast majority of the codebase to use normal Rust atomics for intra-GPU logic, while carefully controlling the GPU-CPU boundary.

---

## 6. Revised Systems-Level Dependency Chain

The atomics finding forces a revision of the critical path from BS1:

```
[A] Toolchain validation (DONE: toolchain.4 — nvptx64 compiles and runs)
       |
[B'] gpu-atomics crate: system-scope intrinsics via llvm.nvvm.membar.sys +
       scoped atomicrmw via inline PTX or NVVM intrinsics  ← NEW BLOCKER
       |
[C] Hostcall protocol correct and race-free (depends on B', not B)
       |
[D] libc shim links correctly against -Zbuild-std=std
       |
[E] GlobalAlloc on GPU (bump allocator first)
       |
[F'] HostcallWaker: system-scope waker using gpu-atomics   ← NEW DEPENDENCY
       |
[F] IntraGpuWaker: RawWaker / Waker through PTX (fat pointer + indirect call)
       |
[G] async fn state machine compiles and round-trips without corruption
```

The new `[B']` step gates everything downstream. It is now the top priority before hostcall.3, async-runtime.2, or any other pending work.

---

## 7. Concrete New Tasks Recommended

### atomics.3 — Implement gpu-atomics crate (system-scope primitives)
- **Kind**: experiment
- **Depends on**: atomics.1 (done), toolchain.4 (done)
- **Goal**: Create `crates/gpu-atomics/` with:
  - `sys_membar()` → `llvm.nvvm.membar.sys`
  - `sys_atomic_store_u32(ptr, val)` → system-scope store via inline PTX (`st.relaxed.sys.global.u32`)
  - `sys_atomic_load_u32(ptr)` → system-scope load via inline PTX (`ld.relaxed.sys.global.u32`)
  - `sys_atomic_cas_u32(ptr, old, new)` → system-scope CAS via inline PTX (`atom.sys.global.cas.b32`)
- **Success**: PTX output inspected to confirm `.sys` scope qualifiers present on every operation.
- **Fallback**: If LLVM intrinsics don't work via `extern "C" #[link_name]`, use `core::arch::asm!` with inline PTX directly.

### atomics.2 — Stress test (existing, elevated priority)
- Now depends on atomics.3 as well as atomics.1 and toolchain.4.
- The stress test must specifically use `gpu-atomics` primitives, not `core::sync::atomic`.

### hostcall.3 — Update dependency
- Add `atomics.3` to `depends_on` list before proceeding.

---

## 8. Risk Matrix Update

| Risk | Severity | Likelihood | Change from BS1 | Mitigation |
|------|----------|------------|-----------------|------------|
| `core::sync::atomic` broken on nvptx64 | **Critical** | **Confirmed** | Promoted from Medium/Medium | Encapsulate in gpu-atomics crate; prohibit raw atomic use at GPU-CPU boundary |
| `atomic::fence()` silently dropped | **Critical** | **Confirmed** | New finding | Replace all fences at GPU-CPU boundary with `llvm.nvvm.membar.sys` |
| LLVM intrinsic `#[link_name]` naming unstable | Medium | Low | New risk | Fail mode is linker error (not silent bug); document exact LLVM version dependency |
| Hostcall protocol races without sys-scope CAS | High | High (if not fixed) | Raised severity | Fix with gpu-atomics.sys_atomic_cas_u32 |
| Waker signaling broken across CPU-GPU | High | High (if not fixed) | New analysis | HostcallWaker uses gpu-atomics; IntraGpuWaker may use narrower scope |
| Switching to Rust-CUDA introduces supply-chain risk | Medium | Medium | Evaluated and rejected | Maintain ADR-1 with amendment |
| LLVM NVPTX backend indirect call bugs | High | Medium | Unchanged | Test RawWaker dispatch early |
| PTX async state machine register overflow | Medium | High | Unchanged | Measure .reg early |

---

## 9. Summary: The Three-Sentence Systems Verdict

The atomics breakage is real and confirmed, but it is a bounded problem: it affects only the GPU-CPU communication layer, not intra-GPU logic. The correct fix is a dedicated `gpu-atomics` crate that replaces `core::sync::atomic` exclusively at that boundary using LLVM NVVM intrinsics or inline PTX, both of which are already accessible via `nvptx64` without abandoning ADR-1. Every other component — hostcall protocol, async executor waker, libc shim — can proceed correctly once `gpu-atomics` is implemented and empirically verified; until then, hostcall.3 and async-runtime.2 are blocked.

---

*Systems Programmer perspective for BS2. See bs2-compiler.md, bs2-gpu.md, bs2-skeptic.md for complementary analyses.*
