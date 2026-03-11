# Brainstorm BS1 — Devil's Advocate / Skeptic Analysis
**Date:** 2026-03-11
**Role:** Skeptic
**Seq:** 1

---

## 1. What VectorWare Isn't Telling Us

VectorWare's blog posts are marketing material first, technical documentation second. Here is what is conspicuously absent:

**The rustc fork is the elephant in the room.** The `extern "gpu-kernel"` ABI does not exist in upstream Rust. VectorWare's blog mentions it casually as if it were a standard annotation. It is not. This is a custom rustc modification, and without knowing its scope — a one-line ABI alias, or deep changes to codegen, calling conventions, and MIR passes — we cannot assess reproducibility. The blog posts do not link to any public fork, any RFC, or any upstream PR. This strongly implies their modifications are proprietary and non-trivial.

**The libc facade scope is understated.** "Reimplements libc calls as hostcalls" sounds minimal. The actual POSIX surface area that `std` touches is enormous: `open`, `read`, `write`, `close`, `mmap`, `munmap`, `brk`, `sbrk`, `mprotect`, `getcwd`, `getenv`, `setenv`, `pthread_*` (all of it), `sigaction`, `getpid`, `gettid`, signal handling, TLS (`__tls_get_addr`, `pthread_key_create`), `dlopen`/`dlsym`, and dozens more. VectorWare claims to have a "facade" but shows only `println!` and basic I/O in demos. The question is: what breaks when you try to use `HashMap`, `Mutex`, `Arc`, `Thread::spawn`, `panic!`, or anything that touches `pthread_mutex_*`?

**Performance numbers are absent.** There are no latency figures for a single hostcall round-trip. No occupancy measurements showing the actual impact of async state machine register pressure. No comparison against a CUDA C++ baseline. This is suspicious — if the numbers were good, they would be front and center.

**The miri verification claim is vague.** "Verified with miri" for what, exactly? Miri cannot model GPU memory semantics, warp divergence, or CUDA memory consistency. Miri runs on the interpreter level for a single-threaded model. Verifying a hostcall protocol that relies on cross-device atomic operations with miri is, at best, verifying the host-side Rust logic in isolation. It proves nothing about the GPU-side correctness.

**CUDA toolkit version dependency is hidden.** The `nanosleep` intrinsic used in the spin-loop executor requires CUDA 11.x+ and specific SM architectures. The blog does not specify which NVIDIA driver version, CUDA toolkit version, or minimum SM level is required. This matters enormously: a setup targeting SM 8.0 (Ampere) may behave differently than SM 7.0 (Volta) for memory ordering.

**No mention of warp divergence in async.** Every async state machine has branches (`match` on the state discriminant). On GPU, branching across threads in a warp causes divergence and serialization. The blog treats GPU threads as if they were independent CPU threads. They are not. 32 threads in a warp that each hold a different async state will serialize every state transition.

---

## 2. Hidden Assumptions in Our Plan

**Assumption: `nvptx64-nvidia-cuda` can be extended with `-Zbuild-std=std`.** This is almost certainly false without the libc shim already in place. The standard library links against `libc`, which on `nvptx64` does not exist. The build will fail during linking before you get anywhere near running code. Our plan assumes `gpu-std` can follow `toolchain` sequentially, but in reality these are tightly coupled — you cannot build `std` for nvptx64 without simultaneously solving the libc problem.

**Assumption: The hostcall design can be done cleanly in isolation.** Our `hostcall.3` (design) depends on `hostcall.1` and `hostcall.2` (investigations), but the design is fundamentally constrained by the async executor architecture from `async-runtime`. If the executor uses a per-warp model, hostcall synchronization is warp-scoped. If it is per-thread, the synchronization domain changes. These decisions are interdependent, but our dependency graph treats them as independent.

**Assumption: Embassy can be ported to GPU with moderate effort.** Embassy was designed for single-core embedded processors. Its critical section model assumes a single hart or single interrupt domain. On a GPU, you have potentially thousands of concurrent "threads" (more accurately: thousands of warps, each running SIMT). Embassy's `CriticalSection` and `GlobalSyncExecutor` have no concept of this. The port is not a "tweak" — it may require a ground-up redesign of the task queue and waker mechanism.

**Assumption: `alloc` is easier than `std`.** The plan assumes we can get `alloc` working relatively early (it is implicit in many tasks). But `alloc` requires a `GlobalAllocator`, which on GPU means either: (a) a bump allocator in shared memory with no deallocation, (b) a GPU-side heap that is non-trivial to implement correctly under warp divergence, or (c) routing `malloc`/`free` through hostcall with terrible latency. None of these are "easy." Option (c) alone would make any `Box<T>` allocation take microseconds, making the entire async state machine model questionable.

**Assumption: Integration is "just" combining components.** Our integration theme treats end-to-end combination as the last step after everything else is done. But integration failures are the most likely source of show-stoppers: symbol resolution conflicts between the libc shim and CUDA runtime libraries, stack size blowup when async state machines are nested, linker issues with multiple `.ptx` sections and `extern "gpu-kernel"` boundaries, and ABI mismatches between the host-compiled shim and the device-compiled executor.

**Assumption: VectorWare's approach is reproducible from blog posts alone.** This is the most dangerous assumption of all. Reproducibility of a system this complex from blog posts — which omit source code, compilation flags, CUDA version requirements, and the actual rustc patch — may be impossible. We may be reverse-engineering, not reproducing.

---

## 3. Toolchain Risks

**The `gpu-kernel` ABI is not in upstream Rust.** As of the knowledge cutoff, `extern "ptx-kernel"` exists in nightly but `extern "gpu-kernel"` does not. If VectorWare's system depends on `gpu-kernel` ABI semantics that differ meaningfully from `ptx-kernel` — different calling conventions, different register assignment, different handling of kernel arguments — then we cannot reproduce their system without their private rustc patch. `toolchain.3` investigates this, but needs to explicitly plan for the outcome: "no public patch exists → we must implement it ourselves → project scope just doubled."

**The LLVM PTX backend has known, unfixed bugs.** The LLVM PTX backend is known to: miscompile certain loop structures, generate incorrect PTX for some SIMD-like operations, and have corner cases around address space inference. VectorWare's blog mentions they "fixed bugs across multiple compiler backends + ptxas." This is a soft admission that they hit these bugs and had to work around them — but they do not say which bugs or which workarounds. We will hit the same bugs and have to discover the workarounds independently.

**`-Zbuild-std` with nightly is a moving target.** The nightly Rust toolchain changes daily. A compilation pipeline that works today may silently break tomorrow due to an unrelated nightly change. Our workflow needs to pin a specific nightly version and document exactly which nightly works. Without version pinning, our experimental results will be unreproducible.

**rust-cuda (codegen_nvvm) may be more practical but is also more fragile.** The `rustc_codegen_nvvm` backend is a third-party codegen that bypasses LLVM's PTX backend entirely and goes through NVVM IR. It supports more CUDA-specific features but: (a) it is not maintained in lockstep with upstream rustc, (b) it requires a specific LLVM/CUDA version pairing, and (c) VectorWare's approach appears to use upstream rustc with nvptx64, not codegen_nvvm. Investigating both is the right call, but the plan should acknowledge they are not drop-in alternatives — they have fundamentally different feature sets and stability profiles.

---

## 4. Hostcall Pitfalls

**Double-buffering does not solve all races.** VectorWare describes a double-buffer + atomic scheme. The critical failure mode not discussed: what happens when the GPU produces a second hostcall request before the host has consumed and responded to the first? If the design uses a two-slot buffer (double-buffer), the second request either blocks (the GPU warp spins waiting for a free slot) or overwrites the first (data loss). Under heavy load — which is normal when 1024 GPU threads are all issuing hostcalls — either outcome is catastrophic. A proper ring buffer is needed, and its size must be carefully chosen as a function of expected GPU parallelism.

**Deadlock between GPU spin and CPU response.** If the GPU thread (warp) spins waiting for a hostcall response, and the CPU host thread is blocked for any reason — OS scheduling jitter, lock contention in the host's I/O library, garbage collector pause if the host is JVM-based, system call delay — the GPU kernel will spin indefinitely. CUDA kernels have a hard watchdog timeout (default 2 seconds on display-attached GPUs). A hostcall that takes more than this will cause the GPU to hang and the driver to kill the kernel. Our design must either: (a) guarantee sub-millisecond host response for all hostcall types, or (b) use a compute-mode GPU (no display) with the watchdog disabled.

**Memory ordering is severely underspecified.** PTX has a complex memory consistency model (PTX ISA 7.x+ with explicit acquire/release scopes). The scopes are: `.cta`, `.cluster`, `.gpu`, `.sys`. A GPU-CPU atomic communication requires `.sys` scope. Using the wrong scope — e.g., `.gpu` scope on a read/write pair — means the CPU may never see the GPU's write. Rust's `core::sync::atomic` does not map cleanly to these scopes on nvptx64. The actual assembly emitted for `Ordering::SeqCst` on nvptx64 may not use `.sys` scope. This needs to be verified at the PTX level, not assumed from the Rust source.

**Multiple-warp serialization destroys throughput.** Even with a correct ring buffer, if multiple warps compete for hostcall slots, they must serialize at the slot-acquisition step. In a kernel with 1024 threads (32 warps), if all warps attempt a hostcall simultaneously, 31 warps are spinning while 1 runs. The effective throughput of the hostcall mechanism is limited to 1 concurrent call per buffer slot. This is not a flaw to fix — it is a fundamental property of the design — but it must be measured and documented. The VectorWare blog implies hostcall is "transparent," which obscures this serialization cost.

**The host-side polling thread has its own hazards.** The host must poll shared memory for new requests. Polling too slowly wastes GPU time (the warp spins longer). Polling too aggressively wastes CPU time (busy-wait loop). The optimal polling strategy is workload-dependent and cannot be determined statically. Adaptive polling (slow → fast on activity) adds complexity and introduces its own timing hazards.

---

## 5. GPU-std Fantasy vs Reality

**Features that are feasible (with significant engineering):**
- `println!` via hostcall write
- `std::fs::File::create`, `write`, `read`, `close` via hostcall fd operations
- `std::env::args`, `std::env::var` via hostcall (read-once at kernel start)
- `Box<T>` with a GPU-side bump allocator (no deallocation)
- `Vec<T>` in global memory (if the allocator is pre-sized)

**Features that are marginal (require deep hacks):**
- `HashMap` — needs a working allocator with reallocation. Bump allocators cannot `realloc`. This breaks HashMap's growth behavior.
- `String` — feasible if allocator works, but string formatting (`format!`) invokes `write_fmt` which has complex lifetimes and may hit stack frame issues.
- `std::io::BufReader`/`BufWriter` — requires heap allocation for the internal buffer.
- `Arc<T>` — requires atomic reference counting. GPU atomics work differently (no `fetch_add` with `Release` ordering in all PTX variants). Need to verify the actual assembly.

**Features that are pipe dreams:**
- `std::thread::spawn` — GPU does not have OS threads. You cannot create a new CUDA thread from within a GPU kernel. Full stop.
- `std::sync::Mutex`, `RwLock`, `Condvar` — these ultimately call `pthread_mutex_*`. There are no POSIX mutexes on GPU. You could implement a spinlock, but then `Mutex::lock()` semantics (blocking sleep, not spinning) are broken.
- `std::net` (networking) — requires socket syscalls. Even if routed through hostcall, the latency makes this useless for GPU workloads.
- `std::process::exit` — calling exit from one GPU thread exits what? The kernel? All threads? The host process? The semantics are undefined.
- **Panic unwinding** — `std::panic::catch_unwind` and stack unwinding via DWARF are not supported in PTX. Panics must terminate the thread (or the kernel). Any `std` code that assumes `?` propagation through `Result` hits a potential hidden `unwrap_or_else(|e| panic!(...))` that silently aborts with no useful diagnostic.

**The TLS problem is critical and unaddressed.** Many `std` internals use thread-local storage (`#[thread_local]`, `std::thread::LocalKey`). On GPU, "thread-local" is ambiguous — does it mean per-CUDA-thread, per-warp, per-block? LLVM's PTX backend can emit `addrspace(0)` for thread-local data but the semantics are not the same as CPU TLS. `errno` in libc is TLS-based. If the libc shim does not correctly handle TLS, errno-setting functions will either crash or produce wrong values silently.

---

## 6. Async on GPU: The Hard Problems

**Warp divergence in state machine polling.** An async state machine compiles to a `match` on a state discriminant. If 32 threads in a warp each hold a different state (because they are at different points in execution), every branch of the `match` executes serially with the other branches masked off. This is not an "async overhead" problem — it is a fundamental incompatibility between async's inherent divergent control flow and SIMT execution. The performance could be worse than a naive synchronous kernel in pathological cases.

**The executor placement problem is a real architectural decision, not an implementation detail.** `async-runtime.2` asks "one executor per warp or per block?" but understates the consequences:
- Per-thread executor: Every thread runs its own executor. If tasks have shared state, that state must be in global memory with atomic access. Register pressure is multiplied by the number of concurrent tasks per thread.
- Per-warp executor: All 32 threads in the warp execute the same task at the same point. This is the SIMT model, but it means no intra-warp async concurrency — the whole warp is stuck waiting if any task awaits.
- Per-block executor: Requires intra-block synchronization (`__syncthreads()`). This is a barrier that stalls all threads in the block, defeating the purpose of async.
- There may be no good answer. The task should surface this as a potential design dead-end.

**Wakers on GPU are fundamentally broken as an abstraction.** Embassy's waker system is built around the concept of "wake me up when this is ready." On a CPU, "wake up" means: put this task back in the run queue, and the executor will re-poll it. On a GPU, the "executor" is just a spin loop. There is no OS scheduler to notify. A `wake()` call on GPU reduces to a no-op or a memory write that the spin loop will notice on the next iteration. This works, but it means every "await" point is a spin-wait. If the awaited event takes 1 millisecond (e.g., a hostcall), the warp spins for 1 millisecond burning registers and blocking other warps from using those execution units.

**Register pressure from async state machines is a hard occupancy cliff.** CUDA's occupancy model is based on registers per thread. An SM on modern NVIDIA GPUs has 65,536 registers. If each thread uses 32 registers, you get 2048 threads (the maximum). If async state machines push per-thread register usage to 64, occupancy halves to 1024 threads. If futures_util combinators (Select, Join, Race) nest multiple futures, register usage could reach 128+ per thread, cutting occupancy to 512 or less. This is not a theoretical concern — VectorWare's blog explicitly mentions it. But they do not give actual numbers, making it impossible to assess severity.

**`nanosleep` is not a real sleep.** The `nanosleep` PTX intrinsic does not yield the warp to another runnable warp. It adds a delay of approximately N nanoseconds to the warp. If the host has not responded to a hostcall within that time, the warp simply wakes up and polls again. It is a power-reduction hint, not cooperative scheduling. The spin loop still runs, it just runs slightly less frequently. Calling this "async" is misleading — there is no preemptive scheduling or cooperative yield to other GPU work happening here.

---

## 7. Integration Complexity

**Symbol resolution is a minefield.** When `-Zbuild-std=std` is used with the nvptx64 target, `std` is compiled as a GPU-side library. But `std` expects a libc. Our libc shim provides the libc symbols. However, CUDA's runtime library (`libcuda`) may also define symbols that conflict with libc shim symbols (e.g., `malloc`, `free` are defined in both CUDA device runtime and libc). The linker behavior when multiple definitions of `malloc` exist in the GPU-side link is unspecified and toolchain-dependent.

**The async executor needs `alloc`, which needs a working allocator.** The async state machine itself is stack-allocated (the Future type), but Embassy's task queue uses heap allocation internally (task storage, `Box<dyn Future>`). If we use a bump allocator for GPU heap, the task queue cannot free completed tasks. This means: (a) we need a fixed maximum number of concurrent tasks known at compile time, or (b) we redesign the task queue to use static storage. Either option requires modifications to Embassy that go beyond a simple port.

**Panic semantics across the GPU/CPU boundary.** If a GPU-side Rust function panics (e.g., index out of bounds), and panics are configured to `abort`, the CUDA thread terminates. But the host may be blocked waiting for a hostcall response that will never come. The host needs a timeout mechanism for every hostcall, and the kernel launch completion check must distinguish between "kernel completed normally" and "kernel aborted due to panic." This error propagation design is not covered in any current task.

**Debug experience is nearly zero.** When something goes wrong in a GPU kernel — wrong output, hang, crash — the debugging tools are: `cuda-gdb` (which barely works for complex kernels), `printf` (which is what we are trying to implement), and NVIDIA Nsight (which requires a GUI and may not integrate with our Rust workflow). If the hostcall implementation is wrong, the first symptom will be a silent hang or corrupted output. There is no stack trace, no panic message, no backtracing. Every bug in the early phases of this project will take 10x longer to diagnose than an equivalent CPU bug.

---

## 8. Show-Stoppers

**Show-Stopper 1: `gpu-kernel` ABI is unavailable without a private rustc patch.**
If `toolchain.3` investigation reveals that `gpu-kernel` ABI requires a proprietary rustc modification with no public equivalent, and if that ABI provides semantics that `ptx-kernel` cannot emulate, then the entire project depends on either (a) implementing the rustc patch ourselves, or (b) finding that ptx-kernel is sufficient. Option (a) is a multi-month compiler engineering effort. If neither alternative is viable, the project cannot reproduce VectorWare's exact technology.

**Show-Stopper 2: Warp divergence makes async on GPU impractical for real workloads.**
If the profiling in `async-runtime.3` reveals that async state machines running in SIMT context consistently cause 8x or greater performance degradation compared to synchronous equivalents — due to divergence + register pressure — then "async on GPU" is a novelty, not a useful technology. The project would technically succeed in reproducing VectorWare's work but produce something no real project would use.

**Show-Stopper 3: Memory ordering in PTX is not safely expressible from Rust.**
If investigation reveals that `core::sync::atomic` on nvptx64 does not emit `.sys` scope atomics (required for GPU-CPU communication), and there is no stable way to force `.sys` scope from Rust without inline PTX assembly, then the hostcall protocol is fundamentally unsafe. Every "working" implementation would be relying on undefined behavior in the memory model, with races that appear to work under light testing but fail unpredictably under load.

**Show-Stopper 4: `-Zbuild-std=std` for nvptx64 is simply broken in current nightly.**
The nvptx64 target has notoriously spotty support for std features. If current nightly Rust cannot compile even a minimal std for nvptx64 without internal compiler errors or linker failures — which has historically been true — then `gpu-std` is blocked until someone upstream fixes the toolchain. This is outside our control.

---

## 9. Missing Themes

**Missing Theme: Memory Model and Atomics Verification.**
Every correctness argument for hostcall and async depends on correct memory ordering. This deserves its own theme: verifying that Rust atomic operations on nvptx64 emit the correct PTX memory scope qualifiers, and establishing a testing methodology for GPU-CPU memory ordering. Without this, all hostcall and async work is built on an unverified foundation.

**Missing Theme: Debugging and Observability Infrastructure.**
Currently, no theme addresses how we will debug GPU kernels. This is not a minor concern — it is a force multiplier for all other themes. A `gpu-dbg` theme covering: PTX-level printf substitutes, CUDA assertions that surface to the host, integration with cuda-gdb, and a strategy for diagnosing hangs vs crashes would save enormous time in later experimental phases.

**Missing Theme: Performance Baseline and Regression Tracking.**
We have no plan for measuring whether our implementations are acceptably performant. The integration theme mentions benchmarks but only at the end. We need a performance theme that establishes: what is a reasonable hostcall latency target? What is acceptable occupancy loss from async? Without a baseline, we will not know when we have regressed or whether our design choices are costing us performance we do not have to spend.

**Missing Theme: Error Handling and Fault Tolerance.**
What happens when a GPU kernel panics? What happens when a hostcall times out? What happens when the host-side polling thread crashes? There is no theme addressing the fault model of the overall system. This will become critical during integration and any real-world use.

---

## 10. Priority Critique

**The current ordering is too sequential and creates a single critical path.** The dependency chain `toolchain.4 → hostcall.1 → hostcall.4 → gpu-std.1 → gpu-std.2 → gpu-std.3 → integration.1` means that a single blocked task halts the entire project. If `toolchain.4` takes two weeks to get working (likely), nothing else meaningful can start. The plan should identify which tasks can be parallelized by relaxing their dependencies.

**`toolchain.3` (VectorWare's rustc modifications) should be the first task, not concurrent with toolchain.1 and toolchain.2.** If toolchain.3 reveals that the entire approach requires an unpublished rustc patch, the scope of toolchain.1 and toolchain.2 changes dramatically. Currently, toolchain.3 has no dependents — meaning its findings are not gating anything. This is backwards. The output of toolchain.3 should determine which toolchain path (nvptx64 upstream, codegen_nvvm, or "we must patch rustc") to pursue. It should be the first task completed, not one of four concurrent investigations.

**`hostcall.2` (studying AMD ROCm hostcall) should have higher priority.** The AMD ROCm hostcall implementation is public and well-documented. It solves the same problem we are trying to solve. Studying it deeply before designing our own (hostcall.3) would save significant design effort. Currently both are at the same priority level, but hostcall.2 is strictly an input to hostcall.3's design quality.

**`async-runtime.1` (Embassy analysis) is correctly prioritized, but its scope is too narrow.** The investigation should also cover: what changes are required to the Embassy source to compile for nvptx64, which Embassy abstractions are fundamentally incompatible with SIMT, and whether a completely custom executor would be simpler than porting Embassy. Porting a complex existing executor to a radically different execution model (SIMT vs single-core embedded) is often harder than writing a simpler custom executor from scratch.

**Performance measurement is missing from all early tasks.** No investigation or experiment task has "measure the performance" as a required output before `integration.1`. We could build a working system that is 1000x slower than native CUDA and not notice until the final phase. Performance measurement should be a requirement of `hostcall.4` (measure round-trip latency) and `async-runtime.3` (measure occupancy and register usage), not deferred to integration.

---

## Summary of Highest-Risk Items

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gpu-kernel` ABI requires private rustc patch | High | Fatal | Investigate first; have ptx-kernel fallback plan |
| Warp divergence kills async performance | High | Severe | Benchmark early in async-runtime.3 |
| PTX memory ordering not safely expressible from Rust | Medium | Fatal | Add atomics verification as new theme |
| `-Zbuild-std=std` for nvptx64 is broken | Medium | Severe | Pin nightly version; test immediately in toolchain.4 |
| Hostcall deadlock under GPU watchdog timeout | Medium | Severe | Design with explicit timeout from day one |
| TLS semantics broken for GPU libc shim | High | Moderate | Audit errno and thread-local usage in gpu-std.1 |
| Integration symbol conflicts (libc shim vs CUDA runtime) | High | Severe | Prototype linking before gpu-std work matures |
