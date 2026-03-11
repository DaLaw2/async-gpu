# Brainstorm BS1 — Systems Programmer Perspective
**Role**: Rust Systems Programmer
**Date**: 2026-03-11
**Focus**: Memory models, ABI compatibility, unsafe boundaries, FFI, low-level systems concerns

---

## 1. Feasibility Assessment

The core question is whether the Rust type system and runtime model can be mapped onto GPU hardware. The answer is: **yes, but with well-defined compromises at every layer**.

### Hard Systems-Level Challenges

**Stack allocation discipline**
GPU threads have a fixed, shallow call stack managed by the hardware (PTX `call` instruction). Rust's default assumption of a growable or at least unbounded stack is unsafe here. Any recursive code, deeply nested futures, or large local variables will overflow or cause register spill to slow local memory. Every `async fn` state machine captures the full coroutine frame — this must fit in registers or be spilled to global memory, raising pressure.

**No OS primitives**
Rust `std` relies on OS abstractions: threads, file descriptors, mutexes backed by `futex`, time via `clock_gettime`. On GPU, none of these exist in the kernel context. The hostcall mechanism is the only escape valve. Any `std` API that reaches a syscall boundary must either be redirected through hostcall or stubbed. There is no partial option: a missing stub causes either a link error or silent UB.

**Atomics scope mismatch**
CUDA hardware provides atomics at multiple scopes: thread, warp, block (CTA), device, and system. Rust's memory model (`core::sync::atomic`) has only one scope — sequentially consistent through "system". Mapping `Ordering::SeqCst` to `atom.sys` PTX instructions is safe but may be unnecessarily expensive. Mapping it to narrower scopes requires custom intrinsics or inline PTX, bypassing Rust's safe atomic API entirely. The hostcall shared memory buffer requires system-scope atomics (`atom.sys`) for GPU-CPU visibility; ordinary `atom.gpu` will not be visible to the host.

**No dynamic dispatch support (initially)**
`dyn Trait` requires fat pointers and vtables. The LLVM PTX backend does support function pointers and indirect calls through PTX `call` with register operands, but this is an area where both the LLVM backend and ptxas have historically had bugs. The `RawWaker` used by `std::task::Waker` is a fat pointer pair (data pointer + vtable pointer). Whether this survives the PTX backend correctly is a critical empirical question for async-runtime.

**No TLS (Thread-Local Storage)**
Rust `std` uses `thread_local!` macros extensively for per-thread state (e.g., panic hooks, allocator state). GPU has no TLS in the POSIX sense. PTX has per-thread registers and per-thread local memory, but they are not addressable via the same TLS mechanism Rust uses. Any `std` code that reads `__rust_no_alloc_shim_is_unstable` or other TLS-backed globals will fail.

**Panic unwinding**
`panic = "abort"` is mandatory. Unwinding on GPU is impossible — there is no exception handling mechanism. The rustc target spec for `nvptx64-nvidia-cuda` forces `panic = "abort"`, which is correct. However, `std` pulls in panic infrastructure. `-Zbuild-std` with careful feature flags must be used to exclude unwinding. The `panic_handler` attribute must be defined in the GPU crate.

---

## 2. Memory Model Concerns

### GPU vs CPU Memory Ordering

CUDA's memory model is a relaxed consistency model with explicit scoping. The key document is the PTX ISA chapter on memory consistency model (CUDA 11+). Key points:

- **Within a warp**: instructions execute in lock-step. Intra-warp ordering is implicit.
- **Within a CTA (block)**: `membar.cta` or `bar.sync` provides a barrier.
- **Device-wide**: `membar.gl` provides a device-level fence.
- **System-wide (GPU+CPU)**: `membar.sys` is required for any data shared with the host.

Rust's `Ordering::SeqCst` compiles to `fence seq_cst` in LLVM IR, which the PTX backend must lower to `membar.sys`. Whether this is done correctly depends on the LLVM version and the specific backend (LLVM NVPTX vs codegen_nvvm). **This must be empirically verified** — incorrect lowering is a silent correctness bug.

For the hostcall shared buffer:
- The GPU writer must use `atom.sys.release` when signaling the host.
- The GPU reader must use `atom.sys.acquire` when reading the response.
- Rust's `store(val, Ordering::Release)` / `load(Ordering::Acquire)` pair should map correctly to these, but the scope (`.sys` vs `.gl`) must be confirmed.

### Cache Coherence

GPU L1 cache is not coherent with the CPU. For host-mapped pinned memory (`cudaHostAllocMapped`), CUDA guarantees coherence only when correct fencing is used. Specifically:
- GPU must issue `membar.sys` before the host can observe writes.
- Host CPU store-load ordering must also include a `mfence` or equivalent on x86.
- Without these fences, the GPU may see stale data from the host response.

### Volatile Semantics

Rust's `std::ptr::read_volatile` / `write_volatile` do not substitute for proper atomic ordering. They prevent compiler reordering but not hardware reordering. The hostcall buffer polling loop must use atomics, not volatile reads.

### Shared Memory (SMEM) vs Global Memory (GMEM)

SMEM is on-chip, shared within a CTA, very fast (similar to L1 cache but software-managed). GMEM is off-chip, device-global, cached through L2. For the async executor task queue:
- If the executor is per-block, SMEM is appropriate for the task queue and waker flags.
- Cross-block communication requires GMEM or a hostcall.
- Putting executor state in SMEM requires knowing its size at compile time (or using dynamic SMEM with `extern __shared__`).

---

## 3. ABI & FFI Boundaries

### ptx-kernel ABI vs gpu-kernel ABI

The current stable Rust `extern "ptx-kernel"` ABI has significant limitations:
- Parameters are passed by value only. No references or pointers as top-level kernel arguments in the ABI — they must be passed inside a struct (pointer-to-struct pattern).
- No return value. The kernel function signature must return `()`.
- The calling convention maps kernel parameters to PTX `.param` space, which is separate from register space and must be explicitly loaded with `ld.param`.

VectorWare's `extern "gpu-kernel"` ABI appears to extend or replace this with a richer interface that supports references in kernel signatures. This is not yet in upstream Rust as of the research date. The PR/RFC status must be tracked (toolchain.3 task).

**Practical impact**: Until `gpu-kernel` ABI is available, pointer arguments must be passed as raw `u64` addresses or packed into a struct, then `transmute`d inside the kernel. This is `unsafe` but manageable.

### Data Layout

Rust does not guarantee struct layout by default (`repr(Rust)` is unspecified). Any struct that crosses the GPU-CPU boundary (hostcall buffer entries, kernel arguments) must be `#[repr(C)]`. This includes:
- Hostcall request/response structs
- Any shared memory data structures visible to both GPU and CPU code
- Waker vtable function pointer tables (if using `dyn Waker` patterns)

`#[repr(C)]` is not sufficient alone for pointer-sized integers: `usize` is 64-bit on both x86_64 and nvptx64, so this is not a hazard here, but must be documented.

### Alignment

GPU hardware enforces alignment for vectorized loads. A 128-bit load (used for float4 or u64x2) requires 16-byte alignment. If the hostcall buffer uses 128-bit atomic operations (e.g., 128-bit CAS for lock-free double-buffering), alignment must be explicitly enforced with `#[repr(C, align(16))]` on the buffer struct. Misalignment causes a hardware exception that manifests as a silent kernel abort on older CUDA versions.

### Calling Conventions for Device Functions

Non-kernel device functions (called within the GPU) use the PTX function call convention, which is similar to the C ABI on NVPTX. Rust functions marked `#[inline(never)]` in GPU code will use this convention. Key constraint: the PTX backend has historically mishandled certain return types (e.g., `Option<NonNull<T>>` where null optimization applies). Prefer `#[repr(C)]` wrapper enums at FFI-adjacent call sites.

### libc Facade Linkage

The libc shim will define `extern "C"` functions with names matching libc symbols (`write`, `open`, `read`, etc.). These must be linked with `--allow-multiple-definition` or by controlling link order, or the PTX linker must be made aware that these are GPU-local definitions. The `-Zbuild-std=std` path patches std to use the target's libc, which on `nvptx64` will resolve to whatever symbols are provided in the GPU crate. The exact linker behavior here must be verified (gpu-std.2 task).

---

## 4. Unsafe Boundaries

### Where Unsafe is Unavoidable

1. **Shared memory initialization**: PTX shared memory starts uninitialized. Accessing it before writing is UB. Any `MaybeUninit<T>` wrapper must be used, and initialization must be sequenced through a `__syncthreads()` equivalent (barrier intrinsic).

2. **Hostcall buffer access**: The shared buffer is accessed concurrently by GPU threads and the CPU host thread. From Rust's perspective, this is a raw pointer shared across threads. The safe API must wrap it in a struct with the appropriate atomic accessor methods. The underlying `*mut u8` manipulation is unavoidably `unsafe`.

3. **PTX intrinsics**: `core::arch::nvptx` intrinsics are `unsafe fn`. Any use of `__syncthreads()`, `__ballot_sync()`, `__activemask()`, or barrier primitives requires `unsafe`. There is no safe abstraction for GPU synchronization primitives.

4. **RawWaker construction**: `std::task::RawWaker::new(data: *const (), vtable: &'static RawWakerVTable)` takes a raw pointer. The executor must manage this pointer's lifetime manually.

5. **Global allocator**: Implementing `GlobalAlloc` for GPU requires `unsafe impl GlobalAlloc`. The allocator itself must use `unsafe` pointer arithmetic and atomics.

6. **Kernel entry point**: The `#[no_mangle] extern "ptx-kernel"` or `extern "gpu-kernel"` function is the unsafe boundary between host CUDA code and Rust GPU code. All invariants must be upheld at this boundary.

### Minimizing Unsafe Surface

- Define a `GpuSharedBuffer<T>` wrapper type that exposes only atomic load/store/CAS operations, hiding the raw pointer inside. Mark the type `!Send` and `!Sync` to prevent accidental cross-warp sharing without explicit unsafe.
- Create a `#[gpu_safe]` proc-macro (or just careful modules) that ensures all synchronization paths go through barrier wrappers.
- Use `#[forbid(unsafe_code)]` on higher-level modules (the async task code, the std facade API layer) to enforce the boundary.
- Audit all uses of `transmute`: prefer `bytemuck::cast` where the trait bounds make the safety obvious, or write explicit `unsafe` blocks with justification comments.

---

## 5. Critical Path Analysis

From a systems perspective, the blockers form a strict dependency chain:

```
[A] Toolchain validation (nvptx64 or codegen_nvvm compiles and runs)
       |
[B] System-scope atomics confirmed correct (memory model empirical verification)
       |
[C] Hostcall protocol correct and race-free (shared buffer, CPU polling loop)
       |
[D] libc shim links correctly against -Zbuild-std=std
       |
[E] GlobalAlloc on GPU (required by std, required by async executor for heap)
       |
[F] RawWaker / Waker works through PTX backend (fat pointer + indirect call)
       |
[G] async fn state machine compiles and round-trips through ptxas without corruption
```

**[A] and [F]** are the highest-risk items. If either fails, the entire project must reassess toolchain or ABI strategy.

**[B]** is silent-failure risk. Wrong memory ordering compiles and runs but produces incorrect results under concurrency. Must be tested with a specific stress test: two GPU threads increment a counter via hostcall, CPU verifies count is exactly N. Any value below N indicates a missed fence.

**[E]** is required earlier than it appears: `std::io::BufWriter` and many other std types heap-allocate. Without a working GPU allocator, `-Zbuild-std=std` will link but panic at runtime on any allocation attempt. A minimal bump allocator in GPU global memory (with a static size limit) should be implemented first.

---

## 6. Recommendations

### Theme: toolchain

- **Start with nvptx64**, not codegen_nvvm. The built-in target requires no additional toolchain setup and the LLVM NVPTX backend is more actively maintained. codegen_nvvm (libnvvm) has lagged on newer LLVM IR features.
- Verify LLVM version: Rust 1.7x+ uses LLVM 17/18. Confirm the NVPTX backend in that version handles `atomicrmw` at system scope, indirect calls through function pointers, and `landingpad`-free builds.
- Compile with `--emit=asm` and inspect PTX output for: correct `.visible .entry` annotation, absence of `__device__`-only intrinsics that the host linker cannot resolve, and correct `ld.param` sequences.
- For `gpu-kernel` ABI: monitor https://github.com/rust-lang/rust for PRs touching `compiler/rustc_target/src/abi/call/nvptx64.rs`. If not yet merged, patch locally against a pinned rustc nightly.

### Theme: hostcall

- Model the protocol after AMD ROCm's hostcall: a ring buffer with per-slot state machine (`FREE` → `PENDING` → `BUSY` → `DONE` → `FREE`). This is more scalable than a single double-buffer when multiple warps are active.
- Use `u32` or `u64` atomics for slot state, not `bool` fields — PTX has no byte-level atomic operations.
- The slot payload must be `#[repr(C, align(16))]` to support 128-bit atomic CAS on the slot header.
- Host polling loop: spin with `std::hint::spin_loop()` on the slot state. Do not use `SIGINT` or futex-based sleep — latency must be minimized as GPU threads spin-wait for the response.
- Define a fixed opcode table (e.g., `OP_WRITE = 1`, `OP_OPEN = 2`) and version the protocol with a header magic number to detect ABI mismatch at runtime.

### Theme: gpu-std

- The minimal libc facade needs to implement at minimum: `write`, `read`, `open`, `close`, `lseek`, `fstat`, `mmap`/`munmap` (or just `malloc`/`free` backed by a GPU allocator), `exit`, `abort`, `memcpy`, `memset`, `strlen`.
- Functions that can safely return an error without hostcall: `getpid` (return 1), `getenv` (return null), `isatty` (return 0).
- `errno` must be a per-thread global. On GPU, this means a per-thread variable in PTX `.local` space, accessed via a `get_errno() -> *mut i32` function. Define it with `#[thread_local]` if supported by the backend, or via inline PTX.
- For `GlobalAlloc`: implement a lock-free slab allocator in GPU global memory. A bump allocator with no `dealloc` is sufficient for initial testing. Size it statically (e.g., 16 MiB) allocated by the host and passed as a kernel parameter.

### Theme: async-runtime

- **Do not start with Embassy's full executor.** Start with a manually-written `block_on` that calls `poll()` in a spin loop. This requires zero infrastructure and proves the state machine compiles correctly.
- Verify `RawWaker` function pointer dispatch works through PTX. Write a test that calls a `RawWaker`'s `wake` method and confirms a flag was set. If this fails, indirect function calls through fat pointers are broken and must be worked around (e.g., by encoding the waker as an integer index into a static dispatch table).
- For the full executor: one executor per warp is the natural granularity. Warp threads are in lock-step, so a single task queue per warp is safe to access without atomics within-warp. Cross-warp task queues require GMEM atomics.
- Register pressure is the primary performance constraint. Measure `.reg` directive in PTX output. If a future state machine uses >128 registers, it will reduce occupancy to 50% or less on most hardware. Consider manually splitting large futures into smaller ones with explicit yield points.
- The `nanosleep` polling in VectorWare's implementation requires PTX `nanosleep` instruction (available on Turing+, SM75+). For Maxwell/Pascal compatibility, a spin loop counter is the fallback.

### Theme: integration

- Build integration incrementally: first `async fn` that calls `println!` (hostcall write), then `async fn` with file I/O, then multiple concurrent tasks.
- The correctness test for concurrent hostcalls is: N async tasks each do `File::create` with a unique filename, `write`, `close`. Host verifies all N files exist with correct content. Any racing bug in the hostcall protocol will manifest as missing or corrupted files.
- Performance benchmark: compare the async hostcall path against synchronous equivalent. Expect 10x–100x overhead per hostcall compared to native GPU memory operations. The goal is correctness, not performance parity.

---

## Summary: Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| LLVM NVPTX backend bugs with indirect calls | High | Medium | Test with simple vtable dispatch early; have codegen_nvvm fallback |
| System-scope atomics lowered incorrectly | High | Medium | Empirical stress test before building higher layers |
| PTX async state machine register overflow | Medium | High | Measure .reg early; split futures if needed |
| libc shim link order issues with -Zbuild-std | Medium | Medium | Study existing no_std + custom allocator crates for precedent |
| gpu-kernel ABI not upstreamed | Medium | Medium | Patch local rustc nightly; document patch for reproducibility |
| nanosleep unavailable on target GPU | Low | Medium | Spin loop fallback; document minimum SM version |

---

*This analysis covers the systems-programmer lens. Other perspectives (compiler internals, distributed systems, proof-of-concept prioritization) are addressed in sibling brainstorm files.*
