# Brainstorm bs1 — Rust Compiler Engineer Analysis
**Role:** Rust Compiler Engineer (rustc, LLVM, codegen)
**Date:** 2026-03-11
**Sequence:** bs1

---

## 1. Compilation Targets: nvptx64-nvidia-cuda vs. Custom Codegen

### nvptx64-nvidia-cuda (Built-in LLVM PTX Backend)

The `nvptx64-nvidia-cuda` target is a Tier 3 target in the upstream rustc. It routes through
LLVM's PTX backend (`lib/Target/NVPTX/`), which has been present in LLVM since version 3.x. The
target triple maps to LLVM IR that is then lowered to PTX assembly by the NVPTX LLVM backend.

**Strengths:**
- Ships with every recent nightly rustc; no additional toolchain required.
- Maintained by the LLVM project; PTX backend bugs get fixes upstream.
- Full LLVM optimization passes apply before PTX emission.
- `-Zbuild-std` integration is well-understood for this target.
- No ABI compatibility surface outside the GPU kernel entry boundary.

**Weaknesses:**
- Tier 3 means no CI guarantee; breakage can sit for weeks.
- `ptx-kernel` calling convention is the only supported kernel ABI — it is frozen and does
  not permit the richer annotations VectorWare's `gpu-kernel` ABI requires.
- No support for stack unwinding (panics must map to `core::intrinsics::abort()`).
- No TLS (thread-local storage), no dynamic dispatch for trait objects backed by vtables
  that reference absolute addresses (relocation model is constrained).
- `alloc` crate works if you supply a custom allocator; `std` does not work out of the box
  because the `std` crate links against `libc` and `unwind` which have no GPU equivalents.

### rustc_codegen_nvvm (rust-cuda project)

`rustc_codegen_nvvm` bypasses the LLVM PTX backend entirely. Instead it:
1. Lowers Rust MIR → LLVM IR (same as the stock backend).
2. Replaces the PTX emission step with NVVM IR emission via NVIDIA's `libnvvm` library.
3. Hands that NVVM IR to `nvcc`/`ptxas` for final PTX and cubin generation.

**Strengths:**
- NVVM IR is closer to what NVIDIA's internal compiler stack expects; fewer ptxas rejections.
- Provides `cuda_std` crate with GPU-side APIs (thread/warp intrinsics, shared memory helpers).
- Historically more capable for features like `alloc` on GPU.
- Some workarounds for ptxas bugs are baked into the codegen.

**Weaknesses:**
- Requires a pinned, specific nightly rustc (often months-old). Upgrading rustc is a multi-day
  porting effort because the codegen backend API is unstable.
- `libnvvm` is a proprietary binary that must match the CUDA toolkit version exactly.
- The project has seen extended periods of low maintenance activity.
- NVVM IR spec lags behind LLVM IR; some LLVM optimization passes produce IR that `libnvvm`
  rejects with opaque errors.
- VectorWare's work is explicitly on the LLVM PTX path, not NVVM; diverging here makes
  reproducing their results harder.

### Verdict

For reproducing VectorWare's work specifically, the `nvptx64-nvidia-cuda` target is the correct
choice. VectorWare's modifications are to `rustc` proper (new ABI keyword, codegen annotations),
not to a separate codegen backend. Using `rustc_codegen_nvvm` would require porting their changes
to a different backend, multiplying the work. The LLVM PTX backend is the right base.

---

## 2. LLVM PTX Backend: Known Bugs and IR Limitations

### Structural Constraints

The NVPTX LLVM backend has a number of hard constraints that differ from x86/ARM targets:

**No recursive function calls in PTX (pre-sm_70).**
PTX prior to `sm_70` does not support a hardware call stack; the NVPTX backend lowers calls
either by inlining (the default and preferred path) or by using hardware call stacks on sm_70+.
Rust's panic machinery, `core::fmt`, and async state machine polling paths can all generate
non-inlineable call graphs if inlining heuristics fail. This will either produce a ptxas error
or silently incorrect code on older targets. Forcing `-Copt-level=3` or `-Cinline-threshold=...`
is often required.

**ABI representation of aggregates.**
The NVPTX backend does not support passing large structs by value on the PTX register file in
the same way CPU backends do. Rustc's default ABI for aggregates (pass-by-value with
`byval` attribute in LLVM IR) triggers an LLVM assertion or mis-lowering in the NVPTX backend
for certain aggregate shapes. This is a known class of bugs:
- Structs with mixed integer/float fields.
- `repr(C)` enums with payloads.
- Async `Future` state machine structs (large, heterogeneous) passed as function arguments.

Workaround: all kernel-boundary and inter-function arguments should be passed via pointer
(`byref` / pointer indirection). VectorWare's kernel annotations enforce this implicitly.

**Volatile and memory ordering.**
PTX has a richer memory consistency model than standard C (scoped atomics with `.sys`, `.gpu`,
`.cta` scopes). LLVM's atomic IR maps to PTX atomics, but the scope is always `.sys` from LLVM's
perspective, which is conservative and slow. Relaxed-order loads/stores may not be correctly
lowered for warp-level communication patterns. This affects the hostcall protocol (shared memory
signaling) directly.

**Known ptxas Rejections from Valid LLVM IR:**
- `alloca` inside a loop body that ptxas cannot prove is constant-size → ptxas error.
- Computed `getelementptr` with dynamic index into a `byval` struct → ptxas ICE on some
  CUDA toolkit versions.
- LLVM `freeze` instruction (generated by Rust for uninit values) is not recognized by all
  ptxas versions; may require `-Zbuild-std` with a patched `core` that avoids `freeze`.

**Register allocation.**
PTX has a large but finite register file. LLVM's register allocator does not model GPU occupancy;
it optimizes for ILP, not for minimizing register count. Large async state machines (see
Section 5) will spill to local memory (which is backed by L1/L2/DRAM) with no warning.

### Workarounds Summary

| Issue | Workaround |
|---|---|
| Non-inlineable calls | `#[inline(always)]` on hot paths; `opt-level=3` |
| Aggregate ABI mis-lowering | Pass via pointer; avoid by-value large structs at kernel boundary |
| `freeze` rejection | Pin to a CUDA toolkit version known to accept it, or patch |
| Dynamic allocas | Convert to fixed-size; avoid `Vec` in hot GPU paths |
| Atomic scope | Accept `.sys` overhead or use inline PTX asm for scoped atomics |

---

## 3. -Zbuild-std: Mechanism, Extension, and GPU Challenges

### How -Zbuild-std Works

`-Zbuild-std` recompiles the standard library (or a subset) from source as part of the current
cargo build. It replaces the pre-compiled artifacts in the sysroot. The relevant flags are:

```
-Zbuild-std=core                     # core only
-Zbuild-std=core,alloc               # core + alloc
-Zbuild-std=core,alloc,std           # core + alloc + std
```

Cargo feeds the standard library's `Cargo.toml` manifests through the same compilation pipeline
as the user crate, with all the same `-C target-cpu`, `--target`, and `-Z` flags applied. This
means the standard library is compiled for the GPU target, including any custom panic handlers,
allocators, and intrinsics.

### Extending to alloc

`-Zbuild-std=core,alloc` compiles the `alloc` crate, which provides `Box`, `Vec`, `String`,
`Arc`, etc. For this to work on GPU:

1. A `#[global_allocator]` must be provided that works on GPU (e.g., a bump allocator backed
   by a pre-allocated device memory buffer, or a hostcall-based `malloc`).
2. The `alloc` crate must be compiled with `--cfg no_global_oom_handling` if OOM panics are
   not supported (GPU kernels cannot truly handle OOM the same way).
3. The `alloc` crate's `__rust_alloc` / `__rust_dealloc` symbols must resolve to the GPU
   allocator, not the host libc `malloc`. With `-Zbuild-std`, these are provided by the
   `#[global_allocator]` impl in the GPU crate.

### Extending to std

`-Zbuild-std=core,alloc,std` is where complexity explodes:

**std's dependencies on the GPU target:**
- `std` depends on `libc` for OS syscalls. On the `nvptx64` target, `libc` has no implementation.
- `std` depends on `unwind` for panic unwinding. PTX has no unwind mechanism.
- `std` depends on platform-specific thread primitives (mutexes, condvars) from the OS.

VectorWare's approach: provide a fake `libc` crate (a `libc` facade) that implements libc
function symbols as hostcalls. This works because:
1. `std` calls into `libc` via FFI (e.g., `extern "C" { fn write(...) }`).
2. The linker resolves these to the GPU-side shim.
3. The GPU-side shim packages the call as a hostcall request and spins until the host responds.

**Challenges for this approach:**
- Panic/unwind: `std` panics call `rust_begin_unwind`, which calls into the panic runtime.
  The `panic_abort` runtime is usable on GPU (it just aborts), but it must be configured
  explicitly via `Cargo.toml` profile `panic = "abort"`.
- TLS (`thread_local!` statics): PTX has no TLS segment; `std`'s use of TLS for errno,
  the current thread handle, etc., must be replaced with per-lane globals or shared memory
  accesses. This requires patching `std` source, not just providing a `libc` shim.
- `std::time`: requires `clock_gettime` or equivalent; must be a hostcall.
- `std::env`: requires `getenv`; reasonable to implement as a hostcall.
- `std::fs`: all syscalls must become hostcalls; this is the primary target of VectorWare's
  gpu-std work.

**The TLS problem is the hardest.** Rust's `std` uses `#[thread_local]` for internal state.
The `nvptx64` target does not support TLS. Without modifications to `std` source, any code
path through `std` that touches a TLS variable will either fail to compile or produce
incorrect PTX. VectorWare must have patched these TLS uses out of `std` in their fork.

---

## 4. ABI Concerns: ptx-kernel vs. gpu-kernel

### Current ptx-kernel ABI

The `extern "ptx-kernel"` calling convention in rustc marks a function as a CUDA kernel entry
point. It translates to `calling_conv = PTX_Kernel` in LLVM IR, which the NVPTX backend handles
by:
- Emitting a `.entry` directive in PTX (vs. `.func` for device functions).
- Marking all parameters as `.param` space arguments in PTX.
- Emitting `ret;` at the end (no return value).

Limitations of `ptx-kernel`:
- No metadata for kernel parameters (no type information, alignment hints, etc.).
- No way to annotate shared memory (`__shared__` in CUDA C) allocation size.
- No way to specify launch bounds (`__launch_bounds__`) which affect register allocation.
- No mechanism for CUDA's `__grid_constant__` or `__restrict__` annotations.
- Single flat calling convention; no way to distinguish kernel variants.

### VectorWare's gpu-kernel ABI

Based on the blog description, `extern "gpu-kernel"` appears to extend `ptx-kernel` with:
- Richer parameter passing annotations (by-value vs. by-pointer decisions at ABI level).
- Possible integration with `#[unsafe(no_mangle)]` in a way that ensures stable symbol naming
  for the CUDA runtime's `cuModuleGetFunction` lookup.
- Potentially: `__launch_bounds__`-equivalent annotations that get embedded as LLVM metadata
  and passed to the NVPTX backend's ptxas invocation.

**What would changing the ABI require in rustc?**

1. `compiler/rustc_target/src/abi/call/nvptx64.rs` — Add new calling convention variant.
2. `compiler/rustc_codegen_llvm/src/abi.rs` — Map the new ABI to an LLVM calling convention
   or to NVPTX-specific function attributes.
3. `compiler/rustc_span/src/symbol.rs` — Register `gpu-kernel` as a known identifier.
4. `compiler/rustc_hir_analysis/` — Parse and validate the new ABI string.
5. Possibly `compiler/rustc_middle/src/ty/layout.rs` — Change how function arguments are
   laid out for the new ABI.

**Has gpu-kernel been upstreamed?**
As of the knowledge cutoff, there is no merged RFC or PR for `extern "gpu-kernel"` in upstream
rustc. The `ptx-kernel` ABI itself was added in a relatively small PR. VectorWare is likely
maintaining a rustc fork with a patch of similar scope.

**Practical implication for this project:**
If we want to reproduce VectorWare's exact kernel annotations, we need either their rustc fork
or to write an equivalent patch ourselves. For initial exploration, using `extern "ptx-kernel"`
with `#[no_mangle]` should be sufficient to verify the overall pipeline; the ABI difference
mainly affects parameter passing efficiency and launch-bounds optimization.

---

## 5. Async State Machines on GPU

### How rustc Generates Future State Machines

When rustc compiles an `async fn`, it desugars it into a state machine struct that implements
`Future`. Each `await` point becomes a state transition. The generated struct:
- Has a discriminant field (typically `u32`) tracking the current state.
- Stores all locals that are live across any `await` point as fields.
- The `poll()` method is a match on the discriminant with a block per state.

Example: an `async fn` that awaits 3 futures and holds a `String` and a `Vec<u8>` will generate
a struct of roughly:
```
struct MyFutureStateMachine {
    state: u32,
    // live across await 0:
    string_val: String,   // 24 bytes on host (ptr + len + cap)
    // live across await 1:
    vec_val: Vec<u8>,     // 24 bytes on host
    // sub-futures:
    fut0: Fut0Type,
    fut1: Fut1Type,
    fut2: Fut2Type,
}
```

On GPU, the layout is the same, but the implications differ dramatically.

### Register Pressure Implications

PTX thread registers are a fixed-size file (typically 255 registers per thread on modern
architectures). The CUDA occupancy model ties the number of simultaneous warps (occupancy)
to the number of registers used per thread. More registers per thread → fewer simultaneous
warps → lower memory latency hiding → lower GPU efficiency.

An async state machine polling function:
- Must load the discriminant and branch.
- Must load/store all live fields at each state transition.
- Sub-future `poll()` calls must be inlined (see Section 2: no recursion without stack).
- The inlined sub-future poll bodies also contribute their register usage.

For a non-trivial async task tree (e.g., `join!` of 4 sub-tasks), the total register
usage can easily exceed 64 registers per thread, dropping occupancy to 25% on typical
hardware (which targets 256 threads/block × 4 warps = acceptable only if the kernel is
compute-bound, not memory-bound).

VectorWare explicitly notes this concern: "Register pressure from futures reduces occupancy."

### What Could Go Wrong in PTX Codegen

**State machine struct passed by value.**
The `poll()` method takes `Pin<&mut Self>` — a pointer. This is fine. But if rustc or LLVM
decides to pass intermediate aggregates by value during inlining optimizations, the NVPTX
backend's aggregate ABI mis-lowering (Section 2) triggers. Careful LLVM IR inspection is
needed.

**Non-constant alloca for state machine storage.**
If the async task spawning infrastructure uses `Box::new(future)` (Embassy does), the `Box`
allocation is a runtime `malloc` call. On GPU this must go through the custom allocator.
The size of the `Box` is determined at compile time (it's `size_of::<FutureStateMachine>()`),
so the `alloca`/`malloc` size is constant — this is fine. The issue is that the pointer
returned from GPU `malloc` may have different alignment requirements than assumed.

**Discriminant branches and PTX divergence.**
The state machine's `match state` generates branches. On GPU, branches across warp lanes
cause divergence: the warp executes both arms serially. If 32 GPU threads are running the
same async task at different states (e.g., a per-thread executor), each state transition
point becomes a divergence event. An async task with N await points introduces up to N
divergence events per poll cycle. This is a correctness-safe but performance-dangerous pattern.
The architecture decision of "one task per thread vs. one task per warp" directly affects this.

**LLVM unwind + async.**
Rust's async state machine interacts with the panic/unwind machinery via `Drop` implementations
on the stored sub-futures. If any `Drop` impl calls a function that requires unwinding,
the LLVM codegen will emit invoke/landingpad IR, which the NVPTX backend cannot lower.
This is avoided by `panic = "abort"` and by ensuring all Drop impls on GPU futures are trivial.

**Workaround for register pressure:**
- Use `futures::future::join_array` over deeply nested `join!` chains to reduce state
  machine nesting depth.
- Mark leaf futures `#[inline(never)]` only if they are tail calls (i.e., they are the
  last thing polled and their result is immediately returned). This is an advanced optimization
  that trades call overhead for register reuse.
- Set `ptxas -maxrregcount=N` to cap register use; this forces spilling to local memory
  but guarantees occupancy. VectorWare likely uses this in their build system.

---

## 6. Fundamental Blockers

These issues **cannot be worked around without rustc changes or custom std patches**:

### 6.1 TLS in std

`std` uses `#[thread_local]` statics internally (errno, current thread info, panic handler
state). The `nvptx64` target has no TLS support at the LLVM level. Compilation of `std` without
source patches will fail or produce incorrect PTX wherever TLS is accessed.

**Required fix:** Patch the affected `std` modules to use per-thread shared memory slots or
to redirect to hostcall-based storage. This is not a linker trick; it requires `std` source
modification.

### 6.2 No Stack Unwinding

PTX has no hardware exception/unwind mechanism. Any `std` code path that can unwind (i.e.,
panics with non-abort semantics) will fail. `panic = "abort"` is mandatory. However, `std`
itself has some code paths that are internally structured assuming unwind is available, even
under `panic = "abort"`. These must be audited and patched.

### 6.3 ptx-kernel ABI Limits

If VectorWare's `gpu-kernel` ABI provides semantics that `ptx-kernel` cannot express (e.g.,
specific parameter space annotations that affect shared memory layout), reproducing their
exact behavior requires an equivalent ABI patch to rustc. This is a moderate-complexity
rustc change (~200-500 lines across the described files).

### 6.4 Inline Asm Fallbacks for Scoped Atomics

Rust's `core::sync::atomic` primitives do not expose GPU-scoped atomics (`.cta`, `.gpu`,
`.sys` scope qualifiers in PTX). For the hostcall protocol's signaling mechanism, naive
use of Rust atomics will emit `.sys`-scoped PTX atomics, which are the most expensive.
There is no way to get cheaper scoped atomics without inline PTX assembly. This is not a
blocker but a performance concern; the hostcall protocol will work correctly with `.sys`
semantics, just at higher latency.

### 6.5 Allocator Bootstrapping

The `alloc` crate requires a `#[global_allocator]` that functions before the first allocation.
On GPU, the allocator must either be:
(a) A static bump allocator over a pre-allocated device buffer — simple, but limited.
(b) A hostcall-based `malloc` — flexible, but adds a round-trip to the host per allocation.

Neither is provided by default. This is a design task, not a fundamental blocker, but it
must be resolved before `alloc` (and therefore async task spawning via `Box`) works.

---

## 7. Recommended Toolchain Path

### Primary Recommendation: nvptx64 + patched nightly rustc

**Step 1: Establish baseline with nvptx64 + ptx-kernel.**
Use the upstream `nvptx64-nvidia-cuda` target with the stock `ptx-kernel` ABI and
`-Zbuild-std=core`. Compile a minimal kernel (`#[no_mangle] extern "ptx-kernel" fn kernel()`),
load it with `cudarc`, and verify execution. This validates the entire host-side pipeline
and the basic toolchain without any custom modifications.

Priority: `toolchain.1` and `toolchain.4` immediately.

**Step 2: Extend to alloc.**
Implement a static bump allocator for GPU (a fixed-size arena over a device buffer).
Compile with `-Zbuild-std=core,alloc`. Verify `Vec`, `Box`, `String` work in a kernel.

**Step 3: Audit std for TLS dependencies, then patch.**
Clone the `rust-lang/rust` repo at a specific nightly commit. Identify all
`#[thread_local]` uses in `std`'s platform-independent and unix-specific code paths.
Replace with GPU-compatible alternatives (per-lane shared memory or panic = abort stubs).
Build this patched rustc. This is the first point where `extern "gpu-kernel"` ABI
extensions can also be added if desired.

**Step 4: Implement libc shim.**
Build the hostcall protocol (theme `hostcall`) in parallel with Step 3.
The libc shim is compiled as a static library for the `nvptx64` target and linked
into the GPU kernel. Key functions: `write`, `read`, `open`, `close`, `malloc`, `free`.

**Step 5: Wire std to libc shim.**
With the patched `std` (no TLS) and the libc shim available, compile
`-Zbuild-std=core,alloc,std` targeting the GPU. This should give `println!` and basic
file I/O in a GPU kernel.

**Step 6: Async executor.**
Port Embassy's executor after `alloc` is working (it needs `Box` for task storage).
Measure register pressure. Apply `ptxas -maxrregcount` as needed.

### Alternative: Evaluate rustc_codegen_nvvm for async

If the LLVM PTX backend proves too bug-prone for the large aggregate types generated by
async state machines (the aggregate ABI issue in Section 2), `rustc_codegen_nvvm` may
produce cleaner PTX for those specific cases. However, it should be treated as a fallback
investigation (`toolchain.2`), not the primary path, because:
- It diverges from VectorWare's documented approach.
- The maintenance situation is uncertain.
- The patched `std` work applies equally to both backends.

### Toolchain Pinning Strategy

Pin to a specific nightly that satisfies all of:
- Stable `-Zbuild-std` behavior for `nvptx64`.
- LLVM version >= 17 (better NVPTX backend, fewer `freeze` issues).
- No known regressions in the `ptx-kernel` ABI handling.

Document the exact toolchain channel (`rust-toolchain.toml`) from the outset. Every team
member and CI run must use exactly this toolchain. Drift here is the single most common
source of "it worked on my machine" failures in GPU Rust projects.

---

## Summary of Key Insights

1. **LLVM PTX backend aggregate ABI mis-lowering is the highest compiler risk** for async
   state machines. Must be validated early (toolchain.4 experiment).

2. **TLS in std is the hardest portability problem** and requires patching std source.
   This work unlocks all of gpu-std.

3. **ptx-kernel ABI is sufficient for initial work**; gpu-kernel ABI is an optimization
   that can be added later as a rustc patch.

4. **panic = "abort" is mandatory** and must be set in all Cargo profiles for the GPU target.

5. **Register pressure from async is real but manageable** with ptxas `-maxrregcount` and
   careful future composition.

6. **The toolchain.1 and toolchain.4 tasks should run immediately** to establish ground
   truth about what the LLVM PTX backend actually accepts with the current nightly, before
   investing in std/async work that depends on a working base.
