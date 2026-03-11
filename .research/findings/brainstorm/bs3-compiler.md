# BS3 Brainstorm — Compiler Engineer Analysis

**Date**: 2026-03-11
**Role**: compiler (Rust compiler engineer: rustc, LLVM, codegen)
**Focus**: Compiler correctness, stability risks, and codegen pitfalls for the async_gpu project

---

## 1. `asm_experimental_arch` Stability Risk

**Assessment: Low-to-medium risk. Plan for stabilization, but do not rely on it.**

The `asm_experimental_arch` feature gate exists because PTX assembly syntax diverges from
the x86/ARM inline asm model. The feature is specifically needed for non-tier-1 targets
(nvptx64, spirv, avr) that require custom register classes and operand constraints. It is
**not** on a path toward removal — it is on a path toward eventual stabilization once the
nvptx64 target matures.

Risk factors:
- The feature could be **renamed** during stabilization (breaking the `#![feature(...)]`
  declaration, easily fixed by s/old/new).
- The feature **constraints API** for PTX registers (`reg32`, `reg64`) could change if the
  nvptx64 target undergoes a register class overhaul.
- If LLVM's PTX backend changes register encoding, the asm! operand types could break silently
  (the compiler accepts the asm, but emits wrong PTX).

Mitigation already in place: All asm use is in `gpu-atomics/src/lib.rs`, which is a small,
auditable crate. The NVVM intrinsic fallback path (`link_llvm_intrinsics`) provides a secondary
path if asm regresses. Keep both paths compilable.

**Recommendation**: Add a CI test that compiles gpu-atomics and checks the PTX output contains
expected instruction patterns (e.g., `atom.cas.sys`, `ld.acquire.sys`). This detects silent
codegen regressions immediately on nightly update.

---

## 2. Complex Control Flow in PTX — CAS Loops and Branches

**Assessment: LLVM's PTX backend handles structured loops well, but watch for divergence bugs.**

CAS spin loops in the hostcall.4 protocol (`sys_cas_u64` within a GPU-side loop) will generate
PTX branch instructions. LLVM's PTX backend has known issues with:

### 2a. Loop unrolling and hoisting across asm blocks

LLVM may hoist a `sys_load_acquire` call (which uses `options(nostack, readonly)`) out of a
spin loop because `readonly` tells LLVM the block does not modify memory. This is the known
pitfall already documented in `gpu-atomics/src/lib.rs` (line 99). It is **not** mitigated by
`#[inline(always)]` alone — it is mitigated by NOT marking the spin-loop variant as `readonly`.

**Concrete fix required before hostcall.4**: The hostcall spin loop (step 6 of the protocol)
uses `sys_load_acquire_u32` to poll `packet.header.control`. This function is marked
`options(nostack, readonly)`. If inlined into a loop, LLVM will hoist the load. The result is
PTX that reads the value once into a register and then loops on the stale register indefinitely
— the warp spins at full clock speed with zero memory traffic. The MAX_SPIN timeout still
increments (it is in the loop), so it will eventually fire, but the warp burns a full warp
slot for the entire timeout duration without making progress.

The fix: provide `sys_spin_load_acquire_u32` (and u64 variant) with:
- No `readonly` option (prevents LICM hoisting)
- `#[inline(never)]` (second line of defense — LICM cannot cross call boundaries)
- An embedded `nanosleep.u32 64` to yield the warp slot to the hardware scheduler

```rust
#[inline(never)]
pub unsafe fn sys_spin_load_acquire_u32(ptr: *const u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "ld.acquire.sys.global.u32 {result}, [{ptr}];",
        "nanosleep.u32 64;",
        result = out(reg32) result,
        ptr = in(reg64) ptr,
        options(nostack),
        // NOTE: no `readonly` — prevents LICM hoisting
    );
    result
}
```

Keep `sys_load_acquire_u32` (with `readonly`) for single-shot non-loop reads where hoisting
is safe. Callers must explicitly choose the correct variant.

### 2b. Warp divergence in CAS retry loops

The CAS retry loop in the hostcall protocol (steps 2 and 4) involves:
```
loop:
    old = load(ptr)
    new = compute(old)
    result = cas(ptr, old, new)
    if result == old: break
    // retry
```

In a SIMT context, different lanes may succeed/fail the CAS at different iterations, causing
warp divergence. LLVM's PTX backend generates `@%pred bra` instructions for Rust `if` branches.
The hardware handles divergence via the predicate mask — correctness is preserved, but
performance degrades.

**More critical**: the hostcall protocol (from hostcall.3-c10) already accounts for this by
using `active_mask` (one CAS per warp, not per lane). This is the correct design. Just ensure
the warp-level CAS is called by exactly one lane (lane 0 or the lowest active lane) with other
lanes predicated off.

### 2c. Branch optimization bugs in LLVM's PTX backend

LLVM has had a class of bugs where `brx.idx` (computed goto / indirect branch) tables are
mis-generated for complex CFGs on PTX. Rust's `match` statements over enums (especially async
state machine discriminants) generate these. This is relevant for the async/await phase.

For the CAS loops specifically: these are simple structured loops with a single back-edge.
LLVM handles these correctly — no known bugs for this pattern on nvptx64.

---

## 3. Cross-Crate Inlining Depth: 3-Level Chain

**Assessment: Works today, but fragile at depth ≥ 3. Verify with PTX output inspection.**

Current chain: `gpu-atomics` → `gpu-kernel` (verified working).

Future chain: `gpu-atomics` → hostcall crate → GPU std shim.

All `gpu-atomics` functions are `#[inline(always)]`. This forces LLVM to inline them regardless
of cost model. For a 3-level chain, the inlining will cascade:
1. hostcall crate calls `sys_cas_u64` → inlined into hostcall
2. gpu-std shim calls hostcall functions → hostcall code (with inlined atomics) folds in

This **should** work because `#[inline(always)]` is a hard directive, not a hint. However:

### 3a. Monomorphization and LTO boundaries

`#[inline(always)]` functions in a dependency are included in the rlib as MIR (via
`codegen_fn_attrs`). They are re-codegen'd at each call site. This works within a single
`--emit=asm` compilation unit (which is how we build gpu-kernel). But if we ever split into
multiple PTX modules compiled separately and linked, cross-module inlining would require ThinLTO.

For the current single-PTX-kernel architecture: **no issue**.

### 3b. Register pressure from deep inlining

Each inlined CAS loop consumes registers. On SM86 (RTX 3060), each SM has 64K registers shared
among all active warps. A deeply inlined kernel with many CAS operations may exceed the per-thread
register budget (255 registers max), forcing register spilling to local memory. This is a
performance concern, not a correctness concern.

If register pressure becomes an issue, the mitigation is to use `#[inline]` (hint) instead of
`#[inline(always)]` and let LLVM decide, accepting that some functions may not inline and will
generate PTX function calls (which is valid and supported by LLVM's PTX backend).

### 3c. Debug vs. release mode difference

In debug mode, `#[inline(always)]` is still respected, but the generated code is much larger
due to lack of optimization. PTX size is not a correctness concern, but debug builds of GPU
kernels can be enormous and slow to JIT-compile via NVRTC/cuModuleLoad.

---

## 4. `-Zbuild-std=std` Feasibility for nvptx64

**Assessment: Currently blocked. The path forward is a custom libc shim, not the std path.**

Building full `std` for nvptx64 requires:
1. A `sys` module implementation (normally provided by the OS, via libc)
2. A panic handler that doesn't call `abort()`
3. An allocator (currently no heap on GPU — must use hostcall-based allocator)
4. Thread-local storage (TLS) — nvptx64 has no TLS mechanism

The fundamental blocker is that `std::sys` for CUDA does not exist in upstream Rust. The
`std::sys::pal` module for unknown/unsupported targets (`std/src/sys/pal/unsupported`) provides
stub implementations that panic, which is not useful.

**The VectorWare approach** (based on the reference articles) is NOT to use `-Zbuild-std=std`
with the default std. Instead, they build a custom std layer:
1. Implement the libc API surface (just the functions used by std: `write`, `read`, `malloc`, etc.)
2. Route those libc calls through the hostcall protocol
3. Compile std against this fake libc

This is conceptually similar to how `wasm32-wasi` provides a WASI libc shim that std compiles
against. The key difference: on nvptx64, the shim is `#[no_std]` Rust code that issues
hostcall RPCs rather than syscalls.

**Concrete steps blocked on**:
- `alloc` crate works today with a custom `#[global_allocator]` — can implement using
  SERVICE_MALLOC hostcall
- `std::io` requires file descriptor operations — implement via SERVICE_WRITE/READ/OPEN/CLOSE
- TLS (`thread_local!`) — completely unsupported on nvptx64, must avoid or stub out
- Unwinding (panic = unwind) — must use `panic = abort` (already required for GPU targets)

**My recommendation**: Target `-Zbuild-std=core,alloc` first (not `std`). This gives access to
`Vec`, `Box`, `String`, `Arc`, etc. on GPU, which covers 80% of use cases. True `std` (with IO)
requires the full hostcall libc shim to be in place first.

---

## 5. Async/Await Codegen on nvptx64

**Assessment: Core mechanism works; indirect calls through trait objects are the key risk.**

### 5a. Async state machine layout

Rust's async/await desugars to a generator-based state machine stored as an enum. Each `.await`
point becomes a variant containing the live variables at that suspension point. This is pure
data structure layout — no platform-specific codegen required. The enum is `repr(Rust)` by default
but the layout algorithm works identically on nvptx64.

**Risk**: The state machine enum contains a discriminant. LLVM must emit PTX code to read/write
the discriminant and branch on it. This generates indirect-like branches (indexed branches via
`brx.idx` in LLVM IR). Known LLVM PTX bug: certain CFG shapes involving these indexed branches
can be mis-compiled in older LLVM versions. LLVM 19.x (our target) has fixed the most egregious
cases but the area is not regression-tested on PTX.

**Mitigation**: Keep async state machines simple (few `.await` points per function). Do not use
deeply nested async blocks. Test by inspecting PTX output for `brx.idx` — replace with explicit
match if bugs are encountered.

### 5b. Trait objects and vtables (dyn Future)

Embassy's executor uses `dyn Future` trait objects for the task queue. On nvptx64, vtable-based
dispatch emits PTX `call` instructions with indirect targets (function pointer call). The LLVM
PTX backend supports indirect calls, but they require the callee to be in the same PTX module
(no dynamic linking). Since we compile everything to a single PTX module, this is fine.

**Critical risk**: if the async executor polls `dyn Future` tasks, and those tasks contain CAS
loops from the hostcall crate, the indirect call path cannot be inlined. The CAS loop runs as
a non-inlined PTX function. This is **correct** but loses the `#[inline(always)]` guarantee.
The alternative is to use static dispatch (`impl Future` or `Pin<&mut impl Future>`) rather than
`dyn Future` for GPU task types.

**Recommendation**: Port Embassy to use static dispatch for the inner task loop on GPU. This
matches the `arch-spin` executor pattern which avoids trait objects for the polling path.
Static dispatch also enables LLVM to merge the poll() body's register frame with the executor's
register frame, reducing total register pressure compared to a separate call frame per vtable
dispatch.

### 5b-addendum. Async state machine + hostcall spin loop — register spill risk

When an async state machine contains a hostcall spin loop and an `.await` point inside that
loop, the live variables at the checkpoint (CAS result buffers, packet pointers, spin_count)
are stored in the state machine enum. If the enum exceeds available registers, LLVM spills it
to GPU local memory (per-thread DRAM-backed private memory, ~100–500 ns latency). The spin
loop then reads `spin_count` from local memory every iteration — adding 100–500 ns overhead to
each 1–2 µs PCIe round-trip. Not catastrophic, but avoidable.

**Design constraint for async-runtime theme**: GPU async functions MUST NOT yield (`.await`)
inside a hostcall spin loop. Hostcall must be a synchronous blocking primitive; `.await` points
appear only after the call completes and its local variables are dead.

```rust
// WRONG — .await inside spin loop saves entire spin state to the enum
async fn bad() { loop { if poll_ready().await { break; } } }

// CORRECT — hostcall_blocking() spins synchronously; .await is outside
async fn good() {
    let result = hostcall_blocking(...);  // no .await inside
    next_async_step().await;             // .await after all spin state is dead
}
```

### 5c. Waker implementation on GPU

Embassy's GPU-compatible executor (arch-spin model) uses a no-op waker or a custom waker that
sets a flag. On GPU, the waker must not use atomic operations from `core::sync::atomic` due to
LLVM bug #173993. If Embassy's waker implementation uses `AtomicUsize::store` for wake signaling,
it must be replaced with `gpu-atomics` primitives.

**This is a concrete correctness issue for the async-runtime theme.**

---

## 6. `link_llvm_intrinsics` Risk Assessment

**Assessment: Higher risk than asm_experimental_arch. Use as fallback only.**

`link_llvm_intrinsics` is explicitly documented as "internal to the compiler." The feature:
- Is not tracked on a stabilization path
- Has no stability guarantee across nightly versions
- The LLVM intrinsic names it references (e.g., `llvm.nvvm.membar.sys`) can change with LLVM
  version bumps (though in practice NVVM intrinsics are very stable)

The more critical risk: the feature is used to call LLVM NVPTX backend-specific intrinsics
directly. If the LLVM NVPTX team renames or removes an intrinsic, the symbol lookup will fail
at compile time with a linker error.

**Current usage**: `link_llvm_intrinsics` is used in `gpu-atomics/src/lib.rs` for
`nvvm_membar_sys` and `nvvm_atomic_add_sys_i32` (lines 211-218), but these are marked
`pub(crate)` and commented as fallbacks. The production code uses `asm_experimental_arch`.

**Recommendation**: Keep the NVVM intrinsic `extern` block always compiled but mark it
`#[allow(dead_code)]`. Do NOT feature-gate it. Reason: a feature-gated path can silently bitrot
without anyone noticing. An always-compiled `extern "C"` declaration will surface a linker error
immediately on nightly upgrade if LLVM renames or removes the intrinsic — exactly the early-
warning behavior we want. DCE will strip the unused symbols from the PTX output. Verify DCE
worked by checking that intrinsic names do not appear in `--emit=asm` output.

---

## 7. Critical Risk Summary

| Risk | Severity | Immediacy | Mitigation |
|------|----------|-----------|------------|
| `sys_load_acquire` hoisted out of spin loop | **CRITICAL** | hostcall.4 | Remove `readonly` from spin-loop variant |
| Async state machine `brx.idx` bugs | HIGH | async-runtime phase | Inspect PTX, keep states minimal |
| `dyn Future` vtable → no inlining of CAS | MEDIUM | async-runtime phase | Use static dispatch in executor |
| Embassy waker uses core::sync::atomic | HIGH | async-runtime phase | Replace with gpu-atomics primitives |
| `link_llvm_intrinsics` removed | MEDIUM | any nightly update | Keep always-compiled + `#[allow(dead_code)]`; DCE strips it |
| Register pressure from deep inlining | MEDIUM | hostcall.4+ | Profile PTX register usage |
| `-Zbuild-std=std` blocked | LOW | not immediate | Use `core,alloc` first, std later |

---

## 8. Immediate Action Items (before hostcall.4 implementation)

1. **Fix `readonly` bug** in `sys_load_acquire_u32/u64`: Add a non-readonly variant for spin
   loop use cases (e.g., `sys_load_acquire_u32_spin`), or remove `readonly` from the existing
   functions.

2. **Add u64 atomics** to `gpu-atomics`: `sys_cas_u64`, `sys_fetch_add_u64`, `sys_exchange_u64`
   as specified in hostcall.3-c10. These follow the identical inline PTX pattern as u32 variants.

3. **Remove `link_llvm_intrinsics` from production path**: Move NVVM intrinsic fallbacks to a
   feature-gated module.

4. **Add PTX output verification**: A test or script that greps the emitted PTX for required
   instruction patterns — ensures no silent codegen regression on nightly update.
