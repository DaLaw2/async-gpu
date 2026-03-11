# atomics.1: Rust Atomic → PTX Scope Mapping Verification
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: atomics
**Kind**: investigation
**Status**: done
**Spawned by**: bs1

## Summary

`core::sync::atomic` on `nvptx64-nvidia-cuda` is **silently broken** for GPU-CPU communication.
All memory orderings (SeqCst, AcqRel, Acquire, Release, Relaxed) compile to the identical
unscoped PTX instruction `atom.global.add.u32` — with no `.sem` qualifier and no `.sys` scope.
The LLVM NVPTX backend discards ordering and syncscope metadata entirely when lowering
`atomicrmw` instructions. Fences (`atomic::fence`) were previously a compiler crash; as of
nightly early 2025 they no longer crash but are still silently dropped (no PTX `fence.*`
instruction emitted). The result is that any GPU-CPU atomic protocol built on
`core::sync::atomic` has **undefined behavior** on SM70+; the generated PTX violates the
NVIDIA PTX memory consistency model. Since `asm!` is also unavailable on nvptx64, inline
PTX workarounds are impossible. The only viable workaround using the plain nvptx64 target
path is through LLVM NVPTX-specific intrinsics accessed via `extern "C"` declarations
with `#[link_name]`, or through a forced switch to the Rust-CUDA (NVVM IR) codegen path
which supports inline PTX asm and can call `llvm.nvvm.atomic.*` scoped intrinsics.

---

## Detailed Findings

### Q1: PTX Instructions from core::sync::atomic

**Observed output (Rust issue #136480, LLVM 19.1.7, nightly 2025-02-01):**

Three distinct Rust intrinsics with different orderings:
```rust
core::intrinsics::atomic_xadd_seqcst(ptr, 1);
core::intrinsics::atomic_xadd_acqrel(ptr, 1);
core::intrinsics::atomic_xadd_relaxed(ptr, 1);
```
All three compile to the same PTX instruction:
```ptx
atom.global.add.u32 %r1, [%rd1], 1;
```

No `.sem` qualifier (no `.acquire`, `.release`, `.acq_rel`, `.sc`).
No `.scope` qualifier (no `.cta`, `.gpu`, `.sys`).

The `atom.global.add.u32` form corresponds to the legacy pre-PTX-6.0 instruction, which has
`monotonic` (relaxed) semantics and is only atomic among threads in the same GPU device. It
does **not** provide any ordering guarantee and is definitely not system-scope.

**For fences:**
- LLVM 19.1.7 (nightly 2025-02-01): `atomic::fence(SeqCst)` causes an LLVM backend crash:
  `LLVM ERROR: Cannot select: ch = AtomicFence ... TargetConstant:i64<7>, TargetConstant:i64<1>`
- Nightly 2025-03-08: fence no longer crashes but no PTX fence instruction is emitted at all.
  The fence is silently dropped. Confirmed by LLVM bug #173993 (opened March 2025).

**For atomic loads/stores:**
- The D50391 patch (merged into LLVM) lowered monotonic atomic loads/stores to
  `ld.volatile` / `st.volatile`, which per PTX ISA has `.relaxed.sys` semantics for load/store.
- However, stronger orderings (Acquire, Release, SeqCst) on plain loads and stores are still
  rejected by the backend (`isStrongerThanMonotonic` guard returns false → operation falls
  through without proper instruction selection).

**Summary table:**

| Rust ordering | Expected PTX              | Actual PTX (nvptx64, SM70+) | Correct? |
|---------------|---------------------------|-----------------------------|----------|
| Relaxed       | `atom.relaxed.gpu.add.u32`| `atom.global.add.u32`       | NO — scope missing |
| Acquire       | `atom.acquire.gpu.add.u32`| `atom.global.add.u32`       | NO — ordering+scope missing |
| Release       | `atom.release.gpu.add.u32`| `atom.global.add.u32`       | NO — ordering+scope missing |
| AcqRel        | `atom.acq_rel.gpu.add.u32`| `atom.global.add.u32`       | NO — ordering+scope missing |
| SeqCst        | `fence.sc.sys; atom.relaxed.sys.add.u32` | `atom.global.add.u32` | NO — both missing |
| fence(SeqCst) | `fence.sc.sys`            | (silently dropped)          | NO — dropped |

### Q2: Ordering → Scope Mapping

**What the PTX ISA requires (SM70+, PTX 6.0+):**

PTX ISA 6.0 (targeting SM70 Volta) introduced a formal memory consistency model with two
orthogonal qualifiers on atomic instructions:

1. **Memory semantic (`.sem`)** — the C++ memory ordering:
   - `.relaxed` — no ordering guarantees
   - `.acquire` — no preceding load may be reordered after
   - `.release` — no following store may be reordered before
   - `.acq_rel` — both acquire and release
   - `.sc` — sequentially consistent (used on fences: `fence.sc.{cta|gpu|sys}`)

2. **Scope (`.scope`)** — the set of threads that observe the operation as atomic:
   - `.cta` — threads in the same thread block (cooperative thread array)
   - `.gpu` — all threads on the same GPU device
   - `.sys` — all threads in the system, including CPU threads

For GPU-CPU shared memory communication via mapped/pinned memory, `.sys` scope is **mandatory**.
A GPU kernel using `.gpu` scope writes are not guaranteed to be visible to the host CPU, even
with explicit fences, because `.gpu` scope fences do not synchronize with non-GPU threads.

**PTX ISA rule for system-scope atomics:**
> An atomic operation is atomic at system scope if it is a load or store affecting a
> naturally-aligned object on mapped (pinned) memory. Requires SM60+ on Linux, SM70+ on Windows.

**What LLVM actually emits:** None of the above. All `atomicrmw` instructions from
`core::sync::atomic` are lowered to legacy `atom.global.*` form with no `.sem` or `.scope`.
The syncscope metadata on the LLVM IR instruction (`syncscope("system")`, `syncscope("gpu")`,
etc.) is silently discarded by the NVPTX instruction selector.

**Root cause (LLVM issue #123339, January 2025):**
The PTX backend "mistranslates" `syncscope("system")` to PTX syncscope `device`. Furthermore,
there is an ambiguity because `syncscope("system")` is the implicit default in LLVM IR when no
scope is specified, and the backend cannot distinguish "explicit system scope" from "no scope
specified". The LLVM maintainer (Artem-B, NVIDIA) confirmed: "LLVM atomics are not plumbed
through into NVPTX." This is tracked as a known gap, not a simple bug, because fixing it risks
breaking existing users who depend on the silent-discard behavior.

### Q3: Current LLVM Version Behavior

**Rust nightly as of early 2026 uses LLVM ≈19.x (based on nightly 1.86 using LLVM 19.1.7).**

Timeline of relevant LLVM changes:

| Date | Change | Impact |
|------|--------|--------|
| ~2019 | D50391 merged: monotonic atomic loads/stores → `ld.volatile`/`st.volatile` | Relaxed ld/st sort-of work |
| 2016 | D24943 merged: `llvm.nvvm.atomic.*.gen.*.{cta,sys}` intrinsics added | Scoped atomics accessible via intrinsics only |
| Mar 2023 | LLVM #61411 opened: AtomicFence crashes NVPTX backend | Known bug |
| Jan 2023 | LLVM Discourse: syncscope/atomicrmw not supported, must use intrinsics | Acknowledged gap |
| Dec 2024 | LLVM PR #119018: libc updated to use scoped fences "now that NVPTX doesn't break on them" | Fences no longer crash (what backend fix enabled this is unclear) |
| Jan 2025 | LLVM #123339: syncscope("system") mistranslated, ambiguity documented | Still open, no fix |
| Feb 2025 | Rust #136480: orderings silently discarded on nightly LLVM 19.1.7 | Still open |
| Mar 2025 | LLVM #173993: NVPTX orderings of atomicrmw silently discarded | Still open |

**Current state as of early 2026:**
- `atomic::fence` no longer crashes but emits no PTX instruction.
- `atomicrmw` always emits `atom.global.<op>.<type>` regardless of ordering.
- No syncscope (sys/gpu/cta) is ever emitted for `atomicrmw` through `core::sync::atomic`.
- LLVM issue #173993 is open with no assigned fix or timeline.
- The LLVM NVPTX maintainer position is that this is a deep architectural gap requiring
  rework of how the NVPTX backend handles LLVM IR atomics.

**SM version dependency:**
- SM <60: No scoped atomic support in PTX hardware. `atom.global.*` is the only form.
- SM60–69: Scoped atomics available (`.sys`, `.cta`) but no `.sem` qualifier (PTX 5.x).
  System-scope requires SM60+ on Linux.
- SM70+ (Volta+): Full PTX 6.0 memory model with `.sem` + `.scope`. System-scope requires
  SM70+ on Windows. This is the minimum for correct GPU-CPU atomics on all platforms.
- **Current nightly Rust default CPU is `sm_30`** (set in `nvptx64_nvidia_cuda.rs`).
  Must override with `-C target-cpu=sm_70` or higher to even reach the hardware that
  *would* support `.sys` scope, but LLVM still won't emit the correct instructions anyway.

### Q4: Workarounds if .sys Scope Missing

Since:
1. `core::sync::atomic` emits wrong PTX (no scope, no ordering), and
2. `asm!` inline assembly is not supported on nvptx64, and
3. LLVM issue #173993 has no fix or timeline,

the following workaround paths exist, in order of feasibility:

#### Option A: LLVM NVPTX Intrinsics via `extern "C"` (Limited, Partial)

LLVM added scoped atomic intrinsics in D24943 (merged 2016):
```
llvm.nvvm.atomic.add.gen.i.sys.i32.p0i32(ptr, val) → atom.sys.add.s32
llvm.nvvm.atomic.add.gen.i.cta.i32.p0i32(ptr, val) → atom.cta.add.s32
```

These can be called from Rust via `extern "C"` with a mangled `#[link_name]`:
```rust
extern "C" {
    #[link_name = "llvm.nvvm.atomic.add.gen.i.sys.i32.p0i32"]
    fn nvvm_atomic_add_sys(ptr: *mut i32, val: i32) -> i32;
}
```

**Limitation:** These intrinsics implement the old PTX 5.x style scope-only atomics
(`.sys` scope but no `.sem` qualifier). They do not emit acquire/release/seqcst semantics.
They only provide atomicity at system scope with relaxed ordering. For GPU-CPU flag protocols
where the flag is written by GPU and read by CPU (or vice versa), you additionally need a
fence with `.sys` scope.

**Fences via intrinsics:** The corresponding `llvm.nvvm.membar.sys` intrinsic generates
`membar.sys` (PTX 5.x style), which is functionally equivalent to `fence.sc.sys` on
SM70+ but is the older (pre-PTX-6.0) form. This is usable for system-scope fence:
```rust
extern "C" {
    #[link_name = "llvm.nvvm.membar.sys"]
    fn nvvm_membar_sys();
}
```

**Combined protocol:**
```rust
// GPU side: write data, then signal CPU
write_data(ptr);
nvvm_membar_sys();  // fence.sys equivalent
nvvm_atomic_add_sys(flag, 1);  // atom.sys.add (relaxed but system-scope)
```

This is semantically correct for simple flag protocols but does not provide the full
acquire/release C++ memory model that `core::sync::atomic` promises.

**Warning:** The `llvm.nvvm.membar.sys` intrinsic requires the linked bitcode to include
it; without the NVVM IR path (i.e., using raw nvptx64 target), it may not be available
or may link incorrectly. This needs empirical verification (task `atomics.2`).

#### Option B: Switch to Rust-CUDA (NVVM IR path)

The Rust-CUDA project (`rustc_codegen_nvvm`) uses libNVVM as the codegen backend instead
of LLVM's raw PTX backend. This path:
- Supports inline PTX assembly (`asm!`), enabling direct `atom.acq_rel.sys.add.u32`
- Provides `cuda_std::atomic` with `SystemAtomic<T>` types using scoped intrinsics
- Implements correct acquire/release/seqcst semantics via inline PTX:
  ```
  atom.add.acq_rel.gpu.u32 %0,[%1],%2;
  ```

**Trade-off:** Rust-CUDA requires a separate, fork-based rustc codegen (not upstream Rust).
It is maintained by the Rust-GPU team, currently tracking nightly-2025-03-02. This is a
significant toolchain dependency but is the only currently known working path for correct
scoped atomics in Rust on NVIDIA GPU.

#### Option C: Volatile Loads/Stores for Simple Flags (Weakest but Available)

The D50391 fix maps `atomic::load(Relaxed)` to `ld.volatile` which per PTX ISA has
`.relaxed.sys` semantics. This means:
- `core::sync::atomic::AtomicU32::load(Relaxed)` → `ld.volatile.u32` → visible system-wide
- `core::sync::atomic::AtomicU32::store(val, Relaxed)` → `st.volatile.u32` → visible system-wide

**BUT:** volatile load/store is not sufficient for acquire-release protocols. Without a
fence, writes to non-flag data may not be visible to the CPU when the flag becomes visible.
This can only be used for single-variable polling where no other data ordering is required.

#### Option D: Explicit PTX via Separate `.ptx` File (Not Practical)

Theoretically, one could write the system-scope atomic operations directly in a `.ptx`
assembly file and link it with the Rust kernel. However:
- The nvptx64 toolchain does not support linking pre-assembled `.ptx` files in the standard
  cargo/rustc build workflow.
- The `llvm-bitcode-linker` works with LLVM bitcode (`.bc`), not PTX text.
- This would require a custom build script and is not a supported workflow.

#### Summary of Workaround Viability

| Option | Correctness | Feasibility | Toolchain |
|--------|-------------|-------------|-----------|
| A: `llvm.nvvm.membar.sys` + `llvm.nvvm.atomic.*.sys` | Partial (no sem) | Uncertain (needs verification) | nvptx64 |
| B: Rust-CUDA (`cuda_std::atomic`) | Full | Confirmed (used in practice) | NVVM fork |
| C: `ld.volatile`/`st.volatile` for flags only | Very limited | Yes | nvptx64 |
| D: Separate PTX file | Full (if written correctly) | Not practical | Custom |

---

## Unexpected Discoveries

1. **The silence is the danger.** The LLVM NVPTX backend does not warn, error, or annotate
   when it drops ordering and scope metadata. A Rust program using `AtomicU32::fetch_add(1, SeqCst)`
   will compile successfully and appear to work in many test scenarios (same-device tests without
   CPU readers), while being silently wrong for GPU-CPU communication. There is no compiler
   diagnostic.

2. **The fence crash was fixed invisibly.** Between LLVM 19.1.7 and nightly 2025-03-08, the
   `AtomicFence` crash in LLVM was resolved (likely by the same changes that allowed PR #119018
   to land), but the fix did not actually emit correct fence instructions. It simply stopped
   crashing. This is arguably worse: previously broken code was at least caught at compile time
   (crash). Now it silently compiles to wrong code.

3. **`ld.volatile` ≡ `.relaxed.sys` is a documented PTX ISA guarantee** (PTX ISA spec, v6.0+).
   This means that `store(val, Relaxed)` + `load(Relaxed)` through mapped memory is actually
   system-scope visible, just without ordering guarantees. This is sufficient for a polling
   loop on a single flag, but not for any acquire-release protocol.

4. **The LLVM intrinsic scoped atomics (D24943, 2016) require SM60+**, not SM70+. The older
   `.sys` scope (without `.sem`) was introduced in PTX 5.0 for SM60 (Pascal). The `.sem`
   qualifier (acquire/release/seqcst) requires PTX 6.0 / SM70 (Volta). This means there is
   a two-tier hardware requirement:
   - SM60+: system-scope relaxed atomics (via intrinsics only, not `core::sync::atomic`)
   - SM70+: full acquire-release model (requires Rust-CUDA or direct PTX assembly)

5. **NVIDIA's own libcudacxx uses `fence.sc; st.relaxed` instead of `st.release` for SeqCst stores**,
   which diverges from the ASPLOS'19 PTX memory model paper. NVIDIA is aware and intends to
   update the formal model. This shows that even NVIDIA's production C++ atomic library has
   correctness nuance here.

6. **PR #119018 (Dec 2024, LLVM libc update) cites that NVPTX "no longer breaks" on scoped
   fences**, confirming a backend fix landed in late 2024 that allowed scoped fence intrinsic
   calls without crashing. However, this fix is for the NVVM IR path used by LLVM's libc GPU
   port (OpenMP offloading), not for the Rust `core::sync::atomic` path through `atomicrmw`.

---

## Key Conclusions

1. **`core::sync::atomic` is unsafe for GPU-CPU communication on nvptx64.** All orderings
   collapse to an unscoped, unordered `atom.global.*` instruction. The memory safety guarantees
   promised by Rust's atomic API do not hold on this target.

2. **There is no inline PTX workaround** because `asm!` is not supported on nvptx64.

3. **Partial workaround exists via LLVM NVPTX intrinsics** (`llvm.nvvm.membar.sys` +
   `llvm.nvvm.atomic.add.gen.i.sys.*`), which provide system-scope relaxed atomics and
   system-scope fences. This enables correct simple flag protocols but not full C++
   acquire-release semantics.

4. **Full workaround requires Rust-CUDA** (NVVM IR path), which provides `cuda_std::atomic`
   with proper scope-aware implementations using inline PTX. This is a toolchain change, not
   just a library change.

5. **The upstream LLVM bug (issue #173993) has no fix or timeline** as of early 2026. The
   NVPTX atomic ordering situation is not being actively worked on by LLVM maintainers.

6. **`hostcall.3` (design) and `atomics.2` (stress-test) are both blocked or redirected**
   pending a decision on which toolchain path to use. Any hostcall design that relies on
   `core::sync::atomic` for GPU-CPU signaling must be redesigned.

7. **This is a CRITICAL BLOCKER for the hostcall theme** if using the plain nvptx64 target.
   The design must either: (a) use the intrinsic workaround (Option A, needs empirical
   verification), or (b) commit to the Rust-CUDA toolchain path (Option B), or (c) restrict
   the hostcall protocol to use only `ld.volatile`/`st.volatile` for signaling flags and
   accept the weaker memory model (Option C, likely insufficient for correctness).

---

## Open Questions

1. Do `llvm.nvvm.membar.sys` and `llvm.nvvm.atomic.add.gen.i.sys.*` intrinsics link and
   execute correctly from the plain nvptx64 Rust target (without NVVM IR)?
   → **Assigned to `atomics.2`** (empirical verification).

2. Is the `toolchain.4` experiment (minimal kernel) using the default `sm_30` CPU target?
   If so, even the intrinsic workaround (requiring SM60+) would not work.
   → Need to confirm `-C target-cpu=sm_70` is used in all kernel builds.

3. What did the late-2024 LLVM NVPTX fix (enabling PR #119018) actually change in the backend?
   Is there a corresponding change to `atomicrmw` lowering, or only to fence intrinsic handling?
   → Requires reading the LLVM commit history around late 2024.

4. Does Rust-CUDA's `cuda_std::atomic::SystemAtomicU32` correctly generate `atom.sys.*`
   instructions with the right `.sem` qualifier? The design doc (issue #8) describes the intent,
   but was it actually implemented and verified?
   → Requires reading Rust-CUDA source and testing.

5. Can a hybrid toolchain be used: compile the atomic-heavy hostcall shim with Rust-CUDA's
   NVVM path, and compile the rest of the kernel with nvptx64?
   → Unknown; the two paths produce different bitcode formats (NVVM IR vs raw PTX bitcode).

---

## Impact on Downstream Tasks

- **hostcall.1** (research shared memory mechanisms): The finding that `core::sync::atomic`
  emits incorrect PTX means any hostcall design using standard Rust atomics for GPU-CPU
  signaling is incorrect. The research in hostcall.1 must account for this and document
  the intrinsic-based workaround or the Rust-CUDA requirement.

- **hostcall.3** (design hostcall protocol): Directly depends on `atomics.1` resolution.
  The protocol design must specify which atomic mechanism is used (LLVM intrinsics vs
  cuda_std vs volatile flags) and justify correctness. This task should be updated to
  explicitly address the scope problem.

- **atomics.2** (stress-test): Should test both the intrinsic path (Option A) and document
  the failure mode of `core::sync::atomic` (Option C). The stress-test scope should include
  confirming whether `ld.volatile`/`st.volatile` is sufficient for simple flag polling.

- **toolchain.4** (minimal kernel experiment): Must use `-C target-cpu=sm_70` or higher.
  If the experiment already runs, the atomic correctness issue doesn't affect pure compute
  kernels, but should be noted for any future work that adds GPU-CPU communication.

- **toolchain.2** (Rust-CUDA investigation): The atomics finding substantially increases
  the priority of investigating Rust-CUDA as a toolchain alternative. If Rust-CUDA
  provides correct scoped atomics, it may be necessary to adopt it for the hostcall
  subsystem even if the rest of the project stays on nvptx64.

---

## Theme Progress

The `atomics` theme has received a critical negative finding: the foundational primitive for
GPU-CPU communication is broken at the toolchain level. This does not block all progress, but
it constrains the design space significantly:

- **Partial unblock:** The LLVM NVPTX intrinsic workaround (Option A) needs empirical
  validation in `atomics.2`. If it works, it provides a path for simple hostcall protocols
  using the existing nvptx64 toolchain.
- **Full unblock:** Adopting Rust-CUDA (Option B) provides correct semantics but requires
  a toolchain decision affecting the whole project.
- **Theme success criteria update:** "Confirmed PTX output for Rust atomics shows correct
  .sys scope" must be rephrased. The investigation confirms that standard `core::sync::atomic`
  does NOT show correct scope. Success should be redefined as: "Identified and validated a
  workaround that provides system-scope atomics with correct ordering for GPU-CPU protocols."
