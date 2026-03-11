# Brainstorm Round 2 — Compiler Engineer Perspective
# Focus: Atomics Breakage on nvptx64 and Workaround Analysis

Date: 2026-03-11
Author: Compiler Engineer (bs2)

---

## 1. The Problem in Detail: LLVM NVPTX Syncscope Erasure

LLVM's NVPTX backend currently strips all `syncscope` metadata from `atomicrmw`,
`cmpxchg`, and `fence` instructions before lowering to PTX. This means:

- Every Rust `atomic::store(..., Ordering::SeqCst)` becomes a bare `st.u32` (or
  similar), not `st.release.sys`.
- `atomic::fence(Ordering::Acquire)` compiles to **zero PTX instructions** — the
  fence is silently dropped.
- All orderings collapse to the same unqualified, non-scoped PTX load/store.

The bug is tracked as LLVM issue #173993 and Rust issue #136480. There is no
committed fix and no timeline for resolution as of early 2026.

Root cause: the NVPTX SelectionDAG lowering path does not consult the `syncscope`
string attached to the LLVM IR instruction when emitting PTX. The syncscope value
exists in IR (LLVM IR text will show `syncscope("agent")`, etc.) but the NVPTX
backend's instruction selection simply ignores it and falls through to the
unscoped variant.

---

## 2. Option A Deep Dive: LLVM NVVM Intrinsics via `#[link_name]`

### How NVVM Intrinsics Survive the Pipeline

LLVM NVVM intrinsics (those in the `llvm.nvvm.*` namespace) are special: they are
treated as "target intrinsics" by the LLVM middle-end. The NVPTX backend's
`NVPTXIntrinsics.td` contains direct patterns that lower them to specific PTX
mnemonics, bypassing SelectionDAG's generic atomic-lowering path entirely. This is
the key advantage over relying on `atomicrmw` with `syncscope`.

Example intrinsic family:

```
llvm.nvvm.membar.gl   → membar.gl   (GPU-scope fence)
llvm.nvvm.membar.sys  → membar.sys  (system-scope fence)
llvm.nvvm.membar.cta  → membar.cta  (thread-block-scope fence)
```

For atomics, the relevant intrinsics are the `llvm.nvvm.atomic.*` family:

```
llvm.nvvm.atomic.load.add.f32.p0f32
llvm.nvvm.atomic.load.add.f32.p1f32   (global address space)
llvm.nvvm.atomic.load.inc.32.p0i32
...
```

However, the NVVM atomic intrinsics are mostly for floating-point and specialized
operations (inc, dec). For integer atomics with explicit scope, the cleaner path
in more recent LLVM is PTX 7.x `atom.sys.*` instructions — but accessing those
requires inline PTX assembly or waiting for LLVM to expose them as intrinsics.

The `membar.*` intrinsics ARE available and work reliably:

```rust
extern "C" {
    #[link_name = "llvm.nvvm.membar.sys"]
    fn membar_sys();
    #[link_name = "llvm.nvvm.membar.gl"]
    fn membar_gl();
    #[link_name = "llvm.nvvm.membar.cta"]
    fn membar_cta();
}
```

**Survival through optimization passes**: Because these are recognized target
intrinsics, LLVM's middle-end passes (instcombine, GVN, etc.) will not fold or
eliminate them unless they can prove the call has no effect — which they cannot,
since the intrinsic has a side-effect annotation. `membar.sys` is memory-barrier
semantics so it will survive `-O3`. This is confirmed behavior, not a guess.

### What Is NOT Available via NVVM Intrinsics

The integer `atom.sys.add`, `atom.sys.cas`, etc. PTX instructions (scoped atomics
from PTX ISA 6.0+) do not have corresponding `llvm.nvvm.*` intrinsics as of LLVM
17/18. The NVVM intrinsic set predates the PTX scoped-atomics feature. This means
Option A covers **fences** robustly but does NOT give us scoped atomic RMW
operations without inline PTX assembly.

### Recommendation for Option A

Use `llvm.nvvm.membar.sys` (and `.gl`) as the fence primitive. This is safe,
portable across LLVM versions that support NVPTX, and will not be misoptimized.
Pair it with `core::ptr::read_volatile` / `write_volatile` for the load/store
sides (see Option C below).

---

## 3. Safe Rust Wrapper Around NVVM Intrinsics

A GPU-side `GpuAtomic<T>` type can provide an `AtomicU32`-like API by composing:

1. `volatile` loads/stores (for the actual memory access)
2. `membar_sys()` calls (for the ordering fence)

Sketch of the wrapper design:

```rust
// In a `gpu_sync` crate, no_std + no linkage to std atomics
use core::cell::UnsafeCell;

pub struct GpuAtomic<T>(UnsafeCell<T>);

unsafe impl<T: Copy> Sync for GpuAtomic<T> {}

impl GpuAtomic<u32> {
    /// Sequentially consistent store
    #[inline(always)]
    pub fn store_seq_cst(&self, val: u32) {
        unsafe {
            membar_sys();                              // pre-fence (release)
            core::ptr::write_volatile(self.0.get(), val);
            membar_sys();                              // post-fence (seq_cst)
        }
    }

    /// Sequentially consistent load
    #[inline(always)]
    pub fn load_seq_cst(&self) -> u32 {
        unsafe {
            membar_sys();                              // acquire fence
            core::ptr::read_volatile(self.0.get())
        }
    }

    /// Release store
    #[inline(always)]
    pub fn store_release(&self, val: u32) {
        unsafe {
            membar_sys();
            core::ptr::write_volatile(self.0.get(), val);
        }
    }

    /// Acquire load
    #[inline(always)]
    pub fn load_acquire(&self) -> u32 {
        unsafe {
            let v = core::ptr::read_volatile(self.0.get());
            membar_sys();
            v
        }
    }
}
```

**Key design decisions**:
- `UnsafeCell` ensures the compiler does not reorder accesses through its alias
  analysis; `write_volatile` prevents store elimination.
- Using `.sys` scope is conservative (correct for host-device shared memory).
  A `.gl` variant would suffice for pure GPU inter-CTA communication.
- No `compare_exchange`: this requires inline PTX `atom.sys.cas.b32` or a
  spinloop around volatile load + conditional volatile store (which is NOT
  atomically safe for CAS semantics). This is a hard limitation of the
  volatile+fence approach for CAS.

For actual atomic RMW (add, CAS, exchange), we need inline PTX assembly:

```rust
#[inline(always)]
pub unsafe fn atomic_cas_sys(ptr: *mut u32, old: u32, new: u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "atom.sys.global.cas.b32 {0}, [{1}], {2}, {3};",
        out(reg32) result,
        in(reg64) ptr,
        in(reg32) old,
        in(reg32) new,
        options(nostack)
    );
    result
}
```

Inline PTX asm is available on `nvptx64-nvidia-cuda` via `core::arch::asm!` in
nightly Rust with the PTX ISA constraint syntax. This is the only reliable path
for scoped integer CAS/RMW.

---

## 4. Volatile Semantics in PTX: Does `read_volatile` Map to `.relaxed.sys`?

This requires careful analysis. The Rust reference says `read_volatile` /
`write_volatile` lower to LLVM `load volatile` / `store volatile`. In LLVM IR,
`volatile` is an access modifier indicating the load/store cannot be reordered
relative to other volatile accesses *by the compiler*. It does NOT carry
`syncscope` or ordering semantics in the hardware sense.

When the NVPTX backend lowers an LLVM `volatile` load:
- It emits `ld.u32` with no scope qualifier (unqualified = `.weak` in PTX ISA
  7.0+ terminology, equivalent to `.relaxed.cta` in older ISA docs).
- It does NOT emit `.relaxed.sys` or `.acquire.sys`.

**Therefore `read_volatile` does NOT map to `.relaxed.sys`.** It maps to
unscoped `ld` which, per PTX ISA 7.5+, has semantics equivalent to
`.relaxed.cta` (block-scope only). This is insufficient for inter-CTA or
host-device communication.

The correct statement is: `volatile` prevents compiler reordering but provides
no hardware memory ordering guarantee beyond the thread's own program order, and
its scope is effectively thread-block-local in PTX semantics.

This is a significant caveat for the volatile+fence strategy. The `membar.sys`
calls provide the ordering fence, but the volatile loads/stores themselves do not
have system scope in the PTX output. Whether this combination (unscoped
load/store + sys fence) achieves the desired semantics depends on the PTX ISA
definition.

Per PTX ISA 7.5, section on memory consistency: a `membar.sys` between an
unscoped `ld` and subsequent computation does establish a system-scope acquire
barrier. The combination:

```
membar.sys
ld.u32 rA, [addr]   // volatile load
```

is functionally equivalent to `ld.relaxed.sys` in terms of the memory model,
because `membar.sys` forces all prior memory operations visible at system scope
before the load proceeds. Similarly:

```
st.u32 [addr], rB   // volatile store
membar.sys
```

approximates `st.release.sys`.

So the volatile+membar approach IS valid for acquire-release semantics, but it
requires careful fence placement and is less efficient than native scoped
atomics (two instructions vs. one).

---

## 5. Combined Approach: Option A (Fences) + Option C (Volatile Loads/Stores)

The combined approach is viable and provides correct acquire-release semantics
for inter-thread and host-device communication. The pattern:

| Rust ordering | Implementation |
|---------------|----------------|
| `Relaxed` load | `read_volatile` (unscoped ld) |
| `Acquire` load | `membar.sys` then `read_volatile` |
| `Release` store | `write_volatile` then `membar.sys` |
| `SeqCst` store | `membar.sys`, `write_volatile`, `membar.sys` |
| `SeqCst` load | `membar.sys`, `read_volatile`, `membar.sys` |
| `fence(SeqCst)` | `membar.sys` |
| CAS / RMW | Inline PTX `atom.sys.*` required |

The main limitation is performance: each ordered access incurs at least one
`membar.sys` which is a full system-scope fence (expensive, ~hundreds of cycles).
For GPU-internal (CTA-to-CTA) communication, `membar.gl` is cheaper and
sufficient.

A production implementation should offer both `GpuAtomic::<T>::sys` and
`GpuAtomic::<T>::gpu` variants to avoid paying the system-scope cost when
host-device sync is not needed.

---

## 6. LLVM Fix Prospects: Upstream Contribution Assessment

The fix required in LLVM is in `lib/Target/NVPTX/NVPTXISelLowering.cpp` (or
`NVPTXAtomicExpand.cpp` if it exists). The lowering of `atomicrmw` and `fence`
instructions needs to:

1. Consult the `syncscope` string on the IR instruction.
2. Map Rust/LLVM syncscope names to PTX scope qualifiers:
   - `""` (default / device scope) → `.gpu` or `.cta` depending on instruction
   - `"agent"` → `.gpu`
   - `"system"` → `.sys`
3. Emit the appropriate PTX mnemonic with scope qualifier.

For `fence` specifically, the fix would need to select between `membar.cta`,
`membar.gl`, and `membar.sys` based on syncscope + ordering.

**Difficulty assessment**: Medium. The NVPTX backend already emits scoped PTX
in some cases (e.g., `ld.acquire.gpu` for certain patterns). The missing piece
is wiring syncscope metadata through SelectionDAG to the final instruction
selection. This is a focused, well-scoped change — not a backend rewrite.

**Contributability**: High in principle. The Rust GPU community (rust-gpu,
Rust-CUDA authors) and NVIDIA both have incentive to fix this. NVIDIA's own
NVVM library correctly emits scoped PTX, which suggests the PTX backend CAN
handle it. The issue is that the LLVM community path (non-NVVM) has not been
maintained to the same standard.

**Proposed contribution path**:
1. Write a minimal LLVM IR test case demonstrating syncscope erasure.
2. Identify the specific SDNode or MI pass that drops the syncscope attribute.
3. Submit an LLVM patch adding syncscope → PTX scope qualifier mapping in
   `NVPTXISelLowering` or `NVPTXAsmPrinter`.
4. Add lit tests for `ld.acquire.sys`, `st.release.gl`, `membar.sys` from IR.

This is a good candidate for a targeted upstream contribution. Estimated effort:
1-2 weeks for someone familiar with LLVM backend development.

---

## 7. Impact on `-Zbuild-std=std`: Arc, Mutex, Internal Atomics

This is a serious concern. `std` uses `core::sync::atomic` extensively:

- `Arc<T>` uses `AtomicUsize` for reference counting (acquire/release).
- `Mutex<T>` (in `parking_lot` or `std`) uses atomic spin state.
- `Once`, `OnceLock`, `LazyLock` use atomics for initialization guards.
- The global allocator (`dlmalloc`, `buddy_system_allocator`) may use atomics for
  internal free-list management.

With the broken atomics, all of these will compile but produce **incorrect PTX**
with no hardware ordering guarantees. Specific failure modes:

- `Arc` reference count races: decrement may not be visible to other threads
  before the destructor runs → use-after-free.
- `Mutex` spin may never observe lock release → deadlock (on GPU: infinite spin
  wasting warp resources).
- `Once` initialization guard may run initializer multiple times across warps.

**For single-warp code** (common in early GPU kernels), this may not manifest
because a single warp executes in program order with no inter-thread races. But
any code using `Arc` across multiple warps or CTAs is silently broken.

**Mitigation without LLVM fix**:
- Avoid `Arc` in multi-warp GPU code; use raw pointers with explicit lifetime
  management.
- Replace `std::sync::Mutex` with `GpuAtomic`-based spinlocks using `membar.sys`.
- Be aware that `-Zbuild-std=std` on GPU is fundamentally unsafe with current
  LLVM until the syncscope bug is fixed.
- Document this clearly in the project: std can be compiled but its sync
  primitives are unsound on GPU.

A pragmatic short-term approach: build a `gpu-std` crate shim that re-exports safe
subsets of std (collections, formatting, etc.) but replaces all sync primitives
with GPU-correct implementations.

---

## 8. Summary and Recommended Actions

| Option | Coverage | Reliability | Effort |
|--------|----------|-------------|--------|
| A: NVVM intrinsics (membar.sys) | Fences only | High (pattern-matched in backend) | Low |
| B: Rust-CUDA / NVVM IR | Full atomics + fences | High | High (toolchain change) |
| C: Volatile loads/stores | Ordering via fence | Medium (needs careful placement) | Low |
| A+C combined | Full acquire-release (no CAS) | High with correct patterns | Medium |
| Inline PTX asm | Full (CAS + RMW + fence) | High | Medium |
| LLVM upstream fix | Full, transparent | High (long-term) | High |

**Immediate recommendation**: Implement `GpuAtomic<T>` using the A+C combined
approach as the project's atomic primitive. Add inline PTX paths for CAS/RMW.
Document that `-Zbuild-std=std` sync types (Arc, Mutex) are unsound until LLVM
is fixed.

**Medium-term**: File or upvote the LLVM issue, provide a minimal reproducer, and
consider drafting the NVPTXISelLowering patch as a contribution — this unblocks the
entire Rust-on-GPU ecosystem, not just this project.

**Longer-term**: Evaluate Rust-CUDA's NVVM IR path (Option B) as an alternative
toolchain if the LLVM fix stalls; it handles scoped atomics correctly because it
goes through NVIDIA's own compiler rather than the community NVPTX backend.
