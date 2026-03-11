# Brainstorm BS2 — Synthesis
**Date:** 2026-03-11
**Sources:** bs2-systems.md, bs2-compiler.md, bs2-gpu.md, bs2-skeptic.md
**Trigger:** atomics.1 confirmed core::sync::atomic broken on nvptx64

---

## Consensus (3+ agree)

### 1. NVVM membar intrinsics work and are the correct fence primitive
All four agree that `llvm.nvvm.membar.sys` via `extern "C" { #[link_name] }` produces correct `membar.sys` PTX. These are target intrinsics with TableGen patterns that survive optimization passes. Systems, compiler, and GPU expert endorse this for the fence layer.

### 2. CAS/RMW operations require inline PTX asm — NVVM intrinsics insufficient
Compiler and systems confirm: the `llvm.nvvm.atomic.*` intrinsic set does NOT cover scoped integer CAS (`atom.sys.global.cas.b32`). For CAS, the only reliable path is inline PTX assembly via `core::arch::asm!`.

**CRITICAL UNRESOLVED**: toolchain.1 reported "no inline asm support on nvptx64," but both compiler and GPU expert wrote code using `core::arch::asm!` with PTX syntax. This contradiction MUST be empirically resolved in atomics.2/atomics.3. If inline PTX asm works, the atomics problem is fully solvable. If not, the ROCm-style lock-free protocol must be redesigned to avoid CAS.

### 3. Intra-GPU atomics (.gpu scope) are correct for executor internals
All agree: `core::sync::atomic` is fine for waker bitmasks, task queues, and all operations that stay within the GPU. Only the GPU-CPU communication boundary needs system-scope primitives. This keeps the unsafe footprint small.

### 4. Build a `gpu-atomics` crate to encapsulate the workaround
Systems proposed, compiler designed the API, GPU expert specified the PTX instructions:
- `sys_release_store_u32(ptr, val)` → `st.release.sys.global.u32`
- `sys_acquire_load_u32(ptr)` → `ld.acquire.sys.global.u32`
- `sys_fence()` → `membar.sys`
- `sys_cas_u32(ptr, old, new)` → `atom.sys.global.cas.b32` (needs inline PTX)

### 5. Maintain ADR-1 (nvptx64) with amendment prohibiting core::sync::atomic at GPU-CPU boundary
Systems, compiler, and GPU expert agree: the benefits of upstream nvptx64 outweigh the atomics workaround cost. Amend ADR-1 to explicitly prohibit `core::sync::atomic` for cross-device communication.

### 6. Per-warp slot design for hostcall (ROCm-aligned)
GPU expert provided detailed design: 128-byte aligned slots split into two 64-byte cache lines (request/response), lane-0 aggregation, `nanosleep.u32 200` between polls. ROCm uses the same scoped-atomic approach.

---

## Dissent

### 1. Should we abandon nvptx64? (Skeptic vs rest)
- **Skeptic**: The broken-atomics + no-inline-asm combination is uniquely bad. ADR-1's stability rationale is weakened when "official support" means "silently incorrect." Rust-CUDA provides correct atomics.
- **Rest**: nvptx64 is upstream, the workaround is bounded, Rust-CUDA has supply-chain risk.
- **Resolution**: The skeptic's concern is valid but conditional. If atomics.3 proves inline PTX asm works on nvptx64, the workaround is clean and complete. If inline PTX asm does NOT work, the skeptic's position strengthens dramatically and ADR-1 must be revisited. **atomics.3 is the decision gate.**

### 2. Volatile semantics disagreement
- **GPU expert**: `ld.volatile` = `.relaxed.sys` per PTX ISA 8.5 §9.7.11 (spec-guaranteed)
- **Compiler**: LLVM's NVPTX backend may emit plain `ld.u32` (no `.volatile` qualifier) from Rust's `read_volatile`, which is `.relaxed.cta`, NOT `.relaxed.sys`
- **Resolution**: Empirical verification needed. Compile a `read_volatile` on nvptx64 and inspect the PTX output. If `.volatile` qualifier is present → GPU expert is correct. If not → compiler's concern is valid and volatile path is weaker than assumed.

### 3. Optimizer reordering around extern "C" membar
- **Skeptic**: `extern "C"` function calls are not recognized as memory barriers by LLVM optimizer — data writes may be reordered past the membar call
- **Compiler**: NVVM intrinsics have side-effect annotations, optimizer cannot eliminate them
- **Resolution**: Needs empirical verification. Check generated PTX to confirm data stores appear before membar in the PTX output. If reordering occurs, inline PTX asm with `"~{memory}"` clobber is the fix.

---

## Unrefuted Skeptic Challenges

1. **std internals (Arc, Mutex, Once, stdout lock) are ALL broken for multi-warp use.** Porting std to GPU with `-Zbuild-std=std` means every sync primitive in std is silently incorrect. `println!` from multiple warps will corrupt the output buffer. This is not fixable with a `gpu-atomics` crate — it requires patching std source or restricting std usage to single-warp.

2. **Workaround accumulation risk.** Each broken component gets a workaround. The system becomes correct by coincidence under test conditions. We will not know which combination of workarounds is safe because none individually have clear semantic guarantees.

3. **Function pointer / dynamic dispatch untested.** Embassy's Pender and RawWakerVTable rely on function pointers. LLVM NVPTX has known issues with dynamic dispatch. The "90% compatible" assessment may miss the 10% that breaks wakers.

4. **toolchain.4 success is misleading.** Trivial kernel tests none of the problematic features. The transition to real workloads will be painful.

---

## Key Decisions and Actions

### ADR-1 Amendment
> `core::sync::atomic` is prohibited in any code path that crosses the GPU-CPU boundary. All GPU-CPU synchronization must use the `gpu-atomics` crate. Intra-GPU synchronization may use `core::sync::atomic`. When porting std, all multi-warp sync primitives must be replaced with gpu-atomics equivalents.

### New Tasks
1. **atomics.3** (experiment, HIGH PRIORITY): Implement `crates/gpu-atomics/` — the CRITICAL DECISION GATE
   - Test inline PTX asm (`core::arch::asm!`) on nvptx64 — does it work?
   - If yes: implement `sys_release_store`, `sys_acquire_load`, `sys_cas`, `membar_sys`
   - If no: re-evaluate ADR-1 vs Rust-CUDA
   - Inspect PTX output to confirm `.sys` scope qualifiers
   - Test volatile PTX output (confirm `ld.volatile` or plain `ld`)

2. **atomics.2** (updated): Stress test using gpu-atomics primitives, NOT core::sync::atomic

3. **hostcall.3** dependency update: now depends on atomics.3 (was atomics.1)

### Key Insight
The project's viability on nvptx64 hinges on a single empirical question: **does `core::arch::asm!` with inline PTX work on the nvptx64 target?** If yes, all atomics problems are solvable cleanly. If no, the toolchain choice must be reconsidered. atomics.3 is the experiment that answers this question.
