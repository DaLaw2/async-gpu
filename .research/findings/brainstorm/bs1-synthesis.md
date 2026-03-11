# Brainstorm BS1 — Synthesis
**Date:** 2026-03-11
**Sources:** bs1-systems.md, bs1-compiler.md, bs1-gpu.md, bs1-skeptic.md

---

## Consensus (3+ agree)

### 1. nvptx64-nvidia-cuda is the correct compilation target
All four analyses converge: the upstream LLVM PTX backend (`nvptx64-nvidia-cuda`) is the right path. `rustc_codegen_nvvm` diverges from VectorWare's approach and should be a fallback only. VectorWare's modifications target `rustc` proper, not a separate codegen backend.

### 2. System-scope atomics MUST be empirically verified before building higher layers
**Critical silent-failure risk.** Rust's `core::sync::atomic` on nvptx64 may not emit `.sys`-scoped PTX atomics, which are required for GPU-CPU shared memory communication. If `Ordering::SeqCst` lowers to `.gpu` or `.cta` scope, the entire hostcall protocol is built on undefined behavior. All four analyses flag this as a top-priority verification item.

### 3. TLS in std is the hardest portability blocker for gpu-std
`std` uses `#[thread_local]` internally (errno, panic state, thread handle). PTX has no TLS segment. Without patching `std` source to remove or redirect TLS usage, `-Zbuild-std=std` for nvptx64 will fail. This is not solvable with a libc shim alone — it requires `std` source modification.

### 4. Register pressure from async state machines is real and severe
Quantitative analysis (GPU expert): adding 3 stacked futures can push register usage to ~95/thread, reducing occupancy to ~33% on Ampere. This is the dominant throughput cost. All analyses agree this must be measured early with `ptxas -v` and managed with `maxrregcount`.

### 5. panic = "abort" is mandatory everywhere
No stack unwinding on GPU. All Cargo profiles for the GPU target must set `panic = "abort"`. All `Drop` impls must be trivial (no unwinding path).

### 6. Start with manual block_on, not full Embassy
Systems and compiler recommend proving the base case first: a manual `block_on` spin-loop executor that validates async state machines compile and `RawWaker` vtable dispatch works through PTX. GPU expert concurs with warp-level task design. Skeptic questions whether Embassy is even portably feasible — a ground-up executor may be simpler.

### 7. Warp divergence fundamentally conflicts with diverse async states
GPU expert quantifies: worst case 3.125% warp utilization with 32 distinct states. Mitigation requires warp-level task assignment (one future per warp) and homogeneous task types. This is not a bug to fix — it is a design constraint to architect around.

### 8. RawWaker / indirect function call through PTX is high risk
Systems and compiler both flag this: `RawWaker` uses a fat pointer (data + vtable). The LLVM NVPTX backend has historical bugs with indirect calls and aggregate ABI lowering. Must be tested empirically before investing in executor infrastructure.

### 9. Minimum GPU target: SM70 (Volta)
System-scope atomics and the revised CUDA memory consistency model require Volta or newer. Pre-Volta hardware lacks the hardware guarantees needed for correct GPU-CPU lock-free communication.

### 10. Toolchain pinning is essential
Pin to a specific nightly rustc with LLVM 17+. Document in `rust-toolchain.toml`. Nightly breakage is a real risk for Tier 3 targets.

---

## Dissent (clear disagreement)

### 1. toolchain.3 priority
- **Skeptic**: toolchain.3 (investigate VectorWare's gpu-kernel ABI) should be the VERY FIRST task. If gpu-kernel ABI requires a proprietary rustc patch with no public equivalent, the entire project scope changes. Currently toolchain.3 gates nothing — this is backwards.
- **Systems/Compiler**: toolchain.1 and toolchain.4 should go first to establish ground truth about what the LLVM PTX backend accepts. gpu-kernel is an optimization over ptx-kernel, not a hard dependency.
- **Resolution**: Both are valid. Compromise: run toolchain.3 in parallel with toolchain.1 (both are investigations with no dependencies). toolchain.3's outcome should be a decision gate before proceeding to experiment tasks.

### 2. Embassy port feasibility
- **Skeptic**: Embassy was designed for single-core embedded. Its CriticalSection, GlobalSyncExecutor, and task queue have no concept of SIMT. The port may require ground-up redesign. A custom minimal executor may be simpler.
- **Others**: Embassy's no_std design makes it a reasonable starting point with modifications.
- **Resolution**: Start with manual block_on (consensus). Embassy evaluation happens in async-runtime.1 investigation, which should explicitly scope the delta between Embassy's assumptions and GPU reality.

---

## Unrefuted Skeptic Challenges (risks to track)

1. **VectorWare's blog posts are marketing, not documentation.** No source code, no compilation flags, no performance numbers, no CUDA version requirements. We may be reverse-engineering, not reproducing.

2. **Performance numbers are absent from all early tasks.** We could build a working system that is 1000x slower than native CUDA and not notice until integration. Hostcall latency and occupancy should be measured in experimental tasks, not deferred.

3. **Hostcall deadlock under GPU watchdog timeout.** Display-attached GPUs have a 2-second watchdog. Any hostcall taking longer kills the kernel. Must design with explicit timeouts or use compute-mode GPU.

4. **Debug experience is near-zero.** No stack traces, no panic messages, no backtrace on GPU. Every early bug will take 10x longer to diagnose. A minimal debugging strategy is needed.

5. **Integration symbol conflicts.** The libc shim and CUDA device runtime may both define `malloc`/`free`. Linker behavior is unspecified. Must prototype linking early.

6. **Hostcall serialization under load.** Even with a ring buffer, multiple warps competing for slots serialize at acquisition. Effective throughput is limited to concurrent slots, not concurrent threads.

---

## Proposed Changes to Themes and Tasks

### New Theme: atomics-verification
The memory model correctness underpins everything. A dedicated theme ensures this is treated as foundational, not an afterthought.

### Task Priority Reordering
1. toolchain.1 and toolchain.3 should run in parallel (both investigation, no deps)
2. toolchain.3 output is a decision gate for the toolchain path
3. Performance measurement required in hostcall.4 and async-runtime.3
4. async-runtime.1 scope expanded to explicitly assess Embassy GPU compatibility

### New Tasks Proposed
- atomics.1: Verify Rust atomic → PTX scope mapping
- atomics.2: Stress-test GPU-CPU atomic communication
- New task in hostcall theme: prototype linking to detect symbol conflicts early
- Expand async-runtime.1 to cover "Embassy vs custom executor" decision

---

## Key Insight

The project's greatest risk is not any single technical challenge — it is the **accumulation of unverified assumptions**. The plan builds each layer on top of the previous one, but verification happens late. The most impactful change is to front-load empirical verification: prove atomics work before designing hostcall, prove indirect calls work before designing the executor, prove linking works before building the libc shim. Each verification either confirms the path or forces a pivot while the cost of pivoting is still low.
