# BS3 Synthesis — Strategic Review at Cycle 10
**Date**: 2026-03-11
**Brainstorm seq**: 3
**Trigger**: interval (completed_tasks 10 - last_brainstorm 7 = 3)

## Consensus Points (3+ Reviewers Agree)

### C1: `readonly` on spin-loop loads is a CRITICAL bug (all 4 agree)
`sys_load_acquire_u32/u64` uses `options(readonly)` which allows LLVM's LICM pass
to hoist the load out of spin loops. The GPU warp reads once into a register and
then loops on the stale value forever. This MUST be fixed before hostcall.4.

**Fix**: Add `sys_spin_load_acquire_u32/u64` variants:
- No `readonly` option (prevents hoisting)
- `#[inline(never)]` (second defense — LICM cannot cross call boundaries)
- Include `nanosleep.u32 64;` to yield warp slot during spin

### C2: u64 inline PTX must be verified as first step (all 4 agree)
The hostcall protocol depends entirely on u64 CAS/fetch_add/exchange. These have
never been compiled or executed. While the pattern is identical to u32, the skeptic
correctly identifies this as unverified — if it fails, ADR-3 collapses.

**Action**: New task `atomics.4` — add u64 atomics + verify compilation + unit test.
This is the first prerequisite for hostcall.4, not deferrable.

### C3: `nanosleep.u32` in spin loops (systems, compiler, gpu agree)
The PTX `nanosleep.u32 N` instruction (SM 7.0+) yields the warp slot to the hardware
scheduler. Without it, a spinning warp burns a full slot at max clock, blocking other
warps from making progress. With it, the scheduler can rotate to other warps.

**Integrate into the spin-load variant** (see C1).

### C4: Warp coordination via lane-0 + shfl_sync broadcast (systems, gpu agree)
The correct pattern for warp-level hostcall: lane 0 executes CAS loops, broadcasts
packet index to all lanes via `shfl.sync.idx.b32`. Other lanes fill their own
payload slots. For hostcall.4 initial test: simplify to single-thread kernel.

### C5: `activemask.b32` PTX instruction for active lane mask (systems, gpu agree)
Use `activemask.b32` (PTX ISA 6.2+, SM 7.0+). Do NOT hardcode 0xFFFFFFFF — partial
warps at kernel grid edges would cause incorrect behavior.

### C6: Protocol is fundamentally implementable (systems, compiler, gpu agree)
The lock-free two-stack design translates correctly from ROCm to NVIDIA + Rust.
No fundamental blockers. The devil is in the details (C1-C5 above).

## Dissent and Unrefuted Skeptic Challenges

### D1: atomics.2 timing — parallel vs prerequisite
- **Skeptic**: MUST run atomics.2 before hostcall.4 (multi-warp CAS unverified)
- **Systems/GPU**: atomics.2 should run in parallel, not block hostcall.4
- **Resolution**: COMPROMISE. hostcall.4 starts with single-thread kernel (no multi-warp
  CAS needed). atomics.2 runs in parallel. When hostcall.4 scales to multi-warp, atomics.2
  must already be complete. This satisfies both concerns.

### D2: ABA tag wraparound (skeptic raises, systems dismisses)
- **Skeptic**: 32-bit tag wraps in ~1.6s at peak (1344 warps × 4 CAS/hostcall)
- **Systems**: "not a practical concern"
- **Analysis**: The skeptic's math is correct at theoretical peak throughput. In practice,
  the PCIe latency bottleneck (~5-10µs per hostcall) limits actual throughput to ~100-200K
  hostcalls/sec, giving ~5-12 hours before tag wraparound. For async workloads with sustained
  hostcall traffic, this is still a concern for long-running kernels.
- **Resolution**: DOCUMENT the limitation. For hostcall.4 (testing), 32-bit tag is fine.
  Add a note to the design that production workloads may need 48-bit or 64-bit tags. Can
  be addressed when the per-SM pool optimization is designed (hostcall.6).

### D3: toolchain.2 (Rust-CUDA) priority
- **Skeptic**: Should be parallelized NOW as insurance for async state machine compilation
- **Compiler**: Raises concrete async codegen concerns (brx.idx bugs, indirect calls)
- **Resolution**: PARTIAL ACCEPT. Don't block on toolchain.2, but elevate its priority.
  Add a small investigation task: compile a minimal async state machine for nvptx64 and
  inspect the PTX output. This costs less than a full toolchain.2 investigation and
  provides the critical data point.

### D4: Embassy desk analysis vs actual compilation (skeptic + compiler)
- **Skeptic**: "90% compatible" is unverified — AtomicU8, vtable dispatch, TLS concerns
- **Compiler**: vtable indirect calls + `brx.idx` PTX bugs are concrete risks
- **Resolution**: ACCEPT. Add a small experiment task: attempt to compile Embassy's
  executor core for nvptx64. This is cheaper than full design work and will immediately
  surface real blockers. If it fails, we know exactly what to fix before async-runtime.2.

### D5: ROCm doorbell != HSA signals (skeptic, partially supported by gpu)
- **Skeptic**: Polling doorbell burns 100% CPU, HSA signals use hardware interrupts
- **GPU**: nanosleep mitigates GPU-side waste, but host-side CPU utilization is real
- **Resolution**: ACKNOWLEDGE. The polling approach is acceptable for testing (hostcall.4).
  For production, investigate `cuStreamWaitValue64` (CUDA 12+) as a hardware-assisted
  alternative. This is not urgent for the research phase.

## New Tasks to Spawn

### atomics.4 (NEW — prerequisite for hostcall.4)
- Theme: atomics
- Title: "Add u64 atomics + spin-load variants to gpu-atomics"
- Kind: experiment
- Depends: atomics.3.1
- Scope:
  1. Add `sys_cas_u64`, `sys_fetch_add_u64`, `sys_exchange_u64`
  2. Add `sys_spin_load_acquire_u32/u64` (no readonly, #[inline(never)], nanosleep)
  3. Add `activemask()` helper
  4. Compile for nvptx64 and verify PTX output
  5. Unit test u64 CAS from a simple kernel

### async-runtime.1.1 (NEW — early Embassy compilation test)
- Theme: async-runtime
- Title: "Experiment: attempt Embassy executor compilation for nvptx64"
- Kind: experiment
- Depends: toolchain.4
- Scope: Try compiling embassy-executor with arch-spin for nvptx64. Document what
  fails. Focus on: AtomicU8, vtable indirect calls, TLS usage.

## Task Priority Order

1. **atomics.4** — u64 atomics + spin-load + activemask (BLOCKS hostcall.4)
2. **hostcall.4** — GPU println via hostcall (single-thread first)
3. **atomics.2** — stress-test GPU-CPU atomics (parallel with hostcall.4)
4. **async-runtime.1.1** — Embassy compilation test (parallel, low effort)

## Theme Status Assessment

| Theme | Status | Next Action |
|-------|--------|-------------|
| toolchain | Nearly complete | toolchain.2 can stay deferred |
| atomics | On track | atomics.4 then atomics.2 |
| hostcall | Design done, implementation next | hostcall.4 after atomics.4 |
| async-runtime | Desk analysis only, untested | async-runtime.1.1 early test |
| gpu-std | Speculative, blocked | No change |
| integration | Blocked | No change |

## Key Insight

The project's confirmed successes (inline PTX asm, u32 atomics, hostcall design) create
a solid foundation, but only for the SIMPLE cases. Every confirmed result has been
single-thread, u32, non-async. The transition to multi-warp, u64, async workloads is
where the real risks lie. The skeptic's most valuable contribution is identifying that
the u64 variants and Embassy compilation are completely unverified — these must be tested
immediately before more design work proceeds.
