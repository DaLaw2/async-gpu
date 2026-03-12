# atomics.6: Research LLVM NVPTX atomics scope fixes in latest versions
**Date**: 2026-03-12
**Cycle**: 34 | **Theme**: atomics | **Kind**: investigation | **Status**: done
**Spawned by**: user

## Summary

LLVM has been incrementally fixing NVPTX atomic scope/ordering support since LLVM 21.
Atomic loads, stores, and fences are fully fixed (`.sys` scope + ordering qualifiers).
`cmpxchg` gained scope support via PR #140812 (merged July 2025, in LLVM 21).
However, **`atomicrmw` (fetch_add, fetch_sub, exchange, etc.) remains broken** -- orderings
and scopes are silently discarded (LLVM Bug #173993, filed Dec 2025, still open).
Our inline PTX workarounds in `gpu-atomics` remain necessary for RMW operations.

## Findings

### Q: Has LLVM fixed missing .sys scope for atomicrmw/cmpxchg on NVPTX?

A: **Partially.** The fix has been applied in stages:

| Operation | Status | LLVM Version | Details |
|-----------|--------|-------------|---------|
| `atomic load` | **Fixed** | LLVM 21+ | Emits `ld.acquire.sys.global.b32` etc. via `tryLoad()` path |
| `atomic store` | **Fixed** | LLVM 21+ | Emits `st.release.sys.global.b32` etc. via `tryStore()` path |
| `fence` | **Fixed** | LLVM 21+ | Emits `fence.sc.sys` etc. |
| `cmpxchg` | **Fixed** | LLVM 21+ | PR [#140812](https://github.com/llvm/llvm-project/pull/140812) merged July 16, 2025. Adds `atom.cas` with sem/scope/addrspace qualifiers for all 4 scopes (cta, cluster, gpu, sys). |
| `atomicrmw` (fetch_add, etc.) | **BROKEN** | All versions through LLVM 22.1.1 | `ISD::ATOMIC_LOAD_ADD` etc. not handled in DAG selection; orderings/scopes silently discarded. Emits bare `atom.global.add.u32` without `.sys` or `.acquire`. |

The core issue: `NVPTXISelDAGToDAG.cpp` has `getMemOrder()`, `getAtomicScope()`, and
`getOperationOrderings()` functions that correctly map LLVM IR syncscopes to PTX scopes
(default `""` -> `.sys`, `"device"` -> `.gpu`, `"block"` -> `.cta`). However, these
functions are only called from `tryLoad()`/`tryStore()` code paths. The `atomicrmw`
operations (`ISD::ATOMIC_LOAD_ADD`, `ISD::ATOMIC_LOAD_SUB`, etc.) go through a separate
lowering path in NVPTXIntrinsics.td that does NOT extract or propagate scope/ordering.

**Confidence**: high (verified by our own empirical PTX output in atomics.5, corroborated
by LLVM Bug #173993 filed Dec 30, 2025, and the Jan 2023 discourse thread which remains
unresolved for atomicrmw)

### Q: Which LLVM version correctly emits system-scope atomics?

A: **LLVM 21** (released ~Aug 2025) is the first version with meaningful scope support:

- **LLVM <= 20**: No scope/ordering on loads, stores, fences, or RMW. Loads/stores used
  `.volatile` (which happens to be `.relaxed.sys` per PTX spec). Fences could crash.
- **LLVM 21.0+**: Loads/stores/fences fully fixed with correct `.sys` scope and ordering.
  `cmpxchg` fixed with scope support (PR #140812, July 2025).
- **LLVM 22.1.0** (Feb 2026): No additional atomicrmw fix found. Bug #173993 still open.
- **LLVM 23 (dev)**: No evidence of an atomicrmw scope fix having landed yet.

Key infrastructure in NVPTX.h defines all necessary scope values:
```
enum Scope { Thread=0, Block=1, Cluster=2, Device=3, System=4, DefaultDevice=5 };
```
The `NVPTXScopes` class in NVPTXISelDAGToDAG.cpp correctly maps LLVM syncscopes:
- `""` (system, LLVM default) -> `NVPTX::Scope::System` -> `.sys`
- `"device"` -> `NVPTX::Scope::Device` -> `.gpu`
- `"block"` -> `NVPTX::Scope::Block` -> `.cta`

The infrastructure is fully in place. The gap is that atomicrmw selection simply doesn't
call into it.

### Q: Are there relevant LLVM bug reports or patches?

A: Yes, several:

1. **LLVM Bug #173993** - "[NVPTX] orderings of atomicrmw instructions are silently discarded"
   - Filed: December 30, 2025
   - Status: **Open** (as of March 2026)
   - Assigned to: Artem-B (NVIDIA LLVM maintainer)
   - Key quote: `ISD::ATOMIC_LOAD_xxx` is not specially handled and lowered without ordering
   - References PR #99709 as the load/store fix baseline
   - Source: http://www.mail-archive.com/llvm-bugs@lists.llvm.org/msg95120.html

2. **PR #99709** - "[NVPTX] Select atomic loads and stores"
   - Merged: July 24, 2024
   - Added scope+ordering for atomic loads/stores on Volta+ (SM 70+)
   - Does NOT cover atomicrmw or cmpxchg
   - Source: https://github.com/llvm/llvm-project/pull/99709

3. **PR #140812** - "[NVPTX] cmpxchg with syncscope"
   - Merged: July 16, 2025 (by akshayrdeodhar, NVIDIA)
   - Adds scope+ordering for `cmpxchg` across all 4 scopes
   - Includes TableGen definitions for 3-operand atomic instructions with sem/scope/addrspace
   - Source: https://github.com/llvm/llvm-project/pull/140812

4. **PR #119018** - "[libc] Update fence to use scoped fence"
   - Merged: December 6, 2024
   - Leverages now-working scoped fences on NVPTX
   - Source: https://github.com/llvm/llvm-project/pull/119018

5. **Discourse thread** - "NVPTX: SyncScope/AtomicOrdering of atomicrmw support?"
   - Date: January 31, 2023
   - First report of the problem; suggested using intrinsics as workaround
   - No resolution for atomicrmw
   - Source: https://discourse.llvm.org/t/nvptx-syncscope-atomicordering-of-atomicrmw-support/68090

### Q: Does the latest Rust nightly emit correct system-scope atomics?

A: **Partially.** Here is the LLVM version mapping for Rust:

| Rust Version | LLVM Version | Release Date |
|-------------|-------------|--------------|
| 1.90.0 | LLVM 21 | 2025-09-18 |
| 1.91.0 - 1.94.0 | LLVM 21.x | 2025-10 to 2026-01 |
| 1.95.0 (beta) | LLVM 21.x (likely) | -- |
| 1.96.0 (nightly) | LLVM 21.x or 22.x (unconfirmed) | -- |

LLVM 22.1.0 was released Feb 24, 2026. No "Update to LLVM 22" PR has been found in
rust-lang/rust as of March 2026, so current Rust stable (1.94) and likely beta/nightly
still use LLVM 21.x. The minimum external LLVM was bumped to 20 (PR #145071).

On LLVM 21 (current Rust), `core::sync::atomic` emits:
- **Loads/stores**: Correct `.sys` scope + ordering. Example: `ld.acquire.sys.global.b32`
- **Fences**: Correct. Example: `fence.sc.sys`
- **`compare_exchange`**: Should be correct with scope (PR #140812 landed before LLVM 21 branch).
  However, our atomics.5 research did not empirically verify this.
- **`fetch_add`, `fetch_sub`, `exchange`**: **BROKEN.** Emits `atom.global.add.u32` without
  scope or ordering. This is confirmed by both our empirical testing (atomics.5) and
  LLVM Bug #173993.

### Q: Can we remove our inline PTX workarounds?

A: **Not yet.** Specifically:

| gpu-atomics function | Can replace with core::sync::atomic? | Reason |
|---------------------|--------------------------------------|--------|
| `sys_store_release_u32/u64` | Yes (on LLVM 21+) | `store(val, Release)` now emits correct PTX |
| `sys_load_acquire_u32/u64` | Yes (on LLVM 21+) | `load(Acquire)` now emits correct PTX |
| `sys_cas_u32/u64` | **Possibly** (on LLVM 21+) | PR #140812 adds cmpxchg scope; needs empirical verification |
| `sys_fetch_add_u32/u64` | **NO** | atomicrmw still broken, no scope emitted |
| `sys_exchange_u64` | **NO** | atomicrmw still broken, no scope emitted |
| `sys_spin_load_acquire_u32/u64` | **NO** | No core::sync::atomic equivalent for spin-loop safety (nanosleep, no LICM hoist) |

**Recommendation**: Keep all gpu-atomics inline PTX for now. The RMW operations are
critical for the hostcall protocol's lock-free two-stack design (`sys_cas_u64` for
stack push/pop, `sys_fetch_add_u32` for doorbell counter). Even for operations where
`core::sync::atomic` now works (loads/stores), the inline PTX provides:
1. Consistency -- all atomics go through the same API
2. Stability -- insulated from future LLVM regressions
3. Spin-loop safety -- nanosleep + no LICM hoisting

**When to reconsider**: When LLVM fixes atomicrmw scope support (Bug #173993 is resolved)
AND that fix lands in a Rust nightly we can test against.

## Impact on Downstream Tasks

- **hostcall protocol**: No change needed. Our inline PTX remains the correct approach
  for all GPU-CPU synchronization primitives.
- **gpu-atomics crate**: Keep as-is. Do not replace any functions with core::sync::atomic.
- **Future monitoring**: Track LLVM Bug #173993 and any PRs from akshayrdeodhar or Artem-B
  that add scope/ordering to the `ISD::ATOMIC_LOAD_ADD` / `ISD::ATOMIC_LOAD_SUB` etc.
  lowering path in NVPTXISelDAGToDAG.cpp or NVPTXIntrinsics.td.
- **Potential future task (atomics.7)**: When atomicrmw is fixed in LLVM, empirically
  verify `compare_exchange` and `fetch_add` emit `.sys` scope, then evaluate whether
  gpu-atomics inline PTX can be simplified or deprecated for some operations.
