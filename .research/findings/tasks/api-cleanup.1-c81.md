# api-cleanup.1: Eliminate gpu-kernel Inline Hostcall Copies
**Cycle**: 81 | **Theme**: api-cleanup | **Kind**: experiment | **Status**: done

## Summary
Removed 5 duplicated hostcall protocol functions from gpu-kernel (hc_pop_free, hc_push, gpu_hostcall_print, gpu_hostcall_request, gpu_hostcall_release) and replaced all call sites with gpu_runtime::hostcall::* equivalents. Also refactored 6 service wrapper functions (open, write, close, read, stdin_read, time) to use gpu_runtime's generic gpu_hostcall_request. All 20+ tests pass. Net reduction: ~115 lines of duplicated protocol code removed.

## Findings

### Q: Which hostcall functions in gpu-kernel are duplicated from gpu-runtime?
A: **5 core protocol functions** were fully duplicated:
1. `hc_pop_free` — CAS-loop pop from free stack
2. `hc_push` — CAS-loop push onto any tagged-pointer stack
3. `gpu_hostcall_print` — full PRINT hostcall cycle
4. `gpu_hostcall_request` — generic hostcall with callback
5. `gpu_hostcall_release` — return packet to free stack

Additionally, 6 service wrappers (open, write, close, read, stdin_read, time) used the local `gpu_hostcall_request` copy.
**Confidence**: high

### Q: Does replacing inline copies with gpu-runtime imports still compile to correct PTX?
A: **Yes.** All functions in gpu-runtime::hostcall are `#[inline(always)]`, so cross-crate calls are fully inlined during fat LTO. The generated PTX is functionally equivalent (same protocol steps), with the bonus of gaining automatic sharding support from gpu-runtime's implementations.
**Confidence**: high

### Q: Are there any cross-crate inlining issues with the consolidated API?
A: **No.** Fat LTO (`-C lto=fat` via nvptx64 profile) ensures all `#[inline(always)]` functions are inlined across crate boundaries. This was already proven in earlier experiments (toolchain.4). The only function kept as a local variant is `hc_pop_free_counted` (benchmark instrumentation with CAS retry counter), which has no equivalent in gpu-runtime.
**Confidence**: high

## Changes Made

### Removed from gpu-kernel
- `hc_pop_free()` (lines 248-262)
- `hc_push()` (lines 266-277)
- `gpu_hostcall_print()` (lines 286-351)
- `gpu_hostcall_request()` (lines 435-488)
- `gpu_hostcall_release()` (lines 492-498)

### Refactored call sites
- `hostcall_print_hello` → uses `gpu_runtime::hostcall::gpu_hostcall_print`
- `hostcall_print_multi` → uses `gpu_runtime::hostcall::gpu_hostcall_print`
- `gpu_hostcall_open/write/close/read` → uses `gpu_runtime::hostcall::gpu_hostcall_request`
- `gpu_hostcall_stdin_read/time` → uses `gpu_runtime::hostcall::gpu_hostcall_request`
- Benchmark kernel `hc_push` calls → uses `gpu_runtime::hostcall::hc_push`

### Kept as-is
- `hc_pop_free_counted` — benchmark-specific instrumented variant (CAS retry counting)
- All WarpFuture code — already used gpu_runtime APIs correctly

## Impact on Downstream Tasks
- **api-cleanup.2**: Public API surface is now cleaner. gpu-kernel only imports from gpu_runtime, no internal protocol duplication.
- **warp-future.5** (proc macro): Generated code can reference gpu_runtime::hostcall directly.
