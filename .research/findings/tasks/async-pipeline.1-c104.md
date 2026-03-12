# async-pipeline.1: Move warp_hostcall_submit and warp_hostcall_wait_u64 to gpu-runtime
**Cycle**: 104 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Moved `warp_hostcall_submit` and `warp_hostcall_wait_u64` from gpu-kernel (private local functions) to gpu-runtime's `warp_future` module as public `#[inline(always)]` functions. This makes them available to any GPU kernel crate and to proc-macro-generated code.

## Findings
### Q: Where are the functions defined and what do they depend on?
A: Both were private `unsafe fn` in `gpu-kernel/src/lib.rs` (lines 2270-2363). They depend on:
- `gpu_runtime::hostcall::{hc_pop_free, pkt_offset, read_shard_info, get_ready_stack_ptr, hc_push, gpu_hostcall_release}` — all already `pub`
- `gpu_runtime::warp_future::{WarpPoll, WarpContext, broadcast_u32}` — same module
- `gpu_atomics::{syncwarp, sys_store_release_u32, sys_fetch_add_u64, sys_spin_load_acquire_u32}` — added to warp_future imports
- `gpu_protocol::*` — added to warp_future imports

**Confidence**: high

### Q: Can they be moved as-is?
A: Yes. Only change needed was replacing `gpu_runtime::` prefixed paths with `crate::` paths (since they're now inside gpu-runtime). Added `use gpu_protocol::*` to the module. All function signatures identical.

**Confidence**: high

### Q: Does the PTX output remain correct?
A: Yes. `#[inline(always)]` ensures the functions are fully inlined into callers. PTX output contains 76 `atom.add.sys.global.u64` and 151 `shfl.sync.idx.b32` — same as before the move.

**Confidence**: high

## Changes Made
1. `crates/gpu-runtime/src/lib.rs`:
   - Added `sys_store_release_u32, sys_fetch_add_u64, sys_spin_load_acquire_u32` to warp_future imports
   - Added `use gpu_protocol::*` to warp_future module
   - Added `pub unsafe fn warp_hostcall_submit(...)` with full doc comments
   - Added `pub unsafe fn warp_hostcall_wait_u64(...)` with full doc comments
   - Added both to prelude re-exports

2. `crates/gpu-kernel/src/lib.rs`:
   - Replaced local function definitions with `use gpu_runtime::warp_future::{warp_hostcall_submit, warp_hostcall_wait_u64}`

## Unexpected Discoveries
None — straightforward move.

## Impact on Downstream Tasks
- **async-pipeline.2** (generalize macro): UNBLOCKED. The proc macro can now generate code that calls `gpu_runtime::warp_future::warp_hostcall_submit` and `warp_hostcall_wait_u64` directly.
- **async-pipeline.3** (branching example): UNBLOCKED.
- **async-pipeline.4** (pipelining example): UNBLOCKED.
