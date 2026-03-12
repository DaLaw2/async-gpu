# api-cleanup.2: Clean Up Public API Surface
**Cycle**: 82 | **Theme**: api-cleanup | **Kind**: design | **Status**: done

## Summary
Restructured gpu-runtime's prelude from a flat dump of all symbols to a curated API surface. Removed low-level atomics and internal protocol helpers from the prelude; added warp intrinsics. Users now get a clean, purpose-driven prelude for common tasks, with explicit module paths for advanced use.

## Findings

### Q: What is the minimal public API surface for gpu-runtime?
A: The prelude should contain:
- **High-level hostcall**: `gpu_hostcall_print`, `gpu_hostcall_request`, `gpu_hostcall_release`
- **Panic**: `gpu_panic_init`
- **Sideband (bulk I/O)**: `gpu_bulk_read`, `gpu_bulk_write`, `sideband_alloc`, `sideband_reset`
- **WarpFuture**: `WarpPoll`, `WarpContext`, `WarpFuture`, `WarpExecutor`, `broadcast_u32`
- **Protocol constants**: SERVICE_*, CONTROL_*, PKT_OFF_PAYLOAD, NULL_INDEX, etc.
- **Warp intrinsics**: `activemask`, `lane_id`, `syncwarp`, `shfl_sync_idx_u32`

Removed from prelude: `membar_sys`, `st_global_u32`, `sys_cas_u32/u64`, `sys_exchange_u64`, `sys_fetch_add_u32/u64`, `sys_load_acquire_u32/u64`, `sys_spin_load_acquire_u32/u64`, `sys_store_release_u32/u64`, `hc_pop_free`, `hc_push`, `read_shard_info`, `pkt_offset`, `get_free_stack_ptr`, `get_ready_stack_ptr`, `GPU_MAX_SPIN`.

These are all still accessible via `gpu_atomics::*` and `gpu_runtime::hostcall::*`.
**Confidence**: high

### Q: Should warp_future be a separate crate or stay as a module?
A: **Stay as a module** in gpu-runtime. The WarpFuture trait and executor are tightly coupled with the hostcall protocol (packet management, control word spinning). Separating into a crate would create a circular dependency (warp_future → hostcall protocol) or require duplicating protocol types. The module provides clean namespacing (`gpu_runtime::warp_future::*`) without the overhead of a separate crate.
**Confidence**: high

### Q: What re-exports should the prelude contain?
A: Replaced `pub use gpu_protocol::*` (which dumped 50+ constants) with selective re-exports of the 20 most commonly needed constants (SERVICE_*, CONTROL_*, packet offsets, file I/O limits). Advanced constants (BUF_OFF_*, shard layout, tagged pointer helpers) remain accessible via `gpu_protocol::*`.
**Confidence**: high

## Changes Made

### Before (prelude)
- All 15 gpu_atomics functions (low-level)
- `gpu_protocol::*` (50+ constants dumped)
- 9 hostcall functions (including internals: hc_pop_free, hc_push, read_shard_info, pkt_offset, get_free_stack_ptr, get_ready_stack_ptr)
- GPU_MAX_SPIN constant
- panic init
- 4 sideband functions
- 5 WarpFuture types

### After (prelude)
- 3 high-level hostcall functions
- panic init
- 4 sideband functions
- 5 WarpFuture types
- 20 commonly needed protocol constants (selective)
- 4 warp intrinsics

### Migration path for advanced users
- Low-level atomics: `use gpu_atomics::{sys_cas_u64, ...};`
- Protocol internals: `use gpu_runtime::hostcall::{hc_pop_free, hc_push, ...};`
- All constants: `use gpu_protocol::*;`

## Impact on Downstream Tasks
- **warp-future.5** (proc macro): Generated code should use `gpu_runtime::hostcall::*` for protocol ops, `gpu_runtime::prelude::*` for high-level types.
- api-cleanup theme is now complete.
