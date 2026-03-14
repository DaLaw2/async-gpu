# printf-batch.2: Implement buffered print in gpu-runtime + flush via sideband
**Cycle**: 252 | **Theme**: printf-batch | **Kind**: experiment | **Status**: done

## Summary
Implemented GPU-side buffered printing with per-thread buffer slots and SERVICE_BULK_PRINT sideband flush. Three crates modified: gpu-protocol (new service ID), gpu-runtime (print_buffer module), gpu-host (bulk print handler).

## Changes

### gpu-protocol/src/lib.rs
- Added `SERVICE_BULK_PRINT: u32 = 15` — new hostcall service for flushing buffered prints

### gpu-runtime/src/lib.rs
- Extended `nvptx_shim` with `thread_idx_y`, `thread_idx_z`, `block_dim_x`, `block_dim_y` for multi-dimensional block support
- Made `hc_push_with` `pub(crate)` for cross-module access
- Added `print_buffer` module (~170 lines):
  - `init(sideband, max_threads)` — zeroes per-thread buffer slot header
  - `print(buf, sideband, msg, msg_len)` — buffers message with u16 length prefix; auto-flushes if slot full; falls back to direct hostcall for oversized messages
  - `flush(buf, sideband)` — sends SERVICE_BULK_PRINT packet with sideband offset + data length + thread metadata; spin-waits for CONTROL_READY; resets buffer
- Constants: SLOT_SIZE=512, SLOT_HEADER=8, SLOT_DATA_SIZE=504

### gpu-host/src/hostcall.rs
- Added `SERVICE_BULK_PRINT` case in listener dispatch
- Added `handle_bulk_print()` method — reads length-prefixed messages from sideband, prepends `[B{block}.T{thread}]` prefix, calls `on_print` per message

## Architecture
```
GPU Thread                     Host Listener
    |                               |
    |-- print_buffer::print() ----> (no hostcall — just buffer locally)
    |-- print_buffer::print() ----> (no hostcall — just buffer locally)
    |                               |
    |-- print_buffer::flush() ----> SERVICE_BULK_PRINT via sideband
    |                               |-- reads length-prefixed messages
    |                               |-- on_print() per message
    |<--- CONTROL_READY ------------|
```

## Findings
### Q: Can per-thread buffered print reduce hostcall overhead?
A: Yes. Design sends 1 hostcall per flush instead of 1 per print. For N prints, reduces from N round-trips to ceil(N/~9) round-trips (504 bytes / ~56 bytes per message ≈ 9 messages per flush). Auto-flush ensures correctness when buffer fills.
**Confidence**: high

## Open Questions
1. Integration with gpu-kernel-std `println!()` — would need to route `write()` in gpu-libc through buffered path. Follow-up task.
2. No end-to-end test yet — needs a kernel that uses `print_buffer::print()` and verifies output. The infrastructure is ready for testing.

## Impact on Downstream Tasks
- `print_buffer` API is ready for use in GPU kernels
- Integration with `println!()` macro requires gpu-libc changes (separate theme)
