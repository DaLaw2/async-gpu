# async-bulk-io.2: Implement GpuBulkReadFuture + GpuBulkWriteFuture
**Cycle**: 285 | **Theme**: async-bulk-io | **Kind**: experiment | **Status**: done

## Summary
Implemented `GpuBulkWriteFuture` and `GpuBulkReadFuture` in `gpu-runtime/src/lib.rs` inside the `std_future` module. Both use the same `HostcallState` three-state machine as existing I/O Futures. Compilation verified with `cargo +nightly check` — zero errors.

## Implementation

### GpuBulkWriteFuture
- Fields: `buf`, `sideband`, `fd`, `src`, `len`, `sideband_offset`, `state`
- Init: allocate sideband space via `sideband_alloc()`, copy data to sideband, submit `SERVICE_BULK_WRITE`
- Waiting: `check_response()`, read bytes-written from payload, release packet
- Done: return `Result<usize, i32>`
- Edge cases: len=0 returns Ok(0) immediately; sideband_alloc failure retries on next poll

### GpuBulkReadFuture
- Fields: `buf`, `sideband`, `fd`, `dst`, `max_len`, `sideband_offset`, `state`
- Init: allocate sideband space, submit `SERVICE_BULK_READ` (no data copy needed — host writes to sideband)
- Waiting: `check_response()`, read bytes-read from payload, copy sideband→dst, release packet
- Done: return `Result<usize, i32>`
- Edge cases: max_len=0 returns Ok(0); bytes_read==0 returns Ok(0) (EOF); FILE_ERROR_SENTINEL returns Err(-1)

### Re-exports
Added all 6 Future types to `prelude`: `GpuOpenFuture`, `GpuWriteFuture`, `GpuReadFuture`, `GpuCloseFuture`, `GpuBulkWriteFuture`, `GpuBulkReadFuture`.

## Verification
- `cargo +nightly check` on gpu-runtime: zero errors, zero warnings (only PTX feature warning)
- Follows exact same pattern as GpuWriteFuture/GpuReadFuture but with sideband buffer
- Protocol matches sync `gpu_bulk_write`/`gpu_bulk_read` exactly (payload slots 0=fd, 1=offset, 2=len)

**Confidence**: high

## Impact on Downstream Tasks
- async-bulk-io.3: Can now write end-to-end test using these Futures in a `#[warp_cooperative] async fn`
