# async-bulk-io.3: End-to-end async bulk I/O test
**Cycle**: 286 | **Theme**: async-bulk-io | **Kind**: experiment | **Status**: done

## Summary
End-to-end test of `GpuBulkReadFuture` and `GpuBulkWriteFuture` — PASSED. Added `bulk_data_pipeline` kernel to the async-pipeline example: reads 77 bytes via sideband bulk read, swaps case, writes via sideband bulk write. Host verification confirms correct output. PTX has 14 `bar.warp.sync` instructions (7 from each pipeline).

## Implementation

### Kernel (`async-pipeline/kernel/src/lib.rs`)
- Added `#[warp_cooperative] pub async fn bulk_data_pipeline(buf, sideband) -> u32`
- 6 `.await` points: open_read, bulk_read, close_read, open_write, bulk_write, close_write
- Transform: swap ASCII case (lowercase↔uppercase)
- Entry point: `async_bulk_pipeline(buf, sideband, output)` with `block_on()`

### Host (`async-pipeline/host/src/main.rs`)
- Refactored into `run_small_io_demo()` + `run_bulk_io_demo()`
- Bulk demo: 77-byte input "The quick brown fox..." → verified case-swapped output
- Shared `verify_file()` helper for both demos

### PTX Verification
- 14 `bar.warp.sync` total (7 per pipeline)
- 0 unresolved `.extern .func`
- 5 `.ptr .align` (handled by build.rs post-processing)
- MIR pass: `bulk_data_pipeline` — 6 polls, 6 suspensions, 7 returns

### Test Results
```
Demo 1: PASSED — 30 bytes, "HELLO FROM GPU ASYNC PIPELINE!"
Demo 2: PASSED — 77 bytes, "tHE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. bULK SIDEBAND i/o TEST FROM gpu!"
```

**Confidence**: high

## Impact
- async-bulk-io theme: ALL 3 success criteria met → theme completed
  - C1: GpuBulkReadFuture implements Future ✓
  - C2: GpuBulkWriteFuture implements Future ✓
  - C3: End-to-end test with #[warp_cooperative] async fn ✓
