# async-bulk-io.1: Investigation — sideband bulk I/O API review
**Cycle**: 285 | **Theme**: async-bulk-io | **Kind**: investigation | **Status**: done

## Summary
Sideband bulk I/O is fully implemented at the sync level. `gpu_bulk_read`/`gpu_bulk_write` use a 1MB bump-allocated sideband buffer. Same HostcallState three-state machine (Init→Waiting→Done) used by existing Futures applies. Implementation is straightforward: wrap the same protocol with Future poll pattern.

## Findings

### Q: What are gpu_bulk_read/gpu_bulk_write signatures?
A: Both take `(buf: *mut u8, sideband: *mut u8, fd: u64, src/dst: *const/mut u8, len: usize) -> usize`. They allocate sideband space, copy data to/from sideband, submit SERVICE_BULK_READ (12) or SERVICE_BULK_WRITE (11), and spin-wait for response.

### Q: Can we reuse HostcallState?
A: **Yes.** Same three-state pattern: Init (alloc sideband + submit hostcall), Waiting (poll CONTROL_READY), Done. The only addition is sideband allocation and data copy in the Init state.

### Q: Design for GpuBulkReadFuture/GpuBulkWriteFuture?
A:
```rust
pub struct GpuBulkReadFuture {
    buf: *mut u8,
    sideband: *mut u8,
    fd: i32,
    out_buf: *mut u8,
    max_len: u32,
    sideband_offset: u64,  // allocated in Init
    state: HostcallState,
}
// Returns Result<usize, i32>
// Init: sideband_alloc() + submit SERVICE_BULK_READ
// Waiting: check_response(), copy sideband→out_buf
// Done: return result
```

**Confidence**: high
