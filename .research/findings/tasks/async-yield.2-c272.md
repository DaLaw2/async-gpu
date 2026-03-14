# async-yield.2: Prototype async hostcall Futures in gpu-runtime
**Cycle**: 272 | **Theme**: async-yield | **Kind**: experiment | **Status**: done

## Summary
Implemented 4 async hostcall Futures in `gpu_runtime::std_future`: `GpuOpenFuture`, `GpuWriteFuture`, `GpuReadFuture`, `GpuCloseFuture`. Each follows the same Init→Waiting→Done pattern as the existing `GpuPrintFuture`. Also extracted shared helpers `submit_hostcall()` and `check_response()` to reduce duplication. All CI lint checks pass.

## Findings

### Q: Can we generalize GpuPrintFuture to all I/O services?
A: **Yes.** All hostcall futures follow the same three-phase pattern:
1. **Init**: pop free packet, fill header + payload, push to ready stack, ring doorbell → Pending
2. **Waiting**: check CONTROL_READY on the packet → Ready or Pending
3. **Done**: terminal state

The only difference is the payload format (which fields go where) and the return type parsing.

**Confidence**: high

### What was implemented
- `submit_hostcall(buf, service, fill_payload)` — shared helper to allocate/fill/submit a packet
- `check_response(buf, pkt_idx)` — shared helper to check if response is ready
- `GpuOpenFuture` — SERVICE_OPEN: `(path, flags) → Result<fd, errno>`
- `GpuWriteFuture` — SERVICE_WRITE: `(fd, data) → Result<bytes_written, errno>`
- `GpuReadFuture` — SERVICE_READ: `(fd, max_len) → Result<bytes_read, errno>` (copies into caller buffer)
- `GpuCloseFuture` — SERVICE_CLOSE: `(fd) → Result<(), errno>`

### Usage in #[warp_cooperative] async fn
```rust
#[warp_cooperative]
async fn pipeline(buf: *mut u8) {
    let fd = GpuOpenFuture::new(buf, b"data.txt\0", FILE_OPEN_READ).await.unwrap();
    let mut data = [0u8; 56];
    let n = GpuReadFuture::new(buf, fd, &mut data).await.unwrap();
    // ... compute on data ...
    let out_fd = GpuOpenFuture::new(buf, b"out.txt\0", FILE_OPEN_WRITE_CREATE).await.unwrap();
    GpuWriteFuture::new(buf, out_fd, &data[..n]).await.unwrap();
    GpuCloseFuture::new(buf, out_fd).await.unwrap();
    GpuCloseFuture::new(buf, fd).await.unwrap();
}
```

Each `.await` is a yield point where the MIR pass inserts warp convergence barriers. Between awaits, compute runs in SIMT lockstep.

## Impact on Downstream Tasks
- async-yield.3 unblocked: can now build the data pipeline demo
- Epic criterion 1 partially met: async hostcall futures exist, but haven't verified on GPU yet
