# async-std-demo.1: Design — demo kernel architecture
**Cycle**: 282 | **Theme**: async-std-demo | **Kind**: design | **Status**: done

## Summary
The demo kernel already exists — `examples/async-pipeline/kernel/src/lib.rs` has a complete `#[warp_cooperative] async fn data_pipeline` using GpuOpenFuture, GpuReadFuture, GpuWriteFuture, GpuCloseFuture, driven by `block_on()`. PTX has 7x `bar.warp.sync` at yield points. No new kernel code needed. Task reduces to running the existing example end-to-end and verifying.

## Findings

### Q: What does the demo architecture look like?
A: **Already implemented.** The async-pipeline example has:

**Kernel** (`examples/async-pipeline/kernel/src/lib.rs`):
```rust
#[warp_cooperative]
pub async fn data_pipeline(buf: *mut u8) -> u32 {
    // 7 await points, each with bar.warp.sync
    let fd = GpuOpenFuture::new(buf, b"pipeline_input.txt", FILE_OPEN_READ).await;
    let data = GpuReadFuture::new(buf, fd, &mut buffer).await;
    // ... transform (uppercase) ...
    let out_fd = GpuOpenFuture::new(buf, b"pipeline_output.txt", FILE_OPEN_WRITE_CREATE).await;
    GpuWriteFuture::new(buf, out_fd, &transformed).await;
    // ... close both ...
}

pub unsafe extern "ptx-kernel" fn async_data_pipeline(buf: *mut u8, output: *mut u32) {
    gpu_panic_init(buf);
    let result = block_on(data_pipeline(buf)).unwrap_or(0xDEAD);
    *output = result;
}
```

**Host** (`examples/async-pipeline/host/src/main.rs`):
- Creates `pipeline_input.txt`
- Launches kernel with hostcall listener
- Verifies `pipeline_output.txt` contains uppercased input
- Full cleanup

**PTX** (verified):
- 7x `bar.warp.sync` at yield points
- 1x `shfl.sync.idx.b32` for discriminant broadcast
- `.version 7.8`, target `sm_86`

### Q: What's needed for async-std C2+C3?
A: **Integration verification:**
1. Run `examples/async-pipeline/host` end-to-end
2. Verify output file has correct uppercased content
3. Confirm PTX has `bar.warp.sync` (already verified: 7 instances)

### Architecture Decision
No new kernel or demo needed. The existing async-pipeline example satisfies:
- **C2**: `#[warp_cooperative] async fn` runs file I/O via Futures with warp yielding (bar.warp.sync in PTX)
- **C3**: Practical demo: read file → compute → write file with I/O yielding

**Confidence**: high

## Impact on Downstream Tasks
- **async-std-demo.2**: Simplified — just run the existing example, no new kernel code
- **async-std-demo.3**: PTX inspection already done (7x bar.warp.sync confirmed)
