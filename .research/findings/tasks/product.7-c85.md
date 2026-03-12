# product.7: Add thread/block/warp metadata to host print output
**Cycle**: 85 | **Theme**: product | **Kind**: experiment | **Status**: done

## Summary
Added automatic `[B{block_idx}.T{thread_idx}]` prefix to all GPU print output received by the host. GPU-side writes block_idx and thread_idx to payload+64/+68 (lane 1 area, unused by PRINT message). Host-side reads and prepends the metadata. Works for all print paths: `gpu_hostcall_print`, hand-written WarpFuture, and proc macro-generated WarpFuture.

## Findings

### Q: Can the host print handler automatically prepend [block.thread] to GPU print output?
A: **Yes.** The PRINT packet payload has 2048 bytes but only uses the first 64 bytes (lane 0 slots: 8 bytes msg_len + 56 bytes message). Writing metadata to payload+64 (lane 1 area) is safe and doesn't interfere with message data.

Implementation:
- **GPU-side**: `gpu_hostcall_print()` writes `block_idx` and `thread_idx` at payload+64/+68 after the message. For WarpFuture, lane 0 writes the metadata before syncwarp.
- **Host-side**: `handle_print()` reads the two u32 values and prepends `[B{block}.T{thread}] ` to the message before passing to the callback.
**Confidence**: high

### Q: Does this work for both gpu-libc and std println paths?
A: It works for all paths that use `gpu_hostcall_print` or hand-written PRINT packet filling. The std println path (`std-build-test`) uses a separate `gpu_hostcall_print_raw` that was not modified — it would need a similar change to get metadata. The panic handler uses SERVICE_PANIC (not SERVICE_PRINT) and already prints block/thread info from a different mechanism.
**Confidence**: high

## Implementation

### GPU-side changes:
1. `gpu_runtime::hostcall::gpu_hostcall_print()` — writes block_idx/thread_idx at payload+64/+68
2. `gpu-kernel: warp_multi_init_hostcall()` — lane 0 writes metadata before syncwarp
3. `gpu-kernel: WarpPrintFuture::poll_warp()` — lane 0 writes metadata before syncwarp
4. `warp-macro: #[warp_async]` generated code — writes metadata in INIT state

### Host-side changes:
1. `hostcall.rs: handle_print()` — reads payload+64/+68, formats `[B{}.T{}] ` prefix, extends message

### Test assertion updates:
- `hostcall_print_hello` check: exact match → `.contains()`
- `WarpFuture PoC` check: `.starts_with()` → `.contains()`

## Impact on Downstream Tasks
- All future GPU print output automatically includes block/thread context
- Useful for debugging multi-block/multi-thread kernels
- The `std-build-test` path needs a similar update if full coverage desired
