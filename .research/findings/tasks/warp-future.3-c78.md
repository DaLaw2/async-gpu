# warp-future.3: Add Warp Intrinsics (shfl_sync, syncwarp) to gpu-atomics
**Cycle**: 78 | **Theme**: warp-future | **Kind**: experiment | **Status**: done

## Summary
Added `syncwarp()` and `shfl_sync_idx_u32()` to the gpu-atomics crate. Both compile to correct PTX instructions (`bar.warp.sync` and `shfl.sync.idx.b32`) and pass hardware verification on SM86 (RTX 3060). All 32 lanes in a full warp correctly receive broadcast values via shfl.sync. These intrinsics survive fat LTO across crate boundaries (gpu-atomics → gpu-kernel).

## Findings

### Q: Does shfl.sync.idx.b32 compile correctly from inline PTX asm on nvptx64?
A: **Yes.** The inline asm `shfl.sync.idx.b32 {result}, {val}, {src}, 0x1f, {mask};` compiles to correct PTX. The `0x1f` parameter specifies warp width = 32 (clamp to lane 31). Hardware test confirms all 32 lanes receive the correct broadcast value (0xCAFE_BABE from lane 0).
**Confidence**: high

### Q: Does bar.warp.sync compile correctly from inline PTX asm on nvptx64?
A: **Yes.** The inline asm `bar.warp.sync {mask};` compiles to correct PTX. No deadlocks observed with 32 active lanes. The instruction is used before shfl.sync in the test to ensure convergence.
**Confidence**: high

### Q: Can lane_id() be implemented via %laneid special register?
A: **Already done.** `lane_id()` was already in gpu-atomics using `mov.u32 {id}, %laneid;`. Confirmed working.
**Confidence**: high

### Q: Do these intrinsics survive fat LTO across crate boundaries?
A: **Yes.** The test kernel in gpu-kernel calls `gpu_atomics::syncwarp()` and `gpu_atomics::shfl_sync_idx_u32()` across crate boundaries. With `#[inline(always)]` and fat LTO, the PTX contains the instructions directly in the kernel function body — no unresolved `.extern .func` references.
**Confidence**: high

## PTX Verification
Generated PTX for `test_warp_intrinsics` kernel:
```ptx
bar.warp.sync %r2;               // syncwarp(mask)
shfl.sync.idx.b32 %r3, %r4, %r5, 0x1f, %r2;  // shfl_sync_idx_u32(mask, val, 0)
```

## Hardware Verification
- Target: SM86 (RTX 3060)
- Launch: 1 block x 32 threads
- Result: all 32 lanes received 0xCAFE_BABE (PASSED)

## New API in gpu-atomics
```rust
pub unsafe fn syncwarp(mask: u32);                          // bar.warp.sync
pub unsafe fn shfl_sync_idx_u32(mask: u32, val: u32, src_lane: u32) -> u32;  // shfl.sync.idx.b32
```

## Impact on Downstream Tasks
- **warp-future.4** (WarpFuture PoC): All prerequisite intrinsics are now available. Can proceed with hand-written WarpFuture state machine.
