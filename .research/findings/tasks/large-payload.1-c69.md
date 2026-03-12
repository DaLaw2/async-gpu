# large-payload.1: Survey bulk data transfer approaches
**Cycle**: 69 | **Theme**: large-payload | **Kind**: investigation | **Status**: done

## Summary

Surveyed three approaches for transferring data larger than the current 56-byte per-slot
limit: multi-packet chaining, sideband mapped buffer with offset+length, and enlarging
packet payload slots. Recommends sideband buffer approach as simplest and most flexible.

## Findings

### Q: Multi-packet chaining vs larger mapped buffer with offset+length?

**Multi-packet chaining:**
- GPU writes data across N packets, host reassembles
- Pro: No new memory allocation; reuses existing pool
- Con: Complex synchronization (must ensure all N packets arrive before processing)
- Con: Deadlock risk — if pool has 64 packets and thread needs 65, it blocks forever
- Con: Reduces available packets for other threads during transfer

**Sideband mapped buffer (RECOMMENDED):**
- Allocate a separate large mapped buffer (e.g., 1MB) alongside the hostcall buffer
- Hostcall packet carries: offset (u64) + length (u64) + service ID
- GPU writes data to sideband buffer at offset, then sends hostcall with metadata
- Host reads data from sideband at offset, processes, writes response back
- Pro: Zero protocol changes to packet format
- Pro: No deadlock risk (one packet per request regardless of data size)
- Pro: Arbitrary data size up to sideband buffer capacity
- Con: Requires new buffer allocation and synchronization

**Confidence**: high

### Q: What synchronization is needed for a sideband data channel?

For sideband buffer approach:
1. GPU writes data to sideband buffer at allocated offset
2. GPU issues `st.release.sys` fence after data write
3. GPU sends hostcall packet with (offset, length, service)
4. Host sees packet → data at offset is guaranteed visible (release-acquire pair)
5. Host processes data, writes response to sideband
6. Host issues normal CONTROL_READY on hostcall packet
7. GPU sees CONTROL_READY → reads response from sideband

The existing hostcall doorbell + CONTROL_FILLED/CONTROL_READY protocol already provides
the necessary synchronization. No new synchronization primitives needed.

**Offset allocation**: Simple bump allocator on GPU side (atomic fetch_add on a shared
offset counter). Host resets counter when buffer wraps or on explicit reclaim.

**Confidence**: high

### Q: How do existing GPU frameworks handle large data transfer?

- **ROCm hostcall (AMD)**: Uses 64-byte payload per packet. Large data via device memory
  pointers passed in packet — host uses DMA to read/write device memory directly. Not
  applicable here (we use mapped memory, not device memory).

- **CUDA cooperative groups**: Not relevant (kernel-to-kernel, not kernel-to-host).

- **VectorWare (inferred)**: Blog mentions "file I/O" working from GPU. Likely uses a
  similar sideband buffer or multi-call chunking approach. No public implementation details.

- **GPU printf**: Uses a circular buffer in device memory. Host drains it after kernel
  completes. Not applicable for interactive hostcall.

**Confidence**: medium (limited public documentation on GPU→host large transfers)

### Q: What is the performance ceiling for mapped memory bandwidth?

- PCIe 3.0 x16: ~12 GB/s unidirectional
- PCIe 4.0 x16: ~25 GB/s unidirectional
- Mapped memory (cuMemHostAlloc): Subject to PCIe bandwidth, not GPU memory bandwidth
- For sequential writes by one GPU thread: limited by single-lane throughput (~1-2 GB/s)
- For warp-coalesced writes: can approach PCIe bandwidth (~10 GB/s)

For our use case (file I/O, typically KB-MB range), even single-thread bandwidth (1 GB/s)
is far more than sufficient. The bottleneck will be host-side file I/O, not PCIe.

**Confidence**: medium (estimates from PCIe spec, not measured)

## Recommended Design (for large-payload.2)

```
┌──────────────────────────────────────────────┐
│ Sideband Buffer (1MB mapped memory)          │
│ ┌──────┬──────┬──────┬───────────────┐       │
│ │ 4KB  │ 4KB  │ 4KB  │ ... (256 slots)│      │
│ └──────┴──────┴──────┴───────────────┘       │
│ Bump allocator: atomic offset counter        │
└──────────────────────────────────────────────┘

New services:
  SERVICE_BULK_WRITE = 11  (GPU→Host: write sideband data to file)
  SERVICE_BULK_READ  = 12  (Host→GPU: read file data into sideband)

Hostcall packet payload:
  Slot 0: fd (u64)
  Slot 1: sideband_offset (u64)
  Slot 2: length (u64)
```

## Impact on Downstream Tasks

- **large-payload.2 (design)**: Use sideband buffer approach. Design offset allocator.
- **large-payload.3 (implement)**: Allocate sideband in HostcallBuffer::new(), add
  SERVICE_BULK_READ/WRITE handlers, GPU-side bulk_read/bulk_write functions.
- **product.8 (parallel grep)**: Depends on bulk read for reading file chunks.
