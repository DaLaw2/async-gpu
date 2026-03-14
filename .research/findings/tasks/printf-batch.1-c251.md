# printf-batch.1: Design GPU-side print buffer — layout, flush protocol, memory budget
**Cycle**: 251 | **Theme**: printf-batch | **Kind**: design | **Status**: done

## Summary
Design for GPU-side buffered printing that accumulates messages and flushes via sideband bulk transfer, reducing hostcall overhead from O(N) round-trips to O(1) per flush.

## Current State
- `gpu_hostcall_print()` sends one 56-byte message per hostcall round-trip (~20-100us each)
- 10 prints = 10 round-trips = 200-1000us overhead
- PRINT_MAX_MSG_LEN = 56 bytes (limited by packet payload size)
- Sideband bulk transfer already exists (gpu_bulk_write) — can send up to 1MB per round-trip

## Design: Buffered Print via Sideband

### Architecture

```
GPU Thread                     Host Listener
    |                               |
    |-- gpu_print_buffered() -----> (no hostcall — just buffer locally)
    |-- gpu_print_buffered() -----> (no hostcall — just buffer locally)
    |-- gpu_print_buffered() -----> (no hostcall — just buffer locally)
    |                               |
    |-- gpu_print_flush() --------> SERVICE_BULK_PRINT via sideband
    |                               |-- reads all buffered messages
    |                               |-- calls on_print for each
    |<--- CONTROL_READY ------------|
```

### Buffer Layout (in sideband memory)

The print buffer lives in a dedicated region of the sideband buffer. Each thread gets a fixed-size buffer slot.

```
Sideband memory (existing):
  [0..8]    alloc offset (AtomicU64)
  [8..data] padding
  [data..]  data region (shared by bulk read/write + print buffer)

Print buffer region (within sideband data region):
  Per-thread slot: [header 8 bytes] [message data up to SLOT_DATA_SIZE bytes]
    header[0..4]: write_offset (u32) — next write position within message data
    header[4..8]: reserved

  Thread N's slot starts at: PRINT_BUF_BASE + N * SLOT_SIZE
  SLOT_SIZE = 512 bytes (8 header + 504 data)
  SLOT_DATA_SIZE = 504 bytes (~9 print messages at 56 bytes each)
```

### Memory Budget

| Config | Threads | Slots | Total Memory |
|--------|---------|-------|-------------|
| 1 thread | 1 | 512B | 512B |
| 32 threads (1 warp) | 32 | 512B × 32 | 16KB |
| 128 threads (4 warps) | 128 | 512B × 128 | 64KB |
| 1024 threads (full block) | 1024 | 512B × 1024 | 512KB |

**Decision**: Use the sideband data region. The default sideband is 1MB, so up to 1024 threads is fine. For larger launches, print buffering can be disabled.

### API

```rust
/// Initialize print buffer for this thread. Call once at kernel start.
/// `sideband` is the sideband buffer pointer.
/// `max_threads` is the launch thread count (for bounds checking).
pub unsafe fn gpu_print_buf_init(sideband: *mut u8, max_threads: u32);

/// Buffer a print message. Does NOT issue a hostcall.
/// If the buffer is full, auto-flushes first.
/// `buf` is the hostcall buffer (needed for auto-flush).
/// `sideband` is the sideband buffer.
pub unsafe fn gpu_print_buffered(
    buf: *mut u8,
    sideband: *mut u8,
    msg: *const u8,
    msg_len: u32,
) -> Result<(), GpuError>;

/// Flush all buffered print messages via a single sideband hostcall.
/// Must be called before kernel exit.
pub unsafe fn gpu_print_flush(
    buf: *mut u8,
    sideband: *mut u8,
) -> Result<(), GpuError>;
```

### Flush Protocol

New service: `SERVICE_BULK_PRINT` (add to gpu-protocol)

Packet payload:
```
[0..8]:   sideband_offset (u64) — start of this thread's buffer in sideband
[8..16]:  data_len (u64) — total bytes of messages in buffer
[16..24]: thread_idx (u32) + block_idx (u32) — for logging
```

Host-side handler:
1. Read sideband data at offset (data_len bytes)
2. Messages are concatenated (no framing needed — each message is a complete line)
3. Pass entire blob to `on_print` callback
4. Set CONTROL_READY

### Alternatives Considered

**A. Shared ring buffer (all threads write to one buffer)**
- Pro: More memory efficient
- Con: Needs atomic write_offset for concurrent access, ordering issues
- Rejected: Atomic contention defeats the purpose of reducing overhead

**B. Per-warp buffer (32 threads share one buffer)**
- Pro: Matches hostcall warp-granular model
- Con: Warp divergence means some lanes may print more than others, buffer imbalance
- Rejected: Per-thread is simpler and avoids divergence issues

**C. Message framing (length-prefix each message in buffer)**
- Pro: Host can separate individual messages
- Con: Extra 4 bytes per message, more complex parsing
- Decision: **Yes, use length-prefix framing** — enables proper per-message on_print callbacks

### Revised Buffer Format (with length-prefix framing)

```
Per-thread slot:
  [0..4]:   write_offset (u32)
  [4..8]:   message_count (u32)
  [8..SLOT_SIZE]: message data

Each message in the data area:
  [0..2]:  msg_len (u16)
  [2..2+msg_len]: message bytes
```

Host parsing:
```rust
fn parse_print_buffer(data: &[u8], on_print: &mut dyn FnMut(&[u8])) {
    let mut pos = 0;
    while pos + 2 <= data.len() {
        let len = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        if len == 0 || pos + 2 + len > data.len() { break; }
        on_print(&data[pos+2..pos+2+len]);
        pos += 2 + len;
    }
}
```

## Open Questions
1. Should auto-flush happen when buffer is full, or should it silently drop messages? (Decision: auto-flush — correctness over performance)
2. Should `SERVICE_BULK_PRINT` be a new service ID or reuse `SERVICE_PRINT` with a flag? (Decision: new service ID — cleaner)
3. Integration with gpu-kernel-std `println!()` — would need to route through buffered path in gpu-libc write(). This is a follow-up task.

## Impact on Downstream Tasks
- printf-batch.2 should implement this design
- New service ID `SERVICE_BULK_PRINT` in gpu-protocol
- GPU-side buffer functions in gpu-runtime
- Host-side handler in gpu-host/hostcall.rs
- Test: kernel with 10+ prints, verify correct output with fewer round-trips
