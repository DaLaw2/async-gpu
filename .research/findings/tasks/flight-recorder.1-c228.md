# flight-recorder.1: Flight Recorder Ring Buffer Design
**Cycle**: 228 | **Theme**: flight-recorder | **Kind**: design | **Status**: done

## Summary
Design a GPU-side ring buffer that stores the last N trace events. When a kernel crashes (trap/assert), the host dumps the buffer for post-mortem analysis.

## Design

### Memory Layout

```
Flight Recorder Buffer (mapped memory):

Header (64 bytes, cache-line aligned):
  Offset  0: write_idx   (u64, atomic) — GPU increments after writing event
  Offset  8: capacity    (u32)         — number of event slots
  Offset 12: flags       (u32)         — bit 0: crashed flag
  Offset 16: reserved    [48 bytes]

Event Slot (64 bytes each):
  Offset  0: metadata    (u64) — same encoding as trace protocol:
             threadIdx:16 | blockIdx:16 | level:8 | msg_len:8 | lane_id:16
  Offset  8: timestamp   (u64) — %clock64
  Offset 16: message     [48 bytes] — trace message (zero-padded)
```

### Protocol

**GPU side (write event):**
```
1. Load write_idx (GPU-local atomic)
2. Compute slot = write_idx % capacity
3. Write metadata + timestamp + message to slot
4. Increment write_idx (Release)
```

No read_idx needed — this is a circular buffer that overwrites old entries.
Only GPU writes, host only reads after synchronize.

**Host side (dump on crash):**
```
1. After cuCtxSynchronize() returns error (or kernel completes with crash flag)
2. Read write_idx and capacity
3. Start from (write_idx - capacity) or 0 (whichever is larger)
4. Read events in order, decode metadata, print formatted output
```

### Constants (gpu-protocol)

```rust
pub const FR_HEADER_SIZE: usize = 64;
pub const FR_SLOT_SIZE: usize = 64;
pub const FR_OFF_WRITE_IDX: usize = 0;
pub const FR_OFF_CAPACITY: usize = 8;
pub const FR_OFF_FLAGS: usize = 12;
pub const FR_SLOT_OFF_META: usize = 0;
pub const FR_SLOT_OFF_TIMESTAMP: usize = 8;
pub const FR_SLOT_OFF_MSG: usize = 16;
pub const FR_MAX_MSG_LEN: usize = 48;
pub const FR_FLAG_CRASHED: u32 = 1;
```

### GPU-Side API

```rust
/// Write a trace event to the flight recorder ring buffer.
/// This is a fire-and-forget write — no hostcall needed.
pub unsafe fn fr_record(fr_buf: *mut u8, level: u8, msg: *const u8, msg_len: u32) {
    let write_idx = sys_load_acquire_u64(fr_buf as *const u64);
    let capacity = read_volatile_u32(fr_buf.add(FR_OFF_CAPACITY));
    let slot_idx = (write_idx % capacity as u64) as usize;
    let slot = fr_buf.add(FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE);

    // Write metadata (thread/block/lane + level + msg_len)
    let meta = encode_trace_metadata(threadIdx.x, blockIdx.x, level, msg_len, lane_id);
    write_volatile(slot as *mut u64, meta);

    // Write timestamp
    let ts = clock64();
    write_volatile(slot.add(8) as *mut u64, ts);

    // Copy message
    let copy_len = min(msg_len, FR_MAX_MSG_LEN);
    memcpy_volatile(slot.add(16), msg, copy_len);

    // Increment write_idx
    sys_store_release_u64(fr_buf as *mut u64, write_idx + 1);
}
```

### Host-Side API

```rust
/// Dump the flight recorder buffer to stderr.
pub fn fr_dump(fr_buf: &FlightRecorder) {
    let write_idx = read_volatile(write_idx_ptr);
    let capacity = fr_buf.capacity;
    let start = if write_idx > capacity as u64 {
        write_idx - capacity as u64
    } else {
        0
    };

    eprintln!("=== Flight Recorder Dump ({} events) ===", write_idx - start);
    for i in start..write_idx {
        let slot_idx = (i % capacity as u64) as usize;
        let slot = &fr_buf.slots[slot_idx];
        let (tid, bid, level, msg_len, lane) = decode_trace_metadata(slot.meta);
        let msg = &slot.msg[..msg_len as usize];
        eprintln!("[{ts}] T{tid}.B{bid}.L{lane} {level}: {msg}",
            ts = slot.timestamp, level = level_str(level), msg = str::from_utf8(msg));
    }
}
```

### Key Design Decisions

1. **No hostcall needed**: Unlike gpu_trace!, the flight recorder writes directly to mapped memory. This is fire-and-forget — no round-trip, no packet allocation, no risk of running out of hostcall packets.

2. **Fixed-size ring buffer**: Overwrites oldest events. Typical capacity: 64 or 128 events. At 64 bytes per event, 128 events = 8KB + 64B header.

3. **Thread safety**: Multiple GPU threads writing concurrently need atomic write_idx. Each thread atomically claims a slot via fetch_add on write_idx. This is safe because each slot is written by exactly one thread.

4. **Reuse trace metadata format**: Same encode/decode as existing trace protocol — code reuse.

## Findings

### Q: Should flight recorder use hostcall or direct mapped memory?
A: Direct mapped memory. Hostcall adds latency and can fail (out of packets). Flight recorder must be ultra-reliable — it's the last resort for crash debugging.
**Confidence**: high

### Q: How to handle multi-thread concurrent writes?
A: Use atomic fetch_add on write_idx to claim a slot. Each thread writes to its own slot. No locks needed. The write_idx may advance past capacity — that's fine, modular arithmetic handles wrap-around.
**Confidence**: high

## Impact on Downstream Tasks
- flight-recorder.2: Implement GPU-side fr_record + host-side FlightRecorder dump
- flight-recorder.3: Test with kernel crash scenario
