# hostcall.3: Design and Implement Hostcall Protocol
**Date**: 2026-03-11
**Cycle**: 10
**Theme**: hostcall
**Kind**: design
**Status**: done
**Spawned by**: initial

## Summary

Design of a lock-free GPU-to-host RPC protocol based on the ROCm hostcall architecture,
adapted for NVIDIA's 32-thread warp model and Rust's type system. Uses pinned mapped
memory for the shared buffer and `gpu-atomics` system-scope primitives for synchronization.

## Design Goals

1. **Any GPU warp can issue an RPC** without coordinating with other warps
2. **Host responds per-RPC** — GPU warp blocks until response is ready
3. **Lock-free** — no mutexes, only CAS and atomic exchange
4. **Bounded latency** — GPU-side timeout, host-side adaptive polling
5. **Extensible services** — printf, file I/O, memory allocation, etc.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Pinned Mapped Memory                    │
│                  (cuMemHostAlloc + DEVICEMAP)              │
│                                                            │
│  ┌─────────────────┐  ┌──────────────────────────────┐    │
│  │  HostcallBuffer │  │  Packet[0..N-1]              │    │
│  │                 │  │  ┌────────┬────────────────┐ │    │
│  │  free_stack: u64│  │  │ Header │ Payload        │ │    │
│  │  ready_stack:u64│  │  │ next   │ slots[32][8]   │ │    │
│  │  doorbell:  u64 │  │  │ active │ (u64 each)     │ │    │
│  │  shutdown:  u32 │  │  │ service│                 │ │    │
│  │  num_packets:u32│  │  │ control│                 │ │    │
│  │  warp_size: u32 │  │  └────────┴────────────────┘ │    │
│  └─────────────────┘  └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
         ▲                          ▲
         │  sys-scope atomics       │  sys-scope stores/loads
    ┌────┴────┐               ┌─────┴─────┐
    │  Host   │               │    GPU    │
    │ Listener│               │   Warp    │
    │ Thread  │               │  (caller) │
    └─────────┘               └───────────┘
```

## Data Structures

### HostcallBuffer (control header)

```rust
#[repr(C, align(64))]  // Cache-line aligned
pub struct HostcallBuffer {
    /// Head of free packet stack (tagged pointer).
    /// GPU pops from here to get a packet slot.
    free_stack: u64,

    /// Head of ready packet stack (tagged pointer).
    /// GPU pushes here after filling a packet.
    /// Host atomically swaps to 0 to grab all pending packets.
    ready_stack: u64,

    /// Doorbell counter. GPU increments after pushing to ready_stack.
    /// Host polls this value for change detection.
    doorbell: u64,

    /// Shutdown flag. Host sets to 1 to signal GPU to stop.
    shutdown: u32,

    /// Number of packets in the pool.
    num_packets: u32,

    /// Warp size (32 for NVIDIA). Determines payload slot count.
    warp_size: u32,

    _padding: [u8; 20],  // Pad to 64 bytes

    /// Packet array follows immediately after this header.
    /// Layout: [Packet; num_packets]
}
```

### Tagged Pointer

Both `free_stack` and `ready_stack` use tagged pointers to prevent ABA problems:

```
Bits 63..32: ABA tag (monotonically increasing)
Bits 31..16: reserved (zero)
Bits 15..0:  packet index (0..N-1), or 0xFFFF = NULL
```

This allows up to 65534 packets and 2^32 tag generations before wraparound.

### Packet

```rust
#[repr(C, align(64))]
pub struct Packet {
    header: PacketHeader,
    payload: PacketPayload,
}

#[repr(C)]
pub struct PacketHeader {
    /// Tagged pointer to next packet in the stack (free or ready list).
    next: u64,

    /// Bitmask of active lanes in the warp (0..31).
    /// Only lanes with bit set have valid data in payload.
    active_mask: u32,

    /// Service ID for this request.
    service: u32,

    /// Control word:
    ///   Bit 0: READY flag (1 = host has written response)
    ///   Bit 1: ERROR flag (1 = host encountered an error)
    ///   Bits 2-31: reserved
    control: u32,

    _padding: [u8; 12],  // Pad header to 32 bytes
}

/// Each warp lane gets 8 × u64 = 64 bytes of argument/response space.
/// Slot 0 is reserved for the message descriptor (future: multi-packet).
/// Slots 1-7 carry arguments (GPU→host) or response data (host→GPU).
#[repr(C)]
pub struct PacketPayload {
    slots: [[u64; 8]; 32],  // 32 lanes × 8 qwords = 2048 bytes
}
```

Total packet size: 32 (header) + 2048 (payload) = 2080 bytes, rounded to 2112 with alignment.

### Service IDs

```rust
pub const SERVICE_NOP: u32 = 0;       // No-op (testing)
pub const SERVICE_PRINT: u32 = 1;     // Printf-style output
pub const SERVICE_WRITE: u32 = 2;     // write(fd, buf, len)
pub const SERVICE_READ: u32 = 3;      // read(fd, buf, len)
pub const SERVICE_OPEN: u32 = 4;      // open(path, flags)
pub const SERVICE_CLOSE: u32 = 5;     // close(fd)
pub const SERVICE_MALLOC: u32 = 6;    // Host malloc (for GPU use)
pub const SERVICE_FREE: u32 = 7;      // Host free
pub const SERVICE_ABORT: u32 = 0xFF;  // Kernel abort
```

## Protocol

### GPU Side: Issue Hostcall

```
fn hostcall(buffer: &HostcallBuffer, service: u32, args: &[u64; 7]) -> Result<[u64; 7], Error>:
    1. Check shutdown flag → if set, return Error::Shutdown
    2. Pop packet from free_stack:
       loop:
           old_head = sys_load_acquire_u64(&buffer.free_stack)
           if old_head == NULL_TAG: return Error::PoolExhausted
           packet = &packets[index_of(old_head)]
           new_head = packet.header.next  // read before CAS
           if sys_cas_u64(&buffer.free_stack, old_head, new_head) == old_head:
               break  // got a packet
    3. Fill packet:
       packet.header.active_mask = __activemask()  // intrinsic
       packet.header.service = service
       packet.header.control = 0  // clear READY
       for each active lane i:
           packet.payload.slots[i][1..8] = args  // via shuffle or direct write
    4. Push packet onto ready_stack:
       loop:
           old_head = sys_load_acquire_u64(&buffer.ready_stack)
           packet.header.next = old_head
           new_tag = tag_of(old_head) + 1
           new_tagged = make_tagged(new_tag, packet_index)
           if sys_cas_u64(&buffer.ready_stack, old_head, new_tagged) == old_head:
               break
    5. Ring doorbell:
       sys_fetch_add_u64(&buffer.doorbell, 1)
    6. Wait for response:
       spin_count = 0
       loop:
           control = sys_load_acquire_u32(&packet.header.control)
           if control & READY_BIT != 0: break
           if spin_count >= MAX_SPIN: return Error::Timeout
           spin_count += 1
    7. Read response from payload slots[lane_id][1..8]
    8. Push packet back to free_stack (same CAS loop as step 2, reversed)
    9. Return response
```

### Host Side: Listener Thread

```
fn host_listener(buffer: &HostcallBuffer):
    timeout = TIMEOUT_FLOOR
    last_doorbell = 0
    loop:
        // Check shutdown
        if buffer.shutdown != 0: break

        // Poll doorbell
        current_doorbell = AtomicU64::load(&buffer.doorbell, Acquire)
        if current_doorbell == last_doorbell:
            spin_loop()
            timeout = min(timeout * 2, TIMEOUT_CEIL)
            continue  // or sleep(timeout) for adaptive backoff

        last_doorbell = current_doorbell
        timeout = max(timeout / 2, TIMEOUT_FLOOR)

        // Grab all ready packets atomically
        ready_head = AtomicU64::swap(&buffer.ready_stack, NULL_TAG, AcqRel)
        if ready_head == NULL_TAG: continue

        // Walk the ready list (LIFO order)
        current = ready_head
        while current != NULL_TAG:
            idx = index_of(current)
            packet = &packets[idx]
            next = packet.header.next

            // Dispatch to service handler
            match packet.header.service:
                SERVICE_PRINT => handle_print(packet)
                SERVICE_WRITE => handle_write(packet)
                ...
                _ => set_error(packet)

            // Signal GPU: response is ready
            packet.header.control = READY_BIT  // with release store
            AtomicU32::store(&packet.header.control, READY_BIT, Release)

            current = next
```

## Synchronization Primitives Used

| Operation | GPU Side | Host Side |
|-----------|----------|-----------|
| Read free_stack | `sys_load_acquire_u64` | N/A |
| CAS free_stack | `sys_cas_u64` (needs impl) | N/A |
| Write ready_stack | `sys_cas_u64` | `AtomicU64::swap(AcqRel)` |
| Doorbell increment | `sys_fetch_add_u64` (needs impl) | `AtomicU64::load(Acquire)` |
| Read control (GPU wait) | `sys_load_acquire_u32` | N/A |
| Write control (host signal) | N/A | `AtomicU32::store(Release)` |
| Read shutdown | `sys_load_acquire_u32` | N/A |
| Write shutdown | N/A | `AtomicU32::store(Release)` |

### Required additions to gpu-atomics

Before hostcall.4 implementation, `gpu-atomics` needs:
1. `sys_cas_u64(ptr, expected, desired) -> u64` — 64-bit system-scope CAS
2. `sys_fetch_add_u64(ptr, val) -> u64` — 64-bit system-scope fetch-and-add
3. `sys_exchange_u64(ptr, val) -> u64` — 64-bit system-scope exchange (optional, CAS can substitute)

These use the same inline PTX pattern as the u32 variants:
```ptx
atom.cas.sys.global.b64 result, [ptr], expected, desired;
atom.add.sys.global.u64 result, [ptr], val;
atom.exch.sys.global.b64 result, [ptr], val;
```

## Memory Layout and Sizing

### Pool Size Formula

```
num_packets = num_SMs × max_warps_per_SM × occupancy_factor
```

For RTX 3060 (28 SMs, 48 warps/SM max):
- Conservative: `28 × 16 × 1 = 448 packets` (~922 KB)
- Default: `28 × 8 = 224 packets` (~461 KB)
- Minimal (for testing): `64 packets` (~132 KB)

### Total Buffer Size

```
buffer_size = sizeof(HostcallBuffer)          // 64 bytes (header)
            + num_packets × sizeof(Packet)    // num_packets × 2112 bytes
```

For 64 packets: 64 + 64 × 2112 = 135,232 bytes ≈ 132 KB.

### Alignment Requirements

- HostcallBuffer: 64-byte aligned (cache line)
- Each Packet: 64-byte aligned (cache line, avoids false sharing)
- Payload slots: naturally aligned (u64 = 8-byte)

All satisfied by `cuMemHostAlloc` which returns page-aligned memory.

## Error Handling

| Error | GPU behavior | Host behavior |
|-------|-------------|---------------|
| Pool exhausted | Return error code | N/A |
| GPU-side timeout | Return error code | N/A |
| Unknown service | N/A | Set ERROR bit in control |
| Host crash | GPU spins until timeout | N/A |
| Shutdown signal | Return error code | Exit listener loop |

The GPU-side MAX_SPIN should be large enough to cover host processing time but
bounded to prevent infinite hangs. Suggested: 1,000,000 iterations (~ms).

## Implementation Plan

### Phase 1: hostcall.4 (Minimal — GPU println)
1. Add u64 CAS/fetch_add/exchange to `gpu-atomics`
2. Implement `HostcallBuffer` allocation on host side
3. Implement minimal GPU-side `hostcall()` function
4. Implement host listener with SERVICE_PRINT only
5. Test: GPU kernel prints "Hello from GPU" via hostcall

### Phase 2: gpu-std.1 (Full libc services)
1. Add SERVICE_WRITE, SERVICE_READ, SERVICE_OPEN, SERVICE_CLOSE
2. Implement libc shim that routes to hostcall
3. Test: `println!` and `File::create` from GPU

### Phase 3: Optimization
1. Warp-cooperative hostcall (reduce per-lane overhead)
2. Adaptive timeout tuning based on benchmarks
3. Multi-packet message support for large payloads

## Design Decisions

### D1: Warp-granular packets (32 lanes per packet)
**Rationale**: Matches NVIDIA warp size. Each warp thread can issue a hostcall
independently, but the packet is allocated once per warp. The `active_mask` field
indicates which lanes have valid data. This minimizes atomic contention on the
free/ready stacks (one CAS per warp, not per thread).

### D2: LIFO (stack) over FIFO (ring) for ready list
**Rationale**: Per ROCm analysis, order doesn't matter for independent RPC requests.
LIFO via atomic CAS is simpler than FIFO with head/tail pointers. The host processes
all pending packets in batch anyway.

### D3: Doorbell counter over signal object
**Rationale**: NVIDIA has no HSA signal equivalent. A simple `fetch_add` on a u64
counter serves as the doorbell. The host detects new work by comparing current value
to last seen value. No kernel driver involvement needed.

### D4: GPU-side bounded spin (timeout)
**Rationale**: ROCm's unbounded spin is a known weakness. Adding a bounded spin
count prevents GPU hangs if the host crashes or becomes unresponsive. The error
propagates back to the kernel, which can abort gracefully.

### D5: Single-packet messages first, multi-packet later
**Rationale**: 7 × u64 = 56 bytes per lane is sufficient for printf args, file
descriptors, and short read/write buffers. Multi-packet reassembly (ROCm-style)
adds significant complexity and should be deferred until a concrete use case demands it.

## Key Conclusions

1. The ROCm two-stack lock-free pool design translates directly to NVIDIA + Rust
2. `gpu-atomics` provides all needed primitives; u64 variants must be added
3. Pinned mapped memory via `cuMemHostAlloc(DEVICEMAP|PORTABLE)` is confirmed correct
4. The protocol is GPU-warp-granular: one packet per warp, CAS for stack operations
5. Host listener uses adaptive polling on the doorbell counter
6. Target buffer: 64-224 packets, ~132 KB-461 KB of pinned memory

## Open Questions

1. **`__activemask()` on nvptx64**: Need to verify how to get the active lane mask.
   Options: `vote.ballot.sync` PTX instruction, or `core::arch::nvptx::_activemask()`.
2. **Warp-level collective for slot filling**: Can we use `shfl.sync` to distribute
   arguments across lanes, or must each lane write its own slots independently?
3. **Host polling vs interrupt**: The adaptive polling approach works for initial
   implementation. If CPU utilization is a concern, `cuStreamWaitValue64` could
   provide an interrupt-driven path (CUDA 12+).

## Theme Progress

The hostcall theme now has a complete protocol design. Success criteria status:
1. "GPU kernel can send structured requests to host" — designed, ready for implementation
2. "Host responds with results" — designed, ready for implementation
3. "Multiple warps can issue concurrent hostcalls correctly" — designed with lock-free CAS
