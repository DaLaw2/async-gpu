# net-protocol.1: TCP Service Protocol Design
**Cycle**: 294 | **Theme**: net-protocol | **Kind**: investigation | **Status**: done

## Summary

Investigated the existing hostcall protocol to understand packet layout, fd table management, sideband buffer usage, and error handling patterns. The current file I/O services (OPEN/READ/WRITE/CLOSE with fds 1..N) and BULK_READ/BULK_WRITE (via sideband buffer) provide a direct template for TCP networking services. The fd table is a simple `HashMap<u64, File>` that can be extended to hold socket handles with minimal changes.

## Findings

### Q: What service IDs exist and what numbering scheme is used?
A: Service IDs are `u32` constants with sequential numbering. Current allocation:

| ID | Constant | Category |
|----|----------|----------|
| 0 | `SERVICE_NOP` | Diagnostics |
| 1 | `SERVICE_PRINT` | Output |
| 2 | `SERVICE_WRITE` | File I/O |
| 3 | `SERVICE_READ` | File I/O |
| 4 | `SERVICE_OPEN` | File I/O |
| 5 | `SERVICE_CLOSE` | File I/O |
| 6 | `SERVICE_MALLOC` | Memory (reserved) |
| 7 | `SERVICE_FREE` | Memory (reserved) |
| 8 | `SERVICE_STDIN` | Input |
| 9 | `SERVICE_TIME` | Misc |
| 10 | `SERVICE_PANIC` | Diagnostics |
| 11 | `SERVICE_BULK_WRITE` | File I/O (bulk) |
| 12 | `SERVICE_BULK_READ` | File I/O (bulk) |
| 13 | `SERVICE_TRACE` | Diagnostics |
| 14 | `SERVICE_ASSERT` | Diagnostics |
| 15 | `SERVICE_BULK_PRINT` | Output (bulk) |
| 0xFF | `SERVICE_ABORT` | Fatal |

IDs 16-254 are available. No range-based grouping is enforced, but functional clusters are visible. TCP services should start at a round number like 16 or 32 for clarity.

### Q: How are hostcall packets structured?
A: Each packet is 2112 bytes (32-byte header + 2048-byte payload, 64-byte aligned):

**Packet header (32 bytes):**
- Offset 0: `next` (u64) — tagged pointer for stack linkage (ABA tag in bits 63..32, index in bits 15..0)
- Offset 8: `active_mask` (u32) — which warp lanes are participating
- Offset 12: `service` (u32) — service ID
- Offset 16: `control` (u32) — flags: CONTROL_FILLED (4), CONTROL_READY (1), CONTROL_ERROR (2)

**Payload (2048 bytes):**
- 32 lanes x 8 slots x 8 bytes each
- Lane 0 payload = slots 0-7 = 64 bytes at offset 32
- Most services use only lane 0 (single-thread requests)
- Slot offset formula: `PKT_OFF_PAYLOAD + lane * 64 + slot * 8`

**Per-lane capacity:** 8 slots x 8 bytes = 64 bytes. For lane-0-only services, this means 64 bytes for request, and the same 64 bytes are overwritten with the response.

**Inline data limits (lane 0):**
- File path: 48 bytes (slots 2-7, after flags+length in slots 0-1)
- Write data: 48 bytes (slots 2-7, after fd+length in slots 0-1)
- Read data: 56 bytes (slots 1-7, after byte-count in slot 0)
- Print message: 56 bytes (slots 1-7, after length in slot 0)

### Q: How does the host manage file descriptors?
A: The fd table lives inside the I/O thread loop as a local `HashMap<u64, File>`:

```rust
let mut fd_table: HashMap<u64, File> = HashMap::new();
let mut next_fd: u64 = 1; // fd 0 is reserved
```

Key properties:
- **Monotonically increasing IDs**: `next_fd` starts at 1, increments on each open. Never reuses IDs.
- **fd 0 is reserved** (not used for anything currently — could be stdin).
- **Typed values**: The map stores `std::fs::File` directly. For sockets, the map type would need to become polymorphic (enum or trait object).
- **Lifetime**: The fd table lives for the duration of `io_thread_loop`. The `reinit_packets()` method explicitly does NOT close file handles — they persist across kernel launches.
- **Lookup pattern**: `fd_table.get_mut(&fd)` for read/write, `fd_table.remove(&fd)` for close. Invalid fd returns `ERR_INVALID_FD`.

**Reuse strategy for sockets**: The simplest approach is to change the map value to an enum:
```rust
enum FdResource {
    File(File),
    TcpStream(TcpStream),
    TcpListener(TcpListener),
}
```
This preserves the fd numbering scheme and all existing fd validation logic.

### Q: How is the sideband buffer used?
A: The sideband buffer is a separate CUDA mapped allocation for bulk data (>56 bytes):

**Layout:**
- Header (64 bytes): `alloc_offset` (u64) at offset 0, `capacity` (u64) at offset 8
- Data region: starts at offset 64, default 1 MB

**Allocation model:**
- GPU side uses a **bump allocator** — atomically increments `alloc_offset` to reserve regions
- Host side reads data at `sideband_host_ptr + SIDEBAND_DATA_OFFSET + offset`
- `reinit_packets()` resets the bump allocator to 0 between kernel launches
- No per-allocation free — the entire buffer is reset at once

**Bulk I/O protocol (BULK_WRITE example):**
1. GPU allocates sideband region (bump alloc), writes data into it
2. GPU sends hostcall with `(fd, sideband_offset, length)` in slots 0-2
3. Host validates `offset + length <= capacity`
4. Host reads data from sideband at the given offset
5. Host writes to the file, returns bytes written in response slot 0

**For TCP**: Sideband is essential for send/recv of payloads larger than 48-56 bytes. The existing BULK_READ/BULK_WRITE pattern maps directly to TCP send/recv with sideband.

### Q: How do services report errors?
A: Two error reporting mechanisms exist:

**1. Structured error (modern, used by file I/O):**
- Host handler returns `true` (has_error flag)
- Control word is set to `CONTROL_READY | CONTROL_ERROR`
- Payload slot 0 contains encoded error: `(raw_errno << 16) | category`
- Error categories: 18 defined constants (`ERR_OTHER` through `ERR_UNSUPPORTED`)
- `io_error_to_category()` maps `std::io::ErrorKind` to `ERR_*` constants
- `write_error_response()` helper encodes and writes in one call
- GPU decodes via `GpuError::from_encoded(slot0)` -> `GpuError { category, raw_errno }`

**2. Legacy sentinel (still used in some paths):**
- `FILE_ERROR_SENTINEL` (u64::MAX) written to slot 0
- Used directly in `handle_write` for invalid fd: `encode_error(ERR_INVALID_FD, 0)`
- Some handlers mix both patterns

**Relevant existing error categories for TCP:**
- `ERR_CONNECTION_REFUSED` (14) — already reserved for networking
- `ERR_TIMED_OUT` (6) — connection/read timeout
- `ERR_WOULD_BLOCK` (7) — non-blocking mode
- `ERR_BROKEN_PIPE` (8) — connection reset
- `ERR_INVALID_FD` (13) — bad socket fd
- `ERR_IO_ERROR` (5) — general I/O
- `ERR_INVALID_INPUT` (4) — bad address/port

**New categories needed for TCP:**
- `ERR_CONNECTION_RESET` — peer reset connection
- `ERR_ADDR_IN_USE` — bind address already in use
- `ERR_ADDR_NOT_AVAILABLE` — bind address not available
- `ERR_NOT_CONNECTED` — socket not connected

### Q: Proposed TCP service protocol
A: Based on the findings, here is the proposed design:

**Service IDs (starting at 16):**

| ID | Constant | Description |
|----|----------|-------------|
| 16 | `SERVICE_TCP_CONNECT` | Connect to remote addr:port, return socket fd |
| 17 | `SERVICE_TCP_WRITE` | Write inline data (up to 48 bytes) to socket fd |
| 18 | `SERVICE_TCP_READ` | Read inline data (up to 56 bytes) from socket fd |
| 19 | `SERVICE_TCP_CLOSE` | Close a socket fd |
| 20 | `SERVICE_TCP_BIND` | Bind+listen on addr:port, return listener fd |
| 21 | `SERVICE_TCP_ACCEPT` | Accept connection on listener fd, return stream fd |
| 22 | `SERVICE_TCP_BULK_WRITE` | Write sideband data to socket fd |
| 23 | `SERVICE_TCP_BULK_READ` | Read data from socket into sideband |

**Packet layouts (all lane 0):**

`SERVICE_TCP_CONNECT` request:
- Slot 0: port (u32) | addr_len (u32) packed as u64
- Slots 1-7: address string (up to 56 bytes, e.g., "192.168.1.1" or hostname)

Response:
- Slot 0: socket fd (u64), or encoded error

`SERVICE_TCP_WRITE` request (mirrors `SERVICE_WRITE`):
- Slot 0: fd (u64)
- Slot 1: data length (u64)
- Slots 2-7: data bytes (up to 48 bytes)

Response:
- Slot 0: bytes written (u64), or encoded error

`SERVICE_TCP_READ` request (mirrors `SERVICE_READ`):
- Slot 0: fd (u64)
- Slot 1: max bytes (u64)

Response:
- Slot 0: bytes read (u64), or encoded error
- Slots 1-7: data bytes (up to 56 bytes)

`SERVICE_TCP_CLOSE` request:
- Slot 0: fd (u64)

Response:
- Slot 0: 0 on success, encoded error on failure

`SERVICE_TCP_BIND` request:
- Slot 0: port (u32) | addr_len (u32)
- Slots 1-7: bind address string (up to 56 bytes)

Response:
- Slot 0: listener fd (u64), or encoded error

`SERVICE_TCP_ACCEPT` request:
- Slot 0: listener fd (u64)

Response:
- Slot 0: stream fd (u64), or encoded error

`SERVICE_TCP_BULK_WRITE` / `SERVICE_TCP_BULK_READ` (mirrors `SERVICE_BULK_WRITE` / `SERVICE_BULK_READ`):
- Same slot layout as file bulk ops, just dispatched to socket instead of file

**Fd table extension:**
```rust
enum FdResource {
    File(File),
    TcpStream(TcpStream),
    TcpListener(TcpListener),
}
```
All fd operations (assign, lookup, remove) remain identical. The `next_fd` counter is shared across files and sockets — a single fd namespace.

**Dispatch path:**
TCP services should go through the **slow path** (I/O thread via mpsc channel), same as file I/O, since they involve blocking network operations. Add them to the existing match arm:
```
SERVICE_OPEN | SERVICE_WRITE | ... | SERVICE_TCP_CONNECT | SERVICE_TCP_WRITE | ... => {
    io_tx.send(IoRequest { pkt_idx: idx, service });
}
```

**Error handling:**
- Reuse `write_error_response()` with `io_error_to_category()` — most `std::net` errors map to the same `ErrorKind` values
- Add new error categories: `ERR_CONNECTION_RESET` (18), `ERR_ADDR_IN_USE` (19), `ERR_ADDR_NOT_AVAILABLE` (20), `ERR_NOT_CONNECTED` (21)
- The existing `ERR_CONNECTION_REFUSED` (14) already covers the most common TCP failure

**Alternative: Shared fd namespace vs. separate close**
Option A (recommended): Use `SERVICE_CLOSE` (5) for both files and sockets — the host just calls `fd_table.remove(&fd)` and drops the resource. No need for `SERVICE_TCP_CLOSE`.
Option B: Separate `SERVICE_TCP_CLOSE` — allows type-checking (refuse to close a file fd via TCP close). Slightly safer but adds complexity.

Recommendation: Option A (shared close) for simplicity, with the fd enum handling drop correctly for both types.

## Impact on Downstream Tasks
- **net-protocol.2** (implementation): Can proceed with the service IDs and packet layouts defined above
- **net-protocol.3** (GPU-side API): Needs to implement `TcpStream` / `TcpListener` types that encode packets per this layout
- **async-std** (epic): TCP services follow the same async hostcall Future pattern as file I/O — `TcpConnectFuture`, `TcpReadFuture`, etc.
- **Sideband capacity**: If TCP workloads transfer large payloads, the default 1 MB sideband may need to be configurable per-launch
