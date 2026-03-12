# large-payload.2: Design bulk transfer protocol extension
**Cycle**: 70 | **Theme**: large-payload | **Kind**: design | **Status**: done

## Summary

Designed a sideband mapped buffer extension for transferring data larger than 56 bytes
between GPU and host. The design adds a second CUDA mapped buffer alongside the existing
hostcall buffer, with two new service IDs (SERVICE_BULK_WRITE, SERVICE_BULK_READ) and a
GPU-side bump allocator for offset management. No changes to the existing packet format.

## Design

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Existing Hostcall Buffer (pinned, device-mapped)            │
│ ┌──────────┬──────────┬──────────┬─────────────────┐        │
│ │ Header   │ Packet 0 │ Packet 1 │ ...             │        │
│ │ (64B)    │ (2112B)  │ (2112B)  │                 │        │
│ └──────────┴──────────┴──────────┴─────────────────┘        │
│                                                             │
│ NEW: Sideband Data Buffer (pinned, device-mapped)           │
│ ┌──────────┬────────────────────────────────────────┐       │
│ │ Header   │ Data region (contiguous)               │       │
│ │ (64B)    │ (configurable, e.g. 1MB)               │       │
│ └──────────┴────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Sideband Buffer Layout

```
Sideband Header (64 bytes):
  Offset 0:  alloc_offset (u64)  — GPU bump allocator position
  Offset 8:  capacity (u64)      — total data region size in bytes
  Offset 16: reserved (48 bytes)

Data Region:
  Starts at offset 64 from sideband base
  Contiguous byte array, capacity bytes
```

Constants (in gpu-protocol):
```rust
pub const SIDEBAND_HEADER_SIZE: usize = 64;
pub const SIDEBAND_OFF_ALLOC: usize = 0;   // u64: bump allocator offset
pub const SIDEBAND_OFF_CAPACITY: usize = 8; // u64: total data region size
pub const SIDEBAND_DATA_OFFSET: usize = SIDEBAND_HEADER_SIZE; // 64

pub const DEFAULT_SIDEBAND_SIZE: usize = 1024 * 1024; // 1MB data region

pub const SERVICE_BULK_WRITE: u32 = 11;
pub const SERVICE_BULK_READ: u32 = 12;

pub const BULK_MAX_LEN: usize = DEFAULT_SIDEBAND_SIZE;
```

### GPU-Side Bump Allocator

GPU threads need to allocate regions within the sideband buffer without coordination
with the host. A simple bump allocator using atomic fetch_add:

```rust
/// Allocate `size` bytes from the sideband buffer. Returns offset from data region
/// start, or u64::MAX if insufficient space.
#[inline(always)]
pub unsafe fn sideband_alloc(sideband: *mut u8, size: u64) -> u64 {
    let alloc_ptr = sideband.add(SIDEBAND_OFF_ALLOC) as *mut u64;
    let capacity_ptr = sideband.add(SIDEBAND_OFF_CAPACITY) as *const u64;
    let capacity = core::ptr::read_volatile(capacity_ptr);

    let old_offset = sys_fetch_add_u64(alloc_ptr, size);
    if old_offset + size > capacity {
        // Rollback: try to restore (best-effort, another thread may have advanced)
        // In practice, if this happens the buffer is full and all subsequent allocs fail too
        return u64::MAX;
    }
    old_offset
}

/// Reset the bump allocator. Called by GPU after host confirms all pending
/// operations are complete (or at kernel start).
#[inline(always)]
pub unsafe fn sideband_reset(sideband: *mut u8) {
    let alloc_ptr = sideband.add(SIDEBAND_OFF_ALLOC) as *mut u64;
    sys_store_release_u64(alloc_ptr, 0);
}
```

**Design choice**: No per-allocation free. The bump allocator is reset in bulk,
either at kernel start or when the GPU knows all pending bulk operations are complete.
This is appropriate because:
1. File I/O is typically sequential (read, process, write)
2. Per-allocation free requires a complex free-list on GPU
3. Bulk reset is simple and matches our usage pattern

### New Service Payloads

#### SERVICE_BULK_WRITE (11) — GPU writes data to host file

GPU writes data to sideband, then sends hostcall:

```
Request payload (lane 0):
  Slot 0: fd (u64)                    — file descriptor
  Slot 1: sideband_offset (u64)       — offset within sideband data region
  Slot 2: length (u64)                — number of bytes to write

Response payload (lane 0):
  Slot 0: bytes_written (u64)         — actual bytes written, or u64::MAX on error
```

Host handler:
1. Read hostcall packet → extract fd, sideband_offset, length
2. Compute data address: `sideband_host_ptr + SIDEBAND_DATA_OFFSET + sideband_offset`
3. Write `length` bytes from sideband to file fd
4. Write bytes_written to response slot 0
5. Set CONTROL_READY

#### SERVICE_BULK_READ (12) — Host reads file data for GPU

GPU allocates space in sideband, sends hostcall requesting read:

```
Request payload (lane 0):
  Slot 0: fd (u64)                    — file descriptor
  Slot 1: sideband_offset (u64)       — offset within sideband data region
  Slot 2: max_length (u64)            — maximum bytes to read

Response payload (lane 0):
  Slot 0: bytes_read (u64)            — actual bytes read, or u64::MAX on error
```

Host handler:
1. Read hostcall packet → extract fd, sideband_offset, max_length
2. Compute data address: `sideband_host_ptr + SIDEBAND_DATA_OFFSET + sideband_offset`
3. Read up to `max_length` bytes from file fd into sideband at computed address
4. Write bytes_read to response slot 0
5. Set CONTROL_READY

GPU then reads data from sideband at `sideband_ptr + SIDEBAND_DATA_OFFSET + offset`.

### Synchronization

The existing hostcall protocol provides all necessary synchronization:

1. **GPU → Host (BULK_WRITE)**:
   - GPU writes data to sideband buffer
   - GPU does `sys_store_release_u32(control, CONTROL_FILLED)` — release fence
   - GPU pushes packet to ready stack (another release)
   - Host sees packet (acquire via atomic load) → data is visible
   - Synchronization is **already provided** by CONTROL_FILLED release-acquire pair

2. **Host → GPU (BULK_READ)**:
   - GPU sends request via hostcall
   - Host reads file data into sideband buffer
   - Host does `AtomicU32::store(control, CONTROL_READY, Release)` — release fence
   - GPU sees CONTROL_READY (acquire via spin-load) → sideband data visible
   - Synchronization is **already provided** by CONTROL_READY release-acquire pair

No new fences or synchronization primitives needed.

### Host-Side Changes

#### HostcallBuffer::new() changes

```rust
pub struct HostcallBuffer {
    pub host_ptr: *mut u8,
    pub dev_ptr: sys::CUdeviceptr,
    pub size: usize,
    pub num_packets: u16,
    // NEW fields
    pub sideband_host_ptr: *mut u8,
    pub sideband_dev_ptr: sys::CUdeviceptr,
    pub sideband_size: usize,
}
```

In `new()`, allocate the sideband buffer as a separate `cuMemHostAlloc` call:
```rust
// Allocate sideband buffer
let sideband_total = SIDEBAND_HEADER_SIZE + DEFAULT_SIDEBAND_SIZE;
let mut sb_host_ptr: *mut c_void = std::ptr::null_mut();
cu.cuMemHostAlloc(&mut sb_host_ptr, sideband_total, flags);
// ... get device pointer, zero-initialize, set capacity field
```

#### Listener changes

Add SERVICE_BULK_WRITE and SERVICE_BULK_READ to the I/O thread dispatch (slow path),
since both involve file I/O operations:

```rust
// In listen_unified, when processing a packet:
SERVICE_BULK_WRITE | SERVICE_BULK_READ => {
    io_tx.send(IoRequest { pkt_idx, service }).ok();
}
```

In the I/O thread, handle bulk operations using `self.sideband_host_ptr`:
```rust
SERVICE_BULK_WRITE => {
    let fd = read_slot(payload, 0);
    let sb_offset = read_slot(payload, 1) as usize;
    let length = read_slot(payload, 2) as usize;
    let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
    let data = std::slice::from_raw_parts(data_ptr, length);
    match files.get_mut(&fd) {
        Some(file) => match file.write_all(data) {
            Ok(()) => write_slot(payload, 0, length as u64),
            Err(e) => { write_error_response(payload, &e); set_error = true; }
        },
        None => { /* ERR_INVALID_FD */ }
    }
}
```

### GPU-Side API (in gpu-runtime)

```rust
/// Write `len` bytes from `src` to file `fd` via sideband bulk transfer.
/// Returns bytes written, or 0 on error.
#[inline(always)]
pub unsafe fn gpu_bulk_write(
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    src: *const u8,
    len: usize,
) -> usize {
    // 1. Allocate space in sideband
    let offset = sideband_alloc(sideband, len as u64);
    if offset == u64::MAX { return 0; }

    // 2. Copy data to sideband
    let dst = sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
    let mut i = 0;
    while i < len {
        core::ptr::write_volatile(dst.add(i), *src.add(i));
        i += 1;
    }

    // 3. Send hostcall with sideband metadata
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_BULK_WRITE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
        core::ptr::write_volatile(payload.add(8) as *mut u64, offset);
        core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
    });
    if pkt.is_null() || !success { return 0; }

    let written = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);
    written as usize
}

/// Read up to `max_len` bytes from file `fd` into `dst` via sideband bulk transfer.
/// Returns bytes read, or 0 on error/EOF.
#[inline(always)]
pub unsafe fn gpu_bulk_read(
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    dst: *mut u8,
    max_len: usize,
) -> usize {
    // 1. Allocate space in sideband for response data
    let offset = sideband_alloc(sideband, max_len as u64);
    if offset == u64::MAX { return 0; }

    // 2. Send hostcall requesting read
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_BULK_READ, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
        core::ptr::write_volatile(payload.add(8) as *mut u64, offset);
        core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
    });
    if pkt.is_null() || !success { return 0; }

    let bytes_read = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

    if bytes_read == u64::MAX || bytes_read == 0 { return 0; }

    // 3. Copy data from sideband to destination
    let src = sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
    let mut i = 0;
    while i < bytes_read as usize {
        core::ptr::write_volatile(dst.add(i), core::ptr::read_volatile(src.add(i)));
        i += 1;
    }

    bytes_read as usize
}
```

### Kernel Interface Changes

Kernels that use bulk transfer need the sideband device pointer as an additional argument:

```rust
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bulk_io_kernel(
    hostcall_buf: *mut u8,
    sideband_buf: *mut u8,
    result: *mut u32,
) {
    gpu_panic_init(hostcall_buf);
    // ... use gpu_bulk_read / gpu_bulk_write
}
```

Host launches with both pointers:
```rust
let params: &[*mut std::ffi::c_void] = &[
    &hcbuf.dev_ptr as *const _ as *mut _,
    &hcbuf.sideband_dev_ptr as *const _ as *mut _,
    &result_dev as *const _ as *mut _,
];
```

### Bounds Checking

Both GPU and host validate bounds:
- **GPU**: `sideband_alloc()` checks `offset + size <= capacity`
- **Host**: Before accessing sideband data, verify `sideband_offset + length <= sideband_capacity`
  If out of bounds, return CONTROL_ERROR with ERR_INVALID_INPUT

### Sizing Guidelines

| Threads | Concurrent Ops | Recommended Sideband |
|---------|---------------|---------------------|
| 1-32    | 1-4           | 256KB               |
| 32-128  | 4-16          | 1MB                 |
| 128-512 | 16-64         | 4MB                 |

Default: 1MB. Configurable via `HostcallBuffer::new_with_sideband(num_packets, sideband_size)`.

### ADR-7: Sideband buffer for bulk data transfer

- **Context**: Hostcall packets have 56-byte payload limit. File I/O needs arbitrary-size transfers.
- **Decision**: Separate CUDA mapped buffer ("sideband") with GPU bump allocator. Two new services
  (BULK_WRITE, BULK_READ) carry offset+length in packet payload, data in sideband.
- **Rationale**: (1) Zero packet format changes. (2) No deadlock (one packet per request regardless
  of data size). (3) Existing CONTROL_FILLED/CONTROL_READY fences provide synchronization. (4) Bump
  allocator is simple and matches sequential I/O patterns.
- **Alternatives**: Multi-packet chaining (deadlock risk), enlarged packets (wastes memory for small ops).

## Impact on Downstream Tasks

- **large-payload.3**: Implement this design. Changes to gpu-protocol (constants + services),
  gpu-runtime (sideband_alloc, gpu_bulk_read, gpu_bulk_write), gpu-host (HostcallBuffer sideband
  allocation + I/O thread handlers). Test with 4KB+ file read/write.
- **product.8**: Can use gpu_bulk_read for parallel file grep once large-payload.3 is done.
- **gpu-libc**: Future enhancement — route libc `read()`/`write()` through bulk path when
  length > 48 bytes.
