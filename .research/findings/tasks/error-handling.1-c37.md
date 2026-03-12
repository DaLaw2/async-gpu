# error-handling.1: Error Propagation Protocol Extension
**Date**: 2026-03-12
**Cycle**: 37
**Theme**: error-handling
**Kind**: design
**Status**: done
**Spawned by**: hostcall error gap analysis

## Summary

Design an error propagation extension to the hostcall protocol that delivers
structured error codes from host to GPU, enabling GPU-side Rust code to
construct `io::Error` values from hostcall failures. The design preserves
backward compatibility, requires no packet layout changes, and uses only
existing protocol fields.

## Problem Statement

The current hostcall protocol has a binary error signal:
- `CONTROL_ERROR` bit (bit 1) tells the GPU "something went wrong"
- `FILE_ERROR_SENTINEL` (`u64::MAX`) in payload slot 0 signals failure

Neither mechanism tells the GPU *what* went wrong. All errors look identical
to GPU code — "file not found" is indistinguishable from "permission denied"
or "disk full." This blocks implementing `io::Error` in gpu-std, because
`ErrorKind` requires knowing the category of failure.

## Design Constraints

1. **No packet layout changes** — existing packet size (2112 bytes) and header
   offsets must remain unchanged. Adding a new header field would break all
   existing code.
2. **Backward compatible** — GPU code that ignores error codes must continue
   to work (it just sees "error" without detail).
3. **Minimal payload overhead** — error codes should not reduce useful data
   capacity except in error responses (where the data is typically unused).
4. **Atomics-safe** — error signaling must work with the existing spin-load
   acquire/release protocol on `PKT_OFF_CONTROL`.

## Design Decision: Where to Encode Error Codes

### Options Evaluated

| Option | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| A: New STATUS header field | Add byte at offset 20 | Clean separation | **Breaks layout**, PACKET_HEADER_SIZE changes |
| B: CONTROL bits 2..7 | Pack errno into bits 2-7 of control word | Zero payload cost | Only 6 bits = 64 codes; overloads control semantics |
| C: Payload slot 0 on error | When CONTROL_ERROR is set, slot 0 = errno | No header changes, 64-bit errno space | Uses slot 0 (already used for FILE_ERROR_SENTINEL) |
| D: CONTROL bits 8..15 + payload | Use bits 8-15 for errno category, payload for detail | Separate category from data | Mixes semantics in control word |

### Selected: Option C — Payload Slot 0 as Error Code

**Rationale:**
- The current protocol already writes `FILE_ERROR_SENTINEL` (`u64::MAX`) to
  slot 0 on error — this is a degenerate case of "error code in slot 0."
- When `CONTROL_ERROR` is set, the response payload is meaningless to the
  caller anyway (no valid data was produced). Repurposing slot 0 costs nothing.
- A full `u64` gives ample space for structured error encoding.
- No header layout changes. No new offsets. No alignment concerns.
- GPU code already checks `CONTROL_ERROR` before reading payload — the
  existing code path naturally separates success (read payload as data) from
  failure (read payload as error code).

## Error Code Encoding

### Slot 0 Layout on Error

When `CONTROL_ERROR` is set in the response, payload slot 0 contains:

```
Bits 63..32: reserved (zero, for future extension)
Bits 31..16: extended errno (raw OS errno, optional, 0 = not provided)
Bits 15..0:  error category (ErrorCategory enum)
```

This gives:
- 16-bit error category: maps directly to Rust `ErrorKind` variants
- 16-bit raw errno: the actual OS errno for debugging/edge cases
- 32-bit reserved: future use (sub-error codes, error source chaining)

### ErrorCategory Enum

```rust
/// Hostcall error categories.
/// These map 1:1 to a subset of `std::io::ErrorKind`.
#[repr(u16)]
pub enum ErrorCategory {
    /// Unspecified error (fallback).
    Other           = 0,
    /// Entity not found (ENOENT).
    NotFound        = 1,
    /// Permission denied (EACCES, EPERM).
    PermissionDenied = 2,
    /// Entity already exists (EEXIST).
    AlreadyExists   = 3,
    /// Invalid input/argument (EINVAL).
    InvalidInput    = 4,
    /// I/O error (EIO).
    IoError         = 5,
    /// Operation timed out (ETIMEDOUT).
    TimedOut        = 6,
    /// Operation would block (EAGAIN, EWOULDBLOCK).
    WouldBlock      = 7,
    /// Broken pipe (EPIPE).
    BrokenPipe      = 8,
    /// Resource busy (EBUSY).
    ResourceBusy    = 9,
    /// No space left on device (ENOSPC).
    StorageFull     = 10,
    /// Too many open files (EMFILE, ENFILE).
    TooManyFiles    = 11,
    /// Out of memory (ENOMEM).
    OutOfMemory     = 12,
    /// Bad file descriptor (EBADF).
    InvalidFd       = 13,
    /// Connection refused (ECONNREFUSED) — future network support.
    ConnectionRefused = 14,
    /// Not a directory / is a directory (ENOTDIR, EISDIR).
    IsADirectory    = 15,
    /// Hostcall timeout — GPU spin limit exceeded.
    HostTimeout     = 16,
    /// Unknown/unsupported service opcode.
    Unsupported     = 17,
}
```

### Helper Functions (gpu-protocol)

```rust
/// Encode an error into payload slot 0 format.
pub const fn encode_error(category: u16, raw_errno: u16) -> u64 {
    ((raw_errno as u64) << 16) | (category as u64)
}

/// Decode error category from payload slot 0.
pub const fn error_category(slot0: u64) -> u16 {
    (slot0 & 0xFFFF) as u16
}

/// Decode raw errno from payload slot 0.
pub const fn error_raw_errno(slot0: u64) -> u16 {
    ((slot0 >> 16) & 0xFFFF) as u16
}
```

## POSIX errno → ErrorCategory Mapping

The host side maps `std::io::Error::raw_os_error()` to `ErrorCategory`:

| errno | POSIX name | ErrorCategory |
|-------|-----------|---------------|
| 2 | ENOENT | NotFound (1) |
| 13 | EACCES | PermissionDenied (2) |
| 1 | EPERM | PermissionDenied (2) |
| 17 | EEXIST | AlreadyExists (3) |
| 22 | EINVAL | InvalidInput (4) |
| 5 | EIO | IoError (5) |
| 110 | ETIMEDOUT | TimedOut (6) |
| 11 | EAGAIN | WouldBlock (7) |
| 32 | EPIPE | BrokenPipe (8) |
| 16 | EBUSY | ResourceBusy (9) |
| 28 | ENOSPC | StorageFull (10) |
| 24 | EMFILE | TooManyFiles (11) |
| 23 | ENFILE | TooManyFiles (11) |
| 12 | ENOMEM | OutOfMemory (12) |
| 9 | EBADF | InvalidFd (13) |
| 21 | EISDIR | IsADirectory (15) |
| 20 | ENOTDIR | IsADirectory (15) |
| * | (other) | Other (0) |

Note: errno values are Linux-specific. On Windows the host would map
`GetLastError()` codes instead, but the ErrorCategory abstraction hides this.

## Protocol Flow Changes

### Current Flow (error case)

```
GPU                          HOST
 |  push packet (SERVICE_OPEN)  |
 |  ─────────────────────────>  |
 |                               |  open() fails with ENOENT
 |                               |  write FILE_ERROR_SENTINEL to slot 0
 |                               |  eprintln!("error: ...")
 |  <─────────────────────────  |  set CONTROL_READY | CONTROL_ERROR
 |  see CONTROL_ERROR            |
 |  return FILE_ERROR_SENTINEL   |
 |  (no idea what went wrong)    |
```

### Proposed Flow (error case)

```
GPU                          HOST
 |  push packet (SERVICE_OPEN)  |
 |  ─────────────────────────>  |
 |                               |  open() fails with ENOENT (errno=2)
 |                               |  slot 0 = encode_error(NotFound, 2)
 |  <─────────────────────────  |  set CONTROL_READY | CONTROL_ERROR
 |  see CONTROL_ERROR            |
 |  read slot 0                  |
 |  category = NotFound          |
 |  raw_errno = 2                |
 |  → io::Error::new(            |
 |      ErrorKind::NotFound,     |
 |      "hostcall: file not found"|
 |    )                          |
```

## Host-Side Changes

### Error Encoding in Handlers

Each `handle_*` method changes from:

```rust
// Before:
Err(e) => {
    eprintln!("  [HOST] FILE OPEN ERROR: {}", e);
    std::ptr::write_volatile(payload as *mut u64, FILE_ERROR_SENTINEL);
    false
}
```

To:

```rust
// After:
Err(e) => {
    eprintln!("  [HOST] FILE OPEN ERROR: {}", e);
    let raw = e.raw_os_error().unwrap_or(0) as u16;
    let cat = errno_to_category(raw);
    std::ptr::write_volatile(payload as *mut u64, encode_error(cat, raw));
    true  // signal CONTROL_ERROR
}
```

Key change: file-level errors now set `has_error = true` so `CONTROL_ERROR`
is set in the control word. Currently, file errors return `false` (no
CONTROL_ERROR) and rely solely on FILE_ERROR_SENTINEL — this means GPU code
must check *both* CONTROL_ERROR and sentinel, which is fragile.

### errno_to_category Function

```rust
fn errno_to_category(raw: u16) -> u16 {
    match raw as i32 {
        2   => 1,   // ENOENT → NotFound
        13  => 2,   // EACCES → PermissionDenied
        1   => 2,   // EPERM → PermissionDenied
        17  => 3,   // EEXIST → AlreadyExists
        22  => 4,   // EINVAL → InvalidInput
        5   => 5,   // EIO → IoError
        110 => 6,   // ETIMEDOUT → TimedOut
        11  => 7,   // EAGAIN → WouldBlock
        32  => 8,   // EPIPE → BrokenPipe
        16  => 9,   // EBUSY → ResourceBusy
        28  => 10,  // ENOSPC → StorageFull
        24 | 23 => 11, // EMFILE/ENFILE → TooManyFiles
        12  => 12,  // ENOMEM → OutOfMemory
        9   => 13,  // EBADF → InvalidFd
        21 | 20 => 15, // EISDIR/ENOTDIR → IsADirectory
        _   => 0,   // Other
    }
}
```

## GPU-Side Changes

### Error Decoding in gpu-kernel

GPU-side hostcall wrappers change from returning `FILE_ERROR_SENTINEL` to
returning a structured error:

```rust
/// Result type for GPU hostcall operations.
/// On error, contains the encoded error value from payload slot 0.
type HostcallResult = Result<u64, u64>;

/// Decode a hostcall error into components.
fn decode_hostcall_error(err: u64) -> (u16, u16) {
    (error_category(err), error_raw_errno(err))
}
```

### gpu_hostcall_open Example

```rust
// Before:
unsafe fn gpu_hostcall_open(...) -> u64 {
    // ... on error:
    return FILE_ERROR_SENTINEL;
}

// After:
unsafe fn gpu_hostcall_open(...) -> HostcallResult {
    let (pkt, success) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| { ... });
    if !success {
        if !pkt.is_null() { gpu_hostcall_release(buf, pkt); }
        return Err(encode_error(ErrorCategory::HostTimeout as u16, 0));
    }
    let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);
    gpu_hostcall_release(buf, pkt);
    if ctrl & CONTROL_ERROR != 0 {
        Err(slot0) // slot0 contains encode_error(category, raw_errno)
    } else {
        Ok(slot0) // slot0 contains fd
    }
}
```

### Integration with gpu-std io::Error

In the gpu-std `File` implementation (future work), errors convert to
`io::Error`:

```rust
fn category_to_error_kind(cat: u16) -> io::ErrorKind {
    match cat {
        1  => io::ErrorKind::NotFound,
        2  => io::ErrorKind::PermissionDenied,
        3  => io::ErrorKind::AlreadyExists,
        4  => io::ErrorKind::InvalidInput,
        6  => io::ErrorKind::TimedOut,
        7  => io::ErrorKind::WouldBlock,
        8  => io::ErrorKind::BrokenPipe,
        12 => io::ErrorKind::OutOfMemory,
        17 => io::ErrorKind::Unsupported,
        _  => io::ErrorKind::Other,
    }
}
```

## Timeout Error Handling

### Current Behavior

GPU spin-waits up to `GPU_MAX_SPIN` (10M iterations). On timeout:
- `gpu_hostcall_request` returns `success = false`
- Callers return `FILE_ERROR_SENTINEL`
- No information about whether it was a timeout vs. other failure

### Proposed Behavior

On timeout, the GPU constructs a timeout error locally:

```rust
if !success {
    // Timeout — host did not respond
    if !pkt.is_null() { gpu_hostcall_release(buf, pkt); }
    return Err(encode_error(ErrorCategory::HostTimeout as u16, 0));
}
```

This is a GPU-local error (no host involvement), so `raw_errno = 0`.

### Timeout Recovery

The packet may still be in the ready stack when the GPU gives up. The host
might process it later and write a response to a packet that has already been
released back to the free stack and potentially reused.

**Mitigation**: On timeout, the GPU must NOT release the packet back to the
free stack. The packet is "leaked" — it stays allocated but unused. This
prevents use-after-free but costs one packet slot permanently.

A more sophisticated approach (future work):
- Add a `CONTROL_CANCELLED` bit (bit 2) that the GPU sets on timeout
- Host checks this bit before processing and skips cancelled packets
- Host releases cancelled packets back to free stack

## Backward Compatibility

### FILE_ERROR_SENTINEL Deprecation Path

1. **Phase 1 (this design)**: Host writes `encode_error(cat, errno)` instead
   of `FILE_ERROR_SENTINEL` on error. Host sets `CONTROL_ERROR` for all file
   errors (not just unknown services).

2. **Phase 2**: GPU code checks `CONTROL_ERROR` first (fast path), then reads
   error details from slot 0. Legacy GPU code that only checks
   `FILE_ERROR_SENTINEL` still works because `encode_error(...)` produces
   values != `u64::MAX` — but this means legacy code sees "success" with
   garbage data, which is a **breaking change**.

3. **Resolution**: To maintain backward compat during transition, the host
   could write `FILE_ERROR_SENTINEL` to slot 0 AND set `CONTROL_ERROR`.
   New GPU code reads error details only when `CONTROL_ERROR` is set.
   Old GPU code checks `slot0 == FILE_ERROR_SENTINEL` and still works.

   However, this loses error detail. Better approach: update GPU code
   atomically with host code (both in this repo, single commit).

### Recommendation

Since both GPU kernel and host handler code live in this repository, update
both sides simultaneously. No backward compatibility concern — there are no
external consumers.

## Implementation Plan

| Step | Component | Change |
|------|-----------|--------|
| 1 | `gpu-protocol` | Add `ErrorCategory` enum, `encode_error`/`error_category`/`error_raw_errno` helpers |
| 2 | `gpu-host` | Add `errno_to_category()`, update `handle_open`/`handle_write`/`handle_read`/`handle_close`/`handle_stdin` to encode errors and return `true` |
| 3 | `gpu-kernel` | Update `gpu_hostcall_request` to return control bits alongside packet; update `gpu_hostcall_open/write/read/close/stdin_read` to propagate error codes |
| 4 | `gpu-kernel` | Update timeout path to encode `HostTimeout` error |
| 5 | Tests | Add host-side unit test: trigger ENOENT, verify encoded error in packet |
| 6 | `gpu-std` (future) | Map `ErrorCategory` → `io::ErrorKind` in `File::open()` etc. |

## Open Questions

1. **Should PRINT errors be propagated?** Currently print is fire-and-forget.
   If the host fails to print, does GPU code care? Probably not — keep
   PRINT as non-error-reporting for now.

2. **Multi-lane errors**: Current protocol is lane-0-only for file I/O. If
   future services use per-lane payloads, each lane might need its own error
   code. Defer until multi-lane services exist.

3. **Error message strings**: Should the host send a human-readable error
   message in addition to the error code? Possible (slots 1-7 = 56 bytes of
   message text on error), but adds complexity. Defer — the category + errno
   is sufficient for `io::Error` construction.

## Key Findings

- The existing `CONTROL_ERROR` bit is already defined and checked by GPU code,
  but file I/O handlers return `false` (no error flag), relying solely on
  `FILE_ERROR_SENTINEL`. This inconsistency should be fixed.
- Repurposing payload slot 0 for structured error codes requires zero layout
  changes and is the most natural extension of the existing sentinel pattern.
- The timeout leak problem (packet stuck in ready stack after GPU gives up) is
  a pre-existing issue, not introduced by this design. The `CONTROL_CANCELLED`
  bit is a clean future fix.
- 18 error categories cover all common POSIX file I/O failures plus
  GPU-specific cases (timeout, unsupported service).
