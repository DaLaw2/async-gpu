//! Hostcall protocol shared definitions for GPU-host communication.
//!
//! This crate defines the wire protocol between GPU kernels running on
//! `nvptx64-nvidia-cuda` and the host CPU. It is `#![no_std]` so it compiles
//! for both targets.
//!
//! # Architecture
//!
//! Communication flows through a **hostcall buffer** — a CUDA mapped memory
//! region visible to both GPU and CPU. The buffer contains:
//!
//! 1. A **header** (64 bytes) with free/ready stacks and metadata
//! 2. A **packet pool** — fixed-size packets containing service ID + payload
//! 3. Optional **per-block shards** for reduced CAS contention at scale
//!
//! Each packet holds a 32-lane × 8-slot payload (2048 bytes), allowing a full
//! warp to communicate with the host in a single round-trip.
//!
//! # Sideband Buffer
//!
//! For bulk data transfer beyond the 56-byte packet payload limit, a separate
//! **sideband buffer** (default 1 MB) is allocated. GPU code uses a bump
//! allocator to reserve regions, then references them by offset in
//! `BULK_READ`/`BULK_WRITE` hostcalls.

#![no_std]

// ============================================================
// Service IDs
// ============================================================

/// No-op service — used for latency benchmarking.
pub const SERVICE_NOP: u32 = 0;
/// Print a message from GPU to host stdout.
pub const SERVICE_PRINT: u32 = 1;
/// Write data to an open file (up to 48 bytes inline).
pub const SERVICE_WRITE: u32 = 2;
/// Read data from an open file (up to 56 bytes inline).
pub const SERVICE_READ: u32 = 3;
/// Open a file by path, returning a file descriptor.
pub const SERVICE_OPEN: u32 = 4;
/// Close a file descriptor.
pub const SERVICE_CLOSE: u32 = 5;
/// Allocate host memory (reserved, not yet implemented).
pub const SERVICE_MALLOC: u32 = 6;
/// Free host memory (reserved, not yet implemented).
pub const SERVICE_FREE: u32 = 7;
/// Abort the GPU kernel (fatal).
pub const SERVICE_ABORT: u32 = 0xFF;
/// Read a line from host stdin.
pub const SERVICE_STDIN: u32 = 8;
/// Get current wall-clock time from host.
pub const SERVICE_TIME: u32 = 9;
/// Report a GPU panic (message + location), then trap.
pub const SERVICE_PANIC: u32 = 10;
/// Write bulk data from sideband buffer to a file.
pub const SERVICE_BULK_WRITE: u32 = 11;
/// Read bulk data from a file into sideband buffer.
pub const SERVICE_BULK_READ: u32 = 12;

// ============================================================
// Control bits (PacketHeader.control)
// ============================================================

/// Set by host when the response is ready for GPU to consume.
pub const CONTROL_READY: u32 = 1;
/// Set by host alongside `CONTROL_READY` to indicate an error occurred.
pub const CONTROL_ERROR: u32 = 2;
/// Set by GPU after filling the packet, before pushing to ready stack.
/// Host checks this bit before processing — skips if not set (stale re-visit).
pub const CONTROL_FILLED: u32 = 4;

// ============================================================
// Tagged pointer constants
// ============================================================

/// Null sentinel for tagged pointers (no packet).
pub const NULL_INDEX: u16 = 0xFFFF;

// ============================================================
// Layout sizes
// ============================================================

/// Number of threads per warp (NVIDIA architecture constant).
pub const WARP_SIZE: u32 = 32;
/// Number of 8-byte slots per lane in a packet payload.
pub const SLOTS_PER_LANE: usize = 8;
/// Size of the buffer header in bytes.
pub const BUFFER_HEADER_SIZE: usize = 64;
/// Size of the packet header in bytes (before payload).
pub const PACKET_HEADER_SIZE: usize = 32;
/// Payload size: 32 lanes × 8 slots × 8 bytes = 2048 bytes.
pub const PACKET_PAYLOAD_SIZE: usize = (WARP_SIZE as usize) * SLOTS_PER_LANE * 8;
/// Total packet size: header (32) + payload (2048), rounded to 64-byte alignment = 2112.
pub const PACKET_SIZE: usize = 2112;

// ============================================================
// Buffer header field offsets (from buffer base pointer)
// ============================================================

/// Offset of the global free stack head (u64, tagged pointer).
pub const BUF_OFF_FREE_STACK: usize = 0;
/// Offset of the global ready stack head (u64, tagged pointer).
pub const BUF_OFF_READY_STACK: usize = 8;
/// Offset of the doorbell counter (u64, incremented by GPU on each push).
pub const BUF_OFF_DOORBELL: usize = 16;
/// Offset of the shutdown flag (u32, non-zero = listener should exit).
pub const BUF_OFF_SHUTDOWN: usize = 24;
/// Offset of the total packet count (u32).
pub const BUF_OFF_NUM_PACKETS: usize = 28;
/// Offset of the warp size field (u32, always 32).
pub const BUF_OFF_WARP_SIZE: usize = 32;
/// Offset of the shard count (u32, 0 = legacy single-stack mode).
pub const BUF_OFF_NUM_SHARDS: usize = 36;
/// Offset of the packets-per-shard count (u32).
pub const BUF_OFF_PKTS_PER_SHARD: usize = 40;
/// Offset of the shard array start offset (u32).
pub const BUF_OFF_SHARD_ARRAY_OFF: usize = 44;

// ============================================================
// Per-block sharding layout
// ============================================================
//
// Each shard entry is 16 bytes:
//   Offset 0: shard_free_stack  (u64) — tagged pointer
//   Offset 8: shard_ready_stack (u64) — tagged pointer

/// Size of one shard entry in bytes (free_stack + ready_stack).
pub const SHARD_ENTRY_SIZE: usize = 16;
/// Offset of the free stack within a shard entry.
pub const SHARD_OFF_FREE_STACK: usize = 0;
/// Offset of the ready stack within a shard entry.
pub const SHARD_OFF_READY_STACK: usize = 8;

/// Byte offset of shard entry `shard_idx` from buffer base.
#[inline(always)]
pub const fn shard_entry_offset(shard_array_offset: usize, shard_idx: u32) -> usize {
    shard_array_offset + (shard_idx as usize) * SHARD_ENTRY_SIZE
}

/// Byte offset of packet `index` in a sharded buffer.
///
/// Packets start after the shard array: `shard_array_offset + num_shards * SHARD_ENTRY_SIZE`.
#[inline(always)]
pub const fn packet_offset_sharded(
    index: u16,
    shard_array_offset: usize,
    num_shards: u32,
) -> usize {
    shard_array_offset + (num_shards as usize) * SHARD_ENTRY_SIZE + (index as usize) * PACKET_SIZE
}

/// Total buffer size for a sharded buffer with `num_packets` packets and `num_shards` shards.
#[inline(always)]
pub const fn buffer_size_sharded(num_packets: u16, num_shards: u32) -> usize {
    BUFFER_HEADER_SIZE
        + (num_shards as usize) * SHARD_ENTRY_SIZE
        + (num_packets as usize) * PACKET_SIZE
}

// ============================================================
// Packet field offsets (from packet base pointer)
// ============================================================

/// Offset of the next-pointer field (u64, tagged pointer for stack linkage).
pub const PKT_OFF_NEXT: usize = 0;
/// Offset of the active mask field (u32, which lanes are participating).
pub const PKT_OFF_ACTIVE_MASK: usize = 8;
/// Offset of the service ID field (u32).
pub const PKT_OFF_SERVICE: usize = 12;
/// Offset of the control flags field (u32).
pub const PKT_OFF_CONTROL: usize = 16;
/// Offset of the payload region (32 lanes × 8 slots × 8 bytes).
pub const PKT_OFF_PAYLOAD: usize = PACKET_HEADER_SIZE;

// ============================================================
// Tagged pointer helpers
// ============================================================
//
// Layout:
//   Bits 63..32: ABA tag (monotonically increasing)
//   Bits 31..16: reserved (zero)
//   Bits 15..0:  packet index (0..N-1), or 0xFFFF = NULL

/// Extract the packet index from a tagged pointer.
///
/// ```
/// use gpu_protocol::{make_tagged, tagged_index};
/// assert_eq!(tagged_index(make_tagged(5, 42)), 42);
/// ```
#[inline(always)]
pub const fn tagged_index(tagged: u64) -> u16 {
    (tagged & 0xFFFF) as u16
}

/// Extract the ABA tag from a tagged pointer.
///
/// ```
/// use gpu_protocol::{make_tagged, tagged_tag};
/// assert_eq!(tagged_tag(make_tagged(5, 42)), 5);
/// ```
#[inline(always)]
pub const fn tagged_tag(tagged: u64) -> u32 {
    (tagged >> 32) as u32
}

/// Construct a tagged pointer from a tag and packet index.
///
/// ```
/// use gpu_protocol::make_tagged;
/// let ptr = make_tagged(1, 0);
/// assert_eq!(ptr, 0x0000_0001_0000_0000);
/// ```
#[inline(always)]
pub const fn make_tagged(tag: u32, index: u16) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

/// Construct a null tagged pointer (index = `NULL_INDEX`).
///
/// ```
/// use gpu_protocol::{null_tagged, tagged_index, NULL_INDEX};
/// assert_eq!(tagged_index(null_tagged()), NULL_INDEX);
/// ```
#[inline(always)]
pub const fn null_tagged() -> u64 {
    make_tagged(0, NULL_INDEX)
}

// ============================================================
// Offset calculators
// ============================================================

/// Byte offset of packet `index` from buffer base (legacy non-sharded layout).
#[inline(always)]
pub const fn packet_offset(index: u16) -> usize {
    BUFFER_HEADER_SIZE + (index as usize) * PACKET_SIZE
}

/// Byte offset of a specific payload slot from the packet base.
///
/// `lane` is the warp lane (0..31), `slot` is the slot index (0..7).
#[inline(always)]
pub const fn payload_slot_offset(lane: u32, slot: usize) -> usize {
    PKT_OFF_PAYLOAD + (lane as usize) * SLOTS_PER_LANE * 8 + slot * 8
}

/// Total buffer size in bytes for `num_packets` packets (legacy non-sharded layout).
#[inline(always)]
pub const fn buffer_size(num_packets: u16) -> usize {
    BUFFER_HEADER_SIZE + (num_packets as usize) * PACKET_SIZE
}

// ============================================================
// PRINT service payload layout (lane 0)
// ============================================================

/// Maximum message length for `SERVICE_PRINT` (7 slots × 8 bytes).
pub const PRINT_MAX_MSG_LEN: usize = 56;

// ============================================================
// FILE I/O service payload layouts (lane 0)
// ============================================================
//
// SERVICE_OPEN request:
//   Slot 0: flags (u64) — see FILE_OPEN_* constants
//   Slot 1: path length (u64)
//   Slots 2-7: path bytes (up to 48 bytes)
// SERVICE_OPEN response:
//   Slot 0: file descriptor (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_WRITE request:
//   Slot 0: fd (u64)
//   Slot 1: data length (u64)
//   Slots 2-7: data bytes (up to 48 bytes)
// SERVICE_WRITE response:
//   Slot 0: bytes written (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_READ request:
//   Slot 0: fd (u64)
//   Slot 1: max bytes to read (u64)
// SERVICE_READ response:
//   Slot 0: bytes read (u64), or FILE_ERROR_SENTINEL on error
//   Slots 1-7: data bytes (up to 56 bytes)
//
// SERVICE_CLOSE request:
//   Slot 0: fd (u64)
// SERVICE_CLOSE response:
//   Slot 0: 0 on success, FILE_ERROR_SENTINEL on error

/// Maximum path length for `SERVICE_OPEN` (7 slots × 8 bytes).
pub const FILE_MAX_PATH_LEN: usize = 56;
/// Maximum inline write length for `SERVICE_WRITE` (6 slots × 8 bytes).
pub const FILE_MAX_WRITE_LEN: usize = 48;
/// Maximum inline read length for `SERVICE_READ` (7 slots × 8 bytes).
pub const FILE_MAX_READ_LEN: usize = 56;
/// Sentinel value indicating a file I/O error in response slot 0.
pub const FILE_ERROR_SENTINEL: u64 = u64::MAX;

/// Open a file for reading.
pub const FILE_OPEN_READ: u32 = 0;
/// Open a file for writing (create if not exists, truncate if exists).
pub const FILE_OPEN_WRITE_CREATE: u32 = 1;
/// Open a file for appending.
pub const FILE_OPEN_APPEND: u32 = 2;

// ============================================================
// Error encoding (structured error propagation)
// ============================================================
//
// When CONTROL_ERROR is set in the response, payload slot 0 contains:
//   Bits 63..32: reserved (zero)
//   Bits 31..16: raw OS errno (optional, 0 = not provided)
//   Bits 15..0:  error category (one of the ERR_* constants)

/// Generic/unknown error.
pub const ERR_OTHER: u16 = 0;
/// File or resource not found.
pub const ERR_NOT_FOUND: u16 = 1;
/// Permission denied.
pub const ERR_PERMISSION_DENIED: u16 = 2;
/// Resource already exists.
pub const ERR_ALREADY_EXISTS: u16 = 3;
/// Invalid input parameter.
pub const ERR_INVALID_INPUT: u16 = 4;
/// General I/O error.
pub const ERR_IO_ERROR: u16 = 5;
/// Operation timed out.
pub const ERR_TIMED_OUT: u16 = 6;
/// Operation would block (non-blocking mode).
pub const ERR_WOULD_BLOCK: u16 = 7;
/// Broken pipe.
pub const ERR_BROKEN_PIPE: u16 = 8;
/// Resource is busy.
pub const ERR_RESOURCE_BUSY: u16 = 9;
/// Storage is full.
pub const ERR_STORAGE_FULL: u16 = 10;
/// Too many open files.
pub const ERR_TOO_MANY_FILES: u16 = 11;
/// Out of memory.
pub const ERR_OUT_OF_MEMORY: u16 = 12;
/// Invalid file descriptor.
pub const ERR_INVALID_FD: u16 = 13;
/// Connection refused (reserved for future networking).
pub const ERR_CONNECTION_REFUSED: u16 = 14;
/// Path is a directory, not a file.
pub const ERR_IS_A_DIRECTORY: u16 = 15;
/// Host-side timeout processing the request.
pub const ERR_HOST_TIMEOUT: u16 = 16;
/// Operation not supported.
pub const ERR_UNSUPPORTED: u16 = 17;

/// Encode an error category and raw errno into payload slot 0 format.
///
/// ```
/// use gpu_protocol::{encode_error, error_category, error_raw_errno, ERR_NOT_FOUND};
/// let encoded = encode_error(ERR_NOT_FOUND, 2);
/// assert_eq!(error_category(encoded), ERR_NOT_FOUND);
/// assert_eq!(error_raw_errno(encoded), 2);
/// ```
#[inline(always)]
pub const fn encode_error(category: u16, raw_errno: u16) -> u64 {
    ((raw_errno as u64) << 16) | (category as u64)
}

/// Decode the error category from an encoded error value.
#[inline(always)]
pub const fn error_category(slot0: u64) -> u16 {
    (slot0 & 0xFFFF) as u16
}

/// Decode the raw OS errno from an encoded error value.
#[inline(always)]
pub const fn error_raw_errno(slot0: u64) -> u16 {
    ((slot0 >> 16) & 0xFFFF) as u16
}

// ============================================================
// GPU-side error types for Result-based error propagation
// ============================================================

/// GPU-side error returned by hostcall helpers.
///
/// Encodes an error category (from the ERR_* constants) and an optional
/// OS errno. Small enough (4 bytes) to be returned by value in Result.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuError {
    /// Error category (one of the ERR_* constants).
    pub category: u16,
    /// Raw OS errno from the host, or 0 if not applicable.
    pub raw_errno: u16,
}

impl GpuError {
    /// Create a new GpuError from category and errno.
    #[inline(always)]
    pub const fn new(category: u16, raw_errno: u16) -> Self {
        Self {
            category,
            raw_errno,
        }
    }

    /// Create a GpuError from an encoded error value (payload slot 0 format).
    #[inline(always)]
    pub const fn from_encoded(slot0: u64) -> Self {
        Self {
            category: error_category(slot0),
            raw_errno: error_raw_errno(slot0),
        }
    }

    /// Pool exhaustion — no free packets available.
    #[inline(always)]
    pub const fn pool_exhausted() -> Self {
        Self::new(ERR_RESOURCE_BUSY, 0)
    }

    /// Timeout waiting for host response.
    #[inline(always)]
    pub const fn timeout() -> Self {
        Self::new(ERR_HOST_TIMEOUT, 0)
    }
}

/// Sentinel values for GpuKernelResult tag field.
pub const TAG_OK: u32 = 0;
/// Kernel returned an error.
pub const TAG_ERR: u32 = 1;
/// Buffer not yet written — kernel may have crashed.
pub const TAG_UNINIT: u32 = 0xDEAD_BEEF;

/// GPU kernel result buffer — 64 bytes, one cache line.
///
/// Passed as the last kernel parameter. Host allocates mapped memory,
/// initializes tag to TAG_UNINIT, launches kernel, then reads result
/// after synchronization.
///
/// Layout:
/// ```text
/// Offset  0: tag        (u32)  — TAG_OK, TAG_ERR, or TAG_UNINIT
/// Offset  4: category   (u16)  — ERR_* constant
/// Offset  6: raw_errno  (u16)  — OS errno
/// Offset  8: thread_idx (u16)  — threadIdx.x that produced error
/// Offset 10: block_idx  (u16)  — blockIdx.x that produced error
/// Offset 12: msg_len    (u32)  — message byte count (0..48)
/// Offset 16: msg_bytes  [48]   — UTF-8 message (truncated if needed)
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct GpuKernelResult {
    /// TAG_OK, TAG_ERR, or TAG_UNINIT.
    pub tag: u32,
    /// ERR_* constant identifying the error category.
    pub category: u16,
    /// OS errno value from the GPU side.
    pub raw_errno: u16,
    /// `threadIdx.x` that produced this result.
    pub thread_idx: u16,
    /// `blockIdx.x` that produced this result.
    pub block_idx: u16,
    /// Byte count of the message in `msg_bytes` (0..48).
    pub msg_len: u32,
    /// UTF-8 message bytes (truncated to 48 bytes if needed).
    pub msg_bytes: [u8; 48],
}

impl GpuKernelResult {
    /// Create an uninitialized result (sentinel value).
    #[inline(always)]
    pub const fn uninit() -> Self {
        Self {
            tag: TAG_UNINIT,
            category: 0,
            raw_errno: 0,
            thread_idx: 0,
            block_idx: 0,
            msg_len: 0,
            msg_bytes: [0u8; 48],
        }
    }

    /// Write OK status.
    #[inline(always)]
    pub fn set_ok(&mut self) {
        self.tag = TAG_OK;
    }

    /// Write error status with a GpuError and optional message.
    #[inline(always)]
    pub fn set_err(&mut self, err: GpuError, thread_idx: u16, block_idx: u16, msg: &[u8]) {
        self.tag = TAG_ERR;
        self.category = err.category;
        self.raw_errno = err.raw_errno;
        self.thread_idx = thread_idx;
        self.block_idx = block_idx;
        let len = if msg.len() > 48 { 48 } else { msg.len() };
        self.msg_len = len as u32;
        let mut i = 0;
        while i < len {
            self.msg_bytes[i] = msg[i];
            i += 1;
        }
    }

    /// Check if the result indicates success.
    #[inline(always)]
    pub const fn is_ok(&self) -> bool {
        self.tag == TAG_OK
    }

    /// Check if the result indicates an error.
    #[inline(always)]
    pub const fn is_err(&self) -> bool {
        self.tag == TAG_ERR
    }

    /// Get the error message as a byte slice.
    #[inline(always)]
    pub fn message(&self) -> &[u8] {
        let len = if (self.msg_len as usize) > 48 {
            48
        } else {
            self.msg_len as usize
        };
        &self.msg_bytes[..len]
    }
}

// ============================================================
// PANIC service payload layout (lane 0)
// ============================================================
//
// Request:
//   Slot 0: metadata (u64)
//     - Bits 15..0:  threadIdx.x (u16)
//     - Bits 31..16: blockIdx.x (u16)
//     - Bits 47..32: message length (u16)
//     - Bits 63..48: reserved (zero)
//   Slots 1-7: panic message bytes (up to 56 bytes, truncated)
// Response:
//   (CONTROL_READY set — GPU thread will trap regardless)

/// Maximum panic message length (7 slots × 8 bytes).
pub const PANIC_MAX_MSG_LEN: usize = 56;

/// Encode panic metadata: thread index, block index, and message length.
///
/// ```
/// use gpu_protocol::{encode_panic_metadata, panic_thread_idx, panic_block_idx, panic_msg_len};
/// let meta = encode_panic_metadata(5, 3, 20);
/// assert_eq!(panic_thread_idx(meta), 5);
/// assert_eq!(panic_block_idx(meta), 3);
/// assert_eq!(panic_msg_len(meta), 20);
/// ```
#[inline(always)]
pub const fn encode_panic_metadata(thread_idx: u16, block_idx: u16, msg_len: u16) -> u64 {
    (thread_idx as u64) | ((block_idx as u64) << 16) | ((msg_len as u64) << 32)
}

/// Decode `threadIdx.x` from panic metadata.
#[inline(always)]
pub const fn panic_thread_idx(meta: u64) -> u16 {
    (meta & 0xFFFF) as u16
}

/// Decode `blockIdx.x` from panic metadata.
#[inline(always)]
pub const fn panic_block_idx(meta: u64) -> u16 {
    ((meta >> 16) & 0xFFFF) as u16
}

/// Decode message length from panic metadata.
#[inline(always)]
pub const fn panic_msg_len(meta: u64) -> u16 {
    ((meta >> 32) & 0xFFFF) as u16
}

// ============================================================
// Sideband buffer (bulk data transfer)
// ============================================================
//
// The sideband buffer is a separate CUDA mapped allocation used for
// transferring data larger than the 56-byte packet payload limit.
//
// Header (64 bytes):
//   Offset 0:  alloc_offset (u64) — GPU bump allocator position
//   Offset 8:  capacity (u64)     — total data region size in bytes
//   Offset 16: reserved (48 bytes)
//
// Data Region:
//   Starts at SIDEBAND_HEADER_SIZE from sideband base
//   Contiguous byte array, `capacity` bytes
//
// SERVICE_BULK_WRITE request (lane 0):
//   Slot 0: fd (u64)
//   Slot 1: sideband_offset (u64) — offset within data region
//   Slot 2: length (u64)
//   Response slot 0: bytes_written (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_BULK_READ request (lane 0):
//   Slot 0: fd (u64)
//   Slot 1: sideband_offset (u64)
//   Slot 2: max_length (u64)
//   Response slot 0: bytes_read (u64), or FILE_ERROR_SENTINEL on error

/// Size of the sideband header in bytes.
pub const SIDEBAND_HEADER_SIZE: usize = 64;
/// Offset of the bump allocator position within the sideband header.
pub const SIDEBAND_OFF_ALLOC: usize = 0;
/// Offset of the capacity field within the sideband header.
pub const SIDEBAND_OFF_CAPACITY: usize = 8;
/// Byte offset where the sideband data region begins.
pub const SIDEBAND_DATA_OFFSET: usize = SIDEBAND_HEADER_SIZE;

/// Default sideband data region size: 1 MB.
pub const DEFAULT_SIDEBAND_SIZE: usize = 1024 * 1024;

// ============================================================
// GPU-side spin limit
// ============================================================

/// Maximum number of poll iterations before the GPU executor traps.
///
/// With a 64ns nanosleep between polls, this gives ~640ms before timeout.
pub const GPU_MAX_SPIN: u32 = 10_000_000;

// ============================================================
// STDIN service payload layout (lane 0)
// ============================================================

/// Maximum bytes readable from stdin in one hostcall (7 slots × 8 bytes).
pub const STDIN_MAX_READ_LEN: usize = 56;

// ============================================================
// TIME service payload layout (lane 0)
// ============================================================
// Request: (no payload needed)
// Response:
//   Slot 0: seconds since Unix epoch (u64)
//   Slot 1: nanoseconds within second (u64)

// ============================================================
// TRACE service — structured GPU trace events
// ============================================================

/// Emit a structured trace event from GPU to host.
///
/// Unlike `SERVICE_PRINT` (plain text), trace events carry structured
/// metadata: thread/block coordinates, severity level, and a GPU
/// timestamp. The host can collect, sort, and filter events.
pub const SERVICE_TRACE: u32 = 13;

/// Emit a GPU assertion failure diagnostic to host (before trap).
///
/// Similar to `SERVICE_PANIC` but includes the assertion expression
/// and is designed for `gpu_assert!()` macro output.
pub const SERVICE_ASSERT: u32 = 14;

// ============================================================
// TRACE service payload layout (lane 0)
// ============================================================
//
// Request:
//   Slot 0: metadata (u64)
//     - Bits 15..0:  threadIdx.x (u16)
//     - Bits 31..16: blockIdx.x (u16)
//     - Bits 39..32: trace level (u8) — see TRACE_LEVEL_* constants
//     - Bits 47..40: message length (u8, 0..48)
//     - Bits 63..48: warp lane ID (u16)
//   Slot 1: GPU timestamp (u64) — from %clock64 or %globaltimer
//   Slots 2-7: message bytes (up to 48 bytes, UTF-8)
// Response:
//   (CONTROL_READY set — no response data needed, fire-and-forget)

/// Trace level: debug (verbose, lowest priority).
pub const TRACE_LEVEL_DEBUG: u8 = 0;
/// Trace level: info (normal events).
pub const TRACE_LEVEL_INFO: u8 = 1;
/// Trace level: warn (potential issues).
pub const TRACE_LEVEL_WARN: u8 = 2;
/// Trace level: error (failures that don't trap).
pub const TRACE_LEVEL_ERROR: u8 = 3;

/// Maximum trace message length (6 slots × 8 bytes).
pub const TRACE_MAX_MSG_LEN: usize = 48;

/// Encode trace metadata: thread index, block index, level, msg_len, lane ID.
///
/// ```
/// use gpu_protocol::*;
/// let meta = encode_trace_metadata(5, 3, TRACE_LEVEL_INFO, 20, 0);
/// assert_eq!(trace_thread_idx(meta), 5);
/// assert_eq!(trace_block_idx(meta), 3);
/// assert_eq!(trace_level(meta), TRACE_LEVEL_INFO);
/// assert_eq!(trace_msg_len(meta), 20);
/// assert_eq!(trace_lane_id(meta), 0);
/// ```
#[inline(always)]
pub const fn encode_trace_metadata(
    thread_idx: u16,
    block_idx: u16,
    level: u8,
    msg_len: u8,
    lane_id: u16,
) -> u64 {
    (thread_idx as u64)
        | ((block_idx as u64) << 16)
        | ((level as u64) << 32)
        | ((msg_len as u64) << 40)
        | ((lane_id as u64) << 48)
}

/// Decode `threadIdx.x` from trace metadata.
#[inline(always)]
pub const fn trace_thread_idx(meta: u64) -> u16 {
    (meta & 0xFFFF) as u16
}

/// Decode `blockIdx.x` from trace metadata.
#[inline(always)]
pub const fn trace_block_idx(meta: u64) -> u16 {
    ((meta >> 16) & 0xFFFF) as u16
}

/// Decode trace level from trace metadata.
#[inline(always)]
pub const fn trace_level(meta: u64) -> u8 {
    ((meta >> 32) & 0xFF) as u8
}

/// Decode message length from trace metadata.
#[inline(always)]
pub const fn trace_msg_len(meta: u64) -> u8 {
    ((meta >> 40) & 0xFF) as u8
}

/// Decode warp lane ID from trace metadata.
#[inline(always)]
pub const fn trace_lane_id(meta: u64) -> u16 {
    ((meta >> 48) & 0xFFFF) as u16
}

/// Decode all fields from trace metadata in one call.
///
/// Returns `(thread_idx, block_idx, level, msg_len, lane_id)`.
#[inline(always)]
pub const fn decode_trace_metadata(meta: u64) -> (u16, u16, u8, u8, u16) {
    (
        trace_thread_idx(meta),
        trace_block_idx(meta),
        trace_level(meta),
        trace_msg_len(meta),
        trace_lane_id(meta),
    )
}

// ============================================================
// ASSERT service payload layout (lane 0)
// ============================================================
//
// Request:
//   Slot 0: metadata (u64) — same format as PANIC
//     - Bits 15..0:  threadIdx.x (u16)
//     - Bits 31..16: blockIdx.x (u16)
//     - Bits 47..32: message length (u16)
//     - Bits 63..48: reserved
//   Slots 1-7: assertion message bytes (up to 56 bytes, UTF-8)
// Response:
//   (CONTROL_READY set — GPU will trap after sending)

/// Maximum assertion message length (7 slots × 8 bytes).
pub const ASSERT_MAX_MSG_LEN: usize = 56;

// ================================================================
// Command Buffer Protocol — Host→GPU command channel
// ================================================================

/// Command buffer header size (64 bytes, cache-line aligned).
pub const CMD_BUF_HEADER_SIZE: usize = 64;

/// Command slot size (64 bytes each).
pub const CMD_SLOT_SIZE: usize = 64;

/// Offset of `write_idx` (u64, atomic) in command buffer header.
/// Host increments after writing a command. GPU reads with Acquire.
pub const CMD_OFF_WRITE_IDX: usize = 0;

/// Offset of `read_idx` (u64, atomic) in command buffer header.
/// GPU increments after processing a command. Host reads for backpressure.
pub const CMD_OFF_READ_IDX: usize = 8;

/// Offset of `capacity` (u32) in command buffer header.
pub const CMD_OFF_CAPACITY: usize = 16;

/// Offset of `cmd_type` (u32) within a command slot.
pub const CMD_SLOT_OFF_TYPE: usize = 0;

/// Offset of payload within a command slot (56 bytes available).
pub const CMD_SLOT_OFF_PAYLOAD: usize = 8;

/// Maximum payload size per command slot.
pub const CMD_MAX_PAYLOAD: usize = CMD_SLOT_SIZE - CMD_SLOT_OFF_PAYLOAD;

/// Command type: no-op (for testing).
pub const CMD_NOP: u32 = 0;

/// Command type: execute computation on device buffer.
/// Payload: input_ptr (u64), output_ptr (u64), count (u32), op_code (u32).
pub const CMD_COMPUTE: u32 = 1;

/// Command type: print a message via hostcall.
/// Payload: msg_len (u32) + message bytes (up to 52 bytes).
pub const CMD_PRINT: u32 = 2;

/// Command type: exit the command processing loop.
pub const CMD_EXIT: u32 = 3;

// ================================================================
// Flight Recorder — GPU-side ring buffer for post-mortem trace events
// ================================================================

/// Size of the flight recorder header in bytes.
pub const FR_HEADER_SIZE: usize = 64;

/// Size of each flight recorder event slot in bytes.
pub const FR_SLOT_SIZE: usize = 64;

/// Offset of `write_idx` (u64, atomic) in the flight recorder header.
pub const FR_OFF_WRITE_IDX: usize = 0;

/// Offset of `capacity` (u32) in the flight recorder header.
pub const FR_OFF_CAPACITY: usize = 8;

/// Offset of `flags` (u32) in the flight recorder header.
pub const FR_OFF_FLAGS: usize = 12;

/// Offset of metadata (u64) in a flight recorder event slot.
pub const FR_SLOT_OFF_META: usize = 0;

/// Offset of timestamp (u64) in a flight recorder event slot.
pub const FR_SLOT_OFF_TIMESTAMP: usize = 8;

/// Offset of message bytes in a flight recorder event slot.
pub const FR_SLOT_OFF_MSG: usize = 16;

/// Maximum message length in a flight recorder event.
pub const FR_MAX_MSG_LEN: usize = FR_SLOT_SIZE - FR_SLOT_OFF_MSG;

/// Flag bit: kernel crashed (set by GPU before trap).
pub const FR_FLAG_CRASHED: u32 = 1;
