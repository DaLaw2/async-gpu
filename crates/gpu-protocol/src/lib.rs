//! Hostcall protocol shared definitions.
//!
//! Used by both GPU kernel (nvptx64) and host (x86_64).
//! This crate is `#![no_std]` so it compiles for both targets.

#![no_std]

// ============================================================
// Service IDs
// ============================================================

pub const SERVICE_NOP: u32 = 0;
pub const SERVICE_PRINT: u32 = 1;
pub const SERVICE_WRITE: u32 = 2;
pub const SERVICE_READ: u32 = 3;
pub const SERVICE_OPEN: u32 = 4;
pub const SERVICE_CLOSE: u32 = 5;
pub const SERVICE_MALLOC: u32 = 6;
pub const SERVICE_FREE: u32 = 7;
pub const SERVICE_ABORT: u32 = 0xFF;
pub const SERVICE_STDIN: u32 = 8;
pub const SERVICE_TIME: u32 = 9;
pub const SERVICE_PANIC: u32 = 10;

// ============================================================
// Control bits (PacketHeader.control)
// ============================================================

pub const CONTROL_READY: u32 = 1;
pub const CONTROL_ERROR: u32 = 2;
/// Set by GPU after filling the packet, before pushing to ready stack.
/// Host checks this bit before processing — skip if not set (stale re-visit).
pub const CONTROL_FILLED: u32 = 4;

// ============================================================
// Tagged pointer constants
// ============================================================

/// Null sentinel for tagged pointers (no packet).
pub const NULL_INDEX: u16 = 0xFFFF;

// ============================================================
// Layout sizes
// ============================================================

pub const WARP_SIZE: u32 = 32;
pub const SLOTS_PER_LANE: usize = 8;
pub const BUFFER_HEADER_SIZE: usize = 64;
pub const PACKET_HEADER_SIZE: usize = 32;
/// 32 lanes × 8 slots × 8 bytes = 2048 bytes.
pub const PACKET_PAYLOAD_SIZE: usize = (WARP_SIZE as usize) * SLOTS_PER_LANE * 8;
/// Header (32) + Payload (2048) = 2080, rounded up to next 64-byte alignment = 2112.
pub const PACKET_SIZE: usize = 2112;

// ============================================================
// Buffer header field offsets (from buffer base pointer)
// ============================================================

pub const BUF_OFF_FREE_STACK: usize = 0;    // u64
pub const BUF_OFF_READY_STACK: usize = 8;   // u64
pub const BUF_OFF_DOORBELL: usize = 16;     // u64
pub const BUF_OFF_SHUTDOWN: usize = 24;     // u32
pub const BUF_OFF_NUM_PACKETS: usize = 28;  // u32
pub const BUF_OFF_WARP_SIZE: usize = 32;    // u32

// ============================================================
// Packet field offsets (from packet base pointer)
// ============================================================

pub const PKT_OFF_NEXT: usize = 0;          // u64
pub const PKT_OFF_ACTIVE_MASK: usize = 8;   // u32
pub const PKT_OFF_SERVICE: usize = 12;      // u32
pub const PKT_OFF_CONTROL: usize = 16;      // u32
pub const PKT_OFF_PAYLOAD: usize = PACKET_HEADER_SIZE; // 32

// ============================================================
// Tagged pointer helpers
// ============================================================
//
// Layout:
//   Bits 63..32: ABA tag (monotonically increasing)
//   Bits 31..16: reserved (zero)
//   Bits 15..0:  packet index (0..N-1), or 0xFFFF = NULL

#[inline(always)]
pub const fn tagged_index(tagged: u64) -> u16 {
    (tagged & 0xFFFF) as u16
}

#[inline(always)]
pub const fn tagged_tag(tagged: u64) -> u32 {
    (tagged >> 32) as u32
}

#[inline(always)]
pub const fn make_tagged(tag: u32, index: u16) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

#[inline(always)]
pub const fn null_tagged() -> u64 {
    make_tagged(0, NULL_INDEX)
}

// ============================================================
// Offset calculators
// ============================================================

/// Byte offset of packet `index` from buffer base.
#[inline(always)]
pub const fn packet_offset(index: u16) -> usize {
    BUFFER_HEADER_SIZE + (index as usize) * PACKET_SIZE
}

/// Byte offset of a specific payload slot from the packet base.
/// `lane` is the warp lane (0..31), `slot` is the slot index (0..7).
#[inline(always)]
pub const fn payload_slot_offset(lane: u32, slot: usize) -> usize {
    PKT_OFF_PAYLOAD + (lane as usize) * SLOTS_PER_LANE * 8 + slot * 8
}

/// Total buffer size in bytes for `num_packets` packets.
#[inline(always)]
pub const fn buffer_size(num_packets: u16) -> usize {
    BUFFER_HEADER_SIZE + (num_packets as usize) * PACKET_SIZE
}

// ============================================================
// PRINT service payload layout (lane 0)
// ============================================================
// Slot 0: message length (u64)
// Slots 1-7: message bytes (up to 56 bytes, packed)

pub const PRINT_MAX_MSG_LEN: usize = 56; // 7 slots × 8 bytes

// ============================================================
// FILE I/O service payload layouts (lane 0)
// ============================================================
//
// SERVICE_OPEN payload (request):
//   Slot 0: path length (u64)
//   Slots 1-7: path bytes (up to 56 bytes, packed)
//   Slot 0 high bits encode flags: 0 = read, 1 = write/create, 2 = append
// SERVICE_OPEN response (in payload):
//   Slot 0: file descriptor (u64), or u64::MAX on error
//
// SERVICE_WRITE payload (request):
//   Slot 0: fd (u64)
//   Slot 1: data length (u64)
//   Slots 2-7: data bytes (up to 48 bytes, packed)
// SERVICE_WRITE response:
//   Slot 0: bytes written (u64), or u64::MAX on error
//
// SERVICE_READ payload (request):
//   Slot 0: fd (u64)
//   Slot 1: max bytes to read (u64)
// SERVICE_READ response:
//   Slot 0: bytes read (u64), or u64::MAX on error
//   Slots 1-7: data bytes (up to 56 bytes)
//
// SERVICE_CLOSE payload (request):
//   Slot 0: fd (u64)
// SERVICE_CLOSE response:
//   Slot 0: 0 on success, u64::MAX on error

pub const FILE_MAX_PATH_LEN: usize = 56;  // 7 slots × 8 bytes (same as PRINT)
pub const FILE_MAX_WRITE_LEN: usize = 48; // 6 slots × 8 bytes (slots 2-7)
pub const FILE_MAX_READ_LEN: usize = 56;  // 7 slots × 8 bytes (slots 1-7)
pub const FILE_ERROR_SENTINEL: u64 = u64::MAX;

// Open flags (stored in high 32 bits of slot 0)
pub const FILE_OPEN_READ: u32 = 0;
pub const FILE_OPEN_WRITE_CREATE: u32 = 1;
pub const FILE_OPEN_APPEND: u32 = 2;

// ============================================================
// Error encoding (error-handling.1 design)
// ============================================================
//
// When CONTROL_ERROR is set in the response, payload slot 0 contains:
//   Bits 63..32: reserved (zero)
//   Bits 31..16: raw OS errno (optional, 0 = not provided)
//   Bits 15..0:  error category (ErrorCategory)

/// Error category codes for hostcall error propagation.
/// Maps to a subset of `std::io::ErrorKind`.
pub const ERR_OTHER: u16 = 0;
pub const ERR_NOT_FOUND: u16 = 1;
pub const ERR_PERMISSION_DENIED: u16 = 2;
pub const ERR_ALREADY_EXISTS: u16 = 3;
pub const ERR_INVALID_INPUT: u16 = 4;
pub const ERR_IO_ERROR: u16 = 5;
pub const ERR_TIMED_OUT: u16 = 6;
pub const ERR_WOULD_BLOCK: u16 = 7;
pub const ERR_BROKEN_PIPE: u16 = 8;
pub const ERR_RESOURCE_BUSY: u16 = 9;
pub const ERR_STORAGE_FULL: u16 = 10;
pub const ERR_TOO_MANY_FILES: u16 = 11;
pub const ERR_OUT_OF_MEMORY: u16 = 12;
pub const ERR_INVALID_FD: u16 = 13;
pub const ERR_CONNECTION_REFUSED: u16 = 14;
pub const ERR_IS_A_DIRECTORY: u16 = 15;
pub const ERR_HOST_TIMEOUT: u16 = 16;
pub const ERR_UNSUPPORTED: u16 = 17;

/// Encode an error into payload slot 0 format.
#[inline(always)]
pub const fn encode_error(category: u16, raw_errno: u16) -> u64 {
    ((raw_errno as u64) << 16) | (category as u64)
}

/// Decode error category from payload slot 0.
#[inline(always)]
pub const fn error_category(slot0: u64) -> u16 {
    (slot0 & 0xFFFF) as u16
}

/// Decode raw errno from payload slot 0.
#[inline(always)]
pub const fn error_raw_errno(slot0: u64) -> u16 {
    ((slot0 >> 16) & 0xFFFF) as u16
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

pub const PANIC_MAX_MSG_LEN: usize = 56; // 7 slots × 8 bytes

/// Encode panic metadata into slot 0.
#[inline(always)]
pub const fn encode_panic_metadata(thread_idx: u16, block_idx: u16, msg_len: u16) -> u64 {
    (thread_idx as u64) | ((block_idx as u64) << 16) | ((msg_len as u64) << 32)
}

/// Decode threadIdx.x from panic metadata slot 0.
#[inline(always)]
pub const fn panic_thread_idx(meta: u64) -> u16 {
    (meta & 0xFFFF) as u16
}

/// Decode blockIdx.x from panic metadata slot 0.
#[inline(always)]
pub const fn panic_block_idx(meta: u64) -> u16 {
    ((meta >> 16) & 0xFFFF) as u16
}

/// Decode message length from panic metadata slot 0.
#[inline(always)]
pub const fn panic_msg_len(meta: u64) -> u16 {
    ((meta >> 32) & 0xFFFF) as u16
}

// ============================================================
// GPU-side spin limit
// ============================================================

pub const GPU_MAX_SPIN: u32 = 10_000_000;

// ============================================================
// STDIN service payload layout (lane 0)
// ============================================================
// Request:
//   Slot 0: max bytes to read (u64)
// Response:
//   Slot 0: bytes read (u64), or u64::MAX on error/EOF
//   Slots 1-7: data bytes (up to 56 bytes)
pub const STDIN_MAX_READ_LEN: usize = 56; // 7 slots × 8 bytes

// ============================================================
// TIME service payload layout (lane 0)
// ============================================================
// Request:
//   (no payload needed)
// Response:
//   Slot 0: seconds since Unix epoch (u64)
//   Slot 1: nanoseconds within second (u64)
