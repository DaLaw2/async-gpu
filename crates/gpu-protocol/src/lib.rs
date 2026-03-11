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

// ============================================================
// Control bits (PacketHeader.control)
// ============================================================

pub const CONTROL_READY: u32 = 1;
pub const CONTROL_ERROR: u32 = 2;

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
// GPU-side spin limit
// ============================================================

pub const GPU_MAX_SPIN: u32 = 10_000_000;
