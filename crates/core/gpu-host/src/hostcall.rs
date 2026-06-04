//! Host-side hostcall listener and buffer management.
//!
//! This module implements the host half of the GPU-host hostcall protocol.
//! It allocates a pinned, device-mapped packet buffer and runs a listener
//! thread that polls for GPU requests and dispatches them to service handlers.
//!
//! # Hostcall protocol overview
//!
//! The GPU and host communicate through a shared memory region containing a
//! pool of fixed-size packet slots. Each packet has a control word, a service
//! ID, and a 56-byte payload. The protocol flow is:
//!
//! 1. **GPU** acquires a free packet slot from the lock-free stack
//! 2. **GPU** writes the service ID + payload, then sets the doorbell flag
//! 3. **Host listener** polls doorbell flags in a tight loop, detects the request
//! 4. **Host** dispatches to the appropriate service handler (print, file I/O,
//!    TCP networking, stdin, etc.)
//! 5. **Host** writes the response payload and clears the doorbell (ACK)
//! 6. **GPU** reads the response and releases the packet back to the free stack
//!
//! For bulk data exceeding 56 bytes, a separate sideband buffer provides a
//! bump-allocated scratch region shared between GPU and host.
//!
//! # Sharding
//!
//! The buffer can be sharded across CUDA blocks (`new_sharded`) so that each
//! block uses `blockIdx.x % num_shards`, reducing contention on the free stack.
//!
//! # FdResource model
//!
//! File descriptors returned to the GPU live in a unified fd table that holds
//! three resource types: [`std::fs::File`], [`std::net::TcpStream`], and
//! [`std::net::TcpListener`]. The GPU uses the same fd namespace for all I/O
//! operations (read, write, close) regardless of the underlying resource type.
//! File handles persist across kernel launches within the same [`HostcallSession`].
//!
//! # Key types
//!
//! - [`HostcallBuffer`] — Pinned shared-memory packet pool with host + device pointers
//! - [`HostcallSession`] — Persistent listener that survives across kernel launches
//! - [`Pipeline`] — Multi-stage kernel pipeline with automatic packet reinitialization
//! - [`CommandBuffer`] — Host-to-GPU command ring buffer
//! - [`FlightRecorder`] — Mapped-memory ring buffer for post-mortem GPU tracing
//! - [`HostcallError`] — Error type for buffer allocation failures

use cudarc::driver::sys::{self, lib as cuda_lib};
use gpu_protocol::*;
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;

// ================================================================
// Unified fd resource table (files + TCP sockets)
// ================================================================

/// A resource held in the fd table — either a file, TCP stream, or TCP listener.
enum FdResource {
    /// An open file handle.
    File(File),
    /// A connected TCP stream.
    TcpStream(TcpStream),
    /// A bound TCP listener.
    TcpListener(TcpListener),
}

// ================================================================
// Stdin abstraction (host-scaling.2 Phase A)
// ================================================================

/// Trait for providing stdin data to the listener.
/// Implementations must be `Send` to allow I/O thread offloading.
pub trait StdinSource: Send {
    /// Read up to `buf.len()` bytes of stdin input. Returns bytes written.
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize;
}

/// Real stdin — reads from `std::io::stdin()`. Blocks until input available.
pub struct RealStdin;

impl StdinSource for RealStdin {
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => 0,
            Ok(n) => {
                let bytes = line.as_bytes();
                let copy = n.min(buf.len());
                buf[..copy].copy_from_slice(&bytes[..copy]);
                copy
            }
        }
    }
}

/// Canned stdin — returns pre-loaded data once, then EOF on subsequent reads.
pub struct CannedStdin {
    data: Vec<u8>,
    consumed: bool,
}

impl CannedStdin {
    /// Create a `CannedStdin` that returns `data` on first read, then EOF.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            consumed: false,
        }
    }
}

impl StdinSource for CannedStdin {
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize {
        if self.consumed {
            return 0;
        }
        self.consumed = true;
        let copy = self.data.len().min(buf.len());
        buf[..copy].copy_from_slice(&self.data[..copy]);
        copy
    }
}

/// Request sent from listener thread to I/O thread for blocking operations.
struct IoRequest {
    pkt_idx: u16,
    service: u32,
}

/// Map a std::io::Error to an error category code for hostcall error propagation.
fn io_error_to_category(e: &std::io::Error) -> u16 {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound => ERR_NOT_FOUND,
        ErrorKind::PermissionDenied => ERR_PERMISSION_DENIED,
        ErrorKind::AlreadyExists => ERR_ALREADY_EXISTS,
        ErrorKind::InvalidInput => ERR_INVALID_INPUT,
        ErrorKind::TimedOut => ERR_TIMED_OUT,
        ErrorKind::WouldBlock => ERR_WOULD_BLOCK,
        ErrorKind::BrokenPipe => ERR_BROKEN_PIPE,
        ErrorKind::OutOfMemory => ERR_OUT_OF_MEMORY,
        ErrorKind::Unsupported => ERR_UNSUPPORTED,
        ErrorKind::ConnectionRefused => ERR_CONNECTION_REFUSED,
        ErrorKind::ConnectionReset => ERR_CONNECTION_RESET,
        ErrorKind::AddrInUse => ERR_ADDR_IN_USE,
        ErrorKind::AddrNotAvailable => ERR_ADDR_NOT_AVAILABLE,
        ErrorKind::NotConnected => ERR_NOT_CONNECTED,
        _ => ERR_OTHER,
    }
}

/// Encode an io::Error into the hostcall error format and write it to payload slot 0.
/// Returns `true` to signal CONTROL_ERROR should be set.
unsafe fn write_error_response(payload: *mut u8, e: &std::io::Error) -> bool {
    let category = io_error_to_category(e);
    let raw_errno = e.raw_os_error().unwrap_or(0) as u16;
    // SAFETY: Caller guarantees payload points to slot 0 within a valid packet's
    // payload region (PKT_OFF_PAYLOAD offset from a valid packet pointer).
    // Volatile write ensures the GPU observes the error value.
    std::ptr::write_volatile(payload as *mut u64, encode_error(category, raw_errno));
    true
}

/// Errors that can occur during hostcall buffer allocation.
#[derive(Debug)]
pub enum HostcallError {
    /// `cuMemHostAlloc` failed — could not allocate pinned GPU-visible memory.
    CudaAlloc(sys::CUresult),
    /// `cuMemHostGetDevicePointer_v2` failed — could not obtain device-side pointer.
    CudaGetDevPtr(sys::CUresult),
}

impl fmt::Display for HostcallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaAlloc(r) => write!(f, "cuMemHostAlloc failed: {r:?}"),
            Self::CudaGetDevPtr(r) => {
                write!(f, "cuMemHostGetDevicePointer_v2 failed: {r:?}")
            }
        }
    }
}

impl std::error::Error for HostcallError {}

/// Hostcall buffer handle with both host and device pointers.
pub struct HostcallBuffer {
    /// Host-side pointer to the pinned hostcall buffer.
    pub host_ptr: *mut u8,
    /// Device-side pointer to the hostcall buffer (for kernel launch args).
    pub dev_ptr: sys::CUdeviceptr,
    /// Total size of the hostcall buffer in bytes.
    pub size: usize,
    /// Number of packet slots in the buffer.
    pub num_packets: u16,
    /// Number of shards (0 = legacy unsharded mode).
    pub num_shards: u32,
    /// Packets assigned to each shard (only meaningful when num_shards > 0).
    pub pkts_per_shard: u32,
    /// Host-side pointer to the sideband buffer for bulk data transfer (>56 bytes).
    pub sideband_host_ptr: *mut u8,
    /// Device-side pointer to the sideband buffer (for kernel launch args).
    pub sideband_dev_ptr: sys::CUdeviceptr,
    /// Total size of the sideband buffer in bytes.
    pub sideband_size: usize,
}

// SAFETY: The buffer is pinned memory shared between host and GPU.
// We ensure single-writer access via the protocol (GPU writes packet,
// host reads; host writes control, GPU reads).
unsafe impl Send for HostcallBuffer {}
unsafe impl Sync for HostcallBuffer {}

impl HostcallBuffer {
    /// Allocate and initialize a hostcall buffer with `num_packets` packet slots
    /// and a default-sized sideband buffer (1MB) for bulk data transfer.
    /// Legacy (unsharded) mode.
    ///
    /// Uses cuMemHostAlloc with DEVICEMAP|PORTABLE flags for GPU-CPU shared access.
    pub fn new(num_packets: u16) -> Result<Self, HostcallError> {
        Self::new_with_sideband(num_packets, DEFAULT_SIDEBAND_SIZE)
    }

    /// Allocate a legacy (unsharded) hostcall buffer with custom sideband size.
    pub fn new_with_sideband(
        num_packets: u16,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        Self::alloc_internal(num_packets, 0, 0, sideband_data_size)
    }

    /// Allocate a sharded hostcall buffer with `num_shards` shards.
    ///
    /// Each shard gets `pkts_per_shard` packets. Total packets = num_shards * pkts_per_shard.
    /// Each CUDA block uses shard `blockIdx.x % num_shards`.
    pub fn new_sharded(num_shards: u32, pkts_per_shard: u32) -> Result<Self, HostcallError> {
        Self::new_sharded_with_sideband(num_shards, pkts_per_shard, DEFAULT_SIDEBAND_SIZE)
    }

    /// Allocate a sharded hostcall buffer with custom sideband size.
    pub fn new_sharded_with_sideband(
        num_shards: u32,
        pkts_per_shard: u32,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        let total_packets = num_shards * pkts_per_shard;
        assert!(total_packets <= 0xFFFE, "too many packets (max 65534)");
        Self::alloc_internal(
            total_packets as u16,
            num_shards,
            pkts_per_shard,
            sideband_data_size,
        )
    }

    /// Internal allocation — handles both legacy and sharded modes.
    fn alloc_internal(
        num_packets: u16,
        num_shards: u32,
        pkts_per_shard: u32,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        let size = if num_shards == 0 {
            buffer_size(num_packets)
        } else {
            buffer_size_sharded(num_packets, num_shards)
        };
        // SAFETY: cuda_lib() returns the lazily-loaded CUDA driver function table.
        let cu = unsafe { cuda_lib() };
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;

        // Allocate hostcall buffer (pinned, device-mapped)
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc writes a valid pointer to host_ptr on success.
        // The allocation is `size` bytes with DEVICEMAP|PORTABLE flags.
        let result = unsafe { cu.cuMemHostAlloc(&mut host_ptr, size, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag, so the driver can
        // provide a GPU-visible address.
        let result = unsafe { cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: host_ptr was allocated above; freeing on error path.
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // SAFETY: host_ptr is valid for `size` bytes. No kernel is running yet.
        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        // Allocate sideband buffer for bulk data transfer
        let sideband_total = SIDEBAND_HEADER_SIZE + sideband_data_size;
        let mut sb_host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: Same CUDA alloc pattern as the hostcall buffer above.
        let result = unsafe { cu.cuMemHostAlloc(&mut sb_host_ptr, sideband_total, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut sb_dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: sb_host_ptr was allocated with DEVICEMAP flag.
        let result = unsafe { cu.cuMemHostGetDevicePointer_v2(&mut sb_dev_ptr, sb_host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: Both pointers were allocated above; freeing on error path.
            unsafe {
                cu.cuMemFreeHost(sb_host_ptr);
                cu.cuMemFreeHost(host_ptr);
            }
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // SAFETY: sb_host_ptr is valid for `sideband_total` bytes. No kernel running.
        // SIDEBAND_OFF_CAPACITY is within the sideband header region.
        unsafe {
            std::ptr::write_bytes(sb_host_ptr as *mut u8, 0, sideband_total);
            let cap_ptr = (sb_host_ptr as *mut u8).add(SIDEBAND_OFF_CAPACITY) as *mut u64;
            std::ptr::write_volatile(cap_ptr, sideband_data_size as u64);
        }

        let buf = Self {
            host_ptr: host_ptr as *mut u8,
            dev_ptr,
            size,
            num_packets,
            num_shards,
            pkts_per_shard,
            sideband_host_ptr: sb_host_ptr as *mut u8,
            sideband_dev_ptr: sb_dev_ptr,
            sideband_size: sideband_total,
        };

        buf.init();

        Ok(buf)
    }

    /// Initialize the hostcall buffer: set up free stack, ready stack, etc.
    /// Handles both legacy (num_shards == 0) and sharded modes.
    fn init(&self) {
        let base = self.host_ptr;
        // SAFETY: All pointer arithmetic below stays within the cuMemHostAlloc region
        // of `self.size` bytes. The buffer layout offsets (BUF_OFF_*, PKT_OFF_*) are
        // computed by gpu_protocol to fit within buffer_size(num_packets). write_volatile
        // is used because the GPU may read this memory once a kernel is launched.
        // This is called during construction, before any kernel launch.
        unsafe {
            // Common header fields
            let doorbell = base.add(BUF_OFF_DOORBELL) as *mut u64;
            let shutdown = base.add(BUF_OFF_SHUTDOWN) as *mut u32;
            let num_packets_field = base.add(BUF_OFF_NUM_PACKETS) as *mut u32;
            let warp_size_field = base.add(BUF_OFF_WARP_SIZE) as *mut u32;
            let num_shards_field = base.add(BUF_OFF_NUM_SHARDS) as *mut u32;
            let pkts_per_shard_field = base.add(BUF_OFF_PKTS_PER_SHARD) as *mut u32;
            let shard_array_off_field = base.add(BUF_OFF_SHARD_ARRAY_OFF) as *mut u32;

            std::ptr::write_volatile(doorbell, 0u64);
            std::ptr::write_volatile(shutdown, 0u32);
            std::ptr::write_volatile(num_packets_field, self.num_packets as u32);
            std::ptr::write_volatile(warp_size_field, WARP_SIZE);
            std::ptr::write_volatile(num_shards_field, self.num_shards);
            std::ptr::write_volatile(pkts_per_shard_field, self.pkts_per_shard);
            std::ptr::write_volatile(shard_array_off_field, BUFFER_HEADER_SIZE as u32);

            if self.num_shards == 0 {
                // Legacy mode: single global free/ready stack
                let free_stack = base.add(BUF_OFF_FREE_STACK) as *mut u64;
                let ready_stack = base.add(BUF_OFF_READY_STACK) as *mut u64;
                std::ptr::write_volatile(ready_stack, null_tagged());

                for i in 0..self.num_packets {
                    let pkt = base.add(packet_offset(i));
                    let next_tagged = if i + 1 < self.num_packets {
                        make_tagged(0, i + 1)
                    } else {
                        null_tagged()
                    };
                    std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next_tagged);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                }

                std::ptr::write_volatile(free_stack, make_tagged(0, 0));
            } else {
                // Sharded mode: per-shard free/ready stacks
                let shard_array_off = BUFFER_HEADER_SIZE;

                // Global stacks empty (not used in sharded mode)
                std::ptr::write_volatile(base.add(BUF_OFF_FREE_STACK) as *mut u64, null_tagged());
                std::ptr::write_volatile(base.add(BUF_OFF_READY_STACK) as *mut u64, null_tagged());

                for s in 0..self.num_shards {
                    let base_pkt = s * self.pkts_per_shard;
                    let entry_off = shard_entry_offset(shard_array_off, s);

                    // Chain packets within this shard
                    for i in 0..self.pkts_per_shard {
                        let pkt_idx = (base_pkt + i) as u16;
                        let pkt = base.add(packet_offset_sharded(
                            pkt_idx,
                            shard_array_off,
                            self.num_shards,
                        ));
                        let next_tagged = if i + 1 < self.pkts_per_shard {
                            make_tagged(0, (base_pkt + i + 1) as u16)
                        } else {
                            null_tagged()
                        };
                        std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next_tagged);
                        std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                    }

                    // Set shard free_stack head
                    std::ptr::write_volatile(
                        base.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64,
                        make_tagged(0, base_pkt as u16),
                    );
                    // Set shard ready_stack to empty
                    std::ptr::write_volatile(
                        base.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64,
                        null_tagged(),
                    );
                }
            }
        }
    }

    /// Reinitialize packet pool for reuse between kernel launches.
    ///
    /// Resets free stacks (all packets available), ready stacks (empty),
    /// and all packet control flags. Resets sideband bump allocator.
    ///
    /// SAFETY: Must only be called after `cuCtxSynchronize()` (GPU is idle).
    /// The listener thread may still be running — this is safe because the
    /// ready stacks are set to NULL (listener will see empty stacks) and
    /// free stacks are only popped by GPU (which is idle).
    pub fn reinit_packets(&self) {
        let base = self.host_ptr;
        // SAFETY: Must only be called after cuCtxSynchronize() — no GPU kernel is
        // accessing the buffer. All pointer arithmetic stays within the allocated
        // region. write_volatile is used because the listener thread may be polling
        // ready stacks concurrently (setting them to NULL is safe — listener sees
        // empty stacks). Free stacks are only popped by GPU code (which is idle).
        unsafe {
            // Reset shutdown flag (may have been set by previous session use)
            std::ptr::write_volatile(base.add(BUF_OFF_SHUTDOWN) as *mut u32, 0);

            if self.num_shards == 0 {
                // Legacy mode
                let free_stack = base.add(BUF_OFF_FREE_STACK) as *mut u64;
                let ready_stack = base.add(BUF_OFF_READY_STACK) as *mut u64;

                // Clear ready stack
                std::ptr::write_volatile(ready_stack, null_tagged());

                // Rebuild free stack chain
                for i in 0..self.num_packets {
                    let pkt = base.add(packet_offset(i));
                    let next_tagged = if i + 1 < self.num_packets {
                        make_tagged(0, i + 1)
                    } else {
                        null_tagged()
                    };
                    std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next_tagged);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                }
                std::ptr::write_volatile(free_stack, make_tagged(0, 0));
            } else {
                // Sharded mode
                let shard_array_off = BUFFER_HEADER_SIZE;

                std::ptr::write_volatile(base.add(BUF_OFF_FREE_STACK) as *mut u64, null_tagged());
                std::ptr::write_volatile(base.add(BUF_OFF_READY_STACK) as *mut u64, null_tagged());

                for s in 0..self.num_shards {
                    let base_pkt = s * self.pkts_per_shard;
                    let entry_off = shard_entry_offset(shard_array_off, s);

                    for i in 0..self.pkts_per_shard {
                        let pkt_idx = (base_pkt + i) as u16;
                        let pkt = base.add(packet_offset_sharded(
                            pkt_idx,
                            shard_array_off,
                            self.num_shards,
                        ));
                        let next_tagged = if i + 1 < self.pkts_per_shard {
                            make_tagged(0, (base_pkt + i + 1) as u16)
                        } else {
                            null_tagged()
                        };
                        std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next_tagged);
                        std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                    }

                    std::ptr::write_volatile(
                        base.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64,
                        make_tagged(0, base_pkt as u16),
                    );
                    std::ptr::write_volatile(
                        base.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64,
                        null_tagged(),
                    );
                }
            }

            // Reset sideband bump allocator
            if !self.sideband_host_ptr.is_null() {
                std::ptr::write_volatile(
                    self.sideband_host_ptr.add(SIDEBAND_OFF_ALLOC) as *mut u64,
                    0u64,
                );
            }
        }
    }

    /// Get a reference to the doorbell as an AtomicU64.
    fn doorbell(&self) -> &AtomicU64 {
        // SAFETY: BUF_OFF_DOORBELL is within the buffer header. The pointer is
        // 8-byte aligned (buffer is page-aligned from cuMemHostAlloc). The
        // resulting reference is valid for the lifetime of HostcallBuffer.
        unsafe { &*(self.host_ptr.add(BUF_OFF_DOORBELL) as *const AtomicU64) }
    }

    /// Get a reference to the ready_stack as an AtomicU64.
    fn ready_stack(&self) -> &AtomicU64 {
        // SAFETY: Same as doorbell() — BUF_OFF_READY_STACK is within the header,
        // 8-byte aligned, valid for the buffer's lifetime.
        unsafe { &*(self.host_ptr.add(BUF_OFF_READY_STACK) as *const AtomicU64) }
    }

    /// Get a reference to the shutdown flag as an AtomicU32.
    fn shutdown(&self) -> &AtomicU32 {
        // SAFETY: BUF_OFF_SHUTDOWN is within the buffer header, 4-byte aligned,
        // valid for the buffer's lifetime.
        unsafe { &*(self.host_ptr.add(BUF_OFF_SHUTDOWN) as *const AtomicU32) }
    }

    /// Get pointer to a packet by index. Handles both legacy and sharded layouts.
    fn packet_ptr(&self, index: u16) -> *mut u8 {
        // SAFETY: packet_offset / packet_offset_sharded compute offsets that stay
        // within the allocated buffer (index must be < num_packets, which is
        // guaranteed by the lock-free stack protocol — only indices that were
        // originally placed in the free stack can appear in the ready stack).
        unsafe {
            if self.num_shards == 0 {
                self.host_ptr.add(packet_offset(index))
            } else {
                self.host_ptr.add(packet_offset_sharded(
                    index,
                    BUFFER_HEADER_SIZE,
                    self.num_shards,
                ))
            }
        }
    }

    /// Signal shutdown to the GPU.
    pub fn signal_shutdown(&self) {
        self.shutdown().store(1, Ordering::Release);
    }

    /// Run the host listener loop with real stdin. Blocks until shutdown is signaled.
    ///
    /// `on_print` is called for each PRINT service request with the message bytes.
    pub fn listen<F>(&self, on_print: F)
    where
        F: FnMut(&[u8]),
    {
        self.listen_unified(on_print, RealStdin);
    }

    /// Unified listener with I/O thread separation (host-scaling.3, ADR-6).
    ///
    /// Fast services (NOP, PRINT, TIME, PANIC) are handled inline on the listener thread.
    /// Blocking services (FILE I/O, STDIN) are offloaded to a dedicated I/O thread via channel.
    pub fn listen_unified<F, S>(&self, mut on_print: F, stdin: S)
    where
        F: FnMut(&[u8]),
        S: StdinSource,
    {
        let (io_tx, io_rx) = mpsc::channel::<IoRequest>();

        std::thread::scope(|scope| {
            // Spawn I/O thread for blocking operations (FILE, STDIN)
            scope.spawn(|| {
                self.io_thread_loop(io_rx, stdin);
            });

            let mut last_doorbell: u64 = 0;
            let mut idle_spins: u32 = 0;

            // Adaptive polling: spin fast for SPIN_PHASE_LIMIT iterations,
            // then switch to sleeping SLEEP_DURATION between polls.
            const SPIN_PHASE_LIMIT: u32 = 1_000; // ~10µs at ~100ns/spin
            const SLEEP_DURATION: std::time::Duration = std::time::Duration::from_micros(100);

            loop {
                if self.shutdown().load(Ordering::Acquire) != 0 {
                    break;
                }

                let current_doorbell = self.doorbell().load(Ordering::Acquire);
                if current_doorbell == last_doorbell {
                    idle_spins += 1;
                    if idle_spins <= SPIN_PHASE_LIMIT {
                        std::hint::spin_loop();
                    } else {
                        std::thread::sleep(SLEEP_DURATION);
                    }
                    continue;
                }

                last_doorbell = current_doorbell;
                idle_spins = 0;

                // Drain all ready stacks (1 global or N shard stacks)
                let stacks_to_scan = if self.num_shards == 0 {
                    1
                } else {
                    self.num_shards
                };
                for s in 0..stacks_to_scan {
                    // Atomically drain the ready stack: swap head with NULL to
                    // claim all enqueued packets. AcqRel ordering ensures we see
                    // all writes the GPU made before pushing to the ready stack.
                    let ready_head = if self.num_shards == 0 {
                        self.ready_stack().swap(null_tagged(), Ordering::AcqRel)
                    } else {
                        let entry_off = shard_entry_offset(BUFFER_HEADER_SIZE, s);
                        // SAFETY: entry_off + SHARD_OFF_READY_STACK is within the
                        // shard array region of the buffer, 8-byte aligned.
                        let shard_ready = unsafe {
                            &*(self.host_ptr.add(entry_off + SHARD_OFF_READY_STACK)
                                as *const AtomicU64)
                        };
                        shard_ready.swap(null_tagged(), Ordering::AcqRel)
                    };
                    if tagged_index(ready_head) == NULL_INDEX {
                        continue;
                    }

                    let mut current = ready_head;
                    while tagged_index(current) != NULL_INDEX {
                        let idx = tagged_index(current);
                        let pkt = self.packet_ptr(idx);

                        // SAFETY: pkt points to a valid packet slot (index came from
                        // the ready stack, which only contains indices < num_packets).
                        // All read_volatile/write calls target offsets within the
                        // packet's fixed-size region (PKT_OFF_NEXT, PKT_OFF_CONTROL,
                        // PKT_OFF_SERVICE are all < PACKET_SIZE).
                        unsafe {
                            let next = std::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);

                            let control = &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                            let ctrl = control.load(Ordering::Acquire);
                            if ctrl & CONTROL_FILLED == 0 {
                                current = next;
                                continue;
                            }

                            let service =
                                std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32);

                            match service {
                                // Fast path — handle inline, set CONTROL_READY immediately
                                SERVICE_NOP => {
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                SERVICE_PRINT => {
                                    self.handle_print(pkt, &mut on_print);
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                SERVICE_TIME => {
                                    let has_error = self.handle_time(pkt);
                                    let flags = if has_error {
                                        CONTROL_READY | CONTROL_ERROR
                                    } else {
                                        CONTROL_READY
                                    };
                                    control.store(flags, Ordering::Release);
                                }
                                SERVICE_PANIC => {
                                    self.handle_panic(pkt);
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                SERVICE_TRACE => {
                                    self.handle_trace(pkt);
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                SERVICE_ASSERT => {
                                    self.handle_assert(pkt);
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                SERVICE_BULK_PRINT => {
                                    self.handle_bulk_print(pkt, &mut on_print);
                                    control.store(CONTROL_READY, Ordering::Release);
                                }
                                // Slow path — offload to I/O thread
                                SERVICE_OPEN
                                | SERVICE_WRITE
                                | SERVICE_READ
                                | SERVICE_CLOSE
                                | SERVICE_STDIN
                                | SERVICE_BULK_WRITE
                                | SERVICE_BULK_READ
                                | SERVICE_TCP_CONNECT
                                | SERVICE_TCP_WRITE
                                | SERVICE_TCP_READ
                                | SERVICE_TCP_CLOSE
                                | SERVICE_TCP_BIND
                                | SERVICE_TCP_ACCEPT
                                | SERVICE_TCP_BULK_WRITE
                                | SERVICE_TCP_BULK_READ => {
                                    let _ = io_tx.send(IoRequest {
                                        pkt_idx: idx,
                                        service,
                                    });
                                }
                                _ => {
                                    control.store(CONTROL_READY | CONTROL_ERROR, Ordering::Release);
                                }
                            }

                            current = next;
                        }
                    }
                }
            }

            // Drop sender to signal I/O thread to exit
            drop(io_tx);
            // I/O thread joins automatically when scope exits
        });
    }

    /// I/O thread loop — processes blocking FILE and STDIN operations.
    ///
    /// Runs until the channel sender is dropped (listener shutdown).
    fn io_thread_loop<S: StdinSource>(&self, rx: mpsc::Receiver<IoRequest>, mut stdin: S) {
        let mut fd_table: HashMap<u64, FdResource> = HashMap::new();
        let mut next_fd: u64 = 1; // fd 0 is reserved

        while let Ok(req) = rx.recv() {
            let pkt = self.packet_ptr(req.pkt_idx);
            // SAFETY: pkt_idx came from the listener's ready stack traversal, so
            // it is a valid packet index. All service handlers read/write within
            // the packet's payload region (offsets < PACKET_SIZE).
            let has_error = unsafe {
                match req.service {
                    SERVICE_OPEN => self.handle_open(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_WRITE => self.handle_write(pkt, &mut fd_table),
                    SERVICE_READ => self.handle_read(pkt, &mut fd_table),
                    SERVICE_CLOSE => self.handle_close(pkt, &mut fd_table),
                    SERVICE_STDIN => self.handle_stdin_from_source(pkt, &mut stdin),
                    SERVICE_BULK_WRITE => self.handle_bulk_write(pkt, &mut fd_table),
                    SERVICE_BULK_READ => self.handle_bulk_read(pkt, &mut fd_table),
                    // TCP services
                    SERVICE_TCP_CONNECT => {
                        self.handle_tcp_connect(pkt, &mut fd_table, &mut next_fd)
                    }
                    SERVICE_TCP_WRITE => self.handle_tcp_write(pkt, &mut fd_table),
                    SERVICE_TCP_READ => self.handle_tcp_read(pkt, &mut fd_table),
                    SERVICE_TCP_CLOSE => self.handle_tcp_close(pkt, &mut fd_table),
                    SERVICE_TCP_BIND => self.handle_tcp_bind(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_TCP_ACCEPT => self.handle_tcp_accept(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_TCP_BULK_WRITE => self.handle_tcp_bulk_write(pkt, &mut fd_table),
                    SERVICE_TCP_BULK_READ => self.handle_tcp_bulk_read(pkt, &mut fd_table),
                    _ => true,
                }
            };

            // SAFETY: PKT_OFF_CONTROL is within the packet, 4-byte aligned.
            let control = unsafe { &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32) };
            let flags = if has_error {
                CONTROL_READY | CONTROL_ERROR
            } else {
                CONTROL_READY
            };
            control.store(flags, Ordering::Release);
        }
    }

    /// Handle a PRINT service request.
    ///
    /// Reads message from lane 0's payload slots and calls the callback.
    unsafe fn handle_print<F>(&self, pkt: *mut u8, on_print: &mut F)
    where
        F: FnMut(&[u8]),
    {
        // SAFETY (applies to all service handlers): pkt points to a valid packet
        // obtained via packet_ptr(). PKT_OFF_PAYLOAD is within the packet. All
        // read_volatile/write_volatile target payload slots 0-7 which occupy bytes
        // [PKT_OFF_PAYLOAD .. PKT_OFF_PAYLOAD + 64) within the packet — well within
        // the packet's total size. Volatile access is required because the GPU wrote
        // these values and we must observe them.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Slot 0 = message length (u64)
        let msg_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let msg_len = msg_len.min(PRINT_MAX_MSG_LEN);

        // Slots 1-7 = message bytes (up to 56 bytes)
        let msg_ptr = payload.add(8); // skip slot 0
        let mut msg_buf = [0u8; PRINT_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        // Read thread/block metadata from payload+64 (lane 1 area)
        let block_idx = std::ptr::read_volatile(payload.add(64) as *const u32);
        let thread_idx = std::ptr::read_volatile(payload.add(68) as *const u32);

        // Format: [B{block}.T{thread}] message
        let prefix = format!("[B{block_idx}.T{thread_idx}] ");
        let mut full_msg = Vec::with_capacity(prefix.len() + msg_len);
        full_msg.extend_from_slice(prefix.as_bytes());
        full_msg.extend_from_slice(&msg_buf[..msg_len]);

        on_print(&full_msg);
    }

    /// Handle SERVICE_BULK_PRINT: flush a buffer of length-prefixed print messages.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: sideband_offset — offset in sideband data region
    ///   Slot 1: data_len — total bytes of length-prefixed messages
    ///   Slot 2: block_idx (high 32) | thread_idx (low 32)
    ///
    /// Message format in sideband: `[u16 len][len bytes data]...`
    unsafe fn handle_bulk_print<F>(&self, pkt: *mut u8, on_print: &mut F)
    where
        F: FnMut(&[u8]),
    {
        // SAFETY: Same payload access pattern as handle_print. Additionally,
        // sideband_host_ptr + SIDEBAND_DATA_OFFSET + sideband_offset is within
        // the sideband buffer (the GPU's bump allocator ensures offset < capacity).
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let sideband_offset = std::ptr::read_volatile(payload as *const u64) as usize;
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let metadata = std::ptr::read_volatile(payload.add(16) as *const u64);
        let thread_idx = (metadata & 0xFFFF_FFFF) as u32;
        let block_idx = (metadata >> 32) as u32;

        if data_len == 0 || self.sideband_host_ptr.is_null() {
            return;
        }

        // Read messages from sideband
        let data_ptr = self
            .sideband_host_ptr
            .add(SIDEBAND_DATA_OFFSET + sideband_offset);
        let prefix = format!("[B{block_idx}.T{thread_idx}] ");

        let mut pos = 0;
        while pos + 2 <= data_len {
            let msg_len = u16::from_le_bytes([*data_ptr.add(pos), *data_ptr.add(pos + 1)]) as usize;
            if msg_len == 0 || pos + 2 + msg_len > data_len {
                break;
            }

            let mut full_msg = Vec::with_capacity(prefix.len() + msg_len);
            full_msg.extend_from_slice(prefix.as_bytes());
            for i in 0..msg_len {
                full_msg.push(*data_ptr.add(pos + 2 + i));
            }
            on_print(&full_msg);

            pos += 2 + msg_len;
        }
    }

    // ================================================================
    // FILE I/O handlers (gpu-std.3)
    // ================================================================

    /// Handle SERVICE_OPEN: open or create a file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: low 32 bits = path length, high 32 bits = flags
    ///   Slots 1-7: path bytes (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: fd on success, FILE_ERROR_SENTINEL on error
    ///
    /// Returns true if the service itself encountered an error (not a file error).
    unsafe fn handle_open(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same payload access pattern as handle_print — all slot reads/writes
        // are within the 64-byte payload region. Path bytes are clamped to
        // FILE_MAX_PATH_LEN (56 bytes = slots 1-7).
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Read slot 0: path_len (low 32) + flags (high 32)
        let slot0 = std::ptr::read_volatile(payload as *const u64);
        let path_len = (slot0 & 0xFFFF_FFFF) as usize;
        let flags = (slot0 >> 32) as u32;
        let path_len = path_len.min(FILE_MAX_PATH_LEN);

        // Read path bytes from slots 1-7
        let path_ptr = payload.add(8);
        let mut path_buf = [0u8; FILE_MAX_PATH_LEN];
        for i in 0..path_len {
            path_buf[i] = std::ptr::read_volatile(path_ptr.add(i));
        }

        let path_str = match std::str::from_utf8(&path_buf[..path_len]) {
            Ok(s) => s,
            Err(_) => {
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 path");
                eprintln!("  [HOST] FILE OPEN ERROR: invalid UTF-8 path");
                return write_error_response(payload, &e);
            }
        };

        let file_result = match flags {
            FILE_OPEN_READ => File::open(path_str),
            FILE_OPEN_WRITE_CREATE => File::create(path_str),
            FILE_OPEN_APPEND => OpenOptions::new().append(true).create(true).open(path_str),
            _ => {
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid open flags");
                eprintln!("  [HOST] FILE OPEN ERROR: invalid flags={flags}");
                return write_error_response(payload, &e);
            }
        };

        match file_result {
            Ok(file) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::File(file));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] FILE OPEN: \"{path_str}\" flags={flags} -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE OPEN ERROR: \"{path_str}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_WRITE: write data to an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: data length (u64)
    ///   Slots 2-7: data bytes (up to 48 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_write(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let data_len = data_len.min(FILE_MAX_WRITE_LEN);

        // Read data bytes from slots 2-7
        let data_ptr = payload.add(16);
        let mut data_buf = [0u8; FILE_MAX_WRITE_LEN];
        for i in 0..data_len {
            data_buf[i] = std::ptr::read_volatile(data_ptr.add(i));
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] FILE WRITE ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] FILE WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match file.write(&data_buf[..data_len]) {
            Ok(n) => {
                // Flush to ensure data is persisted
                let _ = file.flush();
                println!("  [HOST] FILE WRITE: fd={fd} {n} bytes written");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_READ: read data from an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, FILE_ERROR_SENTINEL on error
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_read(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let max_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_len = max_len.min(FILE_MAX_READ_LEN);

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] FILE READ ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] FILE READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let mut read_buf = [0u8; FILE_MAX_READ_LEN];
        match file.read(&mut read_buf[..max_len]) {
            Ok(n) => {
                println!("  [HOST] FILE READ: fd={fd} {n} bytes read");
                // Write response: slot 0 = bytes read
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                // Slots 1-7 = data bytes
                let dst = payload.add(8);
                for i in 0..n {
                    std::ptr::write_volatile(dst.add(i), read_buf[i]);
                }
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TIME: return wall-clock time.
    ///
    /// Response payload (lane 0):
    ///   Slot 0: seconds since Unix epoch (u64)
    ///   Slot 1: nanoseconds within second (u64)
    unsafe fn handle_time(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => {
                let secs = duration.as_secs();
                let nanos = duration.subsec_nanos() as u64;
                std::ptr::write_volatile(payload as *mut u64, secs);
                std::ptr::write_volatile(payload.add(8) as *mut u64, nanos);
                println!("  [HOST] TIME: epoch_secs={secs} nanos={nanos}");
            }
            Err(_) => {
                std::ptr::write_volatile(payload as *mut u64, FILE_ERROR_SENTINEL);
                std::ptr::write_volatile(payload.add(8) as *mut u64, 0);
            }
        }
        false
    }

    /// Handle SERVICE_PANIC: receive and display a GPU panic message.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: metadata (threadIdx.x, blockIdx.x, msg_len packed)
    ///   Slots 1-7: panic message bytes (up to 56 bytes)
    /// Response: CONTROL_READY (no error — GPU will trap regardless)
    unsafe fn handle_panic(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Decode metadata from slot 0
        let meta = std::ptr::read_volatile(payload as *const u64);
        let thread_idx = panic_thread_idx(meta);
        let block_idx = panic_block_idx(meta);
        let msg_len = panic_msg_len(meta) as usize;
        let msg_len = msg_len.min(PANIC_MAX_MSG_LEN);

        // Read message bytes from slots 1-7
        let msg_ptr = payload.add(8);
        let mut msg_buf = [0u8; PANIC_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid UTF-8>");
        eprintln!("\x1b[1;31m[GPU PANIC]\x1b[0m block={block_idx} thread={thread_idx}: {msg}");

        false // No error — GPU thread will trap after receiving response
    }

    /// Handle SERVICE_TRACE: receive and display a structured trace event.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: metadata (threadIdx:16 | blockIdx:16 | level:8 | msg_len:8 | lane_id:16)
    ///   Slot 1: clock64 timestamp (u64)
    ///   Slots 2-7: message bytes (up to 48 bytes)
    /// Response: CONTROL_READY (no error)
    unsafe fn handle_trace(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Decode metadata from slot 0
        let meta = std::ptr::read_volatile(payload as *const u64);
        let thread_idx = trace_thread_idx(meta);
        let block_idx = trace_block_idx(meta);
        let level = trace_level(meta);
        let msg_len = trace_msg_len(meta) as usize;
        let msg_len = msg_len.min(TRACE_MAX_MSG_LEN);

        // Slot 1: timestamp
        let timestamp = std::ptr::read_volatile(payload.add(8) as *const u64);

        // Slots 2-7: message bytes (starting at offset 16)
        let msg_ptr = payload.add(16);
        let mut msg_buf = [0u8; TRACE_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid UTF-8>");
        let level_str = match level {
            TRACE_LEVEL_DEBUG => "DEBUG",
            TRACE_LEVEL_INFO => "INFO",
            TRACE_LEVEL_WARN => "WARN",
            TRACE_LEVEL_ERROR => "ERROR",
            _ => "UNKNOWN",
        };
        let color = match level {
            TRACE_LEVEL_DEBUG => "\x1b[36m",   // cyan
            TRACE_LEVEL_INFO => "\x1b[32m",    // green
            TRACE_LEVEL_WARN => "\x1b[33m",    // yellow
            TRACE_LEVEL_ERROR => "\x1b[1;31m", // bold red
            _ => "\x1b[0m",
        };
        eprintln!("{color}[GPU {level_str}]\x1b[0m B{block_idx}.T{thread_idx} @{timestamp}: {msg}");

        false
    }

    /// Handle SERVICE_ASSERT: receive and display a GPU assertion failure.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: metadata (threadIdx:16 | blockIdx:16 | msg_len:16 — same as PANIC format)
    ///   Slots 1-7: assertion message bytes (up to 56 bytes)
    /// Response: CONTROL_READY (GPU will trap after receiving response)
    unsafe fn handle_assert(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Decode metadata — uses same format as PANIC
        let meta = std::ptr::read_volatile(payload as *const u64);
        let thread_idx = panic_thread_idx(meta);
        let block_idx = panic_block_idx(meta);
        let msg_len = panic_msg_len(meta) as usize;
        let msg_len = msg_len.min(ASSERT_MAX_MSG_LEN);

        // Read message bytes from slots 1-7
        let msg_ptr = payload.add(8);
        let mut msg_buf = [0u8; ASSERT_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid UTF-8>");
        eprintln!(
            "\x1b[1;31m[GPU ASSERT FAILED]\x1b[0m block={block_idx} thread={thread_idx}: {msg}"
        );

        false // GPU will trap after receiving response
    }

    /// Handle SERVICE_CLOSE: close an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: 0 on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_close(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);

        match fd_table.remove(&fd) {
            Some(resource) => {
                // Resource is dropped here, which closes it
                let kind = match &resource {
                    FdResource::File(_) => "FILE",
                    FdResource::TcpStream(_) => "TCP STREAM",
                    FdResource::TcpListener(_) => "TCP LISTENER",
                };
                drop(resource);
                println!("  [HOST] CLOSE: fd={fd} ({kind}) closed");
                std::ptr::write_volatile(payload as *mut u64, 0);
                false
            }
            None => {
                eprintln!("  [HOST] CLOSE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                true
            }
        }
    }
    /// Handle SERVICE_STDIN using a `StdinSource` abstraction.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read (u64)
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_stdin_from_source<S: StdinSource>(&self, pkt: *mut u8, stdin: &mut S) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let max_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let max_len = max_len.min(STDIN_MAX_READ_LEN);

        let mut buf = [0u8; STDIN_MAX_READ_LEN];
        let n = stdin.read_line_bytes(&mut buf[..max_len]);

        println!("  [HOST] STDIN: {n} bytes");
        std::ptr::write_volatile(payload as *mut u64, n as u64);
        let dst = payload.add(8);
        for i in 0..n {
            std::ptr::write_volatile(dst.add(i), buf[i]);
        }
        false
    }

    /// Handle SERVICE_BULK_WRITE: write sideband data to an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_bulk_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Payload slot reads within packet bounds. Sideband access at
        // SIDEBAND_DATA_OFFSET + sb_offset is bounds-checked against sideband
        // capacity below. from_raw_parts creates a slice within the sideband region.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check against sideband capacity
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + length > capacity {
            eprintln!(
                "  [HOST] BULK WRITE ERROR: offset={sb_offset} + len={length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] BULK WRITE ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] BULK WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let data = std::slice::from_raw_parts(data_ptr, length);

        match file.write_all(data) {
            Ok(()) => {
                let _ = file.flush();
                println!("  [HOST] BULK WRITE: fd={fd} {length} bytes written");
                std::ptr::write_volatile(payload as *mut u64, length as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_BULK_READ: read file data into sideband buffer.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: max_length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_bulk_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + max_length > capacity {
            eprintln!(
                "  [HOST] BULK READ ERROR: offset={sb_offset} + len={max_length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] BULK READ ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] BULK READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let buf = std::slice::from_raw_parts_mut(data_ptr, max_length);

        match file.read(buf) {
            Ok(n) => {
                println!("  [HOST] BULK READ: fd={fd} {n} bytes read");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    // ================================================================
    // TCP networking handlers
    // ================================================================

    /// Extract an address string from packet payload slots 1-7.
    ///
    /// Slot 0 contains `port(u32) | addr_len(u32)` packed as `u64`.
    /// Returns `(address_string, port)` or writes an error response and returns `None`.
    unsafe fn extract_tcp_addr(&self, payload: *mut u8) -> Option<(String, u16)> {
        // SAFETY: payload points to a valid packet payload region. Slot reads
        // (0 and 1-7) are within the 64-byte payload. addr_len is clamped to
        // TCP_MAX_ADDR_LEN (56 bytes).
        let slot0 = std::ptr::read_volatile(payload as *const u64);
        let port = (slot0 & 0xFFFF_FFFF) as u32;
        let addr_len = ((slot0 >> 32) & 0xFFFF_FFFF) as usize;
        let addr_len = addr_len.min(TCP_MAX_ADDR_LEN);

        // Read address bytes from slots 1-7
        let addr_ptr = payload.add(8);
        let mut addr_buf = [0u8; TCP_MAX_ADDR_LEN];
        for i in 0..addr_len {
            addr_buf[i] = std::ptr::read_volatile(addr_ptr.add(i));
        }

        match std::str::from_utf8(&addr_buf[..addr_len]) {
            Ok(s) => Some((s.to_string(), port as u16)),
            Err(_) => None,
        }
    }

    /// Handle SERVICE_TCP_CONNECT: connect to a remote TCP address:port.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: `port(u32) | addr_len(u32)` packed as `u64`
    ///   Slots 1-7: address string (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: socket fd on success, encoded error on failure
    unsafe fn handle_tcp_connect(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let (addr, port) = match self.extract_tcp_addr(payload) {
            Some(v) => v,
            None => {
                eprintln!("  [HOST] TCP CONNECT ERROR: invalid UTF-8 address");
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 addr");
                return write_error_response(payload, &e);
            }
        };

        let socket_addr = format!("{addr}:{port}");
        match TcpStream::connect(&socket_addr) {
            Ok(stream) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpStream(stream));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP CONNECT: \"{socket_addr}\" -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP CONNECT ERROR: \"{socket_addr}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_WRITE: write inline data to a TCP socket (up to 48 bytes).
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: data length (u64)
    ///   Slots 2-7: data bytes (up to 48 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, encoded error on failure
    unsafe fn handle_tcp_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let data_len = data_len.min(TCP_MAX_WRITE_LEN);

        // Read data bytes from slots 2-7
        let data_ptr = payload.add(16);
        let mut data_buf = [0u8; TCP_MAX_WRITE_LEN];
        for i in 0..data_len {
            data_buf[i] = std::ptr::read_volatile(data_ptr.add(i));
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP WRITE ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match stream.write(&data_buf[..data_len]) {
            Ok(n) => {
                let _ = stream.flush();
                println!("  [HOST] TCP WRITE: fd={fd} {n} bytes written");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_READ: read inline data from a TCP socket (up to 56 bytes).
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, encoded error on failure
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_tcp_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let max_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_len = max_len.min(TCP_MAX_READ_LEN);

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP READ ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let mut read_buf = [0u8; TCP_MAX_READ_LEN];
        match stream.read(&mut read_buf[..max_len]) {
            Ok(n) => {
                println!("  [HOST] TCP READ: fd={fd} {n} bytes read");
                // Write response: slot 0 = bytes read
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                // Slots 1-7 = data bytes
                let dst = payload.add(8);
                for i in 0..n {
                    std::ptr::write_volatile(dst.add(i), read_buf[i]);
                }
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_CLOSE: close a TCP socket (stream or listener).
    ///
    /// The fd namespace is shared between files and sockets. This handler
    /// specifically expects a TCP resource. For generic close, use SERVICE_CLOSE.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: 0 on success, encoded error on failure
    unsafe fn handle_tcp_close(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);

        match fd_table.remove(&fd) {
            Some(resource) => {
                let kind = match &resource {
                    FdResource::TcpStream(_) => "TCP STREAM",
                    FdResource::TcpListener(_) => "TCP LISTENER",
                    FdResource::File(_) => "FILE (via TCP_CLOSE)",
                };
                drop(resource);
                println!("  [HOST] TCP CLOSE: fd={fd} ({kind}) closed");
                std::ptr::write_volatile(payload as *mut u64, 0);
                false
            }
            None => {
                eprintln!("  [HOST] TCP CLOSE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                true
            }
        }
    }

    /// Handle SERVICE_TCP_BIND: bind and listen on a local TCP address:port.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: `port(u32) | addr_len(u32)` packed as `u64`
    ///   Slots 1-7: bind address string (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: listener fd on success, encoded error on failure
    unsafe fn handle_tcp_bind(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let (addr, port) = match self.extract_tcp_addr(payload) {
            Some(v) => v,
            None => {
                eprintln!("  [HOST] TCP BIND ERROR: invalid UTF-8 address");
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 addr");
                return write_error_response(payload, &e);
            }
        };

        let socket_addr = format!("{addr}:{port}");
        match TcpListener::bind(&socket_addr) {
            Ok(listener) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpListener(listener));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP BIND: \"{socket_addr}\" -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BIND ERROR: \"{socket_addr}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_ACCEPT: accept a connection on a TCP listener fd.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: listener fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: new stream fd on success, encoded error on failure
    unsafe fn handle_tcp_accept(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let listener_fd = std::ptr::read_volatile(payload as *const u64);

        // We need to borrow the listener immutably, then insert the new stream.
        // accept() takes &self on TcpListener, so no mutable borrow conflict.
        let accept_result = match fd_table.get(&listener_fd) {
            Some(FdResource::TcpListener(l)) => l.accept(),
            Some(_) => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: fd={listener_fd} is not a TCP listener");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: invalid fd={listener_fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match accept_result {
            Ok((stream, peer_addr)) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpStream(stream));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP ACCEPT: listener fd={listener_fd} -> stream fd={fd} from {peer_addr}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: fd={listener_fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_BULK_WRITE: write sideband buffer data to a TCP socket.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, encoded error on failure
    unsafe fn handle_tcp_bulk_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check against sideband capacity
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + length > capacity {
            eprintln!(
                "  [HOST] TCP BULK WRITE ERROR: offset={sb_offset} + len={length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let data = std::slice::from_raw_parts(data_ptr, length);

        match stream.write_all(data) {
            Ok(()) => {
                let _ = stream.flush();
                println!("  [HOST] TCP BULK WRITE: fd={fd} {length} bytes written");
                std::ptr::write_volatile(payload as *mut u64, length as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_BULK_READ: read from a TCP socket into sideband buffer.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: max_length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, encoded error on failure
    unsafe fn handle_tcp_bulk_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + max_length > capacity {
            eprintln!(
                "  [HOST] TCP BULK READ ERROR: offset={sb_offset} + len={max_length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP BULK READ ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP BULK READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let buf = std::slice::from_raw_parts_mut(data_ptr, max_length);

        match stream.read(buf) {
            Ok(n) => {
                println!("  [HOST] TCP BULK READ: fd={fd} {n} bytes read");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BULK READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Listen with both a print callback and a canned stdin provider.
    ///
    /// Convenience wrapper around `listen_unified` with `CannedStdin`.
    pub fn listen_with_stdin<F>(&self, on_print: F, stdin_data: Vec<u8>)
    where
        F: FnMut(&[u8]),
    {
        self.listen_unified(on_print, CannedStdin::new(stdin_data));
    }
}

// ================================================================
// HostcallSession — persistent listener across kernel launches
// ================================================================

/// A persistent hostcall session that keeps the listener thread alive
/// across multiple kernel launches.
///
/// Lifecycle:
/// ```text
/// let session = HostcallSession::start(64)?;
/// // Launch kernel A
/// launch_kernel(session.dev_ptr(), ...);
/// dev.synchronize()?;
/// session.reinit_packets();
/// // Launch kernel B (same hostcall buffer, same listener)
/// launch_kernel(session.dev_ptr(), ...);
/// dev.synchronize()?;
/// session.shutdown();
/// ```
///
/// File handles opened by one kernel persist for subsequent kernels
/// within the same session.
pub struct HostcallSession {
    buf: std::sync::Arc<HostcallBuffer>,
    listener_handle: Option<std::thread::JoinHandle<()>>,
}

impl HostcallSession {
    /// Start a new session with the given packet count.
    /// Spawns listener + I/O threads immediately.
    pub fn start(num_packets: u16) -> Result<Self, HostcallError> {
        let buf = HostcallBuffer::new(num_packets)?;
        let buf = std::sync::Arc::new(buf);
        let buf_listener = std::sync::Arc::clone(&buf);

        let listener_handle = std::thread::spawn(move || {
            buf_listener.listen(|_msg| {
                // Print messages handled by handle_print — on_print is for test capture
            });
        });

        Ok(Self {
            buf,
            listener_handle: Some(listener_handle),
        })
    }

    /// Start a new session with a custom print callback.
    pub fn start_with_print<F>(num_packets: u16, on_print: F) -> Result<Self, HostcallError>
    where
        F: FnMut(&[u8]) + Send + 'static,
    {
        let buf = HostcallBuffer::new(num_packets)?;
        let buf = std::sync::Arc::new(buf);
        let buf_listener = std::sync::Arc::clone(&buf);

        let listener_handle = std::thread::spawn(move || {
            buf_listener.listen(on_print);
        });

        Ok(Self {
            buf,
            listener_handle: Some(listener_handle),
        })
    }

    /// Get the device pointer for kernel launch args.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.buf.dev_ptr
    }

    /// Get the sideband device pointer for bulk transfer args.
    pub fn sideband_dev_ptr(&self) -> sys::CUdeviceptr {
        self.buf.sideband_dev_ptr
    }

    /// Reinitialize packet pool between kernel launches.
    ///
    /// MUST be called after `dev.synchronize()` and before the next kernel launch.
    /// Resets free/ready stacks and sideband allocator.
    /// File handles opened by previous kernels are NOT closed.
    pub fn reinit_packets(&self) {
        self.buf.reinit_packets();
    }

    /// Shut down the session. Stops listener + I/O threads, closes all files.
    pub fn shutdown(mut self) {
        self.buf.signal_shutdown();
        if let Some(handle) = self.listener_handle.take() {
            // Wait briefly for listener to drain, then join
            std::thread::sleep(std::time::Duration::from_millis(100));
            let _ = handle.join();
        }
    }
}

impl Drop for HostcallSession {
    fn drop(&mut self) {
        self.buf.signal_shutdown();
        if let Some(handle) = self.listener_handle.take() {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let _ = handle.join();
        }
    }
}

// ================================================================
// Pipeline — Multi-stage kernel launch with shared hostcall session
// ================================================================

type PipelineStage =
    Box<dyn FnOnce(sys::CUdeviceptr) -> std::result::Result<(), crate::GpuHostError>>;

/// A pipeline of kernel stages that share a single [`HostcallSession`].
///
/// Each stage is a closure that launches a kernel using the session's
/// hostcall buffer device pointer. The pipeline handles synchronization
/// and packet reinitialization between stages automatically.
pub struct Pipeline {
    session: HostcallSession,
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new pipeline with the given hostcall packet count.
    pub fn new(num_packets: u16) -> std::result::Result<Self, HostcallError> {
        let session = HostcallSession::start(num_packets)?;
        Ok(Self {
            session,
            stages: Vec::new(),
        })
    }

    /// Add a stage to the pipeline.
    ///
    /// The closure receives the hostcall buffer device pointer and should
    /// launch a kernel + synchronize. The pipeline reinits packets between stages.
    pub fn stage<F>(mut self, f: F) -> Self
    where
        F: FnOnce(sys::CUdeviceptr) -> std::result::Result<(), crate::GpuHostError> + 'static,
    {
        self.stages.push(Box::new(f));
        self
    }

    /// Execute all stages sequentially with automatic synchronization.
    ///
    /// Between stages, the hostcall packet pool is reinitialized.
    /// After all stages complete, the session is shut down.
    pub fn run(self) -> std::result::Result<(), crate::GpuHostError> {
        let hc_ptr = self.session.dev_ptr();
        let stages = self.stages;
        let session = self.session;

        for (i, stage) in stages.into_iter().enumerate() {
            if i > 0 {
                session.reinit_packets();
            }
            stage(hc_ptr)?;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
        session.shutdown();
        Ok(())
    }
}

// ================================================================
// CommandBuffer — Host→GPU command channel
// ================================================================

/// A mapped-memory command buffer for host→GPU command submission.
///
/// The host writes commands to a ring buffer; the GPU kernel polls
/// `write_idx` and processes commands sequentially.
pub struct CommandBuffer {
    host_ptr: *mut u8,
    dev_ptr: sys::CUdeviceptr,
    _size: usize,
    capacity: u32,
}

// SAFETY: CommandBuffer wraps pinned CUDA mapped memory (cuMemHostAlloc with DEVICEMAP).
// The raw pointer is valid for the lifetime of the struct and freed in Drop.
// Thread safety is ensured by the protocol: only one writer (host submit) at a time,
// and the GPU reads via the device pointer after observing write_idx updates.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

/// Command to submit to the GPU via command buffer.
pub enum Command {
    /// No-op (for testing).
    Nop,
    /// Execute a computation. op_code 0 = vector add, 1 = scalar multiply, etc.
    Compute {
        /// Device pointer to input data.
        input_ptr: u64,
        /// Device pointer to output buffer.
        output_ptr: u64,
        /// Number of elements to process.
        count: u32,
        /// Operation code (application-defined).
        op_code: u32,
    },
    /// Print a message via hostcall (max 52 bytes).
    Print {
        /// Message bytes to print.
        msg: Vec<u8>,
    },
    /// Exit the command processing loop.
    Exit,
}

impl CommandBuffer {
    /// Allocate a command buffer with the given slot capacity.
    pub fn new(capacity: u32) -> Result<Self, HostcallError> {
        let size = CMD_BUF_HEADER_SIZE + (capacity as usize) * CMD_SLOT_SIZE;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc allocates `size` bytes of pinned device-mapped memory.
        // write_bytes zero-initializes the region; no kernel is running yet.
        unsafe {
            let cu = cuda_lib();
            let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
            let r = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(HostcallError::CudaAlloc(r));
            }
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag above.
        // On failure, we free host_ptr before returning.
        unsafe {
            let cu = cuda_lib();
            let r = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
            if r != sys::CUresult::CUDA_SUCCESS {
                cu.cuMemFreeHost(host_ptr);
                return Err(HostcallError::CudaGetDevPtr(r));
            }
        }

        let host_ptr = host_ptr as *mut u8;

        // SAFETY: CMD_OFF_CAPACITY is within the header region of the command buffer.
        unsafe {
            std::ptr::write_volatile(host_ptr.add(CMD_OFF_CAPACITY) as *mut u32, capacity);
        }

        Ok(Self {
            host_ptr,
            dev_ptr,
            _size: size,
            capacity,
        })
    }

    /// Get device pointer for kernel arg.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Submit a command to the buffer.
    ///
    /// Blocks if the buffer is full (busy-waits for GPU to drain).
    pub fn submit(&self, cmd: &Command) {
        // SAFETY: All read_volatile/write_volatile target offsets within the command
        // buffer's allocated region (CMD_OFF_WRITE_IDX, CMD_OFF_READ_IDX are in the
        // header; slot_ptr is computed from CMD_BUF_HEADER_SIZE + slot_idx * CMD_SLOT_SIZE
        // where slot_idx < capacity). Volatile access is required because the GPU
        // reads these values concurrently. The final AtomicU64 store with Release
        // ordering ensures the GPU sees the command data before the index update.
        unsafe {
            // Wait for space (backpressure)
            loop {
                let write_idx =
                    std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const u64);
                let read_idx =
                    std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_READ_IDX) as *const u64);
                if (write_idx - read_idx) < self.capacity as u64 {
                    break;
                }
                std::thread::yield_now();
            }

            let write_idx =
                std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const u64);
            let slot_idx = (write_idx % self.capacity as u64) as usize;
            let slot_ptr = self
                .host_ptr
                .add(CMD_BUF_HEADER_SIZE + slot_idx * CMD_SLOT_SIZE);

            // Write command type and payload
            match cmd {
                Command::Nop => {
                    std::ptr::write_volatile(slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32, CMD_NOP);
                }
                Command::Compute {
                    input_ptr,
                    output_ptr,
                    count,
                    op_code,
                } => {
                    std::ptr::write_volatile(
                        slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32,
                        CMD_COMPUTE,
                    );
                    let payload = slot_ptr.add(CMD_SLOT_OFF_PAYLOAD);
                    std::ptr::write_volatile(payload as *mut u64, *input_ptr);
                    std::ptr::write_volatile(payload.add(8) as *mut u64, *output_ptr);
                    std::ptr::write_volatile(payload.add(16) as *mut u32, *count);
                    std::ptr::write_volatile(payload.add(20) as *mut u32, *op_code);
                }
                Command::Print { msg } => {
                    std::ptr::write_volatile(
                        slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32,
                        CMD_PRINT,
                    );
                    let payload = slot_ptr.add(CMD_SLOT_OFF_PAYLOAD);
                    let len = msg.len().min(CMD_MAX_PAYLOAD - 4) as u32;
                    std::ptr::write_volatile(payload as *mut u32, len);
                    for i in 0..len as usize {
                        std::ptr::write_volatile(payload.add(4 + i), msg[i]);
                    }
                }
                Command::Exit => {
                    std::ptr::write_volatile(slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32, CMD_EXIT);
                }
            }

            // Increment write_idx with Release semantics
            let write_idx_ptr = &*(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const AtomicU64);
            write_idx_ptr.store(write_idx + 1, Ordering::Release);
        }
    }

    /// Reset indices to 0 (between kernel launches).
    pub fn reset(&self) {
        // SAFETY: CMD_OFF_WRITE_IDX and CMD_OFF_READ_IDX are within the header.
        // Must only be called when no kernel is accessing the buffer (after sync).
        unsafe {
            std::ptr::write_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *mut u64, 0);
            std::ptr::write_volatile(self.host_ptr.add(CMD_OFF_READ_IDX) as *mut u64, 0);
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        // SAFETY: host_ptr was allocated by cuMemHostAlloc in new() and has not
        // been freed yet (Drop is called exactly once).
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

// ================================================================
// FlightRecorder — Post-mortem trace event ring buffer
// ================================================================

/// A mapped-memory ring buffer that stores the last N trace events.
///
/// Unlike hostcall-based tracing, the flight recorder writes directly to
/// mapped memory with no round-trip. After a kernel crash, call [`dump()`]
/// to print the last N events for post-mortem analysis.
///
/// [`dump()`]: FlightRecorder::dump
pub struct FlightRecorder {
    host_ptr: *mut u8,
    dev_ptr: sys::CUdeviceptr,
    _size: usize,
    capacity: u32,
}

// SAFETY: FlightRecorder wraps pinned CUDA mapped memory (cuMemHostAlloc with DEVICEMAP).
// The raw pointer is valid for the lifetime of the struct and freed in Drop.
// The GPU writes events atomically via write_idx; the host only reads after
// kernel completion (cuCtxSynchronize), so there is no data race.
unsafe impl Send for FlightRecorder {}
unsafe impl Sync for FlightRecorder {}

impl FlightRecorder {
    /// Allocate a flight recorder with the given event slot capacity.
    pub fn new(capacity: u32) -> Result<Self, HostcallError> {
        let size = FR_HEADER_SIZE + (capacity as usize) * FR_SLOT_SIZE;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: Same CUDA alloc pattern as CommandBuffer::new — allocates pinned
        // device-mapped memory of `size` bytes and zero-initializes it.
        unsafe {
            let cu = cuda_lib();
            let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
            let r = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(HostcallError::CudaAlloc(r));
            }
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag. On failure, we free it.
        unsafe {
            let cu = cuda_lib();
            let r = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
            if r != sys::CUresult::CUDA_SUCCESS {
                cu.cuMemFreeHost(host_ptr);
                return Err(HostcallError::CudaGetDevPtr(r));
            }
        }

        let host_ptr = host_ptr as *mut u8;

        // SAFETY: FR_OFF_CAPACITY is within the flight recorder header.
        unsafe {
            std::ptr::write_volatile(host_ptr.add(FR_OFF_CAPACITY) as *mut u32, capacity);
        }

        Ok(Self {
            host_ptr,
            dev_ptr,
            _size: size,
            capacity,
        })
    }

    /// Get device pointer for kernel arg.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Check if the kernel set the crashed flag.
    pub fn crashed(&self) -> bool {
        // SAFETY: FR_OFF_FLAGS is within the flight recorder header, 4-byte aligned.
        let flags =
            unsafe { std::ptr::read_volatile(self.host_ptr.add(FR_OFF_FLAGS) as *const u32) };
        (flags & FR_FLAG_CRASHED) != 0
    }

    /// Get the number of events written (may exceed capacity for wrap-around).
    pub fn write_count(&self) -> u64 {
        // SAFETY: FR_OFF_WRITE_IDX is within the header, 8-byte aligned.
        unsafe { std::ptr::read_volatile(self.host_ptr.add(FR_OFF_WRITE_IDX) as *const u64) }
    }

    /// Dump all recorded events to stderr.
    ///
    /// Events are printed in chronological order. If the buffer has wrapped
    /// around, only the last `capacity` events are shown.
    pub fn dump(&self) {
        let write_idx = self.write_count();
        if write_idx == 0 {
            eprintln!("=== Flight Recorder: no events ===");
            return;
        }

        let start = write_idx.saturating_sub(self.capacity as u64);

        let crashed = self.crashed();
        eprintln!(
            "=== Flight Recorder Dump ({} events{}) ===",
            write_idx - start,
            if crashed { ", CRASHED" } else { "" }
        );

        for i in start..write_idx {
            let slot_idx = (i % self.capacity as u64) as usize;
            // SAFETY: slot_idx < capacity, so FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE
            // is within the allocated flight recorder buffer. All slot offset reads
            // (FR_SLOT_OFF_META, FR_SLOT_OFF_TIMESTAMP, FR_SLOT_OFF_MSG) are within
            // FR_SLOT_SIZE bytes of the slot start.
            let slot = unsafe { self.host_ptr.add(FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE) };

            let meta = unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_META) as *const u64) };
            let timestamp =
                unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_TIMESTAMP) as *const u64) };

            let (tid, bid, level, msg_len, lane) = decode_trace_metadata(meta);

            let msg_len = (msg_len as usize).min(FR_MAX_MSG_LEN);
            let mut msg_buf = vec![0u8; msg_len];
            for j in 0..msg_len {
                msg_buf[j] = unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_MSG + j)) };
            }

            let level_str = match level {
                TRACE_LEVEL_DEBUG => "DEBUG",
                TRACE_LEVEL_INFO => "INFO",
                TRACE_LEVEL_WARN => "WARN",
                TRACE_LEVEL_ERROR => "ERROR",
                _ => "???",
            };

            let msg = String::from_utf8_lossy(&msg_buf);
            eprintln!(
                "  [{ts}] T{tid}.B{bid}.L{lane} {lvl}: {msg}",
                ts = timestamp,
                lvl = level_str,
            );
        }

        eprintln!("=== End Flight Recorder ===");
    }

    /// Reset the flight recorder (between kernel launches).
    pub fn reset(&self) {
        // SAFETY: FR_OFF_WRITE_IDX and FR_OFF_FLAGS are within the header.
        // Must only be called when no kernel is accessing the buffer (after sync).
        unsafe {
            std::ptr::write_volatile(self.host_ptr.add(FR_OFF_WRITE_IDX) as *mut u64, 0);
            std::ptr::write_volatile(self.host_ptr.add(FR_OFF_FLAGS) as *mut u32, 0);
        }
    }
}

impl Drop for FlightRecorder {
    fn drop(&mut self) {
        // SAFETY: host_ptr was allocated by cuMemHostAlloc in new() and has not
        // been freed yet (Drop is called exactly once).
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

impl Drop for HostcallBuffer {
    fn drop(&mut self) {
        // SAFETY: Both host_ptr and sideband_host_ptr were allocated by
        // cuMemHostAlloc in alloc_internal() and have not been freed yet.
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
            if !self.sideband_host_ptr.is_null() {
                cu.cuMemFreeHost(self.sideband_host_ptr as *mut std::ffi::c_void);
            }
        }
    }
}
