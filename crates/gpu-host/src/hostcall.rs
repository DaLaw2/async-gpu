//! Host-side hostcall listener and buffer management.
//!
//! Allocates a pinned, device-mapped hostcall buffer and runs a listener
//! thread that polls for GPU requests and dispatches to service handlers.

use cudarc::driver::sys::{self, lib as cuda_lib};
use gpu_protocol::*;
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;

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
        _ => ERR_OTHER,
    }
}

/// Encode an io::Error into the hostcall error format and write it to payload slot 0.
/// Returns `true` to signal CONTROL_ERROR should be set.
unsafe fn write_error_response(payload: *mut u8, e: &std::io::Error) -> bool {
    let category = io_error_to_category(e);
    let raw_errno = e.raw_os_error().unwrap_or(0) as u16;
    std::ptr::write_volatile(payload as *mut u64, encode_error(category, raw_errno));
    true
}

#[derive(Debug)]
pub enum HostcallError {
    CudaAlloc(sys::CUresult),
    CudaGetDevPtr(sys::CUresult),
}

impl fmt::Display for HostcallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaAlloc(r) => write!(f, "cuMemHostAlloc failed: {:?}", r),
            Self::CudaGetDevPtr(r) => {
                write!(f, "cuMemHostGetDevicePointer_v2 failed: {:?}", r)
            }
        }
    }
}

impl std::error::Error for HostcallError {}

/// Hostcall buffer handle with both host and device pointers.
pub struct HostcallBuffer {
    pub host_ptr: *mut u8,
    pub dev_ptr: sys::CUdeviceptr,
    pub size: usize,
    pub num_packets: u16,
    /// Sideband buffer for bulk data transfer (>56 bytes).
    pub sideband_host_ptr: *mut u8,
    pub sideband_dev_ptr: sys::CUdeviceptr,
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
    ///
    /// Uses cuMemHostAlloc with DEVICEMAP|PORTABLE flags for GPU-CPU shared access.
    pub fn new(num_packets: u16) -> Result<Self, HostcallError> {
        Self::new_with_sideband(num_packets, DEFAULT_SIDEBAND_SIZE)
    }

    /// Allocate a hostcall buffer with custom sideband size.
    pub fn new_with_sideband(
        num_packets: u16,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        let size = buffer_size(num_packets);
        let cu = unsafe { cuda_lib() };
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;

        // Allocate hostcall buffer (pinned, device-mapped)
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cu.cuMemHostAlloc(&mut host_ptr, size, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        let result =
            unsafe { cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        // Allocate sideband buffer for bulk data transfer
        let sideband_total = SIDEBAND_HEADER_SIZE + sideband_data_size;
        let mut sb_host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result =
            unsafe { cu.cuMemHostAlloc(&mut sb_host_ptr, sideband_total, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe { cu.cuMemFreeHost(host_ptr as *mut std::ffi::c_void) };
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut sb_dev_ptr: sys::CUdeviceptr = 0;
        let result =
            unsafe { cu.cuMemHostGetDevicePointer_v2(&mut sb_dev_ptr, sb_host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                cu.cuMemFreeHost(sb_host_ptr);
                cu.cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
            }
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // Zero-initialize sideband and set capacity
        unsafe {
            std::ptr::write_bytes(sb_host_ptr as *mut u8, 0, sideband_total);
            // Set capacity field in sideband header
            let cap_ptr =
                (sb_host_ptr as *mut u8).add(SIDEBAND_OFF_CAPACITY) as *mut u64;
            std::ptr::write_volatile(cap_ptr, sideband_data_size as u64);
        }

        let buf = Self {
            host_ptr: host_ptr as *mut u8,
            dev_ptr,
            size,
            num_packets,
            sideband_host_ptr: sb_host_ptr as *mut u8,
            sideband_dev_ptr: sb_dev_ptr,
            sideband_size: sideband_total,
        };

        buf.init();

        Ok(buf)
    }

    /// Initialize the hostcall buffer: set up free stack, ready stack, etc.
    fn init(&self) {
        let base = self.host_ptr;
        unsafe {
            // Header fields
            let free_stack = base.add(BUF_OFF_FREE_STACK) as *mut u64;
            let ready_stack = base.add(BUF_OFF_READY_STACK) as *mut u64;
            let doorbell = base.add(BUF_OFF_DOORBELL) as *mut u64;
            let shutdown = base.add(BUF_OFF_SHUTDOWN) as *mut u32;
            let num_packets_field = base.add(BUF_OFF_NUM_PACKETS) as *mut u32;
            let warp_size_field = base.add(BUF_OFF_WARP_SIZE) as *mut u32;

            // Initialize header
            std::ptr::write_volatile(ready_stack, null_tagged());
            std::ptr::write_volatile(doorbell, 0u64);
            std::ptr::write_volatile(shutdown, 0u32);
            std::ptr::write_volatile(num_packets_field, self.num_packets as u32);
            std::ptr::write_volatile(warp_size_field, WARP_SIZE);

            // Build the free stack: chain all packets as a linked list.
            // free_stack → packet[0] → packet[1] → ... → packet[N-1] → NULL
            for i in 0..self.num_packets {
                let pkt = base.add(packet_offset(i));
                let next_tagged = if i + 1 < self.num_packets {
                    make_tagged(0, i + 1)
                } else {
                    null_tagged()
                };
                std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next_tagged);
                // Clear control
                std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
            }

            // Set free_stack head to packet[0] with tag 0
            std::ptr::write_volatile(free_stack, make_tagged(0, 0));
        }
    }

    /// Get a reference to the doorbell as an AtomicU64.
    fn doorbell(&self) -> &AtomicU64 {
        unsafe { &*(self.host_ptr.add(BUF_OFF_DOORBELL) as *const AtomicU64) }
    }

    /// Get a reference to the ready_stack as an AtomicU64.
    fn ready_stack(&self) -> &AtomicU64 {
        unsafe { &*(self.host_ptr.add(BUF_OFF_READY_STACK) as *const AtomicU64) }
    }

    /// Get a reference to the shutdown flag as an AtomicU32.
    fn shutdown(&self) -> &AtomicU32 {
        unsafe { &*(self.host_ptr.add(BUF_OFF_SHUTDOWN) as *const AtomicU32) }
    }

    /// Get pointer to a packet by index.
    fn packet_ptr(&self, index: u16) -> *mut u8 {
        unsafe { self.host_ptr.add(packet_offset(index)) }
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
            const SLEEP_DURATION: std::time::Duration =
                std::time::Duration::from_micros(100);

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

                let ready_head =
                    self.ready_stack().swap(null_tagged(), Ordering::AcqRel);
                if tagged_index(ready_head) == NULL_INDEX {
                    continue;
                }

                let mut current = ready_head;
                while tagged_index(current) != NULL_INDEX {
                    let idx = tagged_index(current);
                    let pkt = self.packet_ptr(idx);

                    unsafe {
                        let next = std::ptr::read_volatile(
                            pkt.add(PKT_OFF_NEXT) as *const u64,
                        );

                        let control =
                            &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                        let ctrl = control.load(Ordering::Acquire);
                        if ctrl & CONTROL_FILLED == 0 {
                            current = next;
                            continue;
                        }

                        let service = std::ptr::read_volatile(
                            pkt.add(PKT_OFF_SERVICE) as *const u32,
                        );

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
                            // Slow path — offload to I/O thread
                            SERVICE_OPEN | SERVICE_WRITE | SERVICE_READ
                            | SERVICE_CLOSE | SERVICE_STDIN
                            | SERVICE_BULK_WRITE | SERVICE_BULK_READ => {
                                let _ = io_tx.send(IoRequest {
                                    pkt_idx: idx,
                                    service,
                                });
                            }
                            _ => {
                                control.store(
                                    CONTROL_READY | CONTROL_ERROR,
                                    Ordering::Release,
                                );
                            }
                        }

                        current = next;
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
    fn io_thread_loop<S: StdinSource>(
        &self,
        rx: mpsc::Receiver<IoRequest>,
        mut stdin: S,
    ) {
        let mut fd_table: HashMap<u64, File> = HashMap::new();
        let mut next_fd: u64 = 1; // fd 0 is reserved

        while let Ok(req) = rx.recv() {
            let pkt = self.packet_ptr(req.pkt_idx);
            let has_error = unsafe {
                match req.service {
                    SERVICE_OPEN => {
                        self.handle_open(pkt, &mut fd_table, &mut next_fd)
                    }
                    SERVICE_WRITE => self.handle_write(pkt, &mut fd_table),
                    SERVICE_READ => self.handle_read(pkt, &mut fd_table),
                    SERVICE_CLOSE => self.handle_close(pkt, &mut fd_table),
                    SERVICE_STDIN => {
                        self.handle_stdin_from_source(pkt, &mut stdin)
                    }
                    SERVICE_BULK_WRITE => {
                        self.handle_bulk_write(pkt, &mut fd_table)
                    }
                    SERVICE_BULK_READ => {
                        self.handle_bulk_read(pkt, &mut fd_table)
                    }
                    _ => true,
                }
            };

            let control = unsafe {
                &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32)
            };
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

        on_print(&msg_buf[..msg_len]);
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
        fd_table: &mut HashMap<u64, File>,
        next_fd: &mut u64,
    ) -> bool {
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
                eprintln!("  [HOST] FILE OPEN ERROR: invalid flags={}", flags);
                return write_error_response(payload, &e);
            }
        };

        match file_result {
            Ok(file) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, file);
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] FILE OPEN: \"{}\" flags={} -> fd={}", path_str, flags, fd);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE OPEN ERROR: \"{}\": {}", path_str, e);
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
    unsafe fn handle_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, File>,
    ) -> bool {
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
            Some(f) => f,
            None => {
                eprintln!("  [HOST] FILE WRITE ERROR: invalid fd={}", fd);
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match file.write(&data_buf[..data_len]) {
            Ok(n) => {
                // Flush to ensure data is persisted
                let _ = file.flush();
                println!("  [HOST] FILE WRITE: fd={} {} bytes written", fd, n);
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE WRITE ERROR: fd={}: {}", fd, e);
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
    unsafe fn handle_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, File>,
    ) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let max_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_len = max_len.min(FILE_MAX_READ_LEN);

        let file = match fd_table.get_mut(&fd) {
            Some(f) => f,
            None => {
                eprintln!("  [HOST] FILE READ ERROR: invalid fd={}", fd);
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let mut read_buf = [0u8; FILE_MAX_READ_LEN];
        match file.read(&mut read_buf[..max_len]) {
            Ok(n) => {
                println!("  [HOST] FILE READ: fd={} {} bytes read", fd, n);
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
                eprintln!("  [HOST] FILE READ ERROR: fd={}: {}", fd, e);
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
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => {
                let secs = duration.as_secs();
                let nanos = duration.subsec_nanos() as u64;
                std::ptr::write_volatile(payload as *mut u64, secs);
                std::ptr::write_volatile(payload.add(8) as *mut u64, nanos);
                println!("  [HOST] TIME: epoch_secs={} nanos={}", secs, nanos);
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
        eprintln!(
            "\x1b[1;31m[GPU PANIC]\x1b[0m block={} thread={}: {}",
            block_idx, thread_idx, msg
        );

        false // No error — GPU thread will trap after receiving response
    }

    /// Handle SERVICE_CLOSE: close an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: 0 on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_close(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, File>,
    ) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);

        match fd_table.remove(&fd) {
            Some(_file) => {
                // File is dropped here, which closes it
                println!("  [HOST] FILE CLOSE: fd={} closed", fd);
                std::ptr::write_volatile(payload as *mut u64, 0);
                false
            }
            None => {
                eprintln!("  [HOST] FILE CLOSE ERROR: invalid fd={}", fd);
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
    unsafe fn handle_stdin_from_source<S: StdinSource>(
        &self,
        pkt: *mut u8,
        stdin: &mut S,
    ) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let max_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let max_len = max_len.min(STDIN_MAX_READ_LEN);

        let mut buf = [0u8; STDIN_MAX_READ_LEN];
        let n = stdin.read_line_bytes(&mut buf[..max_len]);

        println!("  [HOST] STDIN: {} bytes", n);
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
        fd_table: &mut HashMap<u64, File>,
    ) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check against sideband capacity
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64,
        ) as usize;
        if sb_offset + length > capacity {
            eprintln!(
                "  [HOST] BULK WRITE ERROR: offset={} + len={} > capacity={}",
                sb_offset, length, capacity
            );
            std::ptr::write_volatile(
                payload as *mut u64,
                encode_error(ERR_INVALID_INPUT, 0),
            );
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(f) => f,
            None => {
                eprintln!("  [HOST] BULK WRITE ERROR: invalid fd={}", fd);
                std::ptr::write_volatile(
                    payload as *mut u64,
                    encode_error(ERR_INVALID_FD, 0),
                );
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let data = std::slice::from_raw_parts(data_ptr, length);

        match file.write_all(data) {
            Ok(()) => {
                let _ = file.flush();
                println!(
                    "  [HOST] BULK WRITE: fd={} {} bytes written",
                    fd, length
                );
                std::ptr::write_volatile(payload as *mut u64, length as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK WRITE ERROR: fd={}: {}", fd, e);
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
        fd_table: &mut HashMap<u64, File>,
    ) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_length =
            std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64,
        ) as usize;
        if sb_offset + max_length > capacity {
            eprintln!(
                "  [HOST] BULK READ ERROR: offset={} + len={} > capacity={}",
                sb_offset, max_length, capacity
            );
            std::ptr::write_volatile(
                payload as *mut u64,
                encode_error(ERR_INVALID_INPUT, 0),
            );
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(f) => f,
            None => {
                eprintln!("  [HOST] BULK READ ERROR: invalid fd={}", fd);
                std::ptr::write_volatile(
                    payload as *mut u64,
                    encode_error(ERR_INVALID_FD, 0),
                );
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let buf = std::slice::from_raw_parts_mut(data_ptr, max_length);

        match file.read(buf) {
            Ok(n) => {
                println!("  [HOST] BULK READ: fd={} {} bytes read", fd, n);
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK READ ERROR: fd={}: {}", fd, e);
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

impl Drop for HostcallBuffer {
    fn drop(&mut self) {
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
            if !self.sideband_host_ptr.is_null() {
                cu.cuMemFreeHost(self.sideband_host_ptr as *mut std::ffi::c_void);
            }
        }
    }
}
