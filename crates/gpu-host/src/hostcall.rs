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
}

// SAFETY: The buffer is pinned memory shared between host and GPU.
// We ensure single-writer access via the protocol (GPU writes packet,
// host reads; host writes control, GPU reads).
unsafe impl Send for HostcallBuffer {}
unsafe impl Sync for HostcallBuffer {}

impl HostcallBuffer {
    /// Allocate and initialize a hostcall buffer with `num_packets` packet slots.
    ///
    /// Uses cuMemHostAlloc with DEVICEMAP|PORTABLE flags for GPU-CPU shared access.
    pub fn new(num_packets: u16) -> Result<Self, HostcallError> {
        let size = buffer_size(num_packets);
        let cu = unsafe { cuda_lib() };

        // Allocate pinned, device-mapped memory
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let result = unsafe { cu.cuMemHostAlloc(&mut host_ptr, size, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(HostcallError::CudaAlloc(result));
        }

        // Get device-side pointer
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let result =
            unsafe { cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // Zero-initialize the entire buffer
        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        let buf = Self {
            host_ptr: host_ptr as *mut u8,
            dev_ptr,
            size,
            num_packets,
        };

        // Initialize the buffer structure
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

    /// Run the host listener loop. Blocks until shutdown is signaled.
    ///
    /// `on_print` is called for each PRINT service request with the message bytes.
    pub fn listen<F>(&self, mut on_print: F)
    where
        F: FnMut(&[u8]),
    {
        let mut last_doorbell: u64 = 0;
        let mut idle_spins: u32 = 0;
        const MAX_IDLE_SPINS: u32 = 1_000_000;

        // File descriptor table for FILE I/O services
        let mut fd_table: HashMap<u64, File> = HashMap::new();
        let mut next_fd: u64 = 1; // fd 0 is reserved

        loop {
            // Check shutdown
            if self.shutdown().load(Ordering::Acquire) != 0 {
                break;
            }

            // Poll doorbell
            let current_doorbell = self.doorbell().load(Ordering::Acquire);
            if current_doorbell == last_doorbell {
                idle_spins += 1;
                if idle_spins > MAX_IDLE_SPINS {
                    // Yield to OS to avoid burning CPU
                    std::thread::yield_now();
                    idle_spins = 0;
                }
                std::hint::spin_loop();
                continue;
            }

            last_doorbell = current_doorbell;
            idle_spins = 0;

            // Atomically grab all ready packets
            let ready_head =
                self.ready_stack().swap(null_tagged(), Ordering::AcqRel);
            if tagged_index(ready_head) == NULL_INDEX {
                continue;
            }

            // Walk the ready list and process each packet
            let mut current = ready_head;
            while tagged_index(current) != NULL_INDEX {
                let idx = tagged_index(current);
                let pkt = self.packet_ptr(idx);

                unsafe {
                    let next =
                        std::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
                    let service =
                        std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32);

                    let has_error = match service {
                        SERVICE_PRINT => {
                            self.handle_print(pkt, &mut on_print);
                            false
                        }
                        SERVICE_NOP => {
                            false
                        }
                        SERVICE_OPEN => {
                            self.handle_open(pkt, &mut fd_table, &mut next_fd)
                        }
                        SERVICE_WRITE => {
                            self.handle_write(pkt, &mut fd_table)
                        }
                        SERVICE_READ => {
                            self.handle_read(pkt, &mut fd_table)
                        }
                        SERVICE_CLOSE => {
                            self.handle_close(pkt, &mut fd_table)
                        }
                        SERVICE_STDIN => {
                            self.handle_stdin(pkt)
                        }
                        SERVICE_TIME => {
                            self.handle_time(pkt)
                        }
                        _ => {
                            // Unknown service — set error bit
                            true
                        }
                    };

                    // Signal GPU: response is ready
                    let control =
                        &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                    let flags = if has_error {
                        CONTROL_READY | CONTROL_ERROR
                    } else {
                        CONTROL_READY
                    };
                    control.store(flags, Ordering::Release);

                    current = next;
                }
            }
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

    /// Handle SERVICE_STDIN: read a line from host stdin.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read (u64), or FILE_ERROR_SENTINEL on error/EOF
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_stdin(&self, pkt: *mut u8) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let max_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let max_len = max_len.min(STDIN_MAX_READ_LEN);

        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => {
                // EOF — not an error, just zero bytes
                println!("  [HOST] STDIN: EOF");
                std::ptr::write_volatile(payload as *mut u64, 0u64);
                false
            }
            Ok(n) => {
                let bytes = line.as_bytes();
                let copy_len = n.min(max_len);
                println!("  [HOST] STDIN: read {} bytes: {:?}", copy_len, &line[..copy_len]);
                std::ptr::write_volatile(payload as *mut u64, copy_len as u64);
                let dst = payload.add(8);
                for i in 0..copy_len {
                    std::ptr::write_volatile(dst.add(i), bytes[i]);
                }
                false
            }
            Err(e) => {
                eprintln!("  [HOST] STDIN ERROR: {}", e);
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
    /// Handle SERVICE_STDIN with canned data instead of reading from real stdin.
    /// Returns the provided data as the stdin response.
    unsafe fn handle_stdin_canned(&self, pkt: *mut u8, data: &[u8]) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let max_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let max_len = max_len.min(STDIN_MAX_READ_LEN);
        let copy_len = data.len().min(max_len);

        println!("  [HOST] STDIN (canned): providing {} bytes", copy_len);
        std::ptr::write_volatile(payload as *mut u64, copy_len as u64);
        let dst = payload.add(8);
        for i in 0..copy_len {
            std::ptr::write_volatile(dst.add(i), data[i]);
        }
        false
    }

    /// Listen with both a print callback and a canned stdin provider.
    pub fn listen_with_stdin<F>(&self, mut on_print: F, stdin_data: Vec<u8>)
    where
        F: FnMut(&[u8]),
    {
        let mut last_doorbell: u64 = 0;
        let mut idle_spins: u32 = 0;
        const MAX_IDLE_SPINS: u32 = 1_000_000;

        let mut fd_table: HashMap<u64, File> = HashMap::new();
        let mut next_fd: u64 = 1;
        let mut stdin_consumed = false;

        loop {
            if self.shutdown().load(Ordering::Acquire) != 0 {
                break;
            }

            let current_doorbell = self.doorbell().load(Ordering::Acquire);
            if current_doorbell == last_doorbell {
                idle_spins += 1;
                if idle_spins > MAX_IDLE_SPINS {
                    std::thread::yield_now();
                    idle_spins = 0;
                }
                std::hint::spin_loop();
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
                    let next =
                        std::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
                    let service =
                        std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32);

                    let has_error = match service {
                        SERVICE_PRINT => {
                            self.handle_print(pkt, &mut on_print);
                            false
                        }
                        SERVICE_STDIN => {
                            if !stdin_consumed {
                                stdin_consumed = true;
                                self.handle_stdin_canned(pkt, &stdin_data)
                            } else {
                                // Subsequent reads return EOF
                                self.handle_stdin_canned(pkt, &[])
                            }
                        }
                        SERVICE_TIME => {
                            self.handle_time(pkt)
                        }
                        SERVICE_OPEN => {
                            self.handle_open(pkt, &mut fd_table, &mut next_fd)
                        }
                        SERVICE_WRITE => {
                            self.handle_write(pkt, &mut fd_table)
                        }
                        SERVICE_READ => {
                            self.handle_read(pkt, &mut fd_table)
                        }
                        SERVICE_CLOSE => {
                            self.handle_close(pkt, &mut fd_table)
                        }
                        _ => true,
                    };

                    let control =
                        &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                    let flags = if has_error {
                        CONTROL_READY | CONTROL_ERROR
                    } else {
                        CONTROL_READY
                    };
                    control.store(flags, Ordering::Release);

                    current = next;
                }
            }
        }
    }
}

impl Drop for HostcallBuffer {
    fn drop(&mut self) {
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}
