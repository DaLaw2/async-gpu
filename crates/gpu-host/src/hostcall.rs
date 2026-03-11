//! Host-side hostcall listener and buffer management.
//!
//! Allocates a pinned, device-mapped hostcall buffer and runs a listener
//! thread that polls for GPU requests and dispatches to service handlers.

use cudarc::driver::sys::{self, lib as cuda_lib};
use gpu_protocol::*;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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

                    match service {
                        SERVICE_PRINT => {
                            self.handle_print(pkt, &mut on_print);
                        }
                        SERVICE_NOP => {
                            // No-op, just acknowledge
                        }
                        _ => {
                            // Unknown service — set error bit
                            let control =
                                &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                            control.store(
                                CONTROL_READY | CONTROL_ERROR,
                                Ordering::Release,
                            );
                            current = next;
                            continue;
                        }
                    }

                    // Signal GPU: response is ready
                    let control =
                        &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
                    control.store(CONTROL_READY, Ordering::Release);

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
}

impl Drop for HostcallBuffer {
    fn drop(&mut self) {
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}
