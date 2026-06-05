// Pipeline/IO kernels — file transform, branching pipeline, pipelined compute, parallel grep.

use crate::hybrid::{hybrid_warp_print_init, hybrid_warp_wait};
use core::arch::nvptx;
use gpu_atomics::{membar_sys, sys_spin_load_acquire_u32};
use gpu_kernel_core::helpers::{gpu_hostcall_close, gpu_hostcall_open, grep_buffer};
use gpu_protocol::*;
use gpu_runtime::warp_future::{warp_hostcall_submit, warp_hostcall_wait_u64};

// ============================================================
// async-pipeline: File transform demo — 16-state WarpFuture
// ============================================================
//
// GPU-autonomous pipeline: open → read → transform → open → write → close → close → print
// All I/O is warp-cooperative. Compute is per-thread divergent.
// One kernel launch, zero CPU intervention between steps.

const FTP_OPEN_IN: u32 = 0;
const FTP_WAIT_OPEN_IN: u32 = 1;
const FTP_BULK_READ: u32 = 2;
const FTP_WAIT_READ: u32 = 3;
const FTP_COMPUTE: u32 = 4;
const FTP_OPEN_OUT: u32 = 5;
const FTP_WAIT_OPEN_OUT: u32 = 6;
const FTP_BULK_WRITE: u32 = 7;
const FTP_WAIT_WRITE: u32 = 8;
const FTP_CLOSE_IN: u32 = 9;
const FTP_WAIT_CLOSE_IN: u32 = 10;
const FTP_CLOSE_OUT: u32 = 11;
const FTP_WAIT_CLOSE_OUT: u32 = 12;
const FTP_PRINT: u32 = 13;
const FTP_WAIT_PRINT: u32 = 14;
const FTP_DONE: u32 = 15;

/// Data size: 32 lanes × 32 bytes = 1024 bytes.
const FTP_DATA_SIZE: u64 = 1024;

struct FileTransformFuture {
    buf: *mut u8,
    sideband: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd_in: u64,
    fd_out: u64,
    sideband_offset: u64,
    bytes_read: u64,
}

impl FileTransformFuture {
    unsafe fn new(buf: *mut u8, sideband: *mut u8) -> Self {
        Self {
            buf,
            sideband,
            state: FTP_OPEN_IN,
            pkt_idx: gpu_protocol::NULL_INDEX,
            fd_in: 0,
            fd_out: 0,
            sideband_offset: 0,
            bytes_read: 0,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for FileTransformFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === Step 1: Open input file ===
            FTP_OPEN_IN => unsafe {
                let path = b"gpu_input.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    FTP_WAIT_OPEN_IN,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_OPEN_IN => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    FTP_BULK_READ,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd_in = fd;
                    }
                }
                WarpPoll::Pending
            },

            // === Step 2: Read data via sideband bulk transfer ===
            FTP_BULK_READ => unsafe {
                if wcx.is_leader() {
                    gpu_runtime::sideband::sideband_reset(self.sideband);
                    self.sideband_offset =
                        gpu_runtime::sideband::sideband_alloc(self.sideband, FTP_DATA_SIZE);
                }
                gpu_atomics::syncwarp(wcx.active_mask);

                let fd = self.fd_in;
                let sb_off = self.sideband_offset;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, FTP_DATA_SIZE);
                    },
                    FTP_WAIT_READ,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_READ => unsafe {
                if let Some(n) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    FTP_COMPUTE,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.bytes_read = n;
                    }
                }
                WarpPoll::Pending
            },

            // === Step 3: Per-thread compute — toggle ASCII case ===
            // Each lane processes its 32-byte slice of the sideband data in-place.
            // Divergent: each lane may process different byte counts.
            FTP_COMPUTE => unsafe {
                let lid = wcx.lane_id;
                let offset = broadcast_u32(wcx.active_mask, self.sideband_offset as u32) as usize;
                let data_base = self
                    .sideband
                    .add(gpu_protocol::SIDEBAND_DATA_OFFSET + offset);
                let lane_base = data_base.add(lid as usize * 32);
                let bytes_read = broadcast_u32(wcx.active_mask, self.bytes_read as u32);
                let lane_start = lid * 32;

                let mut i: u32 = 0;
                while i < 32 && lane_start + i < bytes_read {
                    let b = core::ptr::read_volatile(lane_base.add(i as usize));
                    let toggled = if (b >= b'A' && b <= b'Z') || (b >= b'a' && b <= b'z') {
                        b ^ 0x20
                    } else {
                        b
                    };
                    core::ptr::write_volatile(lane_base.add(i as usize), toggled);
                    i += 1;
                }

                // Flush all lanes' sideband writes to system visibility
                membar_sys();
                gpu_atomics::syncwarp(wcx.active_mask);

                if wcx.is_leader() {
                    self.state = FTP_OPEN_OUT;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Step 4: Open output file ===
            FTP_OPEN_OUT => unsafe {
                let path = b"gpu_output.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    FTP_WAIT_OPEN_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_OPEN_OUT => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    FTP_BULK_WRITE,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd_out = fd;
                    }
                }
                WarpPoll::Pending
            },

            // === Step 5: Write transformed data via sideband ===
            FTP_BULK_WRITE => unsafe {
                let fd = self.fd_out;
                let sb_off = self.sideband_offset;
                let len = self.bytes_read;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_WRITE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, len);
                    },
                    FTP_WAIT_WRITE,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_WRITE => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    FTP_CLOSE_IN,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            // === Step 6: Close input file ===
            FTP_CLOSE_IN => unsafe {
                let fd = self.fd_in;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    FTP_WAIT_CLOSE_IN,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_CLOSE_IN => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    FTP_CLOSE_OUT,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            // === Step 7: Close output file ===
            FTP_CLOSE_OUT => unsafe {
                let fd = self.fd_out;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    FTP_WAIT_CLOSE_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_CLOSE_OUT => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, FTP_PRINT, &mut self.state)
                    .is_some()
                {}
                WarpPoll::Pending
            },

            // === Step 8: Print completion message ===
            FTP_PRINT => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"pipeline: done",
                    FTP_WAIT_PRINT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            FTP_WAIT_PRINT => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, FTP_DONE, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            FTP_DONE => WarpPoll::Ready(true),

            _ => WarpPoll::Ready(false),
        }
    }
}

/// async-pipeline demo kernel: GPU-autonomous file transform pipeline.
///
/// The GPU self-coordinates 8 I/O steps + 1 compute step in a single kernel launch:
///   open(in) → read(in) → transform → open(out) → write(out) → close(in) → close(out) → print
///
/// No CPU intervention between steps — the GPU drives the entire pipeline via WarpFuture.
///
/// `buf`      = hostcall buffer (CUDA mapped memory)
/// `sideband` = sideband buffer for bulk data transfer (CUDA mapped memory)
/// `status`   = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn file_transform_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = FileTransformFuture::new(buf, sideband);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// async-pipeline.3: Branching Pipeline — conditional state transitions
// ============================================================
//
// Demonstrates conditional state transitions based on hostcall response values.
// (Originally hand-written; now achievable via standard async fn + MIR pass.)
//
// Logic:
//   1. Try to OPEN "branch_test.txt" for reading
//   2. If OPEN succeeds (fd != FILE_ERROR_SENTINEL):
//      → CLOSE the file, PRINT "file exists"
//   3. If OPEN fails (fd == FILE_ERROR_SENTINEL):
//      → CREATE the file, WRITE default data, CLOSE, PRINT "file created"
//
// State machine:
//   0: TRY_OPEN       → submit OPEN(read)
//   1: WAIT_OPEN      → if fd ok → state 2 (CLOSE_EXISTING)
//                        if fd err → state 4 (CREATE_FILE)
//   2: CLOSE_EXISTING → submit CLOSE(fd)
//   3: WAIT_CLOSE_1   → state 8 (PRINT_EXISTS)
//   4: CREATE_FILE    → submit OPEN(write|create)
//   5: WAIT_CREATE    → store fd_out
//   6: WRITE_DEFAULT  → submit WRITE(fd_out, "hello from GPU\n")
//   7: WAIT_WRITE     → state 10 (CLOSE_CREATED)
//   8: PRINT_EXISTS   → submit PRINT("branch: file exists")
//   9: WAIT_PRINT_1   → DONE
//  10: CLOSE_CREATED  → submit CLOSE(fd_out)
//  11: WAIT_CLOSE_2   → state 12 (PRINT_CREATED)
//  12: PRINT_CREATED  → submit PRINT("branch: file created")
//  13: WAIT_PRINT_2   → DONE
//  14: DONE

const BP_TRY_OPEN: u32 = 0;
const BP_WAIT_OPEN: u32 = 1;
const BP_CLOSE_EXISTING: u32 = 2;
const BP_WAIT_CLOSE_1: u32 = 3;
const BP_CREATE_FILE: u32 = 4;
const BP_WAIT_CREATE: u32 = 5;
const BP_WRITE_DEFAULT: u32 = 6;
const BP_WAIT_WRITE: u32 = 7;
const BP_PRINT_EXISTS: u32 = 8;
const BP_WAIT_PRINT_1: u32 = 9;
const BP_CLOSE_CREATED: u32 = 10;
const BP_WAIT_CLOSE_2: u32 = 11;
const BP_PRINT_CREATED: u32 = 12;
const BP_WAIT_PRINT_2: u32 = 13;
const BP_DONE: u32 = 14;

struct BranchingPipelineFuture {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
}

impl BranchingPipelineFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: 0,
            pkt_idx: NULL_INDEX,
            fd: 0,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for BranchingPipelineFuture {
    type Output = bool;

    #[inline(always)]
    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === Branch point: try opening the file for reading ===
            BP_TRY_OPEN => unsafe {
                let path = b"branch_test.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    BP_WAIT_OPEN,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            // === CONDITIONAL STATE TRANSITION ===
            // This is the key pattern: the next state depends on the runtime value.
            // All 32 lanes see the same fd (broadcast from lane 0), so all lanes
            // agree on the branch direction — warp convergence is maintained.
            //
            // We inline the wait logic here instead of using warp_hostcall_wait_u64
            // because we need to inspect CONTROL_ERROR to decide the branch.
            // The host sets CONTROL_ERROR when file open fails.
            BP_WAIT_OPEN => unsafe {
                let idx = broadcast_u32(wcx.active_mask, self.pkt_idx as u32) as u16;
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let ctrl = sys_spin_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

                if ctrl & CONTROL_READY != 0 {
                    let has_error = (ctrl & CONTROL_ERROR) != 0;

                    // Broadcast error flag to all lanes
                    let err_flag = broadcast_u32(wcx.active_mask, has_error as u32);

                    let mut fd_val: u64 = 0;
                    if wcx.is_leader() && !has_error {
                        fd_val = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                    }
                    // Broadcast fd to all lanes
                    let lo = broadcast_u32(wcx.active_mask, fd_val as u32) as u64;
                    let hi = broadcast_u32(wcx.active_mask, (fd_val >> 32) as u32) as u64;
                    let fd = lo | (hi << 32);

                    if wcx.is_leader() {
                        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
                        if err_flag != 0 {
                            // File does not exist → take the CREATE branch
                            self.state = BP_CREATE_FILE;
                        } else {
                            // File exists → take the CLOSE+PRINT branch
                            self.fd = fd;
                            self.state = BP_CLOSE_EXISTING;
                        }
                    }
                    gpu_atomics::syncwarp(wcx.active_mask);
                }
                WarpPoll::Pending
            },

            // === Branch A: File exists → close it and print ===
            BP_CLOSE_EXISTING => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    BP_WAIT_CLOSE_1,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_CLOSE_1 => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BP_PRINT_EXISTS,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            // === Branch B: File does not exist → create, write, close ===
            BP_CREATE_FILE => unsafe {
                let path = b"branch_test.txt";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    BP_WAIT_CREATE,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_CREATE => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BP_WRITE_DEFAULT,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                    }
                }
                WarpPoll::Pending
            },

            BP_WRITE_DEFAULT => unsafe {
                let fd = self.fd;
                let msg = b"hello from GPU\n";
                let msg_len = msg.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_WRITE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, msg_len as u64);
                        let dst = payload.add(16);
                        let mut i = 0;
                        while i < msg_len {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                    },
                    BP_WAIT_WRITE,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_WRITE => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BP_CLOSE_CREATED,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            // === Convergence point: both branches end with PRINT ===
            BP_PRINT_EXISTS => unsafe {
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_PRINT,
                    |payload| {
                        let msg = b"branch: file exists";
                        core::ptr::write_volatile(payload as *mut u64, msg.len() as u64);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < msg.len() {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                        core::ptr::write_volatile(
                            payload.add(64) as *mut u32,
                            nvptx::_block_idx_x() as u32,
                        );
                        core::ptr::write_volatile(
                            payload.add(68) as *mut u32,
                            nvptx::_thread_idx_x() as u32,
                        );
                    },
                    BP_WAIT_PRINT_1,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_PRINT_1 => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, BP_DONE, &mut self.state)
                    .is_some()
                {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            },

            BP_CLOSE_CREATED => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    BP_WAIT_CLOSE_2,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_CLOSE_2 => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BP_PRINT_CREATED,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            BP_PRINT_CREATED => unsafe {
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_PRINT,
                    |payload| {
                        let msg = b"branch: file created";
                        core::ptr::write_volatile(payload as *mut u64, msg.len() as u64);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < msg.len() {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                        core::ptr::write_volatile(
                            payload.add(64) as *mut u32,
                            nvptx::_block_idx_x() as u32,
                        );
                        core::ptr::write_volatile(
                            payload.add(68) as *mut u32,
                            nvptx::_thread_idx_x() as u32,
                        );
                    },
                    BP_WAIT_PRINT_2,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BP_WAIT_PRINT_2 => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, BP_DONE, &mut self.state)
                    .is_some()
                {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            },

            BP_DONE => WarpPoll::Ready(true),

            _ => WarpPoll::Ready(false),
        }
    }
}

/// async-pipeline.3: Branching pipeline — conditional state transitions demo.
///
/// Demonstrates that WarpFuture state machines can branch based on runtime values.
/// Try to open a file → if exists, close+print; if not, create+write+close+print.
/// All 32 lanes take the same branch (state is broadcast from lane 0).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn branching_pipeline(buf: *mut u8, status: *mut u32) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = BranchingPipelineFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// async-pipeline.4: Pipelined I/O + Compute
// ============================================================
//
// Demonstrates overlapping computation with pending I/O.
// Instead of: submit → wait → compute → submit → wait
// We do:      submit → compute_while_waiting → wait
//
// The key insight: a WarpFuture state between SUBMIT and WAIT
// can do arbitrary per-thread work. The WAIT state will eventually
// see CONTROL_READY and proceed.
//
// Pipeline:
//   1. PRINT "pipeline: start" (warm up hostcall path)
//   2. Submit PRINT "pipeline: computing..." (I/O operation)
//   3. While print is in-flight, compute FMA reduction (per-thread)
//   4. Wait for print completion
//   5. PRINT the computed result
//
// This shows that GPU threads can do useful FMA work while a hostcall
// is being processed by the host listener thread.

const PP_PRINT_START: u32 = 0;
const PP_WAIT_START: u32 = 1;
const PP_SUBMIT_COMPUTING: u32 = 2;
const PP_COMPUTE_WHILE_IO: u32 = 3; // Compute happens HERE while I/O is pending
const PP_WAIT_COMPUTING: u32 = 4;
const PP_PRINT_RESULT: u32 = 5;
const PP_WAIT_RESULT: u32 = 6;
const PP_DONE: u32 = 7;

struct PipelinedComputeFuture {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    /// Per-lane FMA result computed while I/O is pending
    fma_result: f32,
    /// Iteration counter for the compute state
    compute_iters: u32,
}

impl PipelinedComputeFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: 0,
            pkt_idx: NULL_INDEX,
            fma_result: 0.0,
            compute_iters: 0,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for PipelinedComputeFuture {
    type Output = bool;

    #[inline(always)]
    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // Step 1: Print start message
            PP_PRINT_START => unsafe {
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_PRINT,
                    |payload| {
                        let msg = b"pipelined: start";
                        core::ptr::write_volatile(payload as *mut u64, msg.len() as u64);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < msg.len() {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                        core::ptr::write_volatile(
                            payload.add(64) as *mut u32,
                            nvptx::_block_idx_x() as u32,
                        );
                        core::ptr::write_volatile(
                            payload.add(68) as *mut u32,
                            nvptx::_thread_idx_x() as u32,
                        );
                    },
                    PP_WAIT_START,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            PP_WAIT_START => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    PP_SUBMIT_COMPUTING,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            // Step 2: Submit a PRINT (this is the I/O operation we overlap with compute)
            PP_SUBMIT_COMPUTING => unsafe {
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_PRINT,
                    |payload| {
                        let msg = b"pipelined: computing...";
                        core::ptr::write_volatile(payload as *mut u64, msg.len() as u64);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < msg.len() {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                        core::ptr::write_volatile(
                            payload.add(64) as *mut u32,
                            nvptx::_block_idx_x() as u32,
                        );
                        core::ptr::write_volatile(
                            payload.add(68) as *mut u32,
                            nvptx::_thread_idx_x() as u32,
                        );
                    },
                    PP_COMPUTE_WHILE_IO,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            // Step 3: COMPUTE while the PRINT I/O is still in-flight.
            // Each lane does FMA iterations. Then we check if I/O completed.
            // If not, we return Pending and come back to compute more.
            PP_COMPUTE_WHILE_IO => unsafe {
                // Per-lane divergent compute: FMA reduction
                // lane_id * 1.5 + 0.5, iterated
                let lid = wcx.lane_id as f32;
                let mut acc = self.fma_result;
                // Do a batch of 100 FMA iterations per poll
                let mut i: u32 = 0;
                while i < 100 {
                    acc = acc * 0.999 + lid * 0.001 + 0.0001;
                    i += 1;
                }
                self.fma_result = acc;
                self.compute_iters += 100;

                // Now check if the I/O completed (non-blocking check)
                let idx = broadcast_u32(wcx.active_mask, self.pkt_idx as u32) as u16;
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let ctrl = sys_spin_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

                if ctrl & CONTROL_READY != 0 {
                    // I/O completed! Release packet and move on.
                    if wcx.is_leader() {
                        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
                        self.state = PP_PRINT_RESULT;
                    }
                    gpu_atomics::syncwarp(wcx.active_mask);
                }
                // If not ready, return Pending → executor will call us again
                // and we'll do more FMA iterations
                WarpPoll::Pending
            },

            // Step 4: Print the compute result
            PP_PRINT_RESULT => unsafe {
                // Broadcast lane 0's compute result + iterations for the message
                let iters = broadcast_u32(wcx.active_mask, self.compute_iters);
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_PRINT,
                    |payload| {
                        // Format: "pipelined: done Niter" (N = iteration count)
                        let prefix = b"pipelined: done ";
                        let mut msg = [0u8; 56];
                        let mut len = 0usize;
                        while len < prefix.len() {
                            msg[len] = prefix[len];
                            len += 1;
                        }
                        // Write iteration count as decimal digits
                        let mut n = iters;
                        if n == 0 {
                            msg[len] = b'0';
                            len += 1;
                        } else {
                            let mut digits = [0u8; 10];
                            let mut dlen = 0;
                            while n > 0 {
                                digits[dlen] = b'0' + (n % 10) as u8;
                                dlen += 1;
                                n /= 10;
                            }
                            let mut j = dlen;
                            while j > 0 {
                                j -= 1;
                                msg[len] = digits[j];
                                len += 1;
                            }
                        }
                        // "iter"
                        let suffix = b"iter";
                        let mut k = 0;
                        while k < suffix.len() && len < 56 {
                            msg[len] = suffix[k];
                            len += 1;
                            k += 1;
                        }
                        core::ptr::write_volatile(payload as *mut u64, len as u64);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < len {
                            core::ptr::write_volatile(dst.add(i), msg[i]);
                            i += 1;
                        }
                        core::ptr::write_volatile(
                            payload.add(64) as *mut u32,
                            nvptx::_block_idx_x() as u32,
                        );
                        core::ptr::write_volatile(
                            payload.add(68) as *mut u32,
                            nvptx::_thread_idx_x() as u32,
                        );
                    },
                    PP_WAIT_RESULT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            PP_WAIT_RESULT => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, PP_DONE, &mut self.state)
                    .is_some()
                {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            },

            PP_DONE => WarpPoll::Ready(true),

            _ => WarpPoll::Ready(false),
        }
    }
}

/// async-pipeline.4: Pipelined I/O + compute demo.
///
/// Shows that GPU threads can do useful FMA computation while a hostcall
/// I/O operation is being processed by the host. The number of compute
/// iterations completed during the I/O round-trip demonstrates the overlap.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn pipelined_compute(buf: *mut u8, status: *mut u32) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = PipelinedComputeFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// Parallel file grep kernel (product.8)
// ============================================================

/// Parallel file grep kernel: each thread opens, reads, and searches a file.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn parallel_grep_kernel(
    buf: *mut u8,
    sideband: *mut u8,
    results: *mut u64,
    pattern_ptr: *const u8,
    pattern_len: u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    let mut pattern_buf = [0u8; 32];
    let plen = (pattern_len as usize).min(32);
    let mut pi: usize = 0;
    while pi < plen {
        pattern_buf[pi] = core::ptr::read_volatile(pattern_ptr.add(pi));
        pi += 1;
    }

    let path = b"gpu_grep_test.txt";
    let (fd, err) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, 0);
    if err != 0 || fd == 0 {
        core::ptr::write_volatile(results.add(tid as usize), 0u64);
        return;
    }

    let mut file_buf = [0u8; 4096];
    let bytes_read =
        gpu_runtime::sideband::gpu_bulk_read(buf, sideband, fd, file_buf.as_mut_ptr(), 4096);

    gpu_hostcall_close(buf, fd);

    let match_count = grep_buffer(
        buf,
        file_buf.as_ptr(),
        bytes_read,
        &pattern_buf[..plen],
        tid,
    );

    core::ptr::write_volatile(results.add(tid as usize), match_count as u64);
}
