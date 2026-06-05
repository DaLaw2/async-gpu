// Hybrid executor kernels — WarpFuture + per-thread compute block.

use core::arch::nvptx;
use gpu_atomics::{sys_fetch_add_u64, sys_spin_load_acquire_u32, sys_store_release_u32};

// ============================================================
// Shared helpers for hybrid kernels
// ============================================================

/// Helper: warp-cooperative PRINT init — pops packet, writes message, submits.
/// Returns (WarpPoll::Pending, pkt_idx) on success, or Pending with NULL_INDEX on backpressure.
#[inline(always)]
pub(crate) unsafe fn hybrid_warp_print_init(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    msg: &[u8],
    next_state: u32,
    state_cell: &mut u32,
    pkt_idx_cell: &mut u16,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

    let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = gpu_runtime::hostcall::hc_pop_free(buf) as u32;
    }
    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == gpu_protocol::NULL_INDEX {
        return WarpPoll::Pending;
    }
    *pkt_idx_cell = idx;

    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);
    let msg_len = msg.len() as u32;

    if wcx.is_leader() {
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
    }

    // Cooperative write: all lanes write first 32 bytes
    let msg_base = payload.add(8);
    let lid = wcx.lane_id;
    if lid < msg_len && lid < 32 {
        core::ptr::write_volatile(msg_base.add(lid as usize), msg[lid as usize]);
    }
    // Lane 0 writes remaining bytes
    if wcx.is_leader() && msg_len > 32 {
        let mut j: u32 = 32;
        while j < msg_len {
            core::ptr::write_volatile(msg_base.add(j as usize), msg[j as usize]);
            j += 1;
        }
    }

    // Metadata
    if wcx.is_leader() {
        core::ptr::write_volatile(payload.add(64) as *mut u32, nvptx::_block_idx_x() as u32);
        core::ptr::write_volatile(payload.add(68) as *mut u32, nvptx::_thread_idx_x() as u32);
    }

    gpu_atomics::syncwarp(wcx.active_mask);

    if wcx.is_leader() {
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
            wcx.active_mask,
        );
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
            gpu_protocol::SERVICE_PRINT,
        );
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            gpu_protocol::CONTROL_FILLED,
        );
        let (num_shards, shard_off, _) = gpu_runtime::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64, 1);
        *state_cell = next_state;
    }

    gpu_atomics::syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// Helper: warp-cooperative WAIT — spin on control word, release packet on READY.
#[inline(always)]
pub(crate) unsafe fn hybrid_warp_wait(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state_cell: &mut u32,
) -> Option<()> {
    use gpu_runtime::warp_future::broadcast_u32;

    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32);

    if ctrl & gpu_protocol::CONTROL_READY != 0 {
        if wcx.is_leader() {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            *state_cell = next_state;
        }
        gpu_atomics::syncwarp(wcx.active_mask);
        Some(())
    } else {
        None
    }
}

// ============================================================
// hybrid-executor.1: WarpFuture + per-thread compute block PoC
// ============================================================
//
// Demonstrates mixing warp-cooperative I/O (WarpFuture) with
// per-thread divergent computation in the same state machine.
//
// State machine:
//   0: INIT_PRINT  - warp-cooperative PRINT "hybrid: start"
//   1: WAIT_PRINT  - wait for host response
//   2: COMPUTE     - per-thread block: each lane computes results[lane_id] = lane_id^2 + 1
//   3: INIT_PRINT2 - warp-cooperative PRINT "hybrid: done"
//   4: WAIT_PRINT2 - wait for host response
//   5: DONE        - return true

const HYB_INIT_PRINT: u32 = 0;
const HYB_WAIT_PRINT: u32 = 1;
const HYB_COMPUTE: u32 = 2;
const HYB_INIT_PRINT2: u32 = 3;
const HYB_WAIT_PRINT2: u32 = 4;
const HYB_DONE: u32 = 5;

struct HybridFuture {
    buf: *mut u8,
    results: *mut u32,
    state: u32,
    pkt_idx: u16,
}

impl HybridFuture {
    #[inline(always)]
    fn new(buf: *mut u8, results: *mut u32) -> Self {
        Self {
            buf,
            results,
            state: HYB_INIT_PRINT,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for HybridFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === WarpFuture I/O: cooperative PRINT "hybrid: start" ===
            HYB_INIT_PRINT => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"hybrid: start",
                    HYB_WAIT_PRINT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            HYB_WAIT_PRINT => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, HYB_COMPUTE, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Pending // continue to next state on next poll
                } else {
                    WarpPoll::Pending
                }
            },

            // === Per-thread compute block ===
            // All lanes enter together (state is broadcast), but each lane
            // computes independently. syncwarp() at exit ensures reconvergence
            // before transitioning back to WarpFuture I/O.
            HYB_COMPUTE => unsafe {
                let lid = wcx.lane_id;

                // --- Per-thread divergent computation ---
                // Each lane computes a different value: lane_id^2 + 1
                // In a real workload, this could be any per-lane logic with
                // different iteration counts, branches, etc.
                let value = lid * lid + 1;

                // Each lane writes its result independently
                core::ptr::write_volatile(self.results.add(lid as usize), value);

                // --- End per-thread block ---
                // syncwarp: reconverge all lanes before returning to WarpFuture mode
                gpu_atomics::syncwarp(wcx.active_mask);

                // Lane 0 transitions state
                if wcx.is_leader() {
                    self.state = HYB_INIT_PRINT2;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === WarpFuture I/O: cooperative PRINT "hybrid: done" ===
            HYB_INIT_PRINT2 => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"hybrid: done",
                    HYB_WAIT_PRINT2,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            HYB_WAIT_PRINT2 => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, HYB_DONE, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            HYB_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// hybrid-executor.1 kernel: WarpFuture PRINT → per-thread compute → WarpFuture PRINT
///
/// `buf` = hostcall buffer
/// `results` = output u32[32] array (one per lane)
/// `status` = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn hybrid_executor_test(
    buf: *mut u8,
    results: *mut u32,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = HybridFuture::new(buf, results);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// hybrid-executor.2: Variable-duration + multi-switch stress test
// ============================================================
//
// 3 I/O phases + 2 compute blocks, testing:
// - Variable-duration per-thread work (lane_id-dependent iteration count)
// - Multiple switching points in one state machine
// - 11-state machine: INIT1→WAIT1→COMPUTE1→INIT2→WAIT2→COMPUTE2→INIT3→WAIT3→DONE
//
// COMPUTE1: sum 1..=(lane_id*100+1), ~100x duration variance across lanes
// COMPUTE2: XOR-fold lane_id-dependent seed, different duration per lane

const HYB2_INIT1: u32 = 0;
const HYB2_WAIT1: u32 = 1;
const HYB2_COMPUTE1: u32 = 2;
const HYB2_INIT2: u32 = 3;
const HYB2_WAIT2: u32 = 4;
const HYB2_COMPUTE2: u32 = 5;
const HYB2_INIT3: u32 = 6;
const HYB2_WAIT3: u32 = 7;
const HYB2_DONE: u32 = 8;

struct HybridStressFuture {
    buf: *mut u8,
    results: *mut u32,
    state: u32,
    pkt_idx: u16,
}

impl HybridStressFuture {
    #[inline(always)]
    fn new(buf: *mut u8, results: *mut u32) -> Self {
        Self {
            buf,
            results,
            state: HYB2_INIT1,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for HybridStressFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // === Phase 1: WarpFuture PRINT "stress: phase1" ===
            HYB2_INIT1 => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"stress: phase1",
                    HYB2_WAIT1,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },
            HYB2_WAIT1 => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, HYB2_COMPUTE1, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Pending
                } else {
                    WarpPoll::Pending
                }
            },

            // === COMPUTE1: Variable-duration sum ===
            // Each lane sums 1..=(lane_id*100+1)
            // Lane 0: 1 iteration, Lane 31: 3101 iterations (~3100x variance)
            HYB2_COMPUTE1 => unsafe {
                let lid = wcx.lane_id;
                let iters = lid * 100 + 1;
                let mut sum: u32 = 0;
                let mut i: u32 = 1;
                while i <= iters {
                    sum = sum.wrapping_add(i);
                    i += 1;
                }
                // Write result: results[lane_id]
                core::ptr::write_volatile(self.results.add(lid as usize), sum);

                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = HYB2_INIT2;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Phase 2: WarpFuture PRINT "stress: phase2" ===
            HYB2_INIT2 => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"stress: phase2",
                    HYB2_WAIT2,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },
            HYB2_WAIT2 => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, HYB2_COMPUTE2, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Pending
                } else {
                    WarpPoll::Pending
                }
            },

            // === COMPUTE2: XOR-fold with lane-dependent iteration count ===
            // Each lane XOR-folds a different seed for (lane_id+1)*50 iterations
            HYB2_COMPUTE2 => unsafe {
                let lid = wcx.lane_id;
                let iters = (lid + 1) * 50;
                let mut val: u32 = 0xDEAD_0000 | lid;
                let mut i: u32 = 0;
                while i < iters {
                    val ^= val << 13;
                    val ^= val >> 17;
                    val ^= val << 5;
                    i += 1;
                }
                // Write result: results[32 + lane_id]
                core::ptr::write_volatile(self.results.add(32 + lid as usize), val);

                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = HYB2_INIT3;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // === Phase 3: WarpFuture PRINT "stress: phase3" ===
            HYB2_INIT3 => unsafe {
                hybrid_warp_print_init(
                    self.buf,
                    wcx,
                    b"stress: phase3",
                    HYB2_WAIT3,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },
            HYB2_WAIT3 => unsafe {
                if hybrid_warp_wait(self.buf, wcx, self.pkt_idx, HYB2_DONE, &mut self.state)
                    .is_some()
                {
                    WarpPoll::Ready(true)
                } else {
                    WarpPoll::Pending
                }
            },

            HYB2_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// hybrid-executor.2 kernel: stress test with variable-duration per-thread compute + multi-switch
///
/// `buf` = hostcall buffer
/// `results` = output u32[64] array (32 per compute phase)
/// `status` = output u32 (1 = success)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn hybrid_stress_test(
    buf: *mut u8,
    results: *mut u32,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = HybridStressFuture::new(buf, results);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}
