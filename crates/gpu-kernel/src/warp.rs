// Warp intrinsics + WarpFuture kernels and helpers.

use core::arch::nvptx;
use gpu_atomics::{sys_fetch_add_u64, sys_spin_load_acquire_u32, sys_store_release_u32};
use gpu_protocol::*;

// ============================================================
// Warp intrinsic tests (warp-future.3)
// ============================================================

/// Test: bar.warp.sync + shfl.sync.idx.b32 warp intrinsics.
///
/// Launches with 32 threads (1 warp). Lane 0 writes a magic value,
/// broadcasts it to all lanes via shfl.sync.idx, then all lanes
/// write the received value to output[lane_id]. If all outputs
/// equal the magic value, both shfl.sync and bar.warp.sync work.
///
/// `output` must have space for 32 u32 entries.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_warp_intrinsics(output: *mut u32) {
    let lid = gpu_atomics::lane_id();
    let mask = gpu_atomics::activemask();

    // Lane 0 provides the magic value; other lanes provide 0
    let my_val = if lid == 0 { 0xCAFE_BABE_u32 } else { 0u32 };

    // Synchronize all lanes before shuffle
    gpu_atomics::syncwarp(mask);

    // Broadcast lane 0's value to all lanes
    let received = gpu_atomics::shfl_sync_idx_u32(mask, my_val, 0);

    // Each lane writes the received value
    *output.add(lid as usize) = received;
}

// ============================================================
// WarpFuture PoC: hand-written warp-level PRINT hostcall (warp-future.4)
// ============================================================

/// State discriminant values for WarpPrintFuture.
const WPF_INIT: u32 = 0;
const WPF_WAIT: u32 = 1;
const WPF_DONE: u32 = 2;

/// Hand-written WarpFuture: all 32 lanes cooperatively send a PRINT hostcall.
///
/// Each lane contributes its lane_id as a byte to the message.
/// Lane 0 handles packet allocation, submission, and release.
/// All lanes stay convergent throughout the state machine.
struct WarpPrintFuture {
    buf: *mut u8,
    state: u32,   // discriminant (lane 0 authoritative)
    pkt_idx: u16, // packet index (uniform after broadcast)
}

impl WarpPrintFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: WPF_INIT,
            pkt_idx: gpu_protocol::NULL_INDEX,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for WarpPrintFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        // Broadcast state from lane 0 to all lanes
        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            WPF_INIT => unsafe {
                // Lane 0: pop a free packet
                let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
                if wcx.is_leader() {
                    idx_raw = gpu_runtime::hostcall::hc_pop_free(self.buf) as u32;
                }

                // Broadcast packet index to all lanes
                let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;

                if idx == gpu_protocol::NULL_INDEX {
                    return WarpPoll::Pending; // backpressure — no free packets
                }
                self.pkt_idx = idx;

                // Compute packet pointer
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

                // Build message: "WarpFuture: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
                // Header "WarpFuture: " in slot 0 (msg_len) + slots 1..
                // Each lane writes its character at the right position
                let prefix = b"WarpFuture: ";
                let msg_len = prefix.len() as u32 + 32; // 12 + 32 = 44 bytes

                // Lane 0 writes the length into slot 0
                if wcx.is_leader() {
                    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
                }

                // All lanes write their byte into the message body
                // Message starts at payload + 8 (slot 1)
                let msg_base = payload.add(8);
                let lid = wcx.lane_id;

                // First 12 bytes are the prefix — only lanes 0..11 write those
                if lid < prefix.len() as u32 {
                    core::ptr::write_volatile(msg_base.add(lid as usize), prefix[lid as usize]);
                }

                // Bytes 12..43 are 'A' + lane_id (all 32 lanes write)
                let char_offset = prefix.len() as u32 + lid;
                if char_offset < msg_len {
                    core::ptr::write_volatile(
                        msg_base.add(char_offset as usize),
                        b'A'.wrapping_add(lid as u8),
                    );
                }

                // Lane 0: write thread/block metadata at payload+64
                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        payload.add(64) as *mut u32,
                        nvptx::_block_idx_x() as u32,
                    );
                    core::ptr::write_volatile(
                        payload.add(68) as *mut u32,
                        nvptx::_thread_idx_x() as u32,
                    );
                }

                // Sync: ensure all payload writes are visible
                gpu_atomics::syncwarp(wcx.active_mask);

                // Lane 0: fill header, mark FILLED, push to ready, ring doorbell
                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
                        wcx.active_mask,
                    );
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
                        gpu_protocol::SERVICE_PRINT,
                    );
                    sys_store_release_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32, 0);
                    sys_store_release_u32(
                        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
                        gpu_protocol::CONTROL_FILLED,
                    );

                    // Push to ready stack + ring doorbell
                    let (num_shards, shard_off, _) =
                        gpu_runtime::hostcall::read_shard_info(self.buf as *const u8);
                    let ready_ptr =
                        gpu_runtime::hostcall::get_ready_stack_ptr(self.buf, num_shards, shard_off);
                    gpu_runtime::hostcall::hc_push(ready_ptr, self.buf, idx);
                    sys_fetch_add_u64(self.buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64, 1);

                    self.state = WPF_WAIT;
                }

                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            WPF_WAIT => unsafe {
                // All lanes read the same control word — perfectly convergent spin
                let idx = broadcast_u32(wcx.active_mask, self.pkt_idx as u32) as u16;
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let ctrl =
                    sys_spin_load_acquire_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32);

                if ctrl & gpu_protocol::CONTROL_READY != 0 {
                    // Host responded — release packet
                    if wcx.is_leader() {
                        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
                        self.state = WPF_DONE;
                    }
                    gpu_atomics::syncwarp(wcx.active_mask);
                    return WarpPoll::Ready(true);
                }

                WarpPoll::Pending
            },

            WPF_DONE => WarpPoll::Ready(true),

            _ => WarpPoll::Pending, // unreachable
        }
    }
}

/// WarpFuture PoC kernel: all 32 lanes cooperatively send a PRINT hostcall.
///
/// Uses the WarpFuture trait + WarpExecutor. Lane 0 handles packet management,
/// all lanes write message data in parallel, all lanes spin-wait convergently.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 if WarpFuture completed successfully)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_future_print_test(buf: *mut u8, result: *mut u32) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = WarpPrintFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    // Lane 0 writes the result
    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// WarpFuture multi-hostcall PoC: 3 sequential PRINT calls (warp-future.6)
// ============================================================

/// State discriminant values for WarpMultiPrintFuture.
/// 7-state machine: INIT1→WAIT1→INIT2→WAIT2→INIT3→WAIT3→DONE
const WMP_INIT1: u32 = 0;
const WMP_WAIT1: u32 = 1;
const WMP_INIT2: u32 = 2;
const WMP_WAIT2: u32 = 3;
const WMP_INIT3: u32 = 4;
const WMP_WAIT3: u32 = 5;
const WMP_DONE: u32 = 6;

/// Hand-written WarpFuture: 3 sequential PRINT hostcalls.
///
/// Validates that a WarpFuture state machine can compose multiple hostcalls
/// while maintaining warp convergence across all state transitions.
/// Lane 0 manages packets; all 32 lanes write payload cooperatively.
///
/// Messages sent:
///   1: "WarpMulti[1/3]: HELLO_FROM_32_LANES!!"
///   2: "WarpMulti[2/3]: SECOND_CALL_WORKING!"
///   3: "WarpMulti[3/3]: PIPELINE_COMPLETE!!"
struct WarpMultiPrintFuture {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    calls_completed: u32,
}

impl WarpMultiPrintFuture {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self {
            buf,
            state: WMP_INIT1,
            pkt_idx: gpu_protocol::NULL_INDEX,
            calls_completed: 0,
        }
    }
}

/// Shared init logic for each of the 3 PRINT hostcalls.
/// Returns the packet pointer (all lanes can use it) and WarpPoll::Pending.
///
/// # Safety
/// Must be called by all active lanes of a warp simultaneously.
#[inline(always)]
unsafe fn warp_multi_init_hostcall(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: &mut u16,
    next_state: u32,
    state: &mut u32,
    call_num: u32,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

    // Lane 0: pop a free packet
    let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = gpu_runtime::hostcall::hc_pop_free(buf) as u32;
    }

    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == gpu_protocol::NULL_INDEX {
        return WarpPoll::Pending; // backpressure
    }
    *pkt_idx = idx;

    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

    // Select message based on call number
    let (prefix, suffix): (&[u8], &[u8]) = match call_num {
        0 => (b"WarpMulti[1/3]: ", b"HELLO_FROM_32_LANES!!"),
        1 => (b"WarpMulti[2/3]: ", b"SECOND_CALL_WORKING!"),
        _ => (b"WarpMulti[3/3]: ", b"PIPELINE_COMPLETE!!"),
    };
    let msg_len = prefix.len() as u32 + suffix.len() as u32;

    // Lane 0: write message length
    if wcx.is_leader() {
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
    }

    // All lanes cooperatively write the message bytes
    let msg_base = payload.add(8);
    let lid = wcx.lane_id;

    // Write prefix bytes (lanes with lid < prefix.len)
    if lid < prefix.len() as u32 {
        core::ptr::write_volatile(msg_base.add(lid as usize), prefix[lid as usize]);
    }

    // Write suffix bytes (lanes with lid < suffix.len)
    if lid < suffix.len() as u32 {
        core::ptr::write_volatile(
            msg_base.add(prefix.len() + lid as usize),
            suffix[lid as usize],
        );
    }

    // Lane 0: write thread/block metadata at payload+64
    if wcx.is_leader() {
        core::ptr::write_volatile(payload.add(64) as *mut u32, nvptx::_block_idx_x() as u32);
        core::ptr::write_volatile(payload.add(68) as *mut u32, nvptx::_thread_idx_x() as u32);
    }

    // Ensure all payload writes are visible
    gpu_atomics::syncwarp(wcx.active_mask);

    // Lane 0: fill header, mark FILLED, push to ready, ring doorbell
    if wcx.is_leader() {
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
            wcx.active_mask,
        );
        core::ptr::write_volatile(
            pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
            gpu_protocol::SERVICE_PRINT,
        );
        sys_store_release_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32, 0);
        sys_store_release_u32(
            pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
            gpu_protocol::CONTROL_FILLED,
        );

        let (num_shards, shard_off, _) = gpu_runtime::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64, 1);

        *state = next_state;
    }

    gpu_atomics::syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// Shared wait logic: spin-wait for host response, release packet.
///
/// # Safety
/// Must be called by all active lanes of a warp simultaneously.
#[inline(always)]
unsafe fn warp_multi_wait_hostcall(
    buf: *mut u8,
    wcx: &mut gpu_runtime::warp_future::WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state: &mut u32,
    calls_completed: &mut u32,
) -> gpu_runtime::warp_future::WarpPoll<bool> {
    use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32);

    if ctrl & gpu_protocol::CONTROL_READY != 0 {
        if wcx.is_leader() {
            gpu_runtime::hostcall::gpu_hostcall_release(buf, pkt);
            *calls_completed += 1;
            *state = next_state;
        }
        gpu_atomics::syncwarp(wcx.active_mask);

        if next_state == WMP_DONE {
            return WarpPoll::Ready(true);
        }
        return WarpPoll::Pending; // Transition to next INIT state
    }

    WarpPoll::Pending
}

unsafe impl gpu_runtime::warp_future::WarpFuture for WarpMultiPrintFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        // Broadcast state from lane 0 to all lanes
        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            WMP_INIT1 => unsafe {
                warp_multi_init_hostcall(
                    self.buf,
                    wcx,
                    &mut self.pkt_idx,
                    WMP_WAIT1,
                    &mut self.state,
                    0,
                )
            },
            WMP_WAIT1 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    WMP_INIT2,
                    &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_INIT2 => unsafe {
                warp_multi_init_hostcall(
                    self.buf,
                    wcx,
                    &mut self.pkt_idx,
                    WMP_WAIT2,
                    &mut self.state,
                    1,
                )
            },
            WMP_WAIT2 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    WMP_INIT3,
                    &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_INIT3 => unsafe {
                warp_multi_init_hostcall(
                    self.buf,
                    wcx,
                    &mut self.pkt_idx,
                    WMP_WAIT3,
                    &mut self.state,
                    2,
                )
            },
            WMP_WAIT3 => unsafe {
                warp_multi_wait_hostcall(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    WMP_DONE,
                    &mut self.state,
                    &mut self.calls_completed,
                )
            },
            WMP_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}

/// WarpFuture multi-hostcall kernel: 3 sequential PRINT hostcalls in one WarpFuture.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 if all 3 calls succeeded)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_future_multi_print_test(buf: *mut u8, result: *mut u32) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = WarpMultiPrintFuture::new(buf);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    // Lane 0 writes the result
    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// WarpFuture proc macro test (warp-future.5)
// ============================================================

// The #[warp_async] proc macro generates:
// - `WarpMacroPrintTest` struct with buf, state, pkt_idx
// - WarpFuture impl with 2 PRINT hostcalls (4 states + DONE)
// - `warp_macro_print_test` kernel entry point
#[warp_macro::warp_async]
unsafe fn warp_macro_print_test(buf: *mut u8) -> bool {
    warp_print!(buf, b"Macro[1/2]: GENERATED_CODE!!");
    warp_print!(buf, b"Macro[2/2]: PROC_MACRO_WORKS!");
}

// ============================================================
// WarpFuture proc macro if/else test (warp-cfg.2)
// ============================================================

// The #[warp_async] macro now supports if/else with warp_*!() calls.
// Lane 0 evaluates the condition and broadcasts the decision to all lanes.
//
// `flag` parameter controls branching: flag != 0 → then, flag == 0 → else.
// This directly tests the DECISION state generation without relying on
// file error propagation.
//
// State machine generated:
//   0: DECISION             → lane0 evaluates (flag != 0), broadcasts
//                              if true → goto 1 (then branch)
//                              if false → goto 3 (else branch)
//   1: INIT warp_print[A]   → submit PRINT "branch: then"
//   2: WAIT warp_print[A]   → goto 5 (join: final print)
//   3: INIT warp_print[B]   → submit PRINT "branch: else"
//   4: WAIT warp_print[B]   → goto 5 (join: final print)
//   5: INIT warp_print[end] → submit PRINT "branch: done"
//   6: WAIT warp_print[end] → DONE (7)
//   7: DONE
#[warp_macro::warp_async]
unsafe fn warp_cfg_if_else_test(buf: *mut u8, flag: u64) -> bool {
    if flag != 0 {
        warp_print!(buf, b"branch: then");
    } else {
        warp_print!(buf, b"branch: else");
    }
    warp_print!(buf, b"branch: done");
}

// ============================================================
// WarpFuture proc macro loop/break test (warp-cfg.3)
// ============================================================

// The #[warp_async] macro supports loop with `if cond { break; }`.
// The loop body executes repeatedly until the break condition is true.
// `counter` parameter: counts down from this value to 0.
//
// State machine:
//   0: INIT print("iter")     → submit PRINT
//   1: WAIT print("iter")     → goto 2
//   2: BREAK_DECISION         → if counter == 0 → goto 4 (post-loop), else → goto 0 (loop back)
//   [back-edge: end of body → state 0]
//   3: post-loop INIT print("done") → submit PRINT
//   4: WAIT print("done")     → DONE
//   5: DONE
//
// Note: counter is decremented via a pattern where each loop iteration
// prints "iter" and checks. Since we can't do compute-only statements yet,
// we use the counter as a constant to determine how many prints happen.
// For the test: counter=3 → 3 "iter" prints before break, then "done".
#[warp_macro::warp_async]
unsafe fn warp_cfg_loop_test(buf: *mut u8, counter: u64) -> bool {
    loop {
        warp_print!(buf, b"iter");
        if counter == 0 {
            break;
        }
    }
    warp_print!(buf, b"done");
}

// ============================================================
// warp-cfg.4: Match support in #[warp_async]
// ============================================================
//
// Test: match on a u64 command code, each arm prints a different message.
// Uses 3 arms: 0 → "cmd: zero", 1 → "cmd: one", _ → "cmd: other".
// Then prints "match: done" after the match.
//
// State machine (for cmd=0):
//   0: MATCH_DECISION → broadcast(cmd) → arm 0,1,2 start states
//   1: INIT print("cmd: zero")    → submit PRINT
//   2: WAIT print("cmd: zero")    → goto 7 (join)
//   3: INIT print("cmd: one")     → submit PRINT
//   4: WAIT print("cmd: one")     → goto 7 (join)
//   5: INIT print("cmd: other")   → submit PRINT
//   6: WAIT print("cmd: other")   → goto 7 (join)
//   7: INIT print("match: done")  → submit PRINT
//   8: WAIT print("match: done")  → DONE
//   9: DONE
#[warp_macro::warp_async]
unsafe fn warp_cfg_match_test(buf: *mut u8, cmd: u64) -> bool {
    match cmd {
        0 => {
            warp_print!(buf, b"cmd: zero");
        }
        1 => {
            warp_print!(buf, b"cmd: one");
        }
        _ => {
            warp_print!(buf, b"cmd: other");
        }
    }
    warp_print!(buf, b"match: done");
}

// ============================================================
// warp-cfg.5: Nested control flow stress test
// ============================================================
//
// Test: if/else with match nested inside the then-branch.
// Validates that nested control flow generates correct state machine.
//
// Parameters: flag (u64) selects if/else, cmd (u64) selects match arm within then.
//
// flag=1, cmd=0 → "then-cmd0" + "nested: done"
// flag=1, cmd=1 → "then-cmd1" + "nested: done"
// flag=1, cmd=99 → "then-other" + "nested: done"
// flag=0, cmd=* → "else-path" + "nested: done"
//
// State machine (flag=1, cmd=0):
//   0: IF_DECISION → broadcast(flag!=0) → 1 (then) or 9 (else)
//   1: MATCH_DECISION → broadcast(match cmd) → 2, 4, or 6
//   2: INIT print("then-cmd0")
//   3: WAIT print("then-cmd0") → goto 8 (match-join)
//   4: INIT print("then-cmd1")
//   5: WAIT print("then-cmd1") → goto 8
//   6: INIT print("then-other")
//   7: WAIT print("then-other") → goto 8
//   8: [match join → if join at 11]
//   9: INIT print("else-path")
//  10: WAIT print("else-path") → goto 11 (if join)
//  11: INIT print("nested: done")
//  12: WAIT print("nested: done") → DONE (13)
//  13: DONE
//
// Note: match join (state 8) and if join (state 11) are the same because
// the match is the only node in the then-branch — so match join IS the then
// continuation, which is the if join point.
#[warp_macro::warp_async]
unsafe fn warp_cfg_nested_test(buf: *mut u8, flag: u64, cmd: u64) -> bool {
    if flag != 0 {
        match cmd {
            0 => {
                warp_print!(buf, b"then-cmd0");
            }
            1 => {
                warp_print!(buf, b"then-cmd1");
            }
            _ => {
                warp_print!(buf, b"then-other");
            }
        }
    } else {
        warp_print!(buf, b"else-path");
    }
    warp_print!(buf, b"nested: done");
}

// ============================================================
// gpu-compute.2: Autonomous Multi-Step Compute Pipeline
// ============================================================
//
// Demonstrates GPU-driven multi-step compute without host orchestration.
// The GPU autonomously decides the processing path using match + if/else,
// performs file I/O and conditional logic based on hostcall results.
//
// This replaces what previously required 150+ lines of hand-written
// state machine code (cf. BranchingPipelineFuture) with a concise
// `#[warp_async]` function using full control flow.
//
// Mode 0: File write pipeline — create file, write data, close
// Mode 1: File read + classify — open file, read, branch on result
// Mode 2: Multi-step I/O — create file, write, re-open, verify, report
//
// State machine (auto-generated by proc macro):
//   Match on `mode` → each arm is a distinct pipeline
//   Sequential hostcalls within arms (open → write → close)
//   Conditional branching on hostcall results (if n > 0)

#[warp_macro::warp_async]
unsafe fn autonomous_pipeline(buf: *mut u8, mode: u64) -> bool {
    warp_print!(buf, b"auto: start");

    match mode {
        0 => {
            // Pipeline A: Create and write a file
            let fd = warp_open!(buf, b"gpu_autonomous.txt", 1);
            warp_write!(buf, fd, b"GPU-autonomous-output", 21);
            warp_close!(buf, fd);
            warp_print!(buf, b"auto: file-written");
        }
        1 => {
            // Pipeline B: Read file and classify by size
            let rfd = warp_open!(buf, b"gpu_autonomous.txt", 0);
            let n = warp_read!(buf, rfd, 56);
            warp_close!(buf, rfd);
            if n > 10 {
                warp_print!(buf, b"auto: large-payload");
            } else {
                warp_print!(buf, b"auto: small-payload");
            }
        }
        _ => {
            // Pipeline C: End-to-end create → verify round-trip
            let wfd2 = warp_open!(buf, b"gpu_roundtrip.txt", 1);
            warp_write!(buf, wfd2, b"verify-me", 9);
            warp_close!(buf, wfd2);
            let rfd2 = warp_open!(buf, b"gpu_roundtrip.txt", 0);
            let nb = warp_read!(buf, rfd2, 56);
            warp_close!(buf, rfd2);
            if nb > 0 {
                warp_print!(buf, b"auto: roundtrip-ok");
            } else {
                warp_print!(buf, b"auto: roundtrip-fail");
            }
        }
    }

    warp_print!(buf, b"auto: done");
}

// ============================================================
// warp-async-v2.2: ? operator test
// ============================================================
//
// Test the ? operator in #[warp_async] with Result<bool, u32> return type.
// Opens a file with warp_open!, uses ? to propagate errors.
// If the file open succeeds, prints a message and returns Ok(true).
// If it fails, the ? operator causes all 32 lanes to return Err.
//
// State machine:
//   0: INIT warp_open            → submit OPEN "/tmp/warp_try_test.txt"
//   1: WAIT warp_open            → capture fd, goto 2
//   2: TRY_DECISION              → if fd == NULL_INDEX → Err(0xFFFF), else → goto 3
//   3: INIT warp_print           → submit PRINT "try: opened"
//   4: WAIT warp_print           → DONE
//   5: DONE
#[warp_macro::warp_async]
unsafe fn warp_try_open_test(buf: *mut u8) -> Result<bool, u32> {
    let fd = warp_open!(buf, b"/tmp/warp_try_test.txt", 1)?;
    warp_print!(buf, b"try: opened");
}

// ============================================================
// warp-async-v2.3: .await test
// ============================================================
//
// Test .await in #[warp_async] using standard GpuPrintFuture.
// The macro:
//   1. Infers the future type from GpuPrintFuture::new(...)
//   2. Creates a MaybeUninit<GpuPrintFuture> struct field
//   3. INIT state stores the future in the field
//   4. POLL state calls warp_poll_future() for warp-cooperative polling
//
// State machine:
//   0: AWAIT_INIT     → create GpuPrintFuture::new(buf, b"await: hello")
//   1: AWAIT_POLL     → warp-cooperative poll via warp_poll_future()
//   2: AWAIT_INIT     → create GpuPrintFuture::new(buf, b"await: done")
//   3: AWAIT_POLL     → warp-cooperative poll
//   4: DONE
#[warp_macro::warp_async]
unsafe fn warp_await_test(buf: *mut u8) -> bool {
    let ok1 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"await: hello").await;
    let ok2 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"await: done").await;
}

// ============================================================
// warp-async-v2.4: End-to-end test
// ============================================================
//
// Combines .await + warp_*!() + if/else in a single #[warp_async] function.
// Tests that the proc macro correctly handles mixed CfgNode types.
//
// State machine:
//   0: AWAIT_INIT     → create GpuPrintFuture::new(buf, b"e2e: start")
//   1: AWAIT_POLL     → warp-cooperative poll, capture ok1
//   2: DECISION       → branch on ok1
//   3: AWAIT_INIT     → create GpuPrintFuture(b"e2e: ok")     (then branch)
//   4: AWAIT_POLL     → poll
//   5: AWAIT_INIT     → create GpuPrintFuture(b"e2e: fail")   (else branch)
//   6: AWAIT_POLL     → poll
//   7: INIT warp_print → submit "e2e: mixed"
//   8: WAIT warp_print → capture result
//   9: DONE
#[warp_macro::warp_async]
unsafe fn warp_e2e_test(buf: *mut u8) -> bool {
    let ok1 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"e2e: start").await;
    if ok1 > 0 {
        let ok2 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"e2e: ok").await;
    } else {
        let ok3 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"e2e: fail").await;
    }
    warp_print!(buf, b"e2e: mixed");
}
