use gpu_atomics::{
    activemask, lane_id, shfl_sync_idx_u32, syncwarp, sys_fetch_add_u64, sys_spin_load_acquire_u32,
    sys_store_release_u32,
};
use gpu_protocol::*;

/// Result of polling a warp-level future.
pub enum WarpPoll<T> {
    /// All lanes completed. Output is per-lane.
    Ready(T),
    /// Warp yielded — will be re-polled.
    Pending,
}

/// Context passed to WarpFuture::poll_warp.
///
/// Contains warp metadata needed during polling. Unlike `core::task::Context`,
/// there is no Waker — warp futures use synchronous spin-poll driven by
/// the WarpExecutor.
pub struct WarpContext {
    /// Active lane mask (from `activemask.b32`)
    pub active_mask: u32,
    /// This lane's ID (0..31)
    pub lane_id: u32,
}

impl WarpContext {
    /// Create a new WarpContext by reading hardware registers.
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self {
            active_mask: activemask(),
            lane_id: lane_id(),
        }
    }

    /// Returns true if this is lane 0 (the "leader" lane).
    #[inline(always)]
    pub fn is_leader(&self) -> bool {
        self.lane_id == 0
    }
}

/// A future representing an entire warp (32 lanes) in SIMT lockstep.
///
/// # Contract
/// - All active lanes must call `poll_warp()` simultaneously.
/// - The state discriminant must be uniform across all lanes
///   (broadcast via `shfl.sync.idx.b32` from lane 0).
/// - Divergent control flow within `poll_warp()` is forbidden —
///   all lanes must execute the same code path with different data.
///
/// # Safety
/// Implementing this trait requires maintaining warp convergence.
/// Breaking convergence causes deadlock or incorrect results.
pub unsafe trait WarpFuture {
    /// Per-lane output type.
    type Output;

    /// Poll the warp future. Called by all active lanes simultaneously.
    fn poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<Self::Output>;
}

/// Minimal warp-level executor.
///
/// Polls a single WarpFuture in a loop until completion. All active lanes
/// participate in every poll. No run queue, no waker — just spin-poll
/// with `nanosleep` yield between iterations.
pub struct WarpExecutor;

impl WarpExecutor {
    /// Run a WarpFuture to completion. All active lanes must call this.
    ///
    /// Returns the per-lane output value.
    ///
    /// # Safety
    /// Must be called by all active lanes of a warp simultaneously.
    #[inline(always)]
    pub unsafe fn run<F: WarpFuture>(future: &mut F) -> F::Output {
        let mut wcx = WarpContext::new();
        let mut polls: u32 = 0;
        const MAX_POLLS: u32 = 10_000_000;

        loop {
            match future.poll_warp(&mut wcx) {
                WarpPoll::Ready(output) => return output,
                WarpPoll::Pending => {
                    polls += 1;
                    if polls >= MAX_POLLS {
                        // Timeout — trap to avoid infinite loop
                        #[cfg(target_arch = "nvptx64")]
                        core::arch::asm!("trap;", options(noreturn));
                        #[cfg(not(target_arch = "nvptx64"))]
                        panic!("WarpExecutor timeout");
                    }
                    // Yield warp scheduler slot
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
            // Ensure convergence before next poll
            syncwarp(wcx.active_mask);
        }
    }
}

/// Broadcast a u32 from lane 0 to all lanes. Convenience wrapper.
#[inline(always)]
pub unsafe fn broadcast_u32(mask: u32, val: u32) -> u32 {
    shfl_sync_idx_u32(mask, val, 0)
}

/// Warp-cooperative hostcall submit: pop a free packet, fill payload, push to
/// ready stack, and ring the doorbell. Only lane 0 performs actual memory ops;
/// all lanes participate in broadcasts to maintain warp convergence.
///
/// Returns `WarpPoll::Pending` always — the caller must transition to a WAIT
/// state to collect the response.
///
/// # Arguments
/// * `buf` — hostcall buffer base pointer
/// * `wcx` — warp context (active mask + lane ID)
/// * `service` — service ID (e.g., `SERVICE_OPEN`, `SERVICE_PRINT`)
/// * `fill_payload` — closure called on lane 0 to fill the packet payload
/// * `next_state` — state value to transition to after submit
/// * `state_cell` — mutable reference to the state machine's state field
/// * `pkt_idx_cell` — mutable reference to store the allocated packet index
#[inline(always)]
pub unsafe fn warp_hostcall_submit(
    buf: *mut u8,
    wcx: &mut WarpContext,
    service: u32,
    fill_payload: impl FnOnce(*mut u8),
    next_state: u32,
    state_cell: &mut u32,
    pkt_idx_cell: &mut u16,
) -> WarpPoll<bool> {
    let mut idx_raw: u32 = NULL_INDEX as u32;
    if wcx.is_leader() {
        idx_raw = crate::hostcall::hc_pop_free(buf) as u32;
    }
    let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
    if idx == NULL_INDEX {
        return WarpPoll::Pending;
    }
    *pkt_idx_cell = idx;

    let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let payload = pkt.add(PKT_OFF_PAYLOAD);

    // Only lane 0 fills the payload
    if wcx.is_leader() {
        fill_payload(payload);
    }

    syncwarp(wcx.active_mask);

    if wcx.is_leader() {
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, wcx.active_mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
        let (num_shards, shard_off, _) = crate::hostcall::read_shard_info(buf as *const u8);
        let ready_ptr = crate::hostcall::get_ready_stack_ptr(buf, num_shards, shard_off);
        crate::hostcall::hc_push(ready_ptr, buf, idx);
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);
        *state_cell = next_state;
    }

    syncwarp(wcx.active_mask);
    WarpPoll::Pending
}

/// Warp-cooperative hostcall wait: poll the control word of a previously
/// submitted packet. Returns `Some(u64)` with the first payload slot when
/// the host has responded, or `None` if still pending.
///
/// On completion, releases the packet back to the free pool and transitions
/// to `next_state`. The u64 return value is broadcast to all lanes via
/// two `shfl.sync.idx.b32` operations (low + high halves).
///
/// # Arguments
/// * `buf` — hostcall buffer base pointer
/// * `wcx` — warp context
/// * `pkt_idx` — packet index from a prior `warp_hostcall_submit` call
/// * `next_state` — state to transition to on completion
/// * `state_cell` — mutable reference to the state machine's state field
#[inline(always)]
pub unsafe fn warp_hostcall_wait_u64(
    buf: *mut u8,
    wcx: &mut WarpContext,
    pkt_idx: u16,
    next_state: u32,
    state_cell: &mut u32,
) -> Option<u64> {
    let idx = broadcast_u32(wcx.active_mask, pkt_idx as u32) as u16;
    let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_spin_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

    if ctrl & CONTROL_READY != 0 {
        let mut val: u64 = 0;
        if wcx.is_leader() {
            val = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
            crate::hostcall::gpu_hostcall_release(buf, pkt);
            *state_cell = next_state;
        }
        // Broadcast u64 as two u32 halves
        let lo = broadcast_u32(wcx.active_mask, val as u32) as u64;
        let hi = broadcast_u32(wcx.active_mask, (val >> 32) as u32) as u64;
        syncwarp(wcx.active_mask);
        Some(lo | (hi << 32))
    } else {
        None
    }
}
