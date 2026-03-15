use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

// Poll result encoding for broadcast:
// 0 = Pending, 1 = Ready(true), 2 = Ready(false)
const POLL_PENDING: u32 = 0;
const POLL_READY_TRUE: u32 = 1;
const POLL_READY_FALSE: u32 = 2;

/// Warp-cooperatively poll a standard `impl Future<Output = bool>`.
///
/// All 32 lanes must call this simultaneously. Lane 0 actually polls
/// the future; the result is broadcast via `shfl.sync.idx.b32` so all
/// lanes observe the same `Poll` value.
///
/// Returns the same `Poll<bool>` to all lanes.
///
/// # Safety
/// - Must be called by all active lanes simultaneously
/// - `future` must be safe to poll on lane 0
#[inline(always)]
pub unsafe fn warp_poll_future(
    future: Pin<&mut impl Future<Output = bool>>,
    cx: &mut Context<'_>,
) -> Poll<bool> {
    let mask = activemask();
    let lid = lane_id();

    // Lane 0 polls the actual future
    let mut result_code: u32 = POLL_PENDING;
    if lid == 0 {
        match future.poll(cx) {
            Poll::Ready(true) => result_code = POLL_READY_TRUE,
            Poll::Ready(false) => result_code = POLL_READY_FALSE,
            Poll::Pending => result_code = POLL_PENDING,
        }
    }

    // Broadcast poll result to all lanes
    let broadcast_result = shfl_sync_idx_u32(mask, result_code, 0);

    // All lanes see the same result
    syncwarp(mask);

    match broadcast_result {
        POLL_READY_TRUE => Poll::Ready(true),
        POLL_READY_FALSE => Poll::Ready(false),
        _ => Poll::Pending,
    }
}

/// Warp-cooperative spin executor for a standard `impl Future<Output = bool>`.
///
/// All 32 lanes call this together. Lane 0 polls the future; result is
/// broadcast to all lanes. Returns when the future completes.
///
/// # Safety
/// - Must be called by all active lanes simultaneously
/// - The future must be safe to poll on lane 0
#[inline(always)]
pub unsafe fn warp_run_future(future: &mut impl Future<Output = bool>) -> Option<bool> {
    const MAX_POLLS: u32 = 10_000_000;

    let mask = activemask();
    let lid = lane_id();

    let mut future = Pin::new_unchecked(future);

    // Create a no-op waker (only lane 0 uses it, but all lanes must have it
    // for convergence)
    const VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
        |_| core::task::RawWaker::new(core::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    let raw_waker = core::task::RawWaker::new(core::ptr::null(), &VTABLE);
    let waker = core::task::Waker::from_raw(raw_waker);
    let mut cx = Context::from_waker(&waker);

    let mut polls: u32 = 0;
    loop {
        // Lane 0 polls, broadcasts result
        let mut result_code: u32 = POLL_PENDING;
        if lid == 0 {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(true) => result_code = POLL_READY_TRUE,
                Poll::Ready(false) => result_code = POLL_READY_FALSE,
                Poll::Pending => result_code = POLL_PENDING,
            }
        }

        let broadcast_result = shfl_sync_idx_u32(mask, result_code, 0);
        syncwarp(mask);

        match broadcast_result {
            POLL_READY_TRUE => return Some(true),
            POLL_READY_FALSE => return Some(false),
            _ => {
                polls += 1;
                if polls >= MAX_POLLS {
                    return None;
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 64;", options(nostack));
            }
        }
    }
}
