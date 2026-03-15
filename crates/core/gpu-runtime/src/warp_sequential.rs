use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

const POLL_PENDING: u32 = 0;
const POLL_READY_TRUE: u32 = 1;
const POLL_READY_FALSE: u32 = 2;

/// Run two futures sequentially with warp convergence.
///
/// All lanes participate. Lane 0 polls each future; result is broadcast.
/// Returns `(ok1, ok2)` — the results of both futures.
///
/// # Safety
/// Must be called by all active lanes simultaneously.
#[inline(always)]
pub unsafe fn warp_run_two_futures(
    f1: &mut impl Future<Output = bool>,
    f2: &mut impl Future<Output = bool>,
) -> (Option<bool>, Option<bool>) {
    const MAX_POLLS: u32 = 10_000_000;

    let mask = activemask();
    let lid = lane_id();

    const VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
        |_| core::task::RawWaker::new(core::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    let raw_waker = core::task::RawWaker::new(core::ptr::null(), &VTABLE);
    let waker = core::task::Waker::from_raw(raw_waker);
    let mut cx = Context::from_waker(&waker);

    // Phase 1: poll f1 to completion
    let mut f1 = Pin::new_unchecked(f1);
    let mut polls: u32 = 0;
    let ok1 = loop {
        let mut result_code: u32 = POLL_PENDING;
        if lid == 0 {
            match f1.as_mut().poll(&mut cx) {
                Poll::Ready(true) => result_code = POLL_READY_TRUE,
                Poll::Ready(false) => result_code = POLL_READY_FALSE,
                Poll::Pending => result_code = POLL_PENDING,
            }
        }
        let broadcast = shfl_sync_idx_u32(mask, result_code, 0);
        syncwarp(mask);

        match broadcast {
            POLL_READY_TRUE => break Some(true),
            POLL_READY_FALSE => break Some(false),
            _ => {
                polls += 1;
                if polls >= MAX_POLLS {
                    break None;
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 64;", options(nostack));
            }
        }
    };

    // Convergence barrier between the two "await" points
    syncwarp(mask);

    // Phase 2: poll f2 to completion
    let mut f2 = Pin::new_unchecked(f2);
    polls = 0;
    let ok2 = loop {
        let mut result_code: u32 = POLL_PENDING;
        if lid == 0 {
            match f2.as_mut().poll(&mut cx) {
                Poll::Ready(true) => result_code = POLL_READY_TRUE,
                Poll::Ready(false) => result_code = POLL_READY_FALSE,
                Poll::Pending => result_code = POLL_PENDING,
            }
        }
        let broadcast = shfl_sync_idx_u32(mask, result_code, 0);
        syncwarp(mask);

        match broadcast {
            POLL_READY_TRUE => break Some(true),
            POLL_READY_FALSE => break Some(false),
            _ => {
                polls += 1;
                if polls >= MAX_POLLS {
                    break None;
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 64;", options(nostack));
            }
        }
    };

    (ok1, ok2)
}
