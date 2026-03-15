use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

// Result encoding for broadcast:
// 0 = Pending, 1 = Ready(Ok(true)), 2 = Ready(Ok(false)), 3 = Ready(Err(code))
const POLL_PENDING: u32 = 0;
const POLL_OK_TRUE: u32 = 1;
const POLL_OK_FALSE: u32 = 2;
const POLL_ERR: u32 = 3;

/// Warp-cooperative poll result with error support.
pub enum WarpResult {
    /// Future still pending
    Pending,
    /// Future completed with Ok(true)
    OkTrue,
    /// Future completed with Ok(false)
    OkFalse,
    /// Future completed with Err(error_code)
    Err(u32),
}

/// Run a `Future<Output = Result<bool, u32>>` warp-cooperatively.
///
/// Lane 0 polls; broadcasts both discriminant and error code (if any).
/// All lanes see the same `WarpResult`.
///
/// # Safety
/// Must be called by all active lanes simultaneously.
#[inline(always)]
pub unsafe fn warp_run_result_future(
    future: &mut impl Future<Output = Result<bool, u32>>,
) -> WarpResult {
    const MAX_POLLS: u32 = 10_000_000;

    let mask = activemask();
    let lid = lane_id();

    let mut future = Pin::new_unchecked(future);

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
        let mut result_code: u32 = POLL_PENDING;
        let mut error_code: u32 = 0;
        if lid == 0 {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(Ok(true)) => result_code = POLL_OK_TRUE,
                Poll::Ready(Ok(false)) => result_code = POLL_OK_FALSE,
                Poll::Ready(Err(e)) => {
                    result_code = POLL_ERR;
                    error_code = e;
                }
                Poll::Pending => result_code = POLL_PENDING,
            }
        }

        // Broadcast both discriminant and error code
        let bc_result = shfl_sync_idx_u32(mask, result_code, 0);
        let bc_error = shfl_sync_idx_u32(mask, error_code, 0);
        syncwarp(mask);

        match bc_result {
            POLL_OK_TRUE => return WarpResult::OkTrue,
            POLL_OK_FALSE => return WarpResult::OkFalse,
            POLL_ERR => return WarpResult::Err(bc_error),
            _ => {
                polls += 1;
                if polls >= MAX_POLLS {
                    return WarpResult::Err(0xDEAD); // timeout
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 64;", options(nostack));
            }
        }
    }
}

/// Run two Result futures sequentially with ? semantics.
///
/// If f1 returns Err, f2 is skipped (all lanes return Err together).
/// This is the warp-cooperative equivalent of:
///   f1.await?;
///   f2.await?;
///
/// # Safety
/// Must be called by all active lanes simultaneously.
#[inline(always)]
pub unsafe fn warp_run_two_result_futures(
    f1: &mut impl Future<Output = Result<bool, u32>>,
    f2: &mut impl Future<Output = Result<bool, u32>>,
) -> Result<u32, u32> {
    let mask = activemask();

    // First .await?
    match warp_run_result_future(f1) {
        WarpResult::OkTrue | WarpResult::OkFalse => {} // continue
        WarpResult::Err(e) => return Err(e),           // all lanes early-return
        WarpResult::Pending => return Err(0xDEAD),     // unreachable
    }

    syncwarp(mask);

    // Second .await?
    match warp_run_result_future(f2) {
        WarpResult::OkTrue | WarpResult::OkFalse => {} // continue
        WarpResult::Err(e) => return Err(e),
        WarpResult::Pending => return Err(0xDEAD),
    }

    Ok(2) // both succeeded
}
