#![no_std]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, RawWaker, RawWakerVTable, Waker};

#[warp_cooperative]
pub async fn cooperative_poll(x: u32) -> u32 {
    x + 1
}

// Minimal waker that does nothing
static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |p| RawWaker::new(p, &VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

#[no_mangle]
pub extern "C" fn kernel_entry(x: u32) -> u32 {
    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = cooperative_poll(x);
    let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
    match pinned.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(val) => val,
        core::task::Poll::Pending => 0,
    }
}
