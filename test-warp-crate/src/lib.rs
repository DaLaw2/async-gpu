#![no_std]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

// ---------------------------------------------------------------------------
// A future that yields Pending once, then returns Ready(value).
// This creates an actual suspension point in the coroutine state machine.
// ---------------------------------------------------------------------------

struct YieldOnce {
    yielded: bool,
    value: u32,
}

impl YieldOnce {
    fn new(value: u32) -> Self {
        YieldOnce { yielded: false, value }
    }
}

impl Future for YieldOnce {
    type Output = u32;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.yielded {
            Poll::Ready(this.value)
        } else {
            this.yielded = true;
            Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-await async fn: two .await points → coroutine with 3+ states
// This should trigger shfl.sync discriminant broadcast in the MIR pass.
// ---------------------------------------------------------------------------

#[warp_cooperative]
pub async fn multi_await(x: u32) -> u32 {
    let a = YieldOnce::new(x + 1).await;
    let b = YieldOnce::new(a + 10).await;
    a + b
}

// ---------------------------------------------------------------------------
// Minimal waker
// ---------------------------------------------------------------------------

static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |p| RawWaker::new(p, &VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

// ---------------------------------------------------------------------------
// Kernel entry: polls the multi-await future to completion
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn kernel_entry(x: u32) -> u32 {
    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = multi_await(x);
    let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };

    // Poll up to 10 times (should need 3: Pending, Pending, Ready)
    let mut i = 0;
    loop {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {
                i += 1;
                if i >= 10 {
                    return 0xDEAD; // safety limit
                }
            }
        }
    }
}
