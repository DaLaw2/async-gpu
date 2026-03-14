#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

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
// This triggers shfl.sync discriminant broadcast in the MIR pass.
// ---------------------------------------------------------------------------

#[warp_cooperative]
pub async fn multi_await(x: u32) -> u32 {
    let a = YieldOnce::new(x + 1).await;
    let b = YieldOnce::new(a + 10).await;
    a + b
}

// ---------------------------------------------------------------------------
// Simple async fn (no .await): single-state, only bar.warp.sync
// ---------------------------------------------------------------------------

#[warp_cooperative]
pub async fn simple_add(x: u32) -> u32 {
    x + 1
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
// GPU Kernel: test simple warp-cooperative async fn
// Each thread polls simple_add(thread_idx) and writes result to output[tid].
// Expected: output[tid] = tid + 1
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_simple_warp(output: *mut u32) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);

    let waker = Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE));
    let mut cx = Context::from_waker(&waker);
    let mut fut = simple_add(tid);
    let mut pinned = Pin::new_unchecked(&mut fut);

    match pinned.as_mut().poll(&mut cx) {
        Poll::Ready(val) => *output.add(tid as usize) = val,
        Poll::Pending => *output.add(tid as usize) = 0xDEAD,
    }
}

// ---------------------------------------------------------------------------
// GPU Kernel: test multi-await warp-cooperative async fn
// Each thread polls multi_await(thread_idx) to completion.
// multi_await(x) = (x+1) + (x+1+10) = 2x + 12
// Expected: output[tid] = 2*tid + 12
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_multi_await(output: *mut u32) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);

    let waker = Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE));
    let mut cx = Context::from_waker(&waker);
    let mut fut = multi_await(tid);
    let mut pinned = Pin::new_unchecked(&mut fut);

    // Poll to completion (max 10 iterations)
    let mut result = 0xDEADu32;
    for _ in 0..10 {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(val) => {
                result = val;
                break;
            }
            Poll::Pending => {}
        }
    }
    *output.add(tid as usize) = result;
}
