#![no_std]
#![feature(abi_gpu_kernel)]
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

pub async fn multi_await(x: u32) -> u32 {
    let a = YieldOnce::new(x + 1).await;
    let b = YieldOnce::new(a + 10).await;
    a + b
}

// ---------------------------------------------------------------------------
// Simple async fn (no .await): single-state, only bar.warp.sync
// ---------------------------------------------------------------------------

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
// Async pipeline simulation: 6-await pipeline proving yield pattern
// This simulates the I/O pipeline pattern without actual hostcall.
// Each .await is a yield point where the warp scheduler can run other warps.
// ---------------------------------------------------------------------------

/// Simulates an async "open" — yields once, returns a fake fd.
struct SimOpen { yielded: bool, fd: u32 }
impl SimOpen {
    fn new(fd: u32) -> Self { SimOpen { yielded: false, fd } }
}
impl Future for SimOpen {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.yielded { Poll::Ready(this.fd) }
        else { this.yielded = true; Poll::Pending }
    }
}

/// Simulates an async "write" — yields once, returns bytes written.
struct SimWrite { yielded: bool, len: u32 }
impl SimWrite {
    fn new(len: u32) -> Self { SimWrite { yielded: false, len } }
}
impl Future for SimWrite {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.yielded { Poll::Ready(this.len) }
        else { this.yielded = true; Poll::Pending }
    }
}

/// Simulates an async "read" — yields once, returns bytes read.
struct SimRead { yielded: bool, len: u32 }
impl SimRead {
    fn new(len: u32) -> Self { SimRead { yielded: false, len } }
}
impl Future for SimRead {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.yielded { Poll::Ready(this.len) }
        else { this.yielded = true; Poll::Pending }
    }
}

/// Simulates an async "close" — yields once, returns success.
struct SimClose { yielded: bool }
impl SimClose {
    fn new() -> Self { SimClose { yielded: false } }
}
impl Future for SimClose {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.yielded { Poll::Ready(0) }
        else { this.yielded = true; Poll::Pending }
    }
}

/// 6-await data pipeline: open → write → close → open → read → close
/// Each .await yields the warp. The MIR pass inserts convergence barriers.
/// Result = written * 1000 + read_back (expected: 29000 + 29 = 29029 for all lanes)
pub async fn async_pipeline(tid: u32) -> u32 {
    // "Open for write" — yields once
    let _wfd = SimOpen::new(10 + tid).await;

    // "Write data" — yields once, simulates writing 29 bytes
    let written = SimWrite::new(29).await;

    // "Close" — yields once
    let _ = SimClose::new().await;

    // "Open for read" — yields once
    let _rfd = SimOpen::new(20 + tid).await;

    // "Read data" — yields once, simulates reading back 29 bytes
    let read_back = SimRead::new(29).await;

    // "Close" — yields once
    let _ = SimClose::new().await;

    // Compute result: 29 * 1000 + 29 = 29029
    written * 1000 + read_back
}

// ---------------------------------------------------------------------------
// GPU Kernel: test async pipeline (6 .await points)
// Each thread polls async_pipeline(tid) to completion.
// Expected: output[tid] = 29029 for all lanes
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_async_pipeline(output: *mut u32) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);

    let waker = Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE));
    let mut cx = Context::from_waker(&waker);
    let mut fut = async_pipeline(tid);
    let mut pinned = Pin::new_unchecked(&mut fut);

    // 6 await points = need at least 7 polls
    let mut result = 0xDEADu32;
    for _ in 0..20 {
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

// ---------------------------------------------------------------------------
// GPU Kernel: test simple warp-cooperative async fn
// Each thread polls simple_add(thread_idx) and writes result to output[tid].
// Expected: output[tid] = tid + 1
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_simple_warp(output: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn test_multi_await(output: *mut u32) {
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
