//! Async Pipeline — warp-cooperative data pipeline with real I/O.
//!
//! Demonstrates `#[warp_cooperative] async fn` using real hostcall Futures:
//! 1. Open input file → read data
//! 2. Transform on GPU (add 1 to each byte)
//! 3. Open output file → write transformed data → close
//!
//! Each `.await` is a yield point where the MIR pass inserts warp convergence
//! barriers (`bar.warp.sync`), allowing other warps to run during I/O wait.

#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
#![feature(register_tool)]
#![register_tool(warp_cooperative)]

use gpu_runtime::prelude::*;
use gpu_runtime::std_future::{GpuCloseFuture, GpuOpenFuture, GpuReadFuture, GpuWriteFuture};

gpu_runtime::panic_handler!();

// ---------------------------------------------------------------------------
// Warp-cooperative async data pipeline
// ---------------------------------------------------------------------------

/// Read file → increment each byte → write to output file.
///
/// 5 await points: open_read, read, close_read, open_write, write, close_write
/// The MIR pass inserts `bar.warp.sync` at each, so all lanes in the warp
/// yield together. Between awaits, compute runs in SIMT lockstep.
#[warp_cooperative]
pub async fn data_pipeline(buf: *mut u8) -> u32 {
    // Step 1: Open input file for reading
    let fd = match GpuOpenFuture::new(buf, b"pipeline_input.txt", FILE_OPEN_READ).await {
        Ok(fd) => fd,
        Err(_) => return 0xE001,
    };

    // Step 2: Read up to 48 bytes
    let mut data = [0u8; 48];
    let n = match GpuReadFuture::new(buf, fd, &mut data).await {
        Ok(n) => n,
        Err(_) => return 0xE002,
    };

    // Step 3: Close input file
    if GpuCloseFuture::new(buf, fd).await.is_err() {
        return 0xE003;
    }

    // Step 4: Transform — uppercase on GPU
    let mut out = [0u8; 48];
    let mut i = 0;
    while i < n {
        let ch = data[i];
        out[i] = if ch >= b'a' && ch <= b'z' { ch - 32 } else { ch };
        i += 1;
    }

    // Step 5: Open output file for writing
    let out_fd =
        match GpuOpenFuture::new(buf, b"pipeline_output.txt", FILE_OPEN_WRITE_CREATE).await {
            Ok(fd) => fd,
            Err(_) => return 0xE004,
        };

    // Step 6: Write transformed data
    let written = match GpuWriteFuture::new(buf, out_fd, &out[..n]).await {
        Ok(w) => w,
        Err(_) => return 0xE005,
    };

    // Step 7: Close output file
    if GpuCloseFuture::new(buf, out_fd).await.is_err() {
        return 0xE006;
    }

    // Report: print confirmation via hostcall
    let msg = b"async pipeline done";
    let _ = unsafe { gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32) };

    // Return bytes written as success indicator
    written as u32
}

// ---------------------------------------------------------------------------
// GPU Kernel entry point
// ---------------------------------------------------------------------------

/// Entry point: thread 0 runs the async pipeline via `block_on`.
///
/// The `data_pipeline` async fn is compiled with `#[warp_cooperative]`, so the
/// MIR pass inserts `bar.warp.sync` + `shfl.sync` at each `.await` point.
/// In a multi-warp scenario, other warps can run compute while this warp
/// waits for I/O responses.
///
/// Currently single-thread because each hostcall Future allocates from a
/// shared packet pool — multi-lane requires warp-batched hostcall (future work).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn async_data_pipeline(buf: *mut u8, output: *mut u32) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);
    if tid != 0 {
        return;
    }

    gpu_panic_init(buf);

    // block_on drives the async pipeline to completion with nanosleep yield
    // between polls, replacing the previous 30-line manual poll loop.
    let result = block_on(data_pipeline(buf)).unwrap_or(0xDEAD);
    *output = result;
}
