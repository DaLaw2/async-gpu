//! Async Pipeline — warp-cooperative data pipeline with real I/O.
//!
//! Demonstrates async fn using real hostcall Futures:
//! 1. Open input file → read data
//! 2. Transform on GPU (add 1 to each byte)
//! 3. Open output file → write transformed data → close
//!
//! Each `.await` is a yield point where the MIR pass inserts warp convergence
//! barriers (`bar.warp.sync`), allowing other warps to run during I/O wait.
//! The MIR pass handles this automatically for all async fn on nvptx64 — no
//! annotation needed.

#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(asm_experimental_arch)]

use gpu_runtime::prelude::*;
use gpu_runtime::std_future::{
    GpuBulkReadFuture, GpuBulkWriteFuture, GpuCloseFuture, GpuOpenFuture, GpuReadFuture,
    GpuWriteFuture,
};

gpu_runtime::panic_handler!();

// ---------------------------------------------------------------------------
// Warp-cooperative async data pipeline
// ---------------------------------------------------------------------------

/// Read file → increment each byte → write to output file.
///
/// 5 await points: open_read, read, close_read, open_write, write, close_write
/// The MIR pass inserts `bar.warp.sync` at each, so all lanes in the warp
/// yield together. Between awaits, compute runs in SIMT lockstep.
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
/// The `data_pipeline` async fn is compiled with the MIR pass, which
/// automatically inserts `bar.warp.sync` + `shfl.sync` at each `.await` point.
/// In a multi-warp scenario, other warps can run compute while this warp
/// waits for I/O responses.
///
/// Currently single-thread because each hostcall Future allocates from a
/// shared packet pool — multi-lane requires warp-batched hostcall (future work).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn async_data_pipeline(buf: *mut u8, output: *mut u32) {
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

// ---------------------------------------------------------------------------
// Warp-cooperative async BULK I/O pipeline
// ---------------------------------------------------------------------------

/// Read file via sideband bulk read → transform → write via sideband bulk write.
///
/// Uses `GpuBulkReadFuture` and `GpuBulkWriteFuture` for large data transfers
/// (up to sideband capacity, default 1MB) instead of the 48-byte packet payload.
/// Each `.await` is a yield point with warp convergence barrier.
pub async fn bulk_data_pipeline(buf: *mut u8, sideband: *mut u8) -> u32 {
    // Step 1: Open input file
    let fd = match GpuOpenFuture::new(buf, b"bulk_input.txt", FILE_OPEN_READ).await {
        Ok(fd) => fd,
        Err(_) => return 0xE001,
    };

    // Step 2: Bulk read up to 256 bytes via sideband
    let mut data = [0u8; 256];
    let n = match GpuBulkReadFuture::new(buf, sideband, fd as u64, data.as_mut_ptr(), 256).await {
        Ok(n) => n,
        Err(_) => return 0xE002,
    };

    // Step 3: Close input
    if GpuCloseFuture::new(buf, fd).await.is_err() {
        return 0xE003;
    }

    // Step 4: Transform — XOR each byte with 0x20 (toggles case for ASCII letters)
    let mut out = [0u8; 256];
    let mut i = 0;
    while i < n {
        let ch = data[i];
        out[i] = if ch >= b'a' && ch <= b'z' {
            ch - 32
        } else if ch >= b'A' && ch <= b'Z' {
            ch + 32
        } else {
            ch
        };
        i += 1;
    }

    // Step 5: Open output file
    let out_fd =
        match GpuOpenFuture::new(buf, b"bulk_output.txt", FILE_OPEN_WRITE_CREATE).await {
            Ok(fd) => fd,
            Err(_) => return 0xE004,
        };

    // Step 6: Bulk write via sideband
    let written =
        match GpuBulkWriteFuture::new(buf, sideband, out_fd as u64, out.as_ptr(), n).await {
            Ok(w) => w,
            Err(_) => return 0xE005,
        };

    // Step 7: Close output
    if GpuCloseFuture::new(buf, out_fd).await.is_err() {
        return 0xE006;
    }

    let msg = b"bulk pipeline done";
    let _ = unsafe { gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32) };

    written as u32
}

/// Entry point for async bulk I/O pipeline.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn async_bulk_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    output: *mut u32,
) {
    let tid: u32;
    core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);
    if tid != 0 {
        return;
    }

    gpu_panic_init(buf);
    sideband_reset(sideband);

    let result = block_on(bulk_data_pipeline(buf, sideband)).unwrap_or(0xDEAD);
    *output = result;
}
