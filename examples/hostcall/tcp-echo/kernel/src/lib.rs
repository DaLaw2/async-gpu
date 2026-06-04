//! TCP Echo -- GPU kernel demonstrating TCP networking via hostcall.
//!
//! Connects to a local TCP echo server, sends a message, reads the echo
//! response, and reports the response length to the host.

#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use gpu_runtime::prelude::*;

gpu_runtime::panic_handler!();

/// TCP echo kernel entry point.
///
/// 1. Connects to 127.0.0.1:{port}
/// 2. Sends "Hello from GPU!"
/// 3. Reads the echo response
/// 4. Writes response length to `output` (or 0xDEAD on error)
/// 5. Closes the socket
#[no_mangle]
pub unsafe extern "gpu-kernel" fn tcp_echo_kernel(
    buf: *mut u8,
    _sideband: *mut u8,
    port: u32,
    output: *mut u32,
) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }
    gpu_panic_init(buf);

    // Step 1: Connect to 127.0.0.1:{port}
    let addr = b"127.0.0.1";
    let connect_fut = GpuTcpConnectFuture::new(buf, addr, port);
    let fd = match block_on(connect_fut) {
        Some(Ok(fd)) => fd,
        _ => {
            let msg = b"TCP connect failed";
            let _ = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
            sys_store_release_u32(output, 0xDEAD);
            return;
        }
    };

    // Step 2: Send message
    let message = b"Hello from GPU!";
    let write_fut = GpuTcpWriteFuture::new(buf, fd, message);
    match block_on(write_fut) {
        Some(Ok(_)) => {}
        _ => {
            let msg = b"TCP write failed";
            let _ = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
            // Close socket before returning
            let close_fut = GpuTcpCloseFuture::new(buf, fd);
            let _ = block_on(close_fut);
            sys_store_release_u32(output, 0xDEAD);
            return;
        }
    }

    // Step 3: Read echo response
    let mut read_buf = [0u8; 56];
    let read_fut = GpuTcpReadFuture::new(buf, fd, &mut read_buf);
    let bytes_read = match block_on(read_fut) {
        Some(Ok(n)) => n,
        _ => {
            let msg = b"TCP read failed";
            let _ = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
            let close_fut = GpuTcpCloseFuture::new(buf, fd);
            let _ = block_on(close_fut);
            sys_store_release_u32(output, 0xDEAD);
            return;
        }
    };

    // Print the echoed message back via hostcall
    let print_len = if bytes_read > 56 { 56 } else { bytes_read };
    let _ = gpu_hostcall_print(buf, read_buf.as_ptr(), print_len as u32);

    // Step 4: Close socket
    let close_fut = GpuTcpCloseFuture::new(buf, fd);
    let _ = block_on(close_fut);

    // Report response length
    sys_store_release_u32(output, bytes_read as u32);
}
