// Hostcall test/bench kernels — kernel entry points for hostcall protocol testing.

use crate::helpers::{
    gpu_hostcall_close, gpu_hostcall_open, gpu_hostcall_read, gpu_hostcall_stdin_read,
    gpu_hostcall_time, gpu_hostcall_write, gpu_instant_nanos, hc_pop_free_counted,
    hc_pop_free_counted_v2,
};
use core::arch::nvptx;
use gpu_atomics::{
    activemask, sys_fetch_add_u64, sys_spin_load_acquire_u32, sys_store_release_u32,
};
use gpu_protocol::*;

// ============================================================
// Hostcall protocol — uses gpu-runtime's consolidated API (api-cleanup.1)
// ============================================================
//
// Core protocol functions (hc_pop_free, hc_push, gpu_hostcall_print,
// gpu_hostcall_request, gpu_hostcall_release) are now provided by
// gpu_runtime::hostcall. This eliminates duplicated code and gains
// automatic sharding support.

/// Hostcall kernel: print "Hello from GPU!" via the hostcall protocol.
///
/// Thread 0 of block 0 issues a single PRINT hostcall. The host listener
/// reads the message from the packet payload and prints it to stdout.
///
/// `buf` is the device-side pointer to the hostcall buffer (mapped memory).
/// `result` is a device pointer where thread 0 writes 1 (success) or 0 (failure).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_print_hello(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Hardcoded message — the bytes live in GPU .const memory
    let msg: &[u8; 15] = b"Hello from GPU!";
    let ok = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), 15).is_ok();
    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

/// Hostcall kernel: multiple warps each print a message.
///
/// Each block's thread 0 issues a PRINT hostcall with the block index.
/// Tests concurrent multi-warp hostcall.
///
/// `buf` is the hostcall buffer, `num_msgs` is total number of messages to print.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_print_multi(buf: *mut u8, success_count: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;

    // Only thread 0 of each block does the hostcall
    if thread_x != 0 {
        return;
    }

    // Format: "Block NNN\n" — we write the block index as decimal digits
    // Simple manual formatting since we don't have std::fmt
    let mut msg_buf: [u8; 16] = [0u8; 16];
    // "Block "
    msg_buf[0] = b'B';
    msg_buf[1] = b'l';
    msg_buf[2] = b'o';
    msg_buf[3] = b'c';
    msg_buf[4] = b'k';
    msg_buf[5] = b' ';
    // Format block_x as decimal (max 3 digits for our test)
    let mut n = block_x;
    let mut pos = 6;
    if n >= 100 {
        msg_buf[pos] = b'0' + (n / 100) as u8;
        pos += 1;
        n %= 100;
    }
    if block_x >= 10 {
        msg_buf[pos] = b'0' + (n / 10) as u8;
        pos += 1;
        n %= 10;
    }
    msg_buf[pos] = b'0' + n as u8;
    pos += 1;

    if gpu_runtime::hostcall::gpu_hostcall_print(buf, msg_buf.as_ptr(), pos as u32).is_ok() {
        gpu_atomics::sys_fetch_add_u32(success_count, 1);
    }
}

// ============================================================
// File I/O test kernel (gpu-std.3)
// ============================================================

/// Hostcall kernel: create a file, write content, close, reopen, read back, verify.
///
/// Thread 0 of block 0 performs a full file I/O round-trip:
/// 1. Open "gpu_test_output.txt" for writing
/// 2. Write "Hello from GPU file I/O!\n"
/// 3. Close the file
/// 4. Open "gpu_test_output.txt" for reading
/// 5. Read back the content
/// 6. Close the file
/// 7. Write result codes to output
///
/// `buf` = hostcall buffer, `result` = output array of u32[4]:
///   [0] = overall success (1) or failure (0)
///   [1] = open-write fd (or 0xFFFFFFFF on error)
///   [2] = bytes written (or 0xFFFFFFFF on error)
///   [3] = bytes read back (or 0xFFFFFFFF on error)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_file_test(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Initialize result to all-fail
    *result.add(0) = 0;
    *result.add(1) = 0xFFFF_FFFF;
    *result.add(2) = 0xFFFF_FFFF;
    *result.add(3) = 0xFFFF_FFFF;

    // File path
    let path: &[u8; 20] = b"gpu_test_output.txt\0";
    let path_len: u32 = 19; // excluding null terminator

    // Message to write
    let msg: &[u8; 25] = b"Hello from GPU file I/O!\n";
    let msg_len: u32 = 25;

    // Step 1: Open file for writing (create)
    let (fd, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_WRITE_CREATE);
    if err != 0 {
        return;
    }
    *result.add(1) = fd as u32;

    // Step 2: Write message
    let (written, err) = gpu_hostcall_write(buf, fd, msg.as_ptr(), msg_len);
    if err != 0 {
        // Still try to close
        gpu_hostcall_close(buf, fd);
        return;
    }
    *result.add(2) = written as u32;

    // Step 3: Close file
    let (_, err) = gpu_hostcall_close(buf, fd);
    if err != 0 {
        return;
    }

    // Step 4: Reopen for reading
    let (fd2, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_READ);
    if err != 0 {
        return;
    }

    // Step 5: Read back content
    let mut read_buf: [u8; 48] = [0u8; 48];
    let (bytes_read, err) = gpu_hostcall_read(buf, fd2, read_buf.as_mut_ptr(), 48);
    if err != 0 {
        gpu_hostcall_close(buf, fd2);
        return;
    }
    *result.add(3) = bytes_read as u32;

    // Step 6: Close read file
    gpu_hostcall_close(buf, fd2);

    // Step 7: Verify content matches
    if bytes_read == msg_len as u64 {
        let mut match_ok: bool = true;
        let mut i: u32 = 0;
        while i < msg_len {
            if read_buf[i as usize] != *msg.as_ptr().add(i as usize) {
                match_ok = false;
            }
            i += 1;
        }
        if match_ok {
            *result.add(0) = 1; // Overall success
        }
    }
}

// ============================================================
// Stdin + time test kernel (gpu-std.4)
// ============================================================

/// Test kernel: read GPU Instant, get wall-clock time, and stdin read.
///
/// `buf` = hostcall buffer
/// `result` = output array of u64[4]:
///   [0] = GPU instant (nanoseconds from %globaltimer)
///   [1] = host wall-clock seconds since epoch
///   [2] = host wall-clock nanoseconds
///   [3] = 1 if stdin read succeeded, 0 if skipped/failed
///         (stdin test is optional — host may not provide input)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_stdin_time_test(
    buf: *mut u8,
    result: *mut u64,
    skip_stdin: u32,
) {
    let thread_x = core::arch::nvptx::_thread_idx_x() as u32;
    let block_x = core::arch::nvptx::_block_idx_x() as u32;
    let block_dim_x = core::arch::nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Test 1: GPU Instant (%globaltimer)
    let t0 = gpu_instant_nanos();
    // Do a small amount of work to see non-zero delta
    // Use volatile reads to prevent the compiler from optimizing this loop away
    let mut dummy: u32 = 0;
    let mut i: u32 = 0;
    while core::ptr::read_volatile(&i) < 1000 {
        dummy = core::ptr::read_volatile(&dummy).wrapping_add(core::ptr::read_volatile(&i));
        i += 1;
    }
    // Use volatile write to ensure dummy is not dead-code eliminated
    core::ptr::write_volatile(result.add(3), dummy as u64);
    let t1 = gpu_instant_nanos();
    *result.add(0) = t1 - t0; // nanosecond delta

    // Test 2: Wall-clock time via hostcall
    let (secs, nanos) = gpu_hostcall_time(buf);
    *result.add(1) = secs;
    *result.add(2) = nanos;

    // Test 3: stdin read (optional, skip if skip_stdin != 0)
    if skip_stdin == 0 {
        let mut stdin_buf: [u8; 56] = [0u8; 56];
        let (bytes, err) = gpu_hostcall_stdin_read(buf, stdin_buf.as_mut_ptr(), 56);
        if err == 0 && bytes > 0 {
            *result.add(3) = 1;
        } else {
            *result.add(3) = 0;
        }
    } else {
        *result.add(3) = 0;
    }
}

// ============================================================
// Error propagation test kernel (error-handling.2)
// ============================================================

/// Test kernel: verify that hostcall error codes propagate to GPU.
///
/// Tests:
/// 1. Open a nonexistent file → should get ERR_NOT_FOUND
/// 2. Close an invalid fd → should get an error
/// 3. Read from an invalid fd → should get an error
///
/// `buf` = hostcall buffer, `result` = output array of u32[6]:
///   [0] = overall success (1 if all tests pass)
///   [1] = test1 error category (expected: ERR_NOT_FOUND = 1)
///   [2] = test2 error category (expected: nonzero)
///   [3] = test3 error category (expected: nonzero)
///   [4] = test1 fd (should be 0 since open failed)
///   [5] = number of tests passed
#[no_mangle]
pub unsafe extern "ptx-kernel" fn error_propagation_test(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    if thread_x != 0 {
        return;
    }

    // Initialize results
    let mut i = 0;
    while i < 6 {
        core::ptr::write_volatile(result.add(i), 0);
        i += 1;
    }

    let mut passed: u32 = 0;

    // Test 1: Open a nonexistent file → should return ERR_NOT_FOUND
    let path = b"__nonexistent_file_12345__";
    let (fd, err_cat) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, FILE_OPEN_READ);
    core::ptr::write_volatile(result.add(1), err_cat as u32);
    core::ptr::write_volatile(result.add(4), fd as u32);
    if err_cat == ERR_NOT_FOUND {
        passed += 1;
    }

    // Test 2: Close an invalid fd (99999) → should return an error
    let (_, err_cat) = gpu_hostcall_close(buf, 99999);
    core::ptr::write_volatile(result.add(2), err_cat as u32);
    if err_cat != 0 {
        passed += 1;
    }

    // Test 3: Read from invalid fd → should return an error
    let mut dummy_buf: [u8; 8] = [0u8; 8];
    let (_, err_cat) = gpu_hostcall_read(buf, 99999, dummy_buf.as_mut_ptr(), 8);
    core::ptr::write_volatile(result.add(3), err_cat as u32);
    if err_cat != 0 {
        passed += 1;
    }

    core::ptr::write_volatile(result.add(5), passed);
    if passed == 3 {
        core::ptr::write_volatile(result.add(0), 1); // All tests passed
    }
}

// ============================================================
// Hostcall latency benchmark kernel (benchmark.2)
// ============================================================

/// Benchmark kernel: measure hostcall NOP round-trip latency.
///
/// Each thread performs `num_iters` NOP hostcalls sequentially.
/// Records per-thread: total elapsed ns, total CAS retries, iterations completed.
///
/// Layout of `results` (u64 array, 3 entries per thread):
///   results[tid*3 + 0] = total elapsed nanoseconds for all iterations
///   results[tid*3 + 1] = total CAS retries across all iterations
///   results[tid*3 + 2] = number of completed iterations
///
/// `buf` = hostcall buffer
/// `results` = output array, must have space for num_threads * 3 u64 entries
/// `num_iters` = number of NOP hostcalls per thread
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_latency_bench(
    buf: *mut u8,
    results: *mut u64,
    num_iters: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    let t_start = gpu_instant_nanos();
    let mut total_retries: u64 = 0;
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        // Pop free packet (instrumented)
        let (pkt_idx, retries) = hc_pop_free_counted(buf);
        if pkt_idx == NULL_INDEX {
            // Pool exhaustion — stop this thread
            break;
        }
        total_retries += retries as u64;

        let pkt = buf.add(packet_offset(pkt_idx));

        // Fill NOP packet
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_NOP);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Mark as filled
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Push to ready stack
        let ready_ptr = buf.add(BUF_OFF_READY_STACK) as *mut u64;
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, pkt_idx);

        // Ring doorbell
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                break;
            }
        }

        // Return packet to free stack
        let free_ptr = buf.add(BUF_OFF_FREE_STACK) as *mut u64;
        gpu_runtime::hostcall::hc_push(free_ptr, buf, pkt_idx);

        completed += 1;
        iter += 1;
    }

    let t_end = gpu_instant_nanos();

    // Write results
    let base = (tid as usize) * 3;
    core::ptr::write_volatile(results.add(base), t_end - t_start);
    core::ptr::write_volatile(results.add(base + 1), total_retries);
    core::ptr::write_volatile(results.add(base + 2), completed);
}

// ============================================================
// Sharding benchmark kernel (per-block-sharding.3)
// ============================================================

/// Shard-aware benchmark kernel: measure hostcall NOP round-trip latency.
///
/// Identical to `hostcall_latency_bench` but uses shard-aware stacks.
/// Works with both sharded and unsharded buffers (auto-detects via num_shards).
///
/// Layout of `results` (u64 array, 3 entries per thread):
///   results[tid*3 + 0] = total elapsed nanoseconds for all iterations
///   results[tid*3 + 1] = total CAS retries across all iterations
///   results[tid*3 + 2] = number of completed iterations
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hostcall_latency_bench_v2(
    buf: *mut u8,
    results: *mut u64,
    num_iters: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    // Read shard info once
    let (num_shards, shard_array_off, _) = gpu_runtime::hostcall::read_shard_info(buf as *const u8);
    let free_ptr = gpu_runtime::hostcall::get_free_stack_ptr(buf, num_shards, shard_array_off);
    let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_array_off);

    let t_start = gpu_instant_nanos();
    let mut total_retries: u64 = 0;
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        // Pop free packet (instrumented, shard-aware)
        let (pkt_idx, retries) = hc_pop_free_counted_v2(buf, free_ptr, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            break;
        }
        total_retries += retries as u64;

        let pkt_off = gpu_runtime::hostcall::pkt_offset(buf as *const u8, pkt_idx);
        let pkt = buf.add(pkt_off);

        // Fill NOP packet
        let mask = activemask();
        core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
        core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_NOP);
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

        // Mark as filled
        sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

        // Push to ready stack (shard-aware via hc_push which reads shard info)
        gpu_runtime::hostcall::hc_push(ready_ptr, buf, pkt_idx);

        // Ring doorbell (always global)
        sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

        // Spin-wait for response
        let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
        let mut spins: u32 = 0;
        loop {
            let ctrl = sys_spin_load_acquire_u32(control_ptr);
            if ctrl & CONTROL_READY != 0 {
                break;
            }
            spins += 1;
            if spins >= GPU_MAX_SPIN {
                break;
            }
        }

        // Return packet to free stack (shard-aware)
        gpu_runtime::hostcall::hc_push(free_ptr, buf, pkt_idx);

        completed += 1;
        iter += 1;
    }

    let t_end = gpu_instant_nanos();

    // Write results
    let base = (tid as usize) * 3;
    core::ptr::write_volatile(results.add(base), t_end - t_start);
    core::ptr::write_volatile(results.add(base + 1), total_retries);
    core::ptr::write_volatile(results.add(base + 2), completed);
}

// ============================================================
// GPU panic test kernel (gpu-panic.2)
// ============================================================

/// Test kernel: deliberately panic to verify panic message delivery via hostcall.
///
/// Thread 0 initializes the panic handler and then panics with a test message.
/// Other threads simply return.
///
/// `buf` = hostcall buffer
/// `result` = output u32 (set to 1 before panic — if host sees this AND the
///            panic message, the test passes)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn panic_test_kernel(buf: *mut u8, result: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;

    // Initialize panic handler with hostcall buffer
    gpu_runtime::panic::gpu_panic_init(buf);

    if global_idx != 0 {
        return;
    }

    // Set marker to indicate we reached this point
    core::ptr::write_volatile(result, 1);

    // Deliberately panic — this should send a message via hostcall then trap
    panic!("test panic from GPU thread 0");
}

// ============================================================
// Bulk data transfer test kernel (large-payload.3)
// ============================================================

/// Test kernel: write a 4KB message to a file via sideband bulk transfer,
/// then read it back and verify the content matches.
///
/// `buf` = hostcall buffer
/// `sideband` = sideband data buffer
/// `result` = output array of u32[4]:
///   [0] = overall success (1 if write+read+verify all pass)
///   [1] = bytes written
///   [2] = bytes read back
///   [3] = content match (1 if all bytes match)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bulk_io_test(buf: *mut u8, sideband: *mut u8, result: *mut u32) {
    use gpu_runtime::sideband::{gpu_bulk_read, gpu_bulk_write, sideband_reset};

    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    // Initialize panic handler + reset sideband allocator
    gpu_runtime::panic::gpu_panic_init(buf);
    sideband_reset(sideband);

    // Initialize results to failure
    core::ptr::write_volatile(result.add(0), 0);
    core::ptr::write_volatile(result.add(1), 0);
    core::ptr::write_volatile(result.add(2), 0);
    core::ptr::write_volatile(result.add(3), 0);

    // Generate a 4KB test pattern
    const DATA_SIZE: usize = 4096;
    let mut test_data: [u8; DATA_SIZE] = [0u8; DATA_SIZE];
    let mut i: usize = 0;
    while i < DATA_SIZE {
        test_data[i] = (i & 0xFF) as u8;
        i += 1;
    }

    // Step 1: Open file for writing
    let path = b"gpu_bulk_test.bin";
    let (fd, err) = gpu_hostcall_open(
        buf,
        path.as_ptr(),
        path.len() as u32,
        FILE_OPEN_WRITE_CREATE,
    );
    if err != 0 {
        return;
    }

    // Step 2: Bulk write 4KB
    let written = gpu_bulk_write(buf, sideband, fd, test_data.as_ptr(), DATA_SIZE);
    core::ptr::write_volatile(result.add(1), written as u32);

    // Close write file
    gpu_hostcall_close(buf, fd);

    if written != DATA_SIZE {
        return;
    }

    // Reset sideband allocator for read phase
    sideband_reset(sideband);

    // Step 3: Reopen for reading
    let (fd2, err) = gpu_hostcall_open(buf, path.as_ptr(), path.len() as u32, FILE_OPEN_READ);
    if err != 0 {
        return;
    }

    // Step 4: Bulk read 4KB
    let mut read_buf: [u8; DATA_SIZE] = [0u8; DATA_SIZE];
    let bytes_read = gpu_bulk_read(buf, sideband, fd2, read_buf.as_mut_ptr(), DATA_SIZE);
    core::ptr::write_volatile(result.add(2), bytes_read as u32);

    // Close read file
    gpu_hostcall_close(buf, fd2);

    if bytes_read != DATA_SIZE {
        return;
    }

    // Step 5: Verify content matches
    let mut match_ok: bool = true;
    let mut j: usize = 0;
    while j < DATA_SIZE {
        if read_buf[j] != test_data[j] {
            match_ok = false;
        }
        j += 1;
    }

    if match_ok {
        core::ptr::write_volatile(result.add(3), 1);
        core::ptr::write_volatile(result.add(0), 1); // Overall success
    }
}

// ============================================================
// Sharding-aware print test — uses gpu-runtime's hostcall path
// ============================================================

/// Print a message via gpu-runtime's `gpu_hostcall_print` which auto-detects
/// sharded vs legacy buffers. Thread 0 of each block prints "Shard N".
/// Increments `success_count` atomically on success.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sharded_print_test(buf: *mut u8, success_count: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;

    if thread_x != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(buf);

    // Format: "Shard N" with block index
    let mut msg_buf: [u8; 16] = [0u8; 16];
    msg_buf[0] = b'S';
    msg_buf[1] = b'h';
    msg_buf[2] = b'a';
    msg_buf[3] = b'r';
    msg_buf[4] = b'd';
    msg_buf[5] = b' ';
    let mut n = block_x;
    let mut pos = 6;
    if n >= 100 {
        msg_buf[pos] = b'0' + (n / 100) as u8;
        pos += 1;
        n %= 100;
    }
    if block_x >= 10 {
        msg_buf[pos] = b'0' + (n / 10) as u8;
        pos += 1;
        n %= 10;
    }
    msg_buf[pos] = b'0' + n as u8;
    pos += 1;

    if gpu_runtime::hostcall::gpu_hostcall_print(buf, msg_buf.as_ptr(), pos as u32).is_ok() {
        gpu_atomics::sys_fetch_add_u32(success_count, 1);
    }
}

// ============================================================
// Trace test kernel (trace-protocol.4)
// ============================================================

/// Multi-thread trace test: each of 32 threads emits a trace event.
///
/// Thread N sends: "trace from T{N}" at INFO level.
/// Atomically increments `success_count` on successful trace send.
/// Host should receive 32 distinct trace events with thread IDs 0-31.
///
/// `buf` = hostcall buffer, `success_count` = atomic counter (u32)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn trace_multithread_test(buf: *mut u8, success_count: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;

    gpu_runtime::panic::gpu_panic_init(buf);

    // Each thread emits a trace event with its thread ID
    gpu_runtime::gpu_trace!(buf, INFO, "trace from T{}", tid);

    // Count successful sends (gpu_trace! swallows errors, so count unconditionally)
    gpu_atomics::sys_fetch_add_u32(success_count, 1);
}

/// Trace + assert test: threads trace, then thread 0 asserts a condition.
///
/// All threads emit a DEBUG trace, then thread 0 asserts true (should pass).
/// This verifies assert does NOT trap when condition is true.
///
/// `buf` = hostcall buffer, `success_count` = atomic counter (u32)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn trace_assert_test(buf: *mut u8, success_count: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;

    gpu_runtime::panic::gpu_panic_init(buf);

    // All threads trace
    gpu_runtime::gpu_trace!(buf, DEBUG, "assert test T{}", tid);

    // Thread 0 asserts a true condition (should NOT trap)
    if tid == 0 {
        gpu_runtime::gpu_assert!(buf, 1 + 1 == 2, "math works");
    }

    gpu_atomics::sys_fetch_add_u32(success_count, 1);
}

// ============================================================
// Session test kernels (hc-session.3)
// ============================================================

/// Session test Kernel A: print a message and write a magic value.
///
/// Demonstrates that the hostcall session is active and working.
/// Writes 0xCAFE to `shared_state` so Kernel B can verify persistence.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn session_kernel_a(buf: *mut u8, shared_state: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(buf);

    let msg: &[u8] = b"session kernel A";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);

    // Write magic value for Kernel B to verify
    gpu_atomics::sys_store_release_u32(shared_state, 0xCAFE);
}

/// Session test Kernel B: read the magic value written by Kernel A.
///
/// Verifies that the hostcall session persisted across launches and that
/// shared mapped memory is readable. Writes 1 to `result` if magic matches.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn session_kernel_b(
    buf: *mut u8,
    shared_state: *mut u32,
    result: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(buf);

    // Read the value Kernel A wrote
    let magic = gpu_atomics::sys_load_acquire_u32(shared_state as *const u32);

    let msg: &[u8] = b"session kernel B";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);

    // Write result: 1 if magic matches, 0 otherwise
    gpu_atomics::sys_store_release_u32(result, if magic == 0xCAFE { 1 } else { 0 });
}

/// Multi-command kernel — polls command buffer, dispatches COMPUTE/PRINT/EXIT.
///
/// Thread 0 runs the command loop; all other threads return immediately.
/// Processes commands sequentially from the ring buffer until CMD_EXIT.
///
/// - CMD_COMPUTE: reads `count` u32s from input_ptr, doubles them, writes to output_ptr
/// - CMD_PRINT: forwards message to hostcall print
/// - CMD_EXIT: acknowledges and breaks
/// - CMD_NOP / unknown: acknowledges and continues
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_cmd_kernel(
    hc_buf: *mut u8,
    cmd_buf: *mut u8,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    loop {
        match gpu_runtime::cmd::cmd_poll(cmd_buf) {
            Some((gpu_protocol::CMD_COMPUTE, payload)) => {
                // payload layout: input_ptr(u64), output_ptr(u64), count(u32), op_code(u32)
                let input_ptr =
                    core::ptr::read_volatile(payload as *const u64) as *const u32;
                let output_ptr =
                    core::ptr::read_volatile(payload.add(8) as *const u64) as *mut u32;
                let count =
                    core::ptr::read_volatile(payload.add(16) as *const u32);
                // op_code ignored for now — always "double"
                for i in 0..count as isize {
                    let val = core::ptr::read_volatile(input_ptr.offset(i));
                    core::ptr::write_volatile(output_ptr.offset(i), val * 2);
                }
                gpu_runtime::cmd::cmd_ack(cmd_buf);
            }
            Some((gpu_protocol::CMD_PRINT, payload)) => {
                let msg_len =
                    core::ptr::read_volatile(payload as *const u32);
                let msg_ptr = payload.add(4);
                let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg_ptr, msg_len);
                gpu_runtime::cmd::cmd_ack(cmd_buf);
            }
            Some((gpu_protocol::CMD_EXIT, _)) => {
                gpu_runtime::cmd::cmd_ack(cmd_buf);
                break;
            }
            Some((_, _)) => {
                // Unknown or NOP — just acknowledge
                gpu_runtime::cmd::cmd_ack(cmd_buf);
            }
            None => {
                gpu_runtime::cmd::cmd_yield();
            }
        }
    }
}

/// Cross-launch pipeline: Kernel A writes values to a shared device buffer.
///
/// Writes `buf[i] = (i + 1) * 100` for i in 0..count.
/// Thread 0 only.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn pipeline_writer_kernel(
    hc_buf: *mut u8,
    data: *mut u32,
    count: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    for i in 0..count as isize {
        core::ptr::write_volatile(data.offset(i), (i as u32 + 1) * 100);
    }

    let msg: &[u8] = b"pipeline writer done";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);
}

/// Cross-launch pipeline: Kernel B reads values written by Kernel A and stores results.
///
/// Reads data[i], multiplies by 3, writes to result[i].
/// Thread 0 only.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn pipeline_reader_kernel(
    hc_buf: *mut u8,
    data: *const u32,
    result: *mut u32,
    count: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    for i in 0..count as isize {
        let val = core::ptr::read_volatile(data.offset(i));
        core::ptr::write_volatile(result.offset(i), val * 3);
    }

    let msg: &[u8] = b"pipeline reader done";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);
}

/// Iterative convergence kernel — Newton's method for integer square root.
///
/// For each input[i], computes floor(sqrt(input[i])) using Newton's method.
/// The iteration count is data-dependent (larger values need more iterations).
/// Stores sqrt result in output[i] and iteration count in iters[i].
///
/// Thread 0 only. Demonstrates autonomous data-dependent iteration.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn convergence_kernel(
    hc_buf: *mut u8,
    input: *const u32,
    output: *mut u32,
    iters: *mut u32,
    count: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    for idx in 0..count as isize {
        let n = core::ptr::read_volatile(input.offset(idx));

        if n <= 1 {
            core::ptr::write_volatile(output.offset(idx), n);
            core::ptr::write_volatile(iters.offset(idx), 0);
            continue;
        }

        // Newton's method: x_{k+1} = (x_k + n / x_k) / 2
        let mut x = n; // Initial guess
        let mut iter_count = 0u32;
        loop {
            let x_next = (x + n / x) / 2;
            iter_count += 1;
            if x_next >= x {
                break; // Converged (integer Newton's method converges when x_next >= x)
            }
            x = x_next;
        }

        core::ptr::write_volatile(output.offset(idx), x);
        core::ptr::write_volatile(iters.offset(idx), iter_count);
    }

    let msg: &[u8] = b"convergence done";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);
}

/// Multi-step autonomous pipeline kernel.
///
/// Reads input array, computes iterative sqrt, stores both results and
/// iteration counts. Prints progress via hostcall. Demonstrates autonomous
/// data-dependent computation within a single kernel launch.
///
/// args: hc_buf, input, output, iters, total_iters_ptr, count
/// - input[i]: values to compute sqrt of
/// - output[i]: sqrt results
/// - iters[i]: per-element iteration counts
/// - total_iters_ptr: sum of all iterations (single u32)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn autonomous_pipeline_kernel(
    hc_buf: *mut u8,
    input: *const u32,
    output: *mut u32,
    iters: *mut u32,
    total_iters: *mut u32,
    count: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    let msg: &[u8] = b"autonomous pipeline start";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);

    let mut total = 0u32;

    for idx in 0..count as isize {
        let n = core::ptr::read_volatile(input.offset(idx));

        if n <= 1 {
            core::ptr::write_volatile(output.offset(idx), n);
            core::ptr::write_volatile(iters.offset(idx), 0);
            continue;
        }

        // Newton's method for isqrt
        let mut x = n;
        let mut iter_count = 0u32;
        loop {
            let x_next = (x + n / x) / 2;
            iter_count += 1;
            if x_next >= x {
                break;
            }
            x = x_next;
        }

        core::ptr::write_volatile(output.offset(idx), x);
        core::ptr::write_volatile(iters.offset(idx), iter_count);
        total += iter_count;
    }

    core::ptr::write_volatile(total_iters, total);

    let msg: &[u8] = b"autonomous pipeline done";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);
}

/// Flight recorder test kernel — writes N trace events to the ring buffer.
///
/// Thread 0 writes events with different levels, then optionally "crashes" if
/// the crash flag in params is set.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flight_recorder_test(
    hc_buf: *mut u8,
    fr_buf: *mut u8,
    should_crash: *const u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    // Write several trace events to flight recorder
    let msg1: &[u8] = b"initializing";
    gpu_runtime::flight_recorder::fr_record(
        fr_buf,
        gpu_protocol::TRACE_LEVEL_INFO,
        msg1.as_ptr(),
        msg1.len() as u32,
    );

    let msg2: &[u8] = b"processing data";
    gpu_runtime::flight_recorder::fr_record(
        fr_buf,
        gpu_protocol::TRACE_LEVEL_DEBUG,
        msg2.as_ptr(),
        msg2.len() as u32,
    );

    let msg3: &[u8] = b"checkpoint reached";
    gpu_runtime::flight_recorder::fr_record(
        fr_buf,
        gpu_protocol::TRACE_LEVEL_INFO,
        msg3.as_ptr(),
        msg3.len() as u32,
    );

    let msg4: &[u8] = b"warning: low memory";
    gpu_runtime::flight_recorder::fr_record(
        fr_buf,
        gpu_protocol::TRACE_LEVEL_WARN,
        msg4.as_ptr(),
        msg4.len() as u32,
    );

    let msg5: &[u8] = b"computation complete";
    gpu_runtime::flight_recorder::fr_record(
        fr_buf,
        gpu_protocol::TRACE_LEVEL_INFO,
        msg5.as_ptr(),
        msg5.len() as u32,
    );

    // Check if we should simulate a crash
    let crash = core::ptr::read_volatile(should_crash);
    if crash != 0 {
        gpu_runtime::flight_recorder::fr_set_crashed(fr_buf);
        // Don't actually trap — just set the flag for testing
    }

    let msg: &[u8] = b"flight recorder test done";
    let _ = gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, msg.as_ptr(), msg.len() as u32);
}
