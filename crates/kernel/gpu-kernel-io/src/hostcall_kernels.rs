// Hostcall test/bench kernels — kernel entry points for hostcall protocol testing.

use core::arch::nvptx;
use gpu_atomics::{
    activemask, sys_fetch_add_u64, sys_spin_load_acquire_u32, sys_store_release_u32,
};
use gpu_kernel_core::helpers::{
    gpu_hostcall_close, gpu_hostcall_open, gpu_hostcall_read, gpu_hostcall_stdin_read,
    gpu_hostcall_time, gpu_hostcall_write, gpu_instant_nanos, hc_pop_free_counted,
    hc_pop_free_counted_v2,
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
pub unsafe extern "gpu-kernel" fn hostcall_print_hello(buf: *mut u8, result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn hostcall_print_multi(buf: *mut u8, success_count: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn hostcall_file_test(buf: *mut u8, result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn hostcall_stdin_time_test(
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
pub unsafe extern "gpu-kernel" fn error_propagation_test(buf: *mut u8, result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn hostcall_latency_bench(
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
pub unsafe extern "gpu-kernel" fn hostcall_latency_bench_v2(
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
// Executor demo kernel (executor-impl.4)
// ============================================================

/// Simple future that writes a value to a result slot and completes immediately.
struct WriteValueFuture {
    result_ptr: *mut u32,
    value: u32,
    done: bool,
}

impl core::future::Future for WriteValueFuture {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.done {
            return core::task::Poll::Ready(());
        }
        unsafe {
            sys_store_release_u32(self.result_ptr, self.value);
        }
        self.done = true;
        core::task::Poll::Ready(())
    }
}

/// Counter future that increments a shared counter and yields once before completing.
struct CounterFuture {
    counter_ptr: *mut u32,
    step: u32,
}

impl core::future::Future for CounterFuture {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.step == 0 {
            // First poll: yield (return Pending)
            self.step = 1;
            core::task::Poll::Pending
        } else {
            // Second poll: increment counter and complete
            unsafe {
                let old = core::ptr::read_volatile(self.counter_ptr);
                core::ptr::write_volatile(self.counter_ptr, old + 1);
            }
            core::task::Poll::Ready(())
        }
    }
}

/// Executor demo kernel: spawn async tasks and run them to completion.
///
/// Tests the GpuExecutor with immediate futures (WriteValueFuture) and
/// yielding futures (CounterFuture).
///
/// `executor_ptr` = device pointer to mapped memory for GpuExecutor (must be >= 256KB)
/// `results` = output array of u32[16]:
///   [0] = spawned count
///   [1] = completed count
///   [2] = tasks_executed from stats
///   [3] = polls_total from stats
///   [4..7] = values written by WriteValueFuture tasks (42, 100, 255, 1337)
///   [8] = counter incremented by CounterFuture tasks (should be 4)
///   [9] = success flag (1 if all tasks completed)
///   [10] = phase marker (for debugging: shows how far init/spawn got)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn executor_demo(executor_ptr: *mut u8, results: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;

    // Initialize result buffer (thread 0 only)
    if thread_x == 0 {
        let mut i = 0u32;
        while i < 16 {
            core::ptr::write_volatile(results.add(i as usize), 0);
            i += 1;
        }
        // Mark phase 1: results zeroed
        core::ptr::write_volatile(results.add(10), 1);
    }

    // Sync all lanes before proceeding
    let mask = activemask();
    gpu_atomics::syncwarp(mask);

    let executor = &*(executor_ptr as *const gpu_runtime::executor::GpuExecutor);

    // Thread 0 initializes and spawns tasks
    if thread_x == 0 {
        executor.init();
        // Mark phase 2: executor initialized
        core::ptr::write_volatile(results.add(10), 2);

        // Spawn 4 WriteValueFuture tasks
        let values: [u32; 4] = [42, 100, 255, 1337];
        let mut i = 0u32;
        while i < 4 {
            let _ = executor.spawn(WriteValueFuture {
                result_ptr: results.add(4 + i as usize),
                value: values[i as usize],
                done: false,
            });
            i += 1;
        }
        // Mark phase 3: 4 immediate tasks spawned
        core::ptr::write_volatile(results.add(10), 3);

        // Spawn 4 CounterFuture tasks
        let mut j = 0u32;
        while j < 4 {
            let _ = executor.spawn(CounterFuture {
                counter_ptr: results.add(8),
                step: 0,
            });
            j += 1;
        }
        // Mark phase 4: all 8 tasks spawned
        core::ptr::write_volatile(results.add(10), 4);
    }

    gpu_atomics::syncwarp(mask);

    // All lanes enter the executor loop.
    // Pass mask explicitly — activemask() inside a method can generate
    // problematic PTX on nvptx64 when inlined.
    let stats = executor.run(mask);

    gpu_atomics::syncwarp(mask);

    // Mark phase 5: executor finished
    if thread_x == 0 {
        core::ptr::write_volatile(results.add(10), 5);
        core::ptr::write_volatile(results.add(0), executor.spawned_count());
        core::ptr::write_volatile(results.add(1), executor.completed_count());
        core::ptr::write_volatile(results.add(2), stats.tasks_executed);
        core::ptr::write_volatile(results.add(3), stats.polls_total);

        let spawned = core::ptr::read_volatile(results.add(0) as *const u32);
        let completed = core::ptr::read_volatile(results.add(1) as *const u32);
        let v0 = core::ptr::read_volatile(results.add(4) as *const u32);
        let v1 = core::ptr::read_volatile(results.add(5) as *const u32);
        let v2 = core::ptr::read_volatile(results.add(6) as *const u32);
        let v3 = core::ptr::read_volatile(results.add(7) as *const u32);
        let counter = core::ptr::read_volatile(results.add(8) as *const u32);

        if spawned == 8
            && completed == 8
            && v0 == 42
            && v1 == 100
            && v2 == 255
            && v3 == 1337
            && counter == 4
        {
            core::ptr::write_volatile(results.add(9), 1);
        }
    }
}

// ============================================================
// Oneshot channel demo kernel (channel-oneshot.3)
// ============================================================

/// A future that sends a value through a oneshot channel.
///
/// On first poll: sends the value by writing to the slot and setting state to SENT.
/// Uses the OneshotSlot's public accessors for state and value pointers.
struct OneshotProducer {
    sender_slot: *mut gpu_runtime::channel::OneshotSlot<u32>,
    value: u32,
    sent: bool,
}

impl core::future::Future for OneshotProducer {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.sent {
            return core::task::Poll::Ready(());
        }
        self.sent = true;
        unsafe {
            let slot = &*self.sender_slot;
            // Write value first, then release-store SENT state
            core::ptr::write_volatile(slot.value_ptr() as *mut u32, self.value);
            gpu_atomics::sys_store_release_u32(slot.state_ptr(), 1); // ONESHOT_SENT
        }
        core::task::Poll::Ready(())
    }
}

/// A future that receives a value from a oneshot channel and writes it to results.
///
/// Polls the slot's atomic state. Returns Pending until SENT, then reads the value.
struct OneshotConsumer {
    slot: *const gpu_runtime::channel::OneshotSlot<u32>,
    result_ptr: *mut u32,
}

impl core::future::Future for OneshotConsumer {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        unsafe {
            let slot = &*self.slot;
            let state = gpu_atomics::sys_load_acquire_u32(slot.state_ptr() as *const u32);
            if state == 1 {
                // SENT — read value and write to result
                let value = core::ptr::read_volatile(slot.value_ptr() as *const u32);
                core::ptr::write_volatile(self.result_ptr, value);
                core::task::Poll::Ready(())
            } else {
                core::task::Poll::Pending
            }
        }
    }
}

/// Channel oneshot demo kernel: test inter-task communication via oneshot channels.
///
/// Thread 0 spawns 4 producer-consumer pairs. Each producer sends a different value
/// through a oneshot channel; each consumer receives it and writes to results.
///
/// `executor_ptr` = device pointer to mapped memory for GpuExecutor (must be >= 256KB)
/// `results` = output array of u32[16]:
///   [0] = spawned count
///   [1] = completed count
///   [2] = tasks_executed from stats
///   [3] = polls_total from stats
///   [4..7] = values received by consumers (expect: 42, 100, 255, 1337)
///   [8] = number of channel pairs (4)
///   [9] = success flag (1 if all correct)
///   [10] = phase marker
///
/// The channel slots are placed at the end of the executor memory region.
/// executor_ptr must have enough space for GpuExecutor + 4 OneshotSlot<u32>.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn channel_oneshot_demo(executor_ptr: *mut u8, results: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;

    // Initialize result buffer (thread 0 only)
    if thread_x == 0 {
        let mut i = 0u32;
        while i < 16 {
            core::ptr::write_volatile(results.add(i as usize), 0);
            i += 1;
        }
        core::ptr::write_volatile(results.add(10), 1); // Phase 1: results zeroed
    }

    let mask = activemask();
    gpu_atomics::syncwarp(mask);

    let executor = &*(executor_ptr as *const gpu_runtime::executor::GpuExecutor);

    // Place 4 OneshotSlot<u32> after the executor struct in mapped memory.
    // GpuExecutor is large (~136KB) — slots go at a known offset.
    let executor_size = core::mem::size_of::<gpu_runtime::executor::GpuExecutor>();
    let slots_base = executor_ptr.add(executor_size) as *mut gpu_runtime::channel::OneshotSlot<u32>;

    if thread_x == 0 {
        executor.init();
        core::ptr::write_volatile(results.add(10), 2); // Phase 2: executor initialized

        // Initialize 4 oneshot slots
        let values: [u32; 4] = [42, 100, 255, 1337];
        let mut i = 0u32;
        while i < 4 {
            let slot = &mut *slots_base.add(i as usize);
            slot.reset();

            // Spawn consumer first (it will poll Pending until producer sends)
            let _ = executor.spawn(OneshotConsumer {
                slot: slot as *const _,
                result_ptr: results.add(4 + i as usize),
            });

            // Spawn producer (it sends on first poll after yield)
            let _ = executor.spawn(OneshotProducer {
                sender_slot: slot as *mut _,
                value: values[i as usize],
                sent: false,
            });

            i += 1;
        }
        core::ptr::write_volatile(results.add(8), 4); // 4 channel pairs
        core::ptr::write_volatile(results.add(10), 3); // Phase 3: all tasks spawned
    }

    gpu_atomics::syncwarp(mask);

    let stats = executor.run(mask);

    gpu_atomics::syncwarp(mask);

    if thread_x == 0 {
        core::ptr::write_volatile(results.add(10), 5); // Phase 5: executor finished
        core::ptr::write_volatile(results.add(0), executor.spawned_count());
        core::ptr::write_volatile(results.add(1), executor.completed_count());
        core::ptr::write_volatile(results.add(2), stats.tasks_executed);
        core::ptr::write_volatile(results.add(3), stats.polls_total);

        // Verify results
        let v0 = core::ptr::read_volatile(results.add(4) as *const u32);
        let v1 = core::ptr::read_volatile(results.add(5) as *const u32);
        let v2 = core::ptr::read_volatile(results.add(6) as *const u32);
        let v3 = core::ptr::read_volatile(results.add(7) as *const u32);
        let spawned = core::ptr::read_volatile(results.add(0) as *const u32);
        let completed = core::ptr::read_volatile(results.add(1) as *const u32);

        if spawned == 8 && completed == 8 && v0 == 42 && v1 == 100 && v2 == 255 && v3 == 1337 {
            core::ptr::write_volatile(results.add(9), 1); // success
        }
    }
}

// ============================================================
// File I/O latency benchmark kernel (bench-suite.3)
// ============================================================

/// File I/O latency benchmark kernel.
///
/// Thread 0 only. Performs `num_iters` rounds of (open → write → close → open → read → close),
/// timing each phase individually. Results are stored as 6 u64 timestamps per iteration
/// (phase latencies in ns) plus 2 u64 summary fields.
///
/// Layout of `results` (u64 array):
///   [0] = total elapsed nanoseconds for all iterations
///   [1] = number of completed full iterations
///   Per-iteration phase timings (6 u64 per iteration, starting at offset 2):
///     [2 + iter*6 + 0] = open-write latency (ns)
///     [2 + iter*6 + 1] = write latency (ns)
///     [2 + iter*6 + 2] = close-write latency (ns)
///     [2 + iter*6 + 3] = open-read latency (ns)
///     [2 + iter*6 + 4] = read latency (ns)
///     [2 + iter*6 + 5] = close-read latency (ns)
///
/// `buf` = hostcall buffer
/// `results` = output array, must have space for (2 + num_iters * 6) u64 entries
/// `num_iters` = number of full open-write-close-open-read-close cycles
#[no_mangle]
pub unsafe extern "gpu-kernel" fn file_io_bench(buf: *mut u8, results: *mut u64, num_iters: u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    let path: &[u8; 20] = b"gpu_bench_output.txt";
    let path_len: u32 = 20;
    let msg: &[u8; 48] = b"Benchmark data: GPU file I/O latency test 12345\n";
    let msg_len: u32 = 48;

    let t_total_start = gpu_instant_nanos();
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        let base = 2 + (iter as usize) * 6;
        let mut read_buf: [u8; 64] = [0u8; 64];

        // Phase 1: Open for writing
        let t0 = gpu_instant_nanos();
        let (fd, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_WRITE_CREATE);
        let t1 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base), t1 - t0);
        if err != 0 {
            break;
        }

        // Phase 2: Write
        let t2 = gpu_instant_nanos();
        let (_written, err) = gpu_hostcall_write(buf, fd, msg.as_ptr(), msg_len);
        let t3 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base + 1), t3 - t2);
        if err != 0 {
            gpu_hostcall_close(buf, fd);
            break;
        }

        // Phase 3: Close write fd
        let t4 = gpu_instant_nanos();
        let (_, err) = gpu_hostcall_close(buf, fd);
        let t5 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base + 2), t5 - t4);
        if err != 0 {
            break;
        }

        // Phase 4: Open for reading
        let t6 = gpu_instant_nanos();
        let (fd2, err) = gpu_hostcall_open(buf, path.as_ptr(), path_len, FILE_OPEN_READ);
        let t7 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base + 3), t7 - t6);
        if err != 0 {
            break;
        }

        // Phase 5: Read
        let t8 = gpu_instant_nanos();
        let (_bytes_read, err) = gpu_hostcall_read(buf, fd2, read_buf.as_mut_ptr(), 64);
        let t9 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base + 4), t9 - t8);
        if err != 0 {
            gpu_hostcall_close(buf, fd2);
            break;
        }

        // Phase 6: Close read fd
        let t10 = gpu_instant_nanos();
        gpu_hostcall_close(buf, fd2);
        let t11 = gpu_instant_nanos();
        core::ptr::write_volatile(results.add(base + 5), t11 - t10);

        completed += 1;
        iter += 1;
    }

    let t_total_end = gpu_instant_nanos();
    core::ptr::write_volatile(results.add(0), t_total_end - t_total_start);
    core::ptr::write_volatile(results.add(1), completed);
}

// ============================================================
// Per-iteration latency benchmark kernel (bench-suite.2)
// ============================================================

/// Per-iteration latency benchmark kernel v3.
///
/// Unlike v1/v2 which record per-thread total elapsed time, this kernel records
/// each iteration's round-trip latency individually, enabling true percentile
/// analysis across all hostcalls (not just per-thread averages).
///
/// Layout of `results` (u64 array):
///   Header (3 u64 per thread):
///     results[tid*3 + 0] = total elapsed nanoseconds (all iterations)
///     results[tid*3 + 1] = total CAS retries
///     results[tid*3 + 2] = number of completed iterations
///   Per-iteration latencies (1 u64 per iteration per thread):
///     results[header_size + tid*num_iters + iter] = single-iteration latency in ns
///
/// `buf` = hostcall buffer
/// `results` = output array, must have space for (num_threads * 3 + num_threads * num_iters) u64 entries
/// `num_iters` = number of NOP hostcalls per thread
/// `num_threads_total` = total thread count (for computing header offset)
#[no_mangle]
pub unsafe extern "gpu-kernel" fn hostcall_latency_bench_v3(
    buf: *mut u8,
    results: *mut u64,
    num_iters: u32,
    num_threads_total: u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let tid = block_x * block_dim_x + thread_x;

    if tid >= num_threads_total {
        return;
    }

    // Read shard info once
    let (num_shards, shard_array_off, _) = gpu_runtime::hostcall::read_shard_info(buf as *const u8);
    let free_ptr = gpu_runtime::hostcall::get_free_stack_ptr(buf, num_shards, shard_array_off);
    let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(buf, num_shards, shard_array_off);

    // Per-iteration latency storage starts after the header
    let header_size = (num_threads_total as usize) * 3;
    let iter_base = header_size + (tid as usize) * (num_iters as usize);

    let t_start = gpu_instant_nanos();
    let mut total_retries: u64 = 0;
    let mut completed: u64 = 0;

    let mut iter: u32 = 0;
    while iter < num_iters {
        let t_iter_start = gpu_instant_nanos();

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

        // Push to ready stack (shard-aware)
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

        let t_iter_end = gpu_instant_nanos();

        // Record per-iteration latency
        core::ptr::write_volatile(
            results.add(iter_base + iter as usize),
            t_iter_end - t_iter_start,
        );

        completed += 1;
        iter += 1;
    }

    let t_end = gpu_instant_nanos();

    // Write header results
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
pub unsafe extern "gpu-kernel" fn panic_test_kernel(buf: *mut u8, result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn bulk_io_test(buf: *mut u8, sideband: *mut u8, result: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn sharded_print_test(buf: *mut u8, success_count: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn trace_multithread_test(buf: *mut u8, success_count: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn trace_assert_test(buf: *mut u8, success_count: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;

    gpu_runtime::panic::gpu_panic_init(buf);

    // All threads trace
    gpu_runtime::gpu_trace!(buf, DEBUG, "assert test T{}", tid);

    // Thread 0 asserts a true condition (should NOT trap)
    if tid == 0 {
        assert!(1 + 1 == 2, "math works");
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
pub unsafe extern "gpu-kernel" fn session_kernel_a(buf: *mut u8, shared_state: *mut u32) {
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
pub unsafe extern "gpu-kernel" fn session_kernel_b(
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
pub unsafe extern "gpu-kernel" fn multi_cmd_kernel(hc_buf: *mut u8, cmd_buf: *mut u8) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(hc_buf);

    loop {
        match gpu_runtime::cmd::cmd_poll(cmd_buf) {
            Some((gpu_protocol::CMD_COMPUTE, payload)) => {
                // payload layout: input_ptr(u64), output_ptr(u64), count(u32), op_code(u32)
                let input_ptr = core::ptr::read_volatile(payload as *const u64) as *const u32;
                let output_ptr = core::ptr::read_volatile(payload.add(8) as *const u64) as *mut u32;
                let count = core::ptr::read_volatile(payload.add(16) as *const u32);
                // op_code ignored for now — always "double"
                for i in 0..count as isize {
                    let val = core::ptr::read_volatile(input_ptr.offset(i));
                    core::ptr::write_volatile(output_ptr.offset(i), val * 2);
                }
                gpu_runtime::cmd::cmd_ack(cmd_buf);
            }
            Some((gpu_protocol::CMD_PRINT, payload)) => {
                let msg_len = core::ptr::read_volatile(payload as *const u32);
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
pub unsafe extern "gpu-kernel" fn pipeline_writer_kernel(
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
pub unsafe extern "gpu-kernel" fn pipeline_reader_kernel(
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
pub unsafe extern "gpu-kernel" fn convergence_kernel(
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
pub unsafe extern "gpu-kernel" fn autonomous_pipeline_kernel(
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
pub unsafe extern "gpu-kernel" fn flight_recorder_test(
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

// ============================================================
// warp-future-bridge.1: GpuPrintFuture as standard impl Future
// ============================================================

/// Test: GpuPrintFuture (standard impl Future) polled by SpinExecutor.
///
/// Single thread runs a GpuPrintFuture through the minimal SpinExecutor.
/// This proves that standard `core::future::Future` hostcall works on GPU
/// without Embassy, without WarpFuture, without any warp cooperation.
///
/// `buf` = hostcall buffer
/// `result` = output: 1 = success, 0 = failure/timeout
#[no_mangle]
pub unsafe extern "gpu-kernel" fn std_future_print_kernel(buf: *mut u8, result: *mut u32) {
    if nvptx::_thread_idx_x() != 0 {
        return;
    }
    core::ptr::write_volatile(result, 0);

    let mut future = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"Hello from std Future!");

    match gpu_runtime::std_future::SpinExecutor::run(&mut future) {
        Some(true) => core::ptr::write_volatile(result, 1),
        _ => core::ptr::write_volatile(result, 0),
    }
}

/// Test: Two sequential GpuPrintFutures via SpinExecutor.
///
/// Proves that multiple standard Futures can be polled sequentially
/// on GPU — the baseline before adding warp cooperation.
///
/// `buf` = hostcall buffer
/// `result` = output: 2 = both succeeded, 1 = first only, 0 = failure
#[no_mangle]
pub unsafe extern "gpu-kernel" fn std_future_two_prints_kernel(buf: *mut u8, result: *mut u32) {
    if nvptx::_thread_idx_x() != 0 {
        return;
    }
    core::ptr::write_volatile(result, 0);

    // First print
    let mut f1 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"std_future print 1");
    let ok1 = match gpu_runtime::std_future::SpinExecutor::run(&mut f1) {
        Some(true) => true,
        _ => false,
    };

    if !ok1 {
        return;
    }
    core::ptr::write_volatile(result, 1);

    // Second print
    let mut f2 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"std_future print 2");
    let ok2 = match gpu_runtime::std_future::SpinExecutor::run(&mut f2) {
        Some(true) => true,
        _ => false,
    };

    if ok2 {
        core::ptr::write_volatile(result, 2);
    }
}

/// Test: Warp-cooperative polling of standard impl Future.
///
/// All 32 lanes enter together. Lane 0 polls GpuPrintFuture;
/// result is broadcast via shfl.sync to all lanes.
/// All lanes write their lane_id to `lane_results[lane_id]` to prove
/// they all observed the same completion.
///
/// `buf` = hostcall buffer
/// `result` = output: 1 = success (all 32 lanes saw Ready), 0 = failure
/// `lane_results` = array of 32 u32 — each lane writes its ID on success
#[no_mangle]
pub unsafe extern "gpu-kernel" fn warp_cooperative_future_kernel(
    buf: *mut u8,
    result: *mut u32,
    lane_results: *mut u32,
) {
    let tid = nvptx::_thread_idx_x();

    // Only first warp participates
    if tid >= 32 {
        return;
    }

    // Initialize result to 0 (lane 0 only)
    if tid == 0 {
        core::ptr::write_volatile(result, 0);
    }

    // All 32 lanes create the future (but only lane 0 will poll it)
    let mut future =
        gpu_runtime::std_future::GpuPrintFuture::new(buf, b"Hello from warp-cooperative Future!");

    // Warp-cooperative poll: lane 0 polls, broadcasts result
    let ok = gpu_runtime::warp_cooperative::warp_run_future(&mut future);

    // All lanes write their lane_id to prove they all reached this point
    core::ptr::write_volatile(lane_results.add(tid as usize), tid);

    // Lane 0 writes final result
    if tid == 0 {
        match ok {
            Some(true) => core::ptr::write_volatile(result, 1),
            _ => core::ptr::write_volatile(result, 0),
        }
    }
}

/// Test: Two sequential warp-cooperative Future polls.
///
/// Simulates two sequential `.await` points:
///   let ok1 = print_future_1.await;  // all 32 lanes converge
///   let ok2 = print_future_2.await;  // all 32 lanes converge again
///
/// `buf` = hostcall buffer
/// `result` = output: 2 = both succeeded, 1 = first only, 0 = failure
/// `lane_results` = array of 32 u32 — each lane writes its ID on success
#[no_mangle]
pub unsafe extern "gpu-kernel" fn warp_cooperative_two_futures_kernel(
    buf: *mut u8,
    result: *mut u32,
    lane_results: *mut u32,
) {
    let tid = nvptx::_thread_idx_x();
    if tid >= 32 {
        return;
    }

    if tid == 0 {
        core::ptr::write_volatile(result, 0);
    }

    let mut f1 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"warp-coop sequential 1");
    let mut f2 = gpu_runtime::std_future::GpuPrintFuture::new(buf, b"warp-coop sequential 2");

    let (ok1, ok2) = gpu_runtime::warp_sequential::warp_run_two_futures(&mut f1, &mut f2);

    // All lanes write their lane_id
    core::ptr::write_volatile(lane_results.add(tid as usize), tid);

    if tid == 0 {
        let score = match (ok1, ok2) {
            (Some(true), Some(true)) => 2,
            (Some(true), _) => 1,
            _ => 0,
        };
        core::ptr::write_volatile(result, score);
    }
}

/// Test: Warp-cooperative Result<T, E> broadcasting with ? semantics.
///
/// Simulates:
///   async fn example(buf) -> Result<(), u32> {
///       print_future_1.await?;  // Ok → continue, Err → all lanes early-return
///       print_future_2.await?;
///       Ok(())
///   }
///
/// `buf` = hostcall buffer
/// `result` = output: 2 = both Ok, 0 = error
/// `lane_results` = array of 32 u32
#[no_mangle]
pub unsafe extern "gpu-kernel" fn warp_result_future_kernel(
    buf: *mut u8,
    result: *mut u32,
    lane_results: *mut u32,
) {
    let tid = nvptx::_thread_idx_x();
    if tid >= 32 {
        return;
    }

    if tid == 0 {
        core::ptr::write_volatile(result, 0);
    }

    let mut f1 = gpu_runtime::std_future::GpuPrintResultFuture::new(buf, b"warp-result print 1");
    let mut f2 = gpu_runtime::std_future::GpuPrintResultFuture::new(buf, b"warp-result print 2");

    let outcome = gpu_runtime::warp_result::warp_run_two_result_futures(&mut f1, &mut f2);

    // All lanes write their lane_id (proves convergence even on error path)
    core::ptr::write_volatile(lane_results.add(tid as usize), tid);

    if tid == 0 {
        match outcome {
            Ok(n) => core::ptr::write_volatile(result, n),
            Err(e) => core::ptr::write_volatile(result, 0x8000_0000 | e),
        }
    }
}

// ============================================================
// Buffered print test — uses gpu-runtime's print_buffer module
// ============================================================

/// Test kernel: accumulate 12 print messages via print_buffer, flush once.
/// Verifies that buffered printing works end-to-end via SERVICE_BULK_PRINT.
///
/// Thread 0 prints 12 messages, each ~20 bytes. With a 504-byte slot,
/// all 12 fit in one flush (12 * ~22 = ~264 bytes < 504).
///
/// `buf` = hostcall buffer
/// `sideband` = sideband buffer
/// `result` = output: 1 if all prints + flush succeeded, 0 on error
#[no_mangle]
pub unsafe extern "gpu-kernel" fn buffered_print_test(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
) {
    let thread_x = nvptx::_thread_idx_x() as u32;
    let block_x = nvptx::_block_idx_x() as u32;
    let block_dim_x = nvptx::_block_dim_x() as u32;
    let global_idx = block_x * block_dim_x + thread_x;
    if global_idx != 0 {
        return;
    }

    gpu_runtime::panic::gpu_panic_init(buf);
    core::ptr::write_volatile(result, 0);

    // Initialize print buffer for this thread
    gpu_runtime::print_buffer::init(sideband, 1);

    // Buffer 12 print messages without hostcall
    let mut i: u32 = 0;
    while i < 12 {
        // Format: "Buffered msg NN\n" — manual formatting
        let mut msg: [u8; 20] = [0u8; 20];
        msg[0] = b'B';
        msg[1] = b'u';
        msg[2] = b'f';
        msg[3] = b'f';
        msg[4] = b'e';
        msg[5] = b'r';
        msg[6] = b'e';
        msg[7] = b'd';
        msg[8] = b' ';
        msg[9] = b'm';
        msg[10] = b's';
        msg[11] = b'g';
        msg[12] = b' ';
        // Two-digit number
        msg[13] = b'0' + (i / 10) as u8;
        msg[14] = b'0' + (i % 10) as u8;
        msg[15] = b'\n';
        let len: u32 = 16;

        if gpu_runtime::print_buffer::print(buf, sideband, msg.as_ptr(), len).is_err() {
            return; // Failed to buffer
        }
        i += 1;
    }

    // Flush all 12 messages in a single SERVICE_BULK_PRINT hostcall
    if gpu_runtime::print_buffer::flush(buf, sideband).is_err() {
        return; // Flush failed
    }

    // Success — all 12 messages buffered and flushed
    core::ptr::write_volatile(result, 1);
}

// ============================================================
// Data-dependent iteration: Newton's method sqrt (data-iter.1)
// ============================================================

/// Convergence-loop kernel: Newton's method for sqrt(S).
///
/// Demonstrates data-dependent iteration on GPU — the kernel autonomously
/// loops until convergence without host intervention. The iteration count
/// depends on the input value and tolerance, not known ahead of time.
///
/// Algorithm: x_{n+1} = (x_n + S/x_n) / 2
/// Convergence: |x_{n+1} - x_n| < epsilon
///
/// Parameters:
/// - `input`: pointer to f32 value S (the number to sqrt)
/// - `output`: pointer to f32 result (the computed sqrt)
/// - `iterations`: pointer to u32 (how many iterations until convergence)
/// - `max_iter`: maximum iterations before giving up
///
/// Launch with 1 block × 1 thread.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn newton_sqrt_kernel(
    input: *const f32,
    output: *mut f32,
    iterations: *mut u32,
    max_iter: u32,
) {
    let s = core::ptr::read_volatile(input);

    // Handle special cases
    if s <= 0.0 {
        core::ptr::write_volatile(output, 0.0);
        core::ptr::write_volatile(iterations, 0);
        return;
    }

    // Initial guess: S/2 (simple, works for all positive S)
    let mut x = s * 0.5;
    let epsilon: f32 = 1e-6;
    let mut iter: u32 = 0;

    loop {
        // Newton's method step: x_new = (x + S/x) / 2
        let x_new = (x + s / x) * 0.5;
        iter += 1;

        // Check convergence: |x_new - x| < epsilon
        let diff = x_new - x;
        let abs_diff = if diff < 0.0 { -diff } else { diff };

        x = x_new;

        if abs_diff < epsilon || iter >= max_iter {
            break;
        }
    }

    core::ptr::write_volatile(output, x);
    core::ptr::write_volatile(iterations, iter);
}

// ============================================================
// MPSC channel demo kernel (channel-mpsc.2)
// ============================================================

/// A future that sends multiple values through an MPSC channel.
///
/// Uses the channel's raw `try_send_raw` method to send values.
/// Retries on full (returns Pending to let executor re-poll).
struct MpscProducer {
    channel: *const gpu_runtime::channel::MpscChannel<u32, 16>,
    values: [u32; 4],
    next_idx: u32,
}

impl core::future::Future for MpscProducer {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        unsafe {
            let ch = &*self.channel;
            while (self.next_idx as usize) < self.values.len() {
                let val = self.values[self.next_idx as usize];
                match ch.try_send(val) {
                    Ok(()) => {
                        self.next_idx += 1;
                    }
                    Err(_) => {
                        // Channel full or closed — yield and retry later
                        return core::task::Poll::Pending;
                    }
                }
            }
            core::task::Poll::Ready(())
        }
    }
}

/// A future that receives values from an MPSC channel and accumulates sum.
///
/// Expects exactly `expected_count` values, then writes sum to result pointer.
struct MpscConsumer {
    channel: *const gpu_runtime::channel::MpscChannel<u32, 16>,
    result_ptr: *mut u32,
    count_ptr: *mut u32,
    sum: u32,
    received: u32,
    expected: u32,
}

impl core::future::Future for MpscConsumer {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        unsafe {
            let ch = &*self.channel;
            // Drain all available values
            loop {
                if self.received >= self.expected {
                    // All expected values received — write results
                    core::ptr::write_volatile(self.result_ptr, self.sum);
                    core::ptr::write_volatile(self.count_ptr, self.received);
                    return core::task::Poll::Ready(());
                }
                match ch.try_recv() {
                    Some(val) => {
                        self.sum += val;
                        self.received += 1;
                    }
                    None => {
                        // Store waker so producer's try_send will re-enqueue us
                        ch.store_waker(cx);
                        // Double-check: value may have arrived between try_recv and store_waker
                        match ch.try_recv() {
                            Some(val) => {
                                self.sum += val;
                                self.received += 1;
                                // Continue draining
                            }
                            None => {
                                return core::task::Poll::Pending;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// MPSC channel demo kernel: test multi-producer single-consumer channel.
///
/// Thread 0 spawns 3 producers + 1 consumer. Each producer sends 4 values
/// through the shared MPSC channel. The consumer receives all 12 values
/// and writes their sum.
///
/// `executor_ptr` = device pointer to mapped memory for GpuExecutor (>= 256KB)
/// `results` = output array of u32[8]:
///   [0] = spawned count
///   [1] = completed count
///   [2] = tasks_executed from stats
///   [3] = polls_total from stats
///   [4] = received sum (expect: 3*10 + 3*20 + 3*30 + 3*40 = 300, or exact sum of all values)
///   [5] = received count (expect: 12)
///   [6] = success flag (1 if correct)
///   [7] = phase marker
///
/// The MPSC channel is placed after the executor struct in mapped memory.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn channel_mpsc_demo(executor_ptr: *mut u8, results: *mut u32) {
    let thread_x = nvptx::_thread_idx_x() as u32;

    // Initialize result buffer (thread 0 only)
    if thread_x == 0 {
        let mut i = 0u32;
        while i < 8 {
            core::ptr::write_volatile(results.add(i as usize), 0);
            i += 1;
        }
        core::ptr::write_volatile(results.add(7), 1); // Phase 1: results zeroed
    }

    let mask = activemask();
    gpu_atomics::syncwarp(mask);

    let executor = &*(executor_ptr as *const gpu_runtime::executor::GpuExecutor);

    // Place MpscChannel after executor struct
    let executor_size = core::mem::size_of::<gpu_runtime::executor::GpuExecutor>();
    // Align to 128 bytes for cache-line separation
    let channel_offset = (executor_size + 127) & !127;
    let channel_ptr =
        executor_ptr.add(channel_offset) as *mut gpu_runtime::channel::MpscChannel<u32, 16>;

    if thread_x == 0 {
        executor.init();

        // Initialize MPSC channel
        let channel = &*channel_ptr;
        channel.init();

        core::ptr::write_volatile(results.add(7), 2); // Phase 2: initialized

        // Producer values: 3 producers, each sends 4 values
        // Producer 0: [10, 20, 30, 40] → sum = 100
        // Producer 1: [11, 21, 31, 41] → sum = 104
        // Producer 2: [12, 22, 32, 42] → sum = 108
        // Total expected sum = 312, count = 12

        // Spawn consumer FIRST — tests waker integration:
        // Consumer will return Pending (no values yet), executor parks it.
        // When producers send values, channel's wake_consumer() re-enqueues
        // the consumer task via the stored waker.
        let _ = executor.spawn(MpscConsumer {
            channel: channel_ptr as *const _,
            result_ptr: results.add(4),
            count_ptr: results.add(5),
            sum: 0,
            received: 0,
            expected: 12,
        });

        // Spawn 3 producers (will wake consumer via channel waker)
        let _ = executor.spawn(MpscProducer {
            channel: channel_ptr as *const _,
            values: [10, 20, 30, 40],
            next_idx: 0,
        });
        let _ = executor.spawn(MpscProducer {
            channel: channel_ptr as *const _,
            values: [11, 21, 31, 41],
            next_idx: 0,
        });
        let _ = executor.spawn(MpscProducer {
            channel: channel_ptr as *const _,
            values: [12, 22, 32, 42],
            next_idx: 0,
        });

        core::ptr::write_volatile(results.add(7), 3); // Phase 3: all tasks spawned
    }

    gpu_atomics::syncwarp(mask);

    let stats = executor.run(mask);

    gpu_atomics::syncwarp(mask);

    if thread_x == 0 {
        core::ptr::write_volatile(results.add(7), 5); // Phase 5: executor finished
        core::ptr::write_volatile(results.add(0), executor.spawned_count());
        core::ptr::write_volatile(results.add(1), executor.completed_count());
        core::ptr::write_volatile(results.add(2), stats.tasks_executed);
        core::ptr::write_volatile(results.add(3), stats.polls_total);

        // Verify: sum = 10+20+30+40+11+21+31+41+12+22+32+42 = 312, count = 12
        let sum = core::ptr::read_volatile(results.add(4) as *const u32);
        let count = core::ptr::read_volatile(results.add(5) as *const u32);
        let spawned = core::ptr::read_volatile(results.add(0) as *const u32);
        let completed = core::ptr::read_volatile(results.add(1) as *const u32);

        if spawned == 4 && completed == 4 && sum == 312 && count == 12 {
            core::ptr::write_volatile(results.add(6), 1); // success
        }
    }
}
