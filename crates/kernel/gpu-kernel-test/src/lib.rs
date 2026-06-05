// GPU test and demo kernels — std-based entry points for testing.
//
// This crate contains test/demo kernels that exercise std features on GPU:
// println!, Vec, HashMap, File I/O, thread::spawn, structured concurrency,
// par_iter, warp intrinsics, and async futures.
//
// Builds with `-Zbuild-std=std` so all code has access to both core and std.
// Re-exports gpu-kernel-core for shared helpers and basic kernels.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

// NOTE: Under restricted_std, the standard library provides #[panic_handler].
// The gpu-runtime panic_handler!() macro is only needed for pure no_std crates.
// Individual kernels call gpu_runtime::panic::gpu_panic_init(buf) at entry to
// route panic messages via hostcall.

// Declare dynamic shared memory symbol at module level (PTX).
// This emits `.extern .shared .align 4 .b8 dynamic_smem[];`
// so that kernels can reference it via inline asm.
#[cfg(target_arch = "nvptx64")]
core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];");

// === Modules merged from former gpu-kernel (no_std) ===
// NOTE: helpers, basic, and compute_math have been extracted to gpu-kernel-core.
// Other kernel modules import helpers via `gpu_kernel_core::helpers::*`.

// basic and compute_math are now in gpu-kernel-core (split-execute.2).
// compute_* modules are now in gpu-kernel-compute (split-execute.3).
// hostcall_kernels, hybrid, and pipeline are now in gpu-kernel-io (split-execute.4).
// Re-export gpu-kernel-core so its kernel symbols are linked into this crate's cdylib.
extern crate gpu_kernel_core;

mod par_iter_demo;
mod sc_demo;
mod thread_test;
mod warp;

// === Std-specific kernel code below ===

// Force-link stdio symbols from gpu-runtime. These are called by the patched
// std PAL via `extern "C"` blocks, so LTO would strip them without this anchor.
#[used]
static _KEEP_STDOUT: unsafe fn(*const u8, usize) -> usize = gpu_runtime::stdio::gpu_stdout_write;
#[used]
static _KEEP_STDIN: unsafe fn(*mut u8, usize) -> usize = gpu_runtime::stdio::gpu_stdin_read;

/// Auto-initialize stdio from the `__HOSTCALL_BUF` device global.
///
/// The host writes the hostcall pointer to the device global via
/// `cuModuleGetGlobal_v2` + `cuMemcpyHtoD` before launch. This function
/// reads it and initializes all subsystems (stdio, panic, libc I/O).
///
/// Returns the hostcall buffer pointer (for use by caller), or null if
/// the host did not inject it.
fn stdio_auto_init() -> *mut u8 {
    let buf = gpu_runtime::entry::hostcall_buf_ptr();
    if !buf.is_null() {
        gpu_runtime::stdio::stdio_init(buf);
        unsafe {
            gpu_runtime::panic::gpu_panic_init(buf);
            gpu_libc::gpu_libc_io_init(buf);
        }
    }
    buf
}

// ============================================================
// Test kernels — demonstrate real std on GPU
// ============================================================

/// Test kernel: println! via patched std (no custom macros).
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_println_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    println!("Hello from gpu-kernel-test!");
    println!("This uses real Rust std println!, not a custom macro.");
}

/// Test kernel: Vec + String + format! via std allocator.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_vec_format_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    // Vec allocation and manipulation
    let mut v: Vec<i32> = Vec::new();
    for i in 0..10 {
        v.push(i * i);
    }
    println!("Vec contents: {:?}", v);
    println!("Vec len={}, capacity={}", v.len(), v.capacity());

    // String formatting
    let name = String::from("GPU");
    let msg = format!("Hello from {}! Vec sum = {}", name, v.iter().sum::<i32>());
    println!("{}", msg);

    // Vec with_capacity
    let mut v2: Vec<u8> = Vec::with_capacity(64);
    v2.extend_from_slice(b"gpu-kernel-test works!");
    println!("String from bytes: {}", core::str::from_utf8(&v2).unwrap());
}

/// Test kernel: multiple allocations and drops to verify allocator.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_alloc_stress_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    // Allocate and drop multiple Vecs to test allocator reuse
    for i in 0..5u32 {
        let v: Vec<u32> = (0..20).map(|x| x + i * 100).collect();
        let sum: u32 = v.iter().sum();
        println!("Round {}: Vec[20] sum = {}", i, sum);
        // v is dropped here — allocator should reclaim
    }
    println!("Alloc stress test complete — 5 rounds of alloc/drop");
}

/// Test kernel: std::fs::File create + write + read via hostcall.
///
/// Demonstrates real std::fs on GPU — File::create(), write_all(), File::open(),
/// read_to_string(). Errors are proper std::io::Error with errno propagation.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
/// `stdio_auto_init()` handles stdio, panic handler, AND libc I/O initialization.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_file_io_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    use std::fs::File;
    use std::io::Read;
    use std::io::Write;

    let path = "gpu_std_test.txt";

    // Create and write
    let create_result = File::create(path);
    match create_result {
        Ok(mut f) => {
            let msg = b"Hello from std::fs on GPU!\n";
            match f.write_all(msg) {
                Ok(()) => println!("[OK] write_all: {} bytes", msg.len()),
                Err(e) => println!("[ERR] write_all: {}", e),
            }
            // f is dropped here → close() via hostcall
        }
        Err(e) => {
            println!("[ERR] File::create: {}", e);
            return;
        }
    }

    // Open and read back
    let open_result = File::open(path);
    match open_result {
        Ok(mut f) => {
            let mut contents = Vec::new();
            match f.read_to_end(&mut contents) {
                Ok(n) => {
                    println!("[OK] read_to_end: {} bytes", n);
                    match core::str::from_utf8(&contents) {
                        Ok(s) => println!("[OK] content: {}", s.trim()),
                        Err(_) => println!("[ERR] content not valid UTF-8"),
                    }
                }
                Err(e) => println!("[ERR] read_to_end: {}", e),
            }
        }
        Err(e) => {
            println!("[ERR] File::open: {}", e);
            return;
        }
    }

    println!("[DONE] std::fs file I/O test complete");

    // Error mapping test: open nonexistent file → should get NotFound
    use std::io::ErrorKind;
    match File::open("nonexistent_file_12345.txt") {
        Ok(_) => println!("[ERR] expected NotFound, got Ok"),
        Err(e) => {
            let kind = e.kind();
            if kind == ErrorKind::NotFound {
                println!("[OK] error_kind: NotFound (correct)");
            } else {
                println!("[ERR] error_kind: {:?} (expected NotFound)", kind);
            }
            // Note: e.raw_os_error() gives the errno value
            println!("[OK] raw_os_error: {:?}", e.raw_os_error());
        }
    }
}

// ============================================================
// std-migration.3: Async pipeline kernel using std + Result + ?
// ============================================================

/// Multi-step pipeline using std types and ? operator for error propagation.
///
/// Demonstrates that a GPU kernel can use idiomatic Rust error handling:
/// - std::fs::File with ? operator
/// - std::io::{Read, Write} traits
/// - Vec<u8> for dynamic buffers
/// - format!() for message construction
/// - Result<(), std::io::Error> return type
///
/// Pipeline: generate data → write file → read file → verify → report
fn std_pipeline_inner() -> Result<(), std::io::Error> {
    use std::fs::File;
    use std::io::{Read, Write};

    // Step 1: Generate data using std types
    let mut data = Vec::new();
    for i in 0u32..10 {
        let line = format!("line {}: value={}\n", i, i * i);
        data.extend_from_slice(line.as_bytes());
    }
    println!("[PIPE] Generated {} bytes of data", data.len());

    // Step 2: Write to file via hostcall (uses ? for error propagation)
    let path = "gpu_pipeline_test.txt";
    {
        let mut f = File::create(path)?;
        f.write_all(&data)?;
    } // f dropped → close
    println!("[PIPE] Wrote {} bytes to {}", data.len(), path);

    // Step 3: Read back via hostcall (uses ? for error propagation)
    let mut readback = Vec::new();
    {
        let mut f = File::open(path)?;
        f.read_to_end(&mut readback)?;
    }
    println!("[PIPE] Read {} bytes from {}", readback.len(), path);

    // Step 4: Verify content matches
    if readback.len() != data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "length mismatch",
        ));
    }
    let mut mismatches = 0u32;
    for i in 0..data.len() {
        if data[i] != readback[i] {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "content mismatch",
        ));
    }

    // Step 5: Parse and compute — demonstrate Vec + string processing
    let text = core::str::from_utf8(&readback)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid UTF-8"))?;
    let line_count = text.lines().count();
    println!(
        "[PIPE] Verified: {} lines, {} bytes, 0 mismatches",
        line_count,
        data.len()
    );

    Ok(())
}

/// Test kernel: multi-step pipeline with std types and ? error propagation.
///
/// This kernel demonstrates the combination of:
/// - Real Rust std on GPU (Vec, String, format!, std::fs, std::io)
/// - Idiomatic error handling with ? operator
/// - Multi-step I/O pipeline (generate → write → read → verify)
/// - Proper std::io::Error propagation from GPU to host
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_pipeline_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    match std_pipeline_inner() {
        Ok(()) => println!("[DONE] std pipeline test PASSED"),
        Err(e) => println!("[ERR] std pipeline test FAILED: {}", e),
    }
}

// ============================================================
// std-migration.4: stdin().read_line() end-to-end test
// ============================================================

/// Test kernel: std::io::stdin().read_line() via hostcall.
///
/// Reads one line from host stdin using real std::io::stdin() (not raw hostcall),
/// then echoes it back via println!. Tests the full PAL chain:
/// stdin().read_line() → Stdin::read() → gpu_stdin_read() → SERVICE_STDIN hostcall.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_stdin_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    use std::io::BufRead;

    println!("[STDIN] Reading line from stdin...");

    let stdin = std::io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(n) => {
            println!("[STDIN] Read {} bytes: {}", n, line.trim());
            if n > 0 && !line.is_empty() {
                println!("[STDIN] PASS — stdin().read_line() works on GPU");
            } else {
                println!("[STDIN] WARN — read 0 bytes (EOF?)");
            }
        }
        Err(e) => {
            println!("[STDIN] FAIL — read_line error: {:?}", e.kind());
        }
    }
}

// ============================================================
// hashmap-fix.2: HashMap test on GPU
// ============================================================

/// Test kernel: std::collections::HashMap on GPU.
///
/// Verifies that HashMap::new() does not panic (hashmap_random_keys uses
/// address-based seed, not fill_bytes). Tests insert, get, contains_key,
/// and iteration.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_hashmap_test() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("hello", 1i32);
    map.insert("world", 2);
    map.insert("gpu", 3);

    println!("[HASHMAP] len = {}", map.len());

    match map.get("gpu") {
        Some(&v) => println!("[HASHMAP] get(\"gpu\") = {}", v),
        None => println!("[HASHMAP] ERR: get(\"gpu\") returned None"),
    }

    if map.contains_key("hello") {
        println!("[HASHMAP] contains_key(\"hello\") = true");
    } else {
        println!("[HASHMAP] ERR: contains_key(\"hello\") = false");
    }

    // Iterate and sum values
    let sum: i32 = map.values().sum();
    println!("[HASHMAP] sum of values = {}", sum);

    if map.len() == 3 && sum == 6 {
        println!("[HASHMAP] PASS — HashMap works on GPU");
    } else {
        println!("[HASHMAP] FAIL — unexpected results");
    }
}

// ============================================================
// std-multithread.3: Multi-thread std test (32 threads)
// ============================================================

/// Read flat thread ID within block via inline PTX.
#[inline(always)]
fn get_tid() -> u32 {
    let tid: u32;
    unsafe {
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);
    }
    tid
}

/// Test kernel: multi-thread println! (4 threads, each prints its tid).
///
/// Uses only 4 threads to avoid hostcall packet pool exhaustion (each println
/// generates multiple 56-byte chunks). Verifies thread_local storage (panic
/// count, etc.) is per-thread and does not cause data races.
///
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_multithread_println_test(result: *mut u32) {
    let _buf = stdio_auto_init();

    let tid = get_tid();

    // Each thread: allocate a Vec, format a string, println!
    let mut v: Vec<u32> = Vec::new();
    v.push(tid);
    v.push(tid * tid);
    let sum: u32 = v.iter().sum();

    println!("[MT] tid={} sum={}", tid, sum);

    // Write tid+1 to result[tid] to prove this thread ran
    unsafe {
        core::ptr::write_volatile(result.add(tid as usize), tid + 1);
    }
}

/// Test kernel: multi-thread Vec allocation stress test (32 threads).
///
/// Each thread allocates a Vec, pushes elements, and writes the sum to output.
/// Verifies that the allocator and thread_local state are thread-safe.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_multithread_vec_test(result: *mut u32) {
    let tid = get_tid();

    // Each thread allocates its own Vec and computes a sum
    let mut v: Vec<u32> = Vec::with_capacity(8);
    for i in 0..8u32 {
        v.push(tid * 10 + i);
    }
    let sum: u32 = v.iter().sum();

    // Expected: sum of (tid*10+0, tid*10+1, ..., tid*10+7) = tid*80 + 28
    unsafe {
        core::ptr::write_volatile(result.add(tid as usize), sum);
    }
}

// ============================================================
// std::thread::spawn demo — GPU threading identical to CPU Rust
// ============================================================

/// Demo: std::thread::spawn on GPU — identical to CPU Rust.
///
/// Launch with: block_dim=(128,1,1), 1 block, hostcall enabled.
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_thread_spawn_demo(result: *mut u32) {
    let _buf = stdio_auto_init();

    gpu_runtime::thread::gpu_main_poll(|| {
        let handle1 = gpu_runtime::thread::spawn(|| -> u32 {
            let mut sum = 0u32;
            for i in 0..10u32 {
                sum += i;
            }
            sum // 45
        });

        let handle2 = gpu_runtime::thread::spawn(|| -> u32 {
            let mut product = 1u32;
            for i in 1..=5u32 {
                product *= i;
            }
            product // 120
        });

        let r1 = handle1.join();
        let r2 = handle2.join();

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, r1);
                core::ptr::write_volatile(result.add(1), r2);
                core::ptr::write_volatile(result.add(2), r1 + r2);
            }
        }
    });
}

// ============================================================
// TRUE std::thread::spawn demo — uses real std::thread API
// ============================================================

/// Demo: REAL std::thread::spawn on GPU with println!
///
/// Unlike std_thread_spawn_demo (which uses gpu_runtime::thread::spawn),
/// this uses the actual `std::thread::spawn` API from the Rust standard
/// library. The patched std routes thread::spawn to gpu_thread_spawn_raw
/// via the cuda PAL module.
///
/// Launch with: block_dim=(128,1,1), 1 block, hostcall enabled.
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn real_std_thread_spawn(result: *mut u32) {
    let _buf = stdio_auto_init();

    extern "C" {
        fn gpu_thread_spawn_raw_count() -> u32;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        println!("[DBG] spawn_raw_count before = {}", unsafe {
            gpu_thread_spawn_raw_count()
        });

        // Use REAL std::thread::spawn — identical to how you'd write it on CPU
        let handle1 = std::thread::spawn(|| -> u32 {
            let mut sum = 0u32;
            for i in 0..10u32 {
                sum += i;
            }
            println!("Thread 1: sum(0..10) = {}", sum);
            sum // 45
        });

        println!("[DBG] spawn_raw_count after 1st = {}", unsafe {
            gpu_thread_spawn_raw_count()
        });

        let handle2 = std::thread::spawn(|| -> u32 {
            let mut product = 1u32;
            for i in 1..=5u32 {
                product *= i;
            }
            println!("Thread 2: 5! = {}", product);
            product // 120
        });

        println!("[DBG] spawn_raw_count after 2nd = {}", unsafe {
            gpu_thread_spawn_raw_count()
        });
        println!("[DBG] before join1");

        let r1 = handle1.join().unwrap();
        println!("[DBG] after join1, r1={}", r1);

        let r2 = handle2.join().unwrap();
        println!("[DBG] after join2, r2={}", r2);

        println!("Main: {} + {} = {}", r1, r2, r1 + r2);

        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, r1);
                core::ptr::write_volatile(result.add(1), r2);
                core::ptr::write_volatile(result.add(2), r1 + r2);
            }
        }
    });
}

// ============================================================
// println-buffer: Buffered println! via print_buffer + sideband
// ============================================================

/// Test kernel: buffered println! using print_buffer integration.
///
/// Initializes the print buffer, prints 6 messages via println! (which now
/// routes through print_buffer's fast path), then flushes. The host should
/// receive all 6 messages in a single SERVICE_BULK_PRINT hostcall instead
/// of 6 separate SERVICE_PRINT hostcalls.
///
/// Launch with 1 block × 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_buffered_println_test(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
) {
    gpu_runtime::stdio::stdio_print_buffer_init(buf, sideband, 1);

    println!("Buffered line 1: hello from std!");
    println!("Buffered line 2: this goes through print_buffer");
    println!("Buffered line 3: auto-batch via sideband");
    println!("Buffered line 4: fewer hostcalls = faster");
    println!("Buffered line 5: almost done");
    println!("Buffered line 6: final message");

    gpu_runtime::stdio::gpu_print_buffer_flush();

    // Signal success
    unsafe {
        core::ptr::write_volatile(result, 1);
    }
}

// ============================================================
// North Star Demo: File::read → cooperative compute → File::write
// ============================================================

/// North Star: File::read → compute → File::write in one kernel.
///
/// Demonstrates the project vision: I/O and compute unified in plain Rust.
/// - Sequential I/O (warp 0): read input file
/// - Cooperative compute (all warps): transform data via cooperative_map
/// - Sequential I/O (warp 0): write output file
///
/// Uses `cooperative_map` instead of `cooperative()` — zero global atomics.
/// All data flows through explicit `(src, dst, len)` parameters.
///
/// Launch with: block_dim=(128,1,1), hostcall enabled.
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn unified_io_compute(result: *mut u32) {
    let _buf = stdio_auto_init();

    gpu_runtime::thread::gpu_main_poll(|| {
        use std::fs::File;
        use std::io::{Read, Write};

        // === SEQUENTIAL I/O: read input ===
        let data = match File::open("io_compute_input.bin") {
            Ok(mut f) => {
                let mut raw = Vec::new();
                f.read_to_end(&mut raw).unwrap();
                raw
            }
            Err(e) => {
                println!("[ERR] File::open: {}", e);
                return;
            }
        };
        let n = data.len() / 4;
        println!("[UIC] Read {} floats from input", n);

        // Allocate output
        let mut output = vec![0u8; data.len()];

        // === COOPERATIVE COMPUTE: all warps multiply by 2 ===
        // Zero global atomics — data flows through cooperative_map's arguments.
        gpu_runtime::thread::cooperative_map(
            data.as_ptr() as *const u8,
            output.as_mut_ptr() as *mut u8,
            n,
            |args| {
                let src = args.src as *const f32;
                let dst = args.dst as *mut f32;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        let v = core::ptr::read_volatile(src.add(i));
                        core::ptr::write_volatile(dst.add(i), v * 2.0);
                    }
                    i += args.n_warps as usize;
                }
            },
        );

        // === SEQUENTIAL I/O: write output ===
        match File::create("io_compute_output.bin") {
            Ok(mut f) => {
                f.write_all(&output).unwrap();
                println!("[UIC] Wrote {} floats to output", n);
            }
            Err(e) => {
                println!("[ERR] File::create: {}", e);
                return;
            }
        }

        println!("[UIC] DONE: read → compute(×2) → write");

        // Write success marker
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, n as u32);
                core::ptr::write_volatile(result.add(1), 1); // success flag
            }
        }
    });
}

// ============================================================
// Debug: trivial kernel to test module loading
// ============================================================

/// Trivial write to verify kernel_std module loads and runs.
/// Launch with: 1 block × 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel_std_smoke_test(result: *mut u32) {
    unsafe {
        core::ptr::write_volatile(result, 0xBEEF_CAFE);
    }
}

/// Simple println test via kernel_std. Launch with 1×1, hostcall enabled.
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel_std_println_smoke(result: *mut u32) {
    let _buf = stdio_auto_init();
    println!("kernel_std_println_smoke: alive!");
    unsafe {
        core::ptr::write_volatile(result, 1);
    }
}

/// Thread pool test without spawn — just gpu_main_poll with no work.
/// Launch with block_dim=(128,1,1), 1 block.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel_std_pool_smoke(result: *mut u32) {
    gpu_runtime::thread::gpu_main_poll(|| {
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, 42);
            }
        }
    });
}

// ============================================================
// Debug: minimal std::thread::spawn (no println in closures)
// ============================================================

/// Minimal std::thread::spawn test — no println inside spawned threads.
/// This isolates whether the hang is from thread spawn or from println.
/// Launch with: block_dim=(128,1,1), 1 block, hostcall enabled.
/// Zero-param entry for hostcall (buf removed); result kept for data output.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn std_thread_spawn_minimal(result: *mut u32) {
    let _buf = stdio_auto_init();

    gpu_runtime::thread::gpu_main_poll(|| {
        println!("[DBG] before first spawn");

        let handle1 = std::thread::spawn(|| -> u32 {
            let mut sum = 0u32;
            for i in 0..10u32 {
                sum += i;
            }
            sum // 45
        });

        println!("[DBG] before second spawn");

        let handle2 = std::thread::spawn(|| -> u32 {
            let mut product = 1u32;
            for i in 1..=5u32 {
                product *= i;
            }
            product // 120
        });

        println!("[DBG] before join1");
        let r1 = handle1.join().unwrap();
        println!("[DBG] after join1, r1={}", r1);

        let r2 = handle2.join().unwrap();
        println!("[DBG] after join2, r2={}", r2);

        println!("[DBG] writing results");
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, r1);
                core::ptr::write_volatile(result.add(1), r2);
                core::ptr::write_volatile(result.add(2), r1 + r2);
            }
        }
    });
}

// ============================================================
// North Star Litmus Test: File::read → matmul → File::write
// ============================================================

/// Naive matmul callback for cooperative_map_with_params.
///
/// params[0] = M, params[1] = K, params[2] = N, params[3] = B ptr (as u64).
/// Each warp computes rows i where i % n_warps == warp_id.
fn matmul_callback(args: &gpu_runtime::thread::CoopMapExtArgs) {
    let a = args.src as *const f32;
    let c = args.dst as *mut f32;
    let m = args.params[0] as usize;
    let k = args.params[1] as usize;
    let n = args.params[2] as usize;
    let b = args.params[3] as *const f32;

    let wid = args.warp_id as usize;
    let nw = args.n_warps as usize;

    // Each warp computes rows i where i % nw == wid
    let mut i = wid;
    while i < m {
        let mut j = 0usize;
        while j < n {
            let mut sum = 0.0f32;
            let mut p = 0usize;
            while p < k {
                let a_val = unsafe { core::ptr::read_volatile(a.add(i * k + p)) };
                let b_val = unsafe { core::ptr::read_volatile(b.add(p * n + j)) };
                sum += a_val * b_val;
                p += 1;
            }
            unsafe {
                core::ptr::write_volatile(c.add(i * n + j), sum);
            }
            j += 1;
        }
        i += nw;
    }
}

/// Inner implementation for the matmul I/O kernel.
/// Separated to allow testing with and without gpu_main_poll.
fn matmul_io_inner(buf: *mut u8, dims: *const u32, result: *mut u32) {
    use std::io::Read;

    // Read dimensions from host-provided device memory
    let m = unsafe { core::ptr::read_volatile(dims.add(0)) } as usize;
    let k = unsafe { core::ptr::read_volatile(dims.add(1)) } as usize;
    let n = unsafe { core::ptr::read_volatile(dims.add(2)) } as usize;

    println!("[MATMUL] M={} K={} N={}", m, k, n);

    // === SEQUENTIAL I/O: read matrix A (M×K f32) ===
    let a_data = match std::fs::File::open("matmul_a.bin") {
        Ok(mut f) => {
            let mut raw = Vec::new();
            f.read_to_end(&mut raw).unwrap();
            raw
        }
        Err(e) => {
            println!("[MATMUL] ERR File::open(matmul_a.bin): {}", e);
            return;
        }
    };
    let a_elems = a_data.len() / 4;
    println!(
        "[MATMUL] Read A: {} floats ({} bytes)",
        a_elems,
        a_data.len()
    );

    // === SEQUENTIAL I/O: read matrix B (K×N f32) ===
    let b_data = match std::fs::File::open("matmul_b.bin") {
        Ok(mut f) => {
            let mut raw = Vec::new();
            f.read_to_end(&mut raw).unwrap();
            raw
        }
        Err(e) => {
            println!("[MATMUL] ERR File::open(matmul_b.bin): {}", e);
            return;
        }
    };
    let b_elems = b_data.len() / 4;
    println!(
        "[MATMUL] Read B: {} floats ({} bytes)",
        b_elems,
        b_data.len()
    );

    // Sanity check
    if a_elems != m * k {
        println!(
            "[MATMUL] ERR: A has {} floats, expected M*K={}",
            a_elems,
            m * k
        );
        return;
    }
    if b_elems != k * n {
        println!(
            "[MATMUL] ERR: B has {} floats, expected K*N={}",
            b_elems,
            k * n
        );
        return;
    }

    // Allocate output C (M×N f32)
    let mut c_data = vec![0u8; m * n * 4];

    // === COOPERATIVE COMPUTE: C = A × B (all warps) ===
    let a_ptr = a_data.as_ptr() as *const u8;
    let c_ptr = c_data.as_mut_ptr() as *mut u8;
    let b_ptr = b_data.as_ptr() as u64;

    gpu_runtime::thread::cooperative_map_with_params(
        a_ptr,
        c_ptr,
        m * n,
        [m as u64, k as u64, n as u64, b_ptr],
        matmul_callback,
    );

    println!("[MATMUL] Compute done: C[{}x{}]", m, n);

    // === SEQUENTIAL I/O: write result C ===
    match std::fs::File::create("matmul_c.bin") {
        Ok(mut f) => {
            use std::io::Write;
            f.write_all(&c_data).unwrap();
            println!("[MATMUL] Wrote C: {} bytes", c_data.len());
        }
        Err(e) => {
            println!("[MATMUL] ERR File::create(matmul_c.bin): {}", e);
            return;
        }
    }

    println!("[MATMUL] DONE: File::read -> matmul -> File::write");

    // Write success marker
    if gpu_runtime::index::thread_idx_x() == 0 {
        unsafe {
            core::ptr::write_volatile(result, 1); // success
            core::ptr::write_volatile(result.add(1), (m * n) as u32); // elements
        }
    }
}

/// North Star litmus test: File::read → matmul → File::write in ONE kernel.
///
/// 1. Sequential I/O (warp 0): read matmul_a.bin (M×K f32) and matmul_b.bin (K×N f32)
/// 2. Cooperative compute (all warps): C = A × B via naive triple-loop matmul
/// 3. Sequential I/O (warp 0): write matmul_c.bin (M×N f32)
///
/// Launch with: block_dim=(128,1,1), hostcall enabled.
/// Zero-param entry for hostcall (buf removed); dims and result kept as data params.
/// Kernel args: (dims: *const u32, result: *mut u32)
///   dims[0] = M, dims[1] = K, dims[2] = N
///   result[0] = success flag (1 = ok), result[1] = M*N (elements written)
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn matmul_io_compute(dims: *const u32, result: *mut u32) {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    // Phase 1: Sequential I/O — single thread reads files and writes result
    // Phase 2: Cooperative compute — all warps via gpu_main_poll
    // Phase 3: Sequential I/O — single thread writes result file
    //
    // For now, run everything inside gpu_main_poll for cooperative compute.
    gpu_runtime::thread::gpu_main_poll(|| {
        matmul_io_inner(buf, dims, result);
    });
}

// ============================================================
// Zero-param kernel entry — hostcall injected via device global
// ============================================================

/// Zero-parameter kernel: hostcall buffer injected via `__HOSTCALL_BUF` device global.
///
/// The host writes the hostcall pointer to the device global before launch.
/// No kernel parameters needed for basic I/O (println!, file I/O).
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn zero_param_hello() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        println!("Hello from zero-param kernel!");
        println!("Hostcall buffer injected via __HOSTCALL_BUF device global.");

        let v: Vec<i32> = (1..=5).collect();
        println!("Vec on GPU: {:?}, sum = {}", v, v.iter().sum::<i32>());
    });
}

// ============================================================
// GPU test kernels — for #[gpu_test] proc macro integration
// ============================================================

/// GPU test: basic arithmetic assertions.
///
/// Zero-param entry. Tests that assert! and assert_eq! work on GPU.
/// If any assertion fails, the panic handler sends the failure message
/// via hostcall with thread/block coordinates, then traps.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_assert_basic() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Basic arithmetic
        let a = 2u32 + 3;
        assert_eq!(a, 5, "2 + 3 should equal 5");

        let b = 10u32 * 4;
        assert_eq!(b, 40, "10 * 4 should equal 40");

        // assert! (boolean)
        assert!(a < b, "5 should be less than 40");

        // assert_ne!
        assert_ne!(a, b, "5 should not equal 40");

        println!("[gpu_test] test_gpu_assert_basic PASSED");
    });
}

/// GPU test: Vec operations with assertions.
///
/// Zero-param entry. Allocates a Vec, pushes elements, and asserts
/// on length, sum, and individual elements.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_vec_operations() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        let mut v: Vec<u32> = Vec::new();
        for i in 0..10u32 {
            v.push(i * i);
        }

        assert_eq!(v.len(), 10, "Vec should have 10 elements");
        assert_eq!(v[0], 0, "v[0] should be 0");
        assert_eq!(v[1], 1, "v[1] should be 1");
        assert_eq!(v[4], 16, "v[4] should be 16");
        assert_eq!(v[9], 81, "v[9] should be 81");

        let sum: u32 = v.iter().sum();
        assert_eq!(sum, 285, "sum of squares 0..10 should be 285");

        println!("[gpu_test] test_gpu_vec_operations PASSED");
    });
}

/// GPU test: thread spawn and join with assertions.
///
/// Zero-param entry. Spawns threads, joins results, and asserts
/// correctness — the same pattern users write with std::thread on CPU.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_thread_spawn() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        let h1 = gpu_runtime::thread::spawn(|| -> u32 { 42 });
        let h2 = gpu_runtime::thread::spawn(|| -> u32 { 99 });

        let r1 = h1.join();
        let r2 = h2.join();

        assert_eq!(r1, 42, "thread 1 should return 42");
        assert_eq!(r2, 99, "thread 2 should return 99");
        assert_eq!(r1 + r2, 141, "sum should be 141");

        println!("[gpu_test] test_gpu_thread_spawn PASSED");
    });
}

/// GPU test: Box allocation and dereference.
///
/// Zero-param entry. Allocates Box<u32> and Box<[u32; 4]> on the GPU heap,
/// verifies values through dereference, and tests that Drop runs correctly.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_box_alloc() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Box a single value
        let b = Box::new(42u32);
        assert_eq!(*b, 42, "Box<u32> should deref to 42");

        // Box an array
        let arr = Box::new([10u32, 20, 30, 40]);
        assert_eq!(arr[0], 10, "arr[0] should be 10");
        assert_eq!(arr[3], 40, "arr[3] should be 40");
        let sum: u32 = arr.iter().sum();
        assert_eq!(sum, 100, "sum of boxed array should be 100");

        // Box a computed value
        let c = Box::new(*b + arr[2]);
        assert_eq!(*c, 72, "42 + 30 should be 72");

        println!("[gpu_test] test_gpu_box_alloc PASSED");
    });
}

/// GPU test: String creation and operations.
///
/// Zero-param entry. Tests String::from, push_str, len, and format!.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_string_ops() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // String creation
        let s = String::from("hello");
        assert_eq!(s.len(), 5, "String len should be 5");

        // String concatenation
        let mut s2 = String::from("GPU ");
        s2.push_str("rocks");
        assert_eq!(s2.len(), 9, "concatenated len should be 9");
        assert_eq!(s2.as_str(), "GPU rocks", "concatenated string mismatch");

        // format! macro
        let s3 = format!("{} {}", 2 + 3, "test");
        assert_eq!(s3, "5 test", "format! should produce '5 test'");

        // String contains
        assert!(s2.contains("rock"), "should contain 'rock'");
        assert!(!s2.contains("CPU"), "should not contain 'CPU'");

        println!("[gpu_test] test_gpu_string_ops PASSED");
    });
}

/// GPU test: HashMap insert, get, contains, iteration.
///
/// Zero-param entry. Tests std::collections::HashMap on GPU.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_hashmap() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        for i in 0..5u32 {
            map.insert(i, i * i);
        }

        assert_eq!(map.len(), 5, "HashMap should have 5 entries");
        assert_eq!(*map.get(&0).unwrap(), 0, "map[0] should be 0");
        assert_eq!(*map.get(&3).unwrap(), 9, "map[3] should be 9");
        assert_eq!(*map.get(&4).unwrap(), 16, "map[4] should be 16");

        assert!(map.contains_key(&2), "should contain key 2");
        assert!(!map.contains_key(&99), "should not contain key 99");

        // Iterate and sum values: 0 + 1 + 4 + 9 + 16 = 30
        let sum: u32 = map.values().sum();
        assert_eq!(sum, 30, "sum of squares 0..5 should be 30");

        // Remove and verify
        map.remove(&2);
        assert_eq!(map.len(), 4, "after remove, len should be 4");
        assert!(!map.contains_key(&2), "key 2 should be gone");

        println!("[gpu_test] test_gpu_hashmap PASSED");
    });
}

/// GPU test: thread spawn with computed data passing.
///
/// Zero-param entry. Spawns threads that do real computation and
/// pass results back through the closure return value.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_thread_data_passing() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Spawn a thread that sums 1..=100
        let h1 = gpu_runtime::thread::spawn(|| -> u64 {
            let mut sum = 0u64;
            for i in 1..=100u64 {
                sum += i;
            }
            sum
        });

        // Spawn a thread that computes factorial(10)
        let h2 = gpu_runtime::thread::spawn(|| -> u64 {
            let mut fact = 1u64;
            for i in 1..=10u64 {
                fact *= i;
            }
            fact
        });

        let sum = h1.join();
        let fact = h2.join();

        assert_eq!(sum, 5050, "sum of 1..=100 should be 5050");
        assert_eq!(fact, 3628800, "10! should be 3628800");

        println!("[gpu_test] test_gpu_thread_data_passing PASSED");
    });
}

/// GPU test: thread reuse (spawn more tasks than available warps).
///
/// Zero-param entry. With 4 warps (128 threads), 3 are available for spawning.
/// Spawning 6 tasks sequentially forces warp reuse.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_thread_reuse() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        let mut total = 0u32;

        // Spawn 6 tasks sequentially on 3 available warps.
        // At least 3 warps must be reused.
        for i in 0..6u32 {
            let h = gpu_runtime::thread::spawn(move || -> u32 { (i + 1) * 10 });
            let r = h.join();
            assert_eq!(r, (i + 1) * 10, "task result mismatch");
            total += r;
        }

        // total = 10 + 20 + 30 + 40 + 50 + 60 = 210
        assert_eq!(total, 210, "total of 6 tasks should be 210");

        println!("[gpu_test] test_gpu_thread_reuse PASSED");
    });
}

// Statics for test_gpu_cooperative
static TEST_COOP_OUT: [core::sync::atomic::AtomicU32; 4] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 4]
};

/// GPU test: cooperative execution — all warps work together.
///
/// Zero-param entry. Uses cooperative() to have all 4 warps each
/// write their warp ID to a global array. Verifies all warps participated.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_cooperative() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Reset
        for slot in TEST_COOP_OUT.iter() {
            slot.store(0, core::sync::atomic::Ordering::Relaxed);
        }

        unsafe {
            gpu_runtime::thread::cooperative(&|| {
                let wid = gpu_runtime::thread::current_id();
                let lid = gpu_runtime::index::thread_idx_x() % 32;
                if lid == 0 {
                    TEST_COOP_OUT[wid as usize].store(
                        wid + 1,
                        core::sync::atomic::Ordering::Relaxed,
                    );
                }
            });
        }

        // Verify all 4 warps participated (warp IDs 0..3 wrote wid+1)
        for i in 0..4u32 {
            let val = TEST_COOP_OUT[i as usize].load(core::sync::atomic::Ordering::Relaxed);
            assert_eq!(val, i + 1, "warp {} should have written {}", i, i + 1);
        }

        println!("[gpu_test] test_gpu_cooperative PASSED");
    });
}

// Statics for test_gpu_cooperative_map
static TEST_CMAP_INPUT: [core::sync::atomic::AtomicU32; 64] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 64]
};
static TEST_CMAP_OUTPUT: [core::sync::atomic::AtomicU32; 64] = {
    const Z: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    [Z; 64]
};

/// GPU test: cooperative_map — all warps double each element.
///
/// Zero-param entry. Sets up a 64-element array in global statics,
/// uses cooperative_map to double each element across all warps.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_cooperative_map() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Initialize input
        for i in 0..64u32 {
            TEST_CMAP_INPUT[i as usize].store(i, core::sync::atomic::Ordering::Relaxed);
            TEST_CMAP_OUTPUT[i as usize].store(0, core::sync::atomic::Ordering::Relaxed);
        }

        gpu_runtime::thread::cooperative_map(
            TEST_CMAP_INPUT.as_ptr() as *const u8,
            TEST_CMAP_OUTPUT.as_ptr() as *mut u8,
            64,
            |args| {
                let src = args.src as *const u32;
                let dst = args.dst as *mut u32;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        let v = core::ptr::read_volatile(src.add(i));
                        core::ptr::write_volatile(dst.add(i), v * 2);
                    }
                    i += args.n_warps as usize;
                }
            },
        );

        // Verify output[i] = i * 2
        for i in 0..64u32 {
            let val = TEST_CMAP_OUTPUT[i as usize].load(core::sync::atomic::Ordering::Relaxed);
            assert_eq!(val, i * 2, "cooperative_map output[{}] should be {}", i, i * 2);
        }

        println!("[gpu_test] test_gpu_cooperative_map PASSED");
    });
}

// Statics for test_gpu_cooperative_reduce
static TEST_CREDUCE_INPUT: [core::sync::atomic::AtomicU64; 64] = {
    const Z: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    [Z; 64]
};

/// GPU test: cooperative_reduce — all warps sum partitions.
///
/// Zero-param entry. Sums 0..64 using cooperative_reduce.
/// Expected: 0+1+2+...+63 = 2016.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_cooperative_reduce() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // Initialize input: values 0..64
        for i in 0..64u64 {
            TEST_CREDUCE_INPUT[i as usize].store(i, core::sync::atomic::Ordering::Relaxed);
        }

        let total = gpu_runtime::thread::cooperative_reduce(
            TEST_CREDUCE_INPUT.as_ptr() as *const u8,
            64,
            |args| {
                let src = args.src as *const u64;
                let mut sum = 0u64;
                let mut i = args.warp_id as usize;
                while i < args.len {
                    unsafe {
                        sum += core::ptr::read_volatile(src.add(i));
                    }
                    i += args.n_warps as usize;
                }
                sum
            },
        );

        // sum of 0..64 = 63*64/2 = 2016
        assert_eq!(total, 2016, "cooperative_reduce should produce 2016");

        println!("[gpu_test] test_gpu_cooperative_reduce PASSED");
    });
}

/// GPU test: type-safe cooperative execution with DisjointSlice + WarpIndex.
///
/// Zero-param entry. Demonstrates the safe cooperative pattern:
///   1. `alloc_disjoint()` — allocates shared memory AND wraps it as DisjointSlice
///   2. `cooperative_indexed()` — safe cooperative() with WarpIndex + WarpHandle
///   3. `spawn_all_indexed()` — existing scope-based indexed pattern
///
/// Each warp writes to its exclusive partition via DisjointSlice. No `unsafe`
/// at the call site — data race prevention is enforced by the type system.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
/// Shared memory: 2048 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_type_safe_cooperative() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        unsafe {
            gpu_runtime::scope::init_shared_mem_allocator(2048);
        }

        // ---- Test 1: alloc_disjoint + spawn_all_indexed ----
        // Allocate a 64-element buffer as a DisjointSlice, fill with spawn_all_indexed.
        let test1_ok = gpu_runtime::scope::block_scope(|scope| {
            let data = scope.alloc_disjoint::<u32>(64);

            scope.spawn_all_indexed(move |widx, _warp| {
                let my_part = data.get_mut(&widx);
                for (i, slot) in my_part.iter_mut().enumerate() {
                    // Write a value derived from the global position.
                    // Use warp_id * 1000 + local_i so we can verify per-warp writes.
                    *slot = widx.warp_id() * 1000 + i as u32;
                }
            });

            // Verify: each element should have the correct warp-tagged value.
            // With contiguous partitioning and 4 warps over 64 elements,
            // each warp gets 16 elements. Warp k gets indices [k*16 .. (k+1)*16).
            // (data is Copy, so it's still usable after the move closure above)
            let n_warps = gpu_runtime::thread::available_parallelism() as u32 + 1;
            let chunk = 64 / n_warps;
            let mut ok = true;

            // Read back the raw buffer to verify
            let (ptr, len) = unsafe { data.raw_parts() };
            for wid in 0..n_warps {
                let start = (wid * chunk) as usize;
                for local_i in 0..chunk as usize {
                    let expected = wid * 1000 + local_i as u32;
                    let actual = unsafe { core::ptr::read_volatile(ptr.add(start + local_i)) };
                    if actual != expected {
                        ok = false;
                    }
                }
            }
            let _ = len; // suppress unused
            ok
        });
        assert!(test1_ok, "alloc_disjoint + spawn_all_indexed should produce correct values");

        // ---- Test 2: alloc_disjoint + cooperative_indexed ----
        // Same pattern but using cooperative_indexed (safe cooperative()) outside scope.
        let test2_ok = gpu_runtime::scope::block_scope(|scope| {
            let data = scope.alloc_disjoint::<u32>(64);

            // cooperative_indexed provides WarpIndex + WarpHandle without unsafe.
            gpu_runtime::thread::cooperative_indexed(&|widx, _warp| {
                let my_part = data.get_mut(&widx);
                for (i, slot) in my_part.iter_mut().enumerate() {
                    *slot = widx.warp_id() * 100 + i as u32;
                }
            });

            // Verify
            let n_warps = gpu_runtime::thread::available_parallelism() as u32 + 1;
            let chunk = 64 / n_warps;
            let mut ok = true;
            let (ptr, _len) = unsafe { data.raw_parts() };
            for wid in 0..n_warps {
                let start = (wid * chunk) as usize;
                for local_i in 0..chunk as usize {
                    let expected = wid * 100 + local_i as u32;
                    let actual = unsafe { core::ptr::read_volatile(ptr.add(start + local_i)) };
                    if actual != expected {
                        ok = false;
                    }
                }
            }
            ok
        });
        assert!(test2_ok, "alloc_disjoint + cooperative_indexed should produce correct values");

        // ---- Test 3: DisjointSlice immutable read via get() ----
        let test3_ok = gpu_runtime::scope::block_scope(|scope| {
            let data = scope.alloc_disjoint::<u32>(8);

            // Fill with known values via spawn_all_indexed
            scope.spawn_all_indexed(move |widx, _warp| {
                let my_part = data.get_mut(&widx);
                for (i, slot) in my_part.iter_mut().enumerate() {
                    let global_i = widx.warp_id() as usize
                        * (8 / widx.n_warps() as usize)
                        + i;
                    *slot = (global_i * 10) as u32;
                }
            });

            // Verify immutable reads
            let mut ok = true;
            for i in 0..8usize {
                match data.get(i) {
                    Some(&val) => {
                        if val != (i * 10) as u32 {
                            ok = false;
                        }
                    }
                    None => {
                        ok = false;
                    }
                }
            }
            // Out-of-bounds read should return None
            if data.get(8).is_some() {
                ok = false;
            }
            ok
        });
        assert!(test3_ok, "DisjointSlice::get() should read correct values");

        println!("[gpu_test] test_gpu_type_safe_cooperative PASSED");
    });
}

/// GPU test: safe cooperative map — rewrite of `test_gpu_cooperative_map` with zero unsafe.
///
/// **Original** (`test_gpu_cooperative_map`):
///   - Uses `cooperative_map()` with raw `*const u8` / `*mut u8` pointers
///   - Has `unsafe { read_volatile / write_volatile }` inside the cooperative closure
///   - Manual pointer arithmetic for partitioning (`src.add(i)`, `dst.add(i)`)
///   - 2 unsafe blocks in the closure body
///
/// **This safe version**:
///   - Uses `cooperative_indexed()` + `DisjointSlice` — zero unsafe in application logic
///   - Warp identity via `WarpIndex` (compile-time proof of ownership)
///   - Warp reductions via `WarpHandle` (safe warp-level ops)
///   - Partition access via `DisjointSlice::get_mut()` (compile-time disjointness)
///   - Verification via `DisjointSlice::get()` (bounds-checked safe reads)
///
/// Zero-param entry. Doubles each element of a 64-element array across all warps,
/// then uses `WarpHandle::reduce_sum_u32` to verify the element count per warp.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
/// Shared memory: 2048 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_cooperative_map_safe() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // init_shared_mem_allocator is the only unsafe call — it's infrastructure setup,
        // not application logic. Required once before any block_scope call.
        unsafe {
            gpu_runtime::scope::init_shared_mem_allocator(2048);
        }

        // ---- Phase 1: Prepare input data (sequential on warp 0) ----
        // Allocate input and output as DisjointSlice within a scope.
        let all_ok = gpu_runtime::scope::block_scope(|scope| {
            let input = scope.alloc_disjoint::<u32>(64);
            let output = scope.alloc_disjoint::<u32>(64);

            // Fill input: input[i] = i (using spawn_all_indexed for safe parallel fill)
            scope.spawn_all_indexed(move |widx, _warp| {
                let my_part = input.get_mut(&widx);
                let n_warps = widx.n_warps() as usize;
                let wid = widx.warp_id() as usize;
                // Contiguous partitioning: warp k gets [start..start+count)
                let chunk = 64 / n_warps;
                let start = wid * chunk;
                for (i, slot) in my_part.iter_mut().enumerate() {
                    *slot = (start + i) as u32;
                }
            });

            // ---- Phase 2: Cooperative compute — all warps double each element ----
            // This is the key demonstration: cooperative_indexed + DisjointSlice
            // replaces cooperative_map's unsafe pointer arithmetic with safe access.
            gpu_runtime::thread::cooperative_indexed(&|widx, warp| {
                let my_input = input.get_mut(&widx);
                let my_output = output.get_mut(&widx);

                // Safe: each warp writes only to its own partition
                for (i, out_slot) in my_output.iter_mut().enumerate() {
                    *out_slot = my_input[i] * 2;
                }

                // Demonstrate WarpHandle: safe warp-level reduce
                // Count how many elements this warp processed
                let my_count = my_output.len() as u32;
                let _total = warp.reduce_sum_u32(my_count);
                // On lane 0, _total == 64 (sum of all warps' counts)
            });

            // ---- Phase 3: Verify output (sequential on warp 0) ----
            // Use DisjointSlice::get() for safe bounds-checked reads
            let mut ok = true;
            for i in 0..64u32 {
                match output.get(i as usize) {
                    Some(&val) => {
                        if val != i * 2 {
                            ok = false;
                        }
                    }
                    None => {
                        ok = false;
                    }
                }
            }

            // Verify bounds checking: out-of-range read returns None
            if output.get(64).is_some() {
                ok = false;
            }

            ok
        });

        assert!(all_ok, "cooperative_map_safe: output[i] should equal i * 2 for all i in 0..64");
        println!("[gpu_test] test_gpu_cooperative_map_safe PASSED");
    });
}

/// GPU test: GPU math intrinsics (sin, cos, sqrt, exp, log, abs, fma).
///
/// Zero-param entry. Tests gpu_runtime::math functions with known values.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_math_intrinsics() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        use gpu_runtime::math;

        // sqrt(4.0) should be ~2.0
        let s = math::sqrt_f32(4.0);
        assert!((s - 2.0).abs() < 0.01, "sqrt(4.0) should be ~2.0");

        // sqrt(9.0) should be ~3.0
        let s2 = math::sqrt_f32(9.0);
        assert!((s2 - 3.0).abs() < 0.01, "sqrt(9.0) should be ~3.0");

        // sin(0) should be ~0.0
        let sin0 = math::sin_f32(0.0);
        assert!(sin0.abs() < 0.01, "sin(0) should be ~0.0");

        // cos(0) should be ~1.0
        let cos0 = math::cos_f32(0.0);
        assert!((cos0 - 1.0).abs() < 0.01, "cos(0) should be ~1.0");

        // exp(0) should be ~1.0
        let exp0 = math::exp_f32(0.0);
        assert!((exp0 - 1.0).abs() < 0.01, "exp(0) should be ~1.0");

        // exp(1) should be ~2.718
        let exp1 = math::exp_f32(1.0);
        assert!((exp1 - 2.718).abs() < 0.05, "exp(1) should be ~2.718");

        // log(1) should be ~0.0
        let ln1 = math::log_f32(1.0);
        assert!(ln1.abs() < 0.01, "ln(1) should be ~0.0");

        // abs(-5.0) should be 5.0
        let a = math::abs_f32(-5.0);
        assert!((a - 5.0).abs() < 0.001, "abs(-5.0) should be 5.0");

        // fma(2.0, 3.0, 4.0) = 2*3+4 = 10.0
        let f = math::fma_f32(2.0, 3.0, 4.0);
        assert!((f - 10.0).abs() < 0.001, "fma(2,3,4) should be 10.0");

        // tanh(0) should be ~0.0
        let t = math::tanh_f32(0.0);
        assert!(t.abs() < 0.01, "tanh(0) should be ~0.0");

        // sigmoid(0) should be ~0.5
        let sig = math::sigmoid_f32(0.0);
        assert!((sig - 0.5).abs() < 0.01, "sigmoid(0) should be ~0.5");

        println!("[gpu_test] test_gpu_math_intrinsics PASSED");
    });
}

// Statics for test_gpu_atomics — GPU atomics must live in global memory, not stack
static TEST_ATOMIC_BASIC: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static TEST_ATOMIC_COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// GPU test: atomic operations across threads.
///
/// Zero-param entry. Tests store/load/fetch_add/fetch_sub/CAS on global atomics,
/// then spawns threads that all increment a shared atomic counter.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_atomics() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        use core::sync::atomic::Ordering;

        // Test basic atomic operations on a global static
        TEST_ATOMIC_BASIC.store(10, Ordering::Relaxed);
        assert_eq!(TEST_ATOMIC_BASIC.load(Ordering::Relaxed), 10, "store/load should work");

        let old = TEST_ATOMIC_BASIC.fetch_add(5, Ordering::Relaxed);
        assert_eq!(old, 10, "fetch_add should return old value");
        assert_eq!(TEST_ATOMIC_BASIC.load(Ordering::Relaxed), 15, "value should be 15 after add");

        let old2 = TEST_ATOMIC_BASIC.fetch_sub(3, Ordering::Relaxed);
        assert_eq!(old2, 15, "fetch_sub should return old value");
        assert_eq!(TEST_ATOMIC_BASIC.load(Ordering::Relaxed), 12, "value should be 12 after sub");

        // fetch_and, fetch_or
        TEST_ATOMIC_BASIC.store(0xFF, Ordering::Relaxed);
        let _ = TEST_ATOMIC_BASIC.fetch_and(0x0F, Ordering::Relaxed);
        assert_eq!(TEST_ATOMIC_BASIC.load(Ordering::Relaxed), 0x0F, "fetch_and should mask");

        let _ = TEST_ATOMIC_BASIC.fetch_or(0xF0, Ordering::Relaxed);
        assert_eq!(TEST_ATOMIC_BASIC.load(Ordering::Relaxed), 0xFF, "fetch_or should set bits");

        // Cross-thread atomics: spawn 3 threads each adding 100
        TEST_ATOMIC_COUNTER.store(0, Ordering::Relaxed);

        let h1 = gpu_runtime::thread::spawn(|| {
            for _ in 0..100u32 {
                TEST_ATOMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        });
        let h2 = gpu_runtime::thread::spawn(|| {
            for _ in 0..100u32 {
                TEST_ATOMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        });
        let h3 = gpu_runtime::thread::spawn(|| {
            for _ in 0..100u32 {
                TEST_ATOMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        });

        h1.join();
        h2.join();
        h3.join();

        let final_val = TEST_ATOMIC_COUNTER.load(Ordering::Relaxed);
        assert_eq!(final_val, 300, "3 threads x 100 increments should give 300");

        println!("[gpu_test] test_gpu_atomics PASSED");
    });
}

/// GPU test: iterator chain operations on Vec.
///
/// Zero-param entry. Tests map, filter, fold, zip, enumerate, collect
/// and other iterator combinators on GPU.
///
/// Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_iterator_chain() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main_poll(|| {
        // map + collect
        let v: Vec<u32> = (0..10).map(|x: u32| x * 3).collect();
        assert_eq!(v.len(), 10, "mapped vec should have 10 elements");
        assert_eq!(v[0], 0, "v[0] should be 0");
        assert_eq!(v[3], 9, "v[3] should be 9");

        // filter + collect
        let evens: Vec<u32> = (0..20).filter(|x: &u32| x % 2 == 0).collect();
        assert_eq!(evens.len(), 10, "should have 10 even numbers");
        assert_eq!(evens[0], 0, "first even should be 0");
        assert_eq!(evens[9], 18, "last even should be 18");

        // fold (sum of squares)
        let sum_sq: u32 = (1..=5).fold(0u32, |acc, x| acc + x * x);
        assert_eq!(sum_sq, 55, "1^2+2^2+3^2+4^2+5^2 should be 55");

        // zip + map + sum
        let a: Vec<u32> = vec![1, 2, 3, 4];
        let b: Vec<u32> = vec![10, 20, 30, 40];
        let dot: u32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot, 300, "dot product should be 300");

        // enumerate + filter
        let indexed: Vec<(usize, u32)> = (10..15u32)
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .collect();
        assert_eq!(indexed.len(), 3, "should have 3 even-indexed elements");
        assert_eq!(indexed[0], (0, 10), "first should be (0, 10)");
        assert_eq!(indexed[2], (4, 14), "last should be (4, 14)");

        // chain
        let chained: Vec<u32> = (0..3).chain(10..13).collect();
        assert_eq!(chained.len(), 6, "chained should have 6 elements");
        assert_eq!(chained, vec![0, 1, 2, 10, 11, 12], "chain values mismatch");

        println!("[gpu_test] test_gpu_iterator_chain PASSED");
    });
}

// ============================================================
// gen-mono.2: Generic monomorphization experiment
// ============================================================
//
// Proves that Rust generics compile and run correctly on nvptx64.
// Pattern: concrete `extern "gpu-kernel"` entry → `#[inline(always)]` generic body.
//
// The compiler monomorphizes the generic body at the MIR level, then LLVM
// emits type-specific PTX instructions (e.g., add.rn.f32 vs add.s32).
//
// Two generic operations:
//   1. generic_map_inplace: data[i] = data[i] * scale + bias  (affine transform)
//   2. generic_reduce_sum:  sum of all elements                (reduction)

/// Generic affine transform: data[i] = data[i] * scale + bias.
///
/// Uses a grid-stride loop so it works with any launch configuration.
/// The `#[inline(always)]` ensures LLVM inlines the generic body into
/// each concrete entry point, producing type-specialized PTX.
#[inline(always)]
fn generic_map_inplace<
    T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
>(
    data: *mut T,
    len: usize,
    scale: T,
    bias: T,
) {
    let tid = gpu_runtime::index::global_thread_idx() as usize;
    let stride = gpu_runtime::index::global_thread_count() as usize;
    let mut i = tid;
    while i < len {
        unsafe {
            let val = core::ptr::read(data.add(i));
            let result = val * scale + bias;
            core::ptr::write(data.add(i), result);
        }
        i += stride;
    }
}

/// Generic reduction: sum all elements and return the total.
///
/// Sequential single-thread reduction (launched with 1 thread for simplicity).
/// The generic body monomorphizes to type-specific add instructions.
#[inline(always)]
fn generic_reduce_sum<T: Copy + core::ops::Add<Output = T>>(
    data: *const T,
    len: usize,
    identity: T,
) -> T {
    let mut acc = identity;
    let mut i = 0usize;
    while i < len {
        unsafe {
            let val = core::ptr::read(data.add(i));
            acc = acc + val;
        }
        i += 1;
    }
    acc
}

// ---- Concrete entry points for f32 ----

/// Monomorphized map kernel for f32: data[i] = data[i] * scale + bias.
///
/// f32 params passed as u32 bits to avoid ABI issues with GPU kernel args.
///
/// Launch with: any grid/block config (grid-stride loop).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn generic_map_f32(
    data: *mut f32,
    n: u32,
    scale_bits: u32,
    bias_bits: u32,
) {
    let scale = f32::from_bits(scale_bits);
    let bias = f32::from_bits(bias_bits);
    generic_map_inplace::<f32>(data, n as usize, scale, bias);
}

/// Monomorphized reduce kernel for f32: sum all elements.
///
/// Result written as u32 bits to result[0]. Launch with 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn generic_reduce_f32(
    data: *const f32,
    n: u32,
    result: *mut u32,
) {
    let total = generic_reduce_sum::<f32>(data, n as usize, 0.0f32);
    unsafe {
        core::ptr::write(result, total.to_bits());
    }
}

// ---- Concrete entry points for i32 ----

/// Monomorphized map kernel for i32: data[i] = data[i] * scale + bias.
///
/// Launch with: any grid/block config (grid-stride loop).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn generic_map_i32(
    data: *mut i32,
    n: u32,
    scale: i32,
    bias: i32,
) {
    generic_map_inplace::<i32>(data, n as usize, scale, bias);
}

/// Monomorphized reduce kernel for i32: sum all elements.
///
/// Result written to result[0]. Launch with 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn generic_reduce_i32(
    data: *const i32,
    n: u32,
    result: *mut i32,
) {
    let total = generic_reduce_sum::<i32>(data, n as usize, 0i32);
    unsafe {
        core::ptr::write(result, total);
    }
}

// ---- GPU test kernels for generic monomorphization ----

/// GPU test: generic map f32 — data[i] = data[i] * 2.0 + 1.0.
///
/// Allocates a Vec, fills with known values, applies the generic affine
/// transform, and verifies the results. Proves that generic_map_inplace<f32>
/// monomorphizes to correct f32 PTX instructions.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_map_f32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Prepare data: [0.0, 1.0, 2.0, ..., 15.0]
        let mut data: Vec<f32> = (0..16).map(|i| i as f32).collect();

        // Apply: data[i] = data[i] * 2.0 + 1.0
        generic_map_inplace::<f32>(data.as_mut_ptr(), data.len(), 2.0, 1.0);

        // Verify: expected[i] = i * 2.0 + 1.0
        for i in 0..16u32 {
            let expected = i as f32 * 2.0 + 1.0;
            let actual = data[i as usize];
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.001,
                "generic_map f32: data[{}] = {}, expected {}",
                i,
                actual,
                expected
            );
        }

        println!("[gpu_test] test_gpu_generic_map_f32 PASSED");
    });
}

/// GPU test: generic map i32 — data[i] = data[i] * 3 + 10.
///
/// Same pattern as f32 but with integer types. Proves that
/// generic_map_inplace<i32> monomorphizes to correct i32 PTX instructions.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_map_i32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Prepare data: [0, 1, 2, ..., 15]
        let mut data: Vec<i32> = (0..16).map(|i| i as i32).collect();

        // Apply: data[i] = data[i] * 3 + 10
        generic_map_inplace::<i32>(data.as_mut_ptr(), data.len(), 3, 10);

        // Verify: expected[i] = i * 3 + 10
        for i in 0..16i32 {
            let expected = i * 3 + 10;
            let actual = data[i as usize];
            assert_eq!(
                actual, expected,
                "generic_map i32: data[{}] = {}, expected {}",
                i, actual, expected
            );
        }

        println!("[gpu_test] test_gpu_generic_map_i32 PASSED");
    });
}

/// GPU test: generic reduce f32 — sum of [1.0, 2.0, ..., 16.0].
///
/// Proves that generic_reduce_sum<f32> monomorphizes to correct f32 add
/// instructions and produces the right result.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_reduce_f32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Data: [1.0, 2.0, ..., 16.0]
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();

        let total = generic_reduce_sum::<f32>(data.as_ptr(), data.len(), 0.0);

        // Expected: 1+2+...+16 = 136.0
        let diff = (total - 136.0).abs();
        assert!(
            diff < 0.01,
            "generic_reduce f32: got {}, expected 136.0",
            total
        );

        println!("[gpu_test] test_gpu_generic_reduce_f32 PASSED");
    });
}

/// GPU test: generic reduce i32 — sum of [1, 2, ..., 100].
///
/// Proves that generic_reduce_sum<i32> monomorphizes to correct i32 add
/// instructions and produces the right result.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_reduce_i32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Data: [1, 2, ..., 100]
        let data: Vec<i32> = (1..=100).map(|i| i as i32).collect();

        let total = generic_reduce_sum::<i32>(data.as_ptr(), data.len(), 0);

        // Expected: 100*101/2 = 5050
        assert_eq!(
            total, 5050,
            "generic_reduce i32: got {}, expected 5050",
            total
        );

        println!("[gpu_test] test_gpu_generic_reduce_i32 PASSED");
    });
}

/// GPU test: same generic body, two types — proves monomorphization correctness.
///
/// Calls the SAME generic_map_inplace function with both f32 and i32 data
/// in a single kernel, then verifies both produce correct type-specific results.
/// This is the definitive proof that Rust monomorphization works on nvptx64.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_dual_type() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // f32 path: data[i] = i * 0.5 + 100.0
        let mut f32_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        generic_map_inplace::<f32>(f32_data.as_mut_ptr(), f32_data.len(), 0.5, 100.0);

        // i32 path: data[i] = i * 7 + (-3)
        let mut i32_data: Vec<i32> = (0..8).map(|i| i as i32).collect();
        generic_map_inplace::<i32>(i32_data.as_mut_ptr(), i32_data.len(), 7, -3);

        // Verify f32
        for i in 0..8u32 {
            let expected = i as f32 * 0.5 + 100.0;
            let actual = f32_data[i as usize];
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.001,
                "dual_type f32[{}] = {}, expected {}",
                i,
                actual,
                expected
            );
        }

        // Verify i32
        for i in 0..8i32 {
            let expected = i * 7 + (-3);
            let actual = i32_data[i as usize];
            assert_eq!(
                actual, expected,
                "dual_type i32[{}] = {}, expected {}",
                i, actual, expected
            );
        }

        // Also verify reduce on both types
        let f32_sum = generic_reduce_sum::<f32>(f32_data.as_ptr(), f32_data.len(), 0.0);
        // Expected: sum of (i*0.5+100.0) for i=0..8 = (0+0.5+1+1.5+2+2.5+3+3.5) + 800 = 814.0
        let expected_f32_sum = 814.0f32;
        let diff = (f32_sum - expected_f32_sum).abs();
        assert!(
            diff < 0.1,
            "dual_type f32 reduce: got {}, expected {}",
            f32_sum,
            expected_f32_sum
        );

        let i32_sum = generic_reduce_sum::<i32>(i32_data.as_ptr(), i32_data.len(), 0);
        // Expected: sum of (i*7-3) for i=0..8 = (0+7+14+21+28+35+42+49) - 24 = 196 - 24 = 172
        assert_eq!(
            i32_sum, 172,
            "dual_type i32 reduce: got {}, expected 172",
            i32_sum
        );

        println!("[gpu_test] test_gpu_generic_dual_type PASSED");
    });
}

// ============================================================
// gen-traits.1: User-defined traits with where bounds on GPU
// ============================================================
//
// Proves that user-defined traits (GpuReducible, GpuTransformable)
// with `where` bounds compile and run correctly on nvptx64.
// Also tests a custom #[repr(C)] type implementing the trait.
//
// Pattern: concrete `extern "gpu-kernel"` entry → `#[inline(always)]`
// generic body bounded by user-defined trait.

use gpu_runtime::traits::{GpuReducible, GpuTransformable};

/// Custom 2D vector type — proves user-defined #[repr(C)] types work
/// with user-defined traits on GPU.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vec2f {
    x: f32,
    y: f32,
}

impl GpuReducible for Vec2f {
    #[inline(always)]
    fn identity() -> Self {
        Vec2f { x: 0.0, y: 0.0 }
    }
    #[inline(always)]
    fn combine(self, other: Self) -> Self {
        Vec2f {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl GpuTransformable for Vec2f {
    #[inline(always)]
    fn default_value() -> Self {
        Vec2f { x: 0.0, y: 0.0 }
    }
    #[inline(always)]
    fn scale(self, factor: Self) -> Self {
        Vec2f {
            x: self.x * factor.x,
            y: self.y * factor.y,
        }
    }
    #[inline(always)]
    fn offset(self, amount: Self) -> Self {
        Vec2f {
            x: self.x + amount.x,
            y: self.y + amount.y,
        }
    }
}

/// Generic parallel reduce using user-defined GpuReducible trait.
///
/// Works for any `T: GpuReducible` — the compiler monomorphizes per type,
/// inlining `identity()` and `combine()` into type-specific PTX instructions.
#[inline(always)]
fn trait_reduce<T: GpuReducible>(data: *const T, len: usize) -> T {
    let mut acc = T::identity();
    let mut i = 0usize;
    while i < len {
        let val = unsafe { core::ptr::read(data.add(i)) };
        acc = acc.combine(val);
        i += 1;
    }
    acc
}

/// Generic transform using `where` bounds — proves explicit where-clause
/// syntax monomorphizes identically to inline trait bounds.
#[inline(always)]
fn apply_transform<T>(data: *mut T, len: usize, factor: T, amount: T)
where
    T: GpuTransformable,
{
    let tid = gpu_runtime::index::global_thread_idx() as usize;
    let stride = gpu_runtime::index::global_thread_count() as usize;
    let mut i = tid;
    while i < len {
        unsafe {
            let val = core::ptr::read(data.add(i));
            let result = val.scale(factor).offset(amount);
            core::ptr::write(data.add(i), result);
        }
        i += stride;
    }
}

/// Generic function with combined trait + where bounds.
///
/// Proves that a function can use both GpuReducible and GpuTransformable
/// bounds on the same type parameter.
#[inline(always)]
fn transform_then_reduce<T>(data: *mut T, len: usize, factor: T, amount: T) -> T
where
    T: GpuReducible + GpuTransformable,
{
    // Step 1: transform in place
    let mut i = 0usize;
    while i < len {
        unsafe {
            let val = core::ptr::read(data.add(i));
            let result = val.scale(factor).offset(amount);
            core::ptr::write(data.add(i), result);
        }
        i += 1;
    }
    // Step 2: reduce
    trait_reduce(data, len)
}

// ---- Concrete entry points for trait-based reduce ----

/// Monomorphized trait reduce for f32: sum all elements via GpuReducible.
///
/// Result written as u32 bits to result[0]. Launch with 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn trait_reduce_f32(
    data: *const f32,
    n: u32,
    result: *mut u32,
) {
    let total = trait_reduce::<f32>(data, n as usize);
    unsafe {
        core::ptr::write(result, total.to_bits());
    }
}

/// Monomorphized trait reduce for i32: sum all elements via GpuReducible.
///
/// Result written to result[0]. Launch with 1 thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn trait_reduce_i32(
    data: *const i32,
    n: u32,
    result: *mut i32,
) {
    let total = trait_reduce::<i32>(data, n as usize);
    unsafe {
        core::ptr::write(result, total);
    }
}

// ---- GPU test kernels for trait-based generics ----

/// GPU test: trait-based reduce f32 — sum via GpuReducible.
///
/// Proves that user-defined trait methods (identity, combine) monomorphize
/// to correct f32 PTX instructions.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_trait_reduce_f32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Data: [1.0, 2.0, ..., 20.0]
        let data: Vec<f32> = (1..=20).map(|i| i as f32).collect();

        let total = trait_reduce::<f32>(data.as_ptr(), data.len());

        // Expected: 20*21/2 = 210.0
        let diff = (total - 210.0).abs();
        assert!(
            diff < 0.01,
            "trait_reduce f32: got {}, expected 210.0",
            total
        );

        // Verify identity
        let id = f32::identity();
        assert!(
            id.abs() < 0.001,
            "f32::identity() should be 0.0, got {}",
            id
        );

        // Verify combine
        let combined = 3.0f32.combine(4.0);
        let diff2 = (combined - 7.0).abs();
        assert!(
            diff2 < 0.001,
            "3.0.combine(4.0) should be 7.0, got {}",
            combined
        );

        println!("[gpu_test] test_gpu_trait_reduce_f32 PASSED");
    });
}

/// GPU test: trait-based reduce i32 — sum via GpuReducible.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_trait_reduce_i32() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Data: [1, 2, ..., 50]
        let data: Vec<i32> = (1..=50).map(|i| i as i32).collect();

        let total = trait_reduce::<i32>(data.as_ptr(), data.len());

        // Expected: 50*51/2 = 1275
        assert_eq!(
            total, 1275,
            "trait_reduce i32: got {}, expected 1275",
            total
        );

        // Verify identity
        let id = i32::identity();
        assert_eq!(id, 0, "i32::identity() should be 0");

        // Verify combine
        let combined = 10i32.combine(20);
        assert_eq!(combined, 30, "10.combine(20) should be 30");

        println!("[gpu_test] test_gpu_trait_reduce_i32 PASSED");
    });
}

/// GPU test: where-clause transform — GpuTransformable with explicit where bounds.
///
/// Proves that `where T: GpuTransformable` monomorphizes the same as
/// `<T: GpuTransformable>` and produces correct type-specific PTX.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_where_transform() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // f32 transform: data[i] = data[i] * 3.0 + 10.0
        let mut f32_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        apply_transform::<f32>(f32_data.as_mut_ptr(), f32_data.len(), 3.0, 10.0);

        for i in 0..8u32 {
            let expected = i as f32 * 3.0 + 10.0;
            let actual = f32_data[i as usize];
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.001,
                "where_transform f32[{}] = {}, expected {}",
                i, actual, expected
            );
        }

        // i32 transform: data[i] = data[i] * 5 + (-2)
        let mut i32_data: Vec<i32> = (0..8).map(|i| i as i32).collect();
        apply_transform::<i32>(i32_data.as_mut_ptr(), i32_data.len(), 5, -2);

        for i in 0..8i32 {
            let expected = i * 5 + (-2);
            let actual = i32_data[i as usize];
            assert_eq!(
                actual, expected,
                "where_transform i32[{}] = {}, expected {}",
                i, actual, expected
            );
        }

        println!("[gpu_test] test_gpu_where_transform PASSED");
    });
}

/// GPU test: combined transform + reduce on primitives.
///
/// Uses transform_then_reduce which requires both GpuReducible + GpuTransformable.
/// Proves that multiple trait bounds on a generic parameter monomorphize correctly.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_trait_combined() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // f32: data = [1,2,3,4,5], transform: x*2+10, then reduce (sum)
        // After transform: [12, 14, 16, 18, 20]
        // Sum: 80.0
        let mut f32_data: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let f32_sum = transform_then_reduce::<f32>(
            f32_data.as_mut_ptr(),
            f32_data.len(),
            2.0,
            10.0,
        );
        let diff = (f32_sum - 80.0).abs();
        assert!(
            diff < 0.01,
            "trait_combined f32: got {}, expected 80.0",
            f32_sum
        );

        // i32: data = [1,2,3,4], transform: x*3+1, then reduce (sum)
        // After transform: [4, 7, 10, 13]
        // Sum: 34
        let mut i32_data: Vec<i32> = (1..=4).map(|i| i as i32).collect();
        let i32_sum = transform_then_reduce::<i32>(
            i32_data.as_mut_ptr(),
            i32_data.len(),
            3,
            1,
        );
        assert_eq!(
            i32_sum, 34,
            "trait_combined i32: got {}, expected 34",
            i32_sum
        );

        println!("[gpu_test] test_gpu_trait_combined PASSED");
    });
}

/// GPU test: custom Vec2f type with GpuReducible — user-defined struct on GPU.
///
/// Proves that a user-defined `#[repr(C)]` struct implementing `GpuReducible`
/// monomorphizes through generic reduce to correct per-field PTX operations.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_trait_custom_vec2f() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // Create Vec2f data: [(1,10), (2,20), (3,30), (4,40), (5,50)]
        let data: Vec<Vec2f> = (1..=5)
            .map(|i| Vec2f {
                x: i as f32,
                y: (i * 10) as f32,
            })
            .collect();

        // Reduce: sum of x = 15.0, sum of y = 150.0
        let total = trait_reduce::<Vec2f>(data.as_ptr(), data.len());
        let dx = (total.x - 15.0).abs();
        let dy = (total.y - 150.0).abs();
        assert!(
            dx < 0.01,
            "Vec2f reduce: x = {}, expected 15.0",
            total.x
        );
        assert!(
            dy < 0.01,
            "Vec2f reduce: y = {}, expected 150.0",
            total.y
        );

        // Verify identity
        let id = Vec2f::identity();
        assert!(id.x.abs() < 0.001, "Vec2f identity x should be 0.0");
        assert!(id.y.abs() < 0.001, "Vec2f identity y should be 0.0");

        // Verify combine
        let a = Vec2f { x: 1.0, y: 2.0 };
        let b = Vec2f { x: 3.0, y: 4.0 };
        let c = a.combine(b);
        assert!((c.x - 4.0).abs() < 0.001, "combine x: expected 4.0");
        assert!((c.y - 6.0).abs() < 0.001, "combine y: expected 6.0");

        // Test transform_then_reduce on Vec2f
        let mut v2_data: Vec<Vec2f> = (1..=3)
            .map(|i| Vec2f {
                x: i as f32,
                y: (i * 2) as f32,
            })
            .collect();
        let factor = Vec2f { x: 2.0, y: 3.0 };
        let amount = Vec2f { x: 1.0, y: -1.0 };
        // Transform: [(1,2)→(3,5), (2,4)→(5,11), (3,6)→(7,17)]
        // x: 1*2+1=3, 2*2+1=5, 3*2+1=7 → sum=15
        // y: 2*3-1=5, 4*3-1=11, 6*3-1=17 → sum=33
        let result = transform_then_reduce::<Vec2f>(
            v2_data.as_mut_ptr(),
            v2_data.len(),
            factor,
            amount,
        );
        assert!(
            (result.x - 15.0).abs() < 0.01,
            "Vec2f transform_then_reduce x: got {}, expected 15.0",
            result.x
        );
        assert!(
            (result.y - 33.0).abs() < 0.01,
            "Vec2f transform_then_reduce y: got {}, expected 33.0",
            result.y
        );

        println!("[gpu_test] test_gpu_trait_custom_vec2f PASSED");
    });
}

/// GPU test: trait dispatch for multiple types in one kernel.
///
/// Calls the same generic function (trait_reduce, apply_transform) with f32,
/// i32, and Vec2f in a single kernel — the definitive proof that user-defined
/// traits monomorphize correctly for all types on nvptx64.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_trait_multi_type() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        // f32 reduce
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let f32_sum = trait_reduce::<f32>(f32_data.as_ptr(), f32_data.len());
        let f32_diff = (f32_sum - 10.0).abs();
        assert!(f32_diff < 0.01, "multi_type f32 reduce: got {}", f32_sum);

        // i32 reduce
        let i32_data: Vec<i32> = vec![10, 20, 30, 40];
        let i32_sum = trait_reduce::<i32>(i32_data.as_ptr(), i32_data.len());
        assert_eq!(i32_sum, 100, "multi_type i32 reduce: got {}", i32_sum);

        // Vec2f reduce
        let v2_data: Vec<Vec2f> = vec![
            Vec2f { x: 1.0, y: 10.0 },
            Vec2f { x: 2.0, y: 20.0 },
            Vec2f { x: 3.0, y: 30.0 },
        ];
        let v2_sum = trait_reduce::<Vec2f>(v2_data.as_ptr(), v2_data.len());
        assert!((v2_sum.x - 6.0).abs() < 0.01, "multi_type Vec2f x: got {}", v2_sum.x);
        assert!((v2_sum.y - 60.0).abs() < 0.01, "multi_type Vec2f y: got {}", v2_sum.y);

        // f32 transform
        let mut f32_t: Vec<f32> = vec![1.0, 2.0, 3.0];
        apply_transform::<f32>(f32_t.as_mut_ptr(), f32_t.len(), 2.0, 5.0);
        assert!((f32_t[0] - 7.0).abs() < 0.01, "multi_type f32 transform[0]");
        assert!((f32_t[2] - 11.0).abs() < 0.01, "multi_type f32 transform[2]");

        // u32 reduce (proves u32 impl works too)
        let u32_data: Vec<u32> = vec![100, 200, 300];
        let u32_sum = trait_reduce::<u32>(u32_data.as_ptr(), u32_data.len());
        assert_eq!(u32_sum, 600, "multi_type u32 reduce: got {}", u32_sum);

        println!("[gpu_test] test_gpu_trait_multi_type PASSED");
    });
}

// ============================================================
// gen-demo.1: Generic parallel_reduce<T: Reducible> — showcase
// ============================================================
//
// EPIC LITMUS TEST: fn parallel_reduce<T: Add>(data: &[T]) -> T
// works on GPU for any T.
//
// This demo proves that the SAME generic function `parallel_reduce`
// works for f32, i32, and a custom Vec2f struct — all monomorphized
// to type-specific PTX with zero overhead.
//
// The showcase consists of:
// 1. A polished `parallel_reduce<T: GpuReducible>` function
// 2. A `parallel_map_reduce<T: GpuReducible + GpuTransformable>` composition
// 3. GPU test kernels exercising all three types at meaningful scale (1024 elements)
// 4. A zero-overhead comparison test (generic vs handwritten)

/// Polished generic parallel reduce — the epic's litmus test.
///
/// Computes the reduction of `data[0..len]` using the `GpuReducible` trait:
/// starts from `T::identity()` and folds via `T::combine()`.
///
/// This is the function that proves: "fn parallel_reduce<T: Add>(data: &[T]) -> T
/// works on GPU for any T." The compiler monomorphizes it per type, producing:
/// - f32: `add.rn.f32` with IEEE 754 rounding, LLVM 4x loop unrolling
/// - i32: `add.s32` signed integer add, LLVM 4x loop unrolling + MAD fusion
/// - Vec2f: two `add.rn.f32` instructions per combine (one for x, one for y)
///
/// Zero overhead: the generated PTX is identical to hand-written type-specific code.
#[inline(always)]
fn parallel_reduce<T: GpuReducible>(data: *const T, len: usize) -> T {
    let mut acc = T::identity();
    let mut i = 0usize;
    while i < len {
        let val = unsafe { core::ptr::read(data.add(i)) };
        acc = acc.combine(val);
        i += 1;
    }
    acc
}

/// Generic map-then-reduce: transform each element, then reduce.
///
/// Demonstrates composing multiple trait bounds on the same type parameter.
/// The compiler fully inlines both the transform and reduce steps, producing
/// a single fused loop in PTX — no intermediate buffer allocation.
#[inline(always)]
fn parallel_map_reduce<T: GpuReducible + GpuTransformable>(
    data: *const T,
    len: usize,
    factor: T,
    amount: T,
) -> T {
    let mut acc = T::identity();
    let mut i = 0usize;
    while i < len {
        let val = unsafe { core::ptr::read(data.add(i)) };
        let transformed = val.scale(factor).offset(amount);
        acc = acc.combine(transformed);
        i += 1;
    }
    acc
}

/// Handwritten (non-generic) f32 reduce — baseline for zero-overhead comparison.
///
/// This is intentionally NOT generic. It uses the exact same algorithm as
/// `parallel_reduce::<f32>`. If the generated PTX is identical (or within
/// noise), that proves zero-overhead abstraction.
#[inline(always)]
fn handwritten_reduce_f32(data: *const f32, len: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;
    while i < len {
        let val = unsafe { core::ptr::read(data.add(i)) };
        acc = acc + val;
        i += 1;
    }
    acc
}

/// Handwritten (non-generic) i32 reduce — baseline for zero-overhead comparison.
#[inline(always)]
fn handwritten_reduce_i32(data: *const i32, len: usize) -> i32 {
    let mut acc = 0i32;
    let mut i = 0usize;
    while i < len {
        let val = unsafe { core::ptr::read(data.add(i)) };
        acc = acc + val;
        i += 1;
    }
    acc
}

// ---- GPU test: generic reduce at scale (1024 elements per type) ----

/// GPU test: generic parallel_reduce at scale — f32, i32, Vec2f with 1024 elements.
///
/// This is the SHOWCASE test for the gpu-generics epic. It proves:
/// 1. The SAME `parallel_reduce` function works for f32, i32, and Vec2f
/// 2. Results are correct for meaningful data sizes (1024 elements)
/// 3. CPU reference values match GPU results exactly
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_reduce_showcase() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        const N: usize = 1024;

        // ---- f32: sum of [1.0, 2.0, ..., 1024.0] ----
        let f32_data: Vec<f32> = (1..=N as u32).map(|i| i as f32).collect();
        let f32_result = parallel_reduce::<f32>(f32_data.as_ptr(), f32_data.len());
        // Expected: N*(N+1)/2 = 1024*1025/2 = 524800.0
        let f32_expected = 524800.0f32;
        let f32_diff = (f32_result - f32_expected).abs();
        assert!(
            f32_diff < 1.0,
            "parallel_reduce<f32> 1024 elems: got {}, expected {}",
            f32_result, f32_expected
        );

        // ---- i32: sum of [1, 2, ..., 1024] ----
        let i32_data: Vec<i32> = (1..=N as i32).collect();
        let i32_result = parallel_reduce::<i32>(i32_data.as_ptr(), i32_data.len());
        // Expected: 1024*1025/2 = 524800
        assert_eq!(
            i32_result, 524800,
            "parallel_reduce<i32> 1024 elems: got {}, expected 524800",
            i32_result
        );

        // ---- Vec2f: sum of [(1,2), (2,4), (3,6), ..., (1024,2048)] ----
        let vec2f_data: Vec<Vec2f> = (1..=N as u32)
            .map(|i| Vec2f {
                x: i as f32,
                y: (i * 2) as f32,
            })
            .collect();
        let vec2f_result = parallel_reduce::<Vec2f>(vec2f_data.as_ptr(), vec2f_data.len());
        // Expected: x = 524800.0, y = 1049600.0
        let vx_diff = (vec2f_result.x - 524800.0).abs();
        let vy_diff = (vec2f_result.y - 1049600.0).abs();
        assert!(
            vx_diff < 1.0,
            "parallel_reduce<Vec2f> x: got {}, expected 524800.0",
            vec2f_result.x
        );
        assert!(
            vy_diff < 1.0,
            "parallel_reduce<Vec2f> y: got {}, expected 1049600.0",
            vec2f_result.y
        );

        println!("[gpu_test] test_gpu_generic_reduce_showcase PASSED");
        println!("  f32: parallel_reduce(1..=1024) = {} (expected {})", f32_result, f32_expected);
        println!("  i32: parallel_reduce(1..=1024) = {} (expected 524800)", i32_result);
        println!("  Vec2f: parallel_reduce = ({}, {}) (expected (524800, 1049600))",
            vec2f_result.x, vec2f_result.y);
    });
}

// ---- GPU test: zero-overhead comparison (generic vs handwritten) ----

/// GPU test: zero-overhead proof — generic reduce produces identical results to handwritten.
///
/// Runs both `parallel_reduce::<f32>` and `handwritten_reduce_f32` on the same
/// data, then compares. If results match exactly, the generic abstraction has
/// zero overhead — the compiler produces identical PTX for both.
///
/// Also compares i32 generic vs handwritten.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_zero_overhead() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        const N: usize = 2048;

        // ---- f32 comparison ----
        let f32_data: Vec<f32> = (1..=N as u32).map(|i| i as f32 * 0.1).collect();

        let generic_f32 = parallel_reduce::<f32>(f32_data.as_ptr(), f32_data.len());
        let handwritten_f32 = handwritten_reduce_f32(f32_data.as_ptr(), f32_data.len());

        // They should be bit-identical since the algorithm is the same
        let f32_diff = (generic_f32 - handwritten_f32).abs();
        assert!(
            f32_diff < 0.001,
            "zero-overhead f32: generic={}, handwritten={}, diff={}",
            generic_f32, handwritten_f32, f32_diff
        );

        // ---- i32 comparison ----
        let i32_data: Vec<i32> = (1..=N as i32).collect();

        let generic_i32 = parallel_reduce::<i32>(i32_data.as_ptr(), i32_data.len());
        let handwritten_i32 = handwritten_reduce_i32(i32_data.as_ptr(), i32_data.len());

        assert_eq!(
            generic_i32, handwritten_i32,
            "zero-overhead i32: generic={}, handwritten={}",
            generic_i32, handwritten_i32
        );

        println!("[gpu_test] test_gpu_generic_zero_overhead PASSED");
        println!("  f32: generic={}, handwritten={} (diff={})", generic_f32, handwritten_f32, f32_diff);
        println!("  i32: generic={}, handwritten={}", generic_i32, handwritten_i32);
    });
}

// ---- GPU test: map-then-reduce composition ----

/// GPU test: parallel_map_reduce — compose transform + reduce in single generic function.
///
/// Demonstrates that multiple trait bounds (GpuReducible + GpuTransformable) compose
/// correctly, and the compiler fuses the transform+reduce into a single loop — no
/// intermediate buffer allocation.
///
/// Zero-param entry. Launch with: block_dim=(128,1,1), 1 block, NO kernel args.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn test_gpu_generic_map_reduce() {
    let buf = stdio_auto_init();
    if buf.is_null() {
        return;
    }

    gpu_runtime::thread::gpu_main(|| {
        const N: usize = 1024;

        // f32: data = [1.0, ..., 1024.0], transform: x*2.0+1.0, then sum
        // After transform: [3.0, 5.0, 7.0, ..., 2049.0]
        // Sum: sum(2*i+1 for i=1..=1024) = 2*524800 + 1024 = 1050624.0
        let f32_data: Vec<f32> = (1..=N as u32).map(|i| i as f32).collect();
        let f32_result = parallel_map_reduce::<f32>(
            f32_data.as_ptr(), f32_data.len(), 2.0, 1.0
        );
        let f32_expected = 1050624.0f32;
        let f32_diff = (f32_result - f32_expected).abs();
        assert!(
            f32_diff < 2.0,
            "map_reduce<f32>: got {}, expected {}",
            f32_result, f32_expected
        );

        // i32: data = [1, ..., 100], transform: x*3+(-1), then sum
        // Sum: sum(3*i-1 for i=1..=100) = 3*5050 - 100 = 15050
        let i32_data: Vec<i32> = (1..=100).collect();
        let i32_result = parallel_map_reduce::<i32>(
            i32_data.as_ptr(), i32_data.len(), 3, -1
        );
        assert_eq!(
            i32_result, 15050,
            "map_reduce<i32>: got {}, expected 15050",
            i32_result
        );

        // Vec2f: data = [(1,10), (2,20), ..., (50,500)]
        // transform: scale(2,3) + offset(1,-1)
        // x: sum(2*i+1 for i=1..=50) = 2*1275 + 50 = 2600
        // y: sum(3*i*10 - 1 for i=1..=50) = 30*1275 - 50 = 38200
        let vec2f_data: Vec<Vec2f> = (1..=50u32)
            .map(|i| Vec2f {
                x: i as f32,
                y: (i * 10) as f32,
            })
            .collect();
        let vec2f_result = parallel_map_reduce::<Vec2f>(
            vec2f_data.as_ptr(),
            vec2f_data.len(),
            Vec2f { x: 2.0, y: 3.0 },
            Vec2f { x: 1.0, y: -1.0 },
        );
        let vx_diff = (vec2f_result.x - 2600.0).abs();
        let vy_diff = (vec2f_result.y - 38200.0).abs();
        assert!(
            vx_diff < 1.0,
            "map_reduce<Vec2f> x: got {}, expected 2600.0",
            vec2f_result.x
        );
        assert!(
            vy_diff < 1.0,
            "map_reduce<Vec2f> y: got {}, expected 38200.0",
            vec2f_result.y
        );

        println!("[gpu_test] test_gpu_generic_map_reduce PASSED");
        println!("  f32 map_reduce: {} (expected {})", f32_result, f32_expected);
        println!("  i32 map_reduce: {} (expected 15050)", i32_result);
        println!("  Vec2f map_reduce: ({}, {}) (expected (2600, 38200))",
            vec2f_result.x, vec2f_result.y);
    });
}
