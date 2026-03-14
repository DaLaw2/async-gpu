// GPU kernels using real Rust std (println!, Vec, String, format!).
//
// Unlike std-build-test which duplicates 430+ lines of hostcall inline PTX,
// this crate depends on gpu-runtime for the hostcall protocol implementation.
// This eliminates code duplication and makes std kernels first-class citizens.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_ptx)]

use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Global hostcall buffer pointer for stdio. Set by kernel at entry.
static STDIO_HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);

/// External function called by std's CUDA PAL Stdout::write().
/// Routes through gpu-runtime's hostcall PRINT implementation.
#[unsafe(no_mangle)]
pub fn gpu_stdout_write(buf: *const u8, len: usize) -> usize {
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || buf.is_null() || len == 0 {
        return len; // silently discard if no hostcall buffer set
    }
    // Send via gpu-runtime's hostcall PRINT service (56-byte chunks)
    const MAX_CHUNK: usize = 56;
    let mut offset = 0usize;
    while offset < len {
        let chunk_len = core::cmp::min(len - offset, MAX_CHUNK);
        let result = unsafe {
            gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, buf.add(offset), chunk_len as u32)
        };
        if result.is_err() {
            return offset; // partial write on failure
        }
        offset += chunk_len;
    }
    len
}

/// External function called by std's CUDA PAL Stdin::read().
/// Routes through gpu-runtime's hostcall SERVICE_STDIN implementation.
#[unsafe(no_mangle)]
pub fn gpu_stdin_read(out_buf: *mut u8, max_len: usize) -> usize {
    use gpu_runtime::prelude::{PKT_OFF_PAYLOAD, SERVICE_STDIN};

    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || out_buf.is_null() || max_len == 0 {
        return 0;
    }
    // SERVICE_STDIN payload slots 1-7 = 56 bytes max
    const STDIN_MAX: usize = 56;
    let request_len = core::cmp::min(max_len, STDIN_MAX) as u32;

    // Stdin is blocking on host — use extended timeout (100M spins vs default 10M)
    const STDIN_MAX_SPIN: u32 = 100_000_000;
    let pkt = match unsafe {
        gpu_runtime::hostcall::gpu_hostcall_request_with_timeout(
            hc_buf,
            SERVICE_STDIN,
            STDIN_MAX_SPIN,
            |payload| {
                // Slot 0: max bytes to read
                core::ptr::write_volatile(payload as *mut u64, request_len as u64);
            },
        )
    } {
        Ok(p) => p,
        Err(_) => return 0,
    };

    let bytes_read = unsafe {
        let slot0 = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
        let src = pkt.add(PKT_OFF_PAYLOAD).add(8); // slots 1-7
        let copy_len = core::cmp::min(slot0, request_len as u64) as usize;
        let mut i = 0usize;
        while i < copy_len {
            *out_buf.add(i) = core::ptr::read_volatile(src.add(i));
            i += 1;
        }
        gpu_runtime::hostcall::gpu_hostcall_release(hc_buf, pkt);
        copy_len
    };

    bytes_read
}

/// Set the hostcall buffer pointer for stdio. Must be called at kernel entry.
fn stdio_init(buf: *mut u8) {
    STDIO_HOSTCALL_BUF.store(buf as u64, AtomicOrdering::Relaxed);
}

// ============================================================
// Test kernels — demonstrate real std on GPU
// ============================================================

/// Test kernel: println! via patched std (no custom macros).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_println_test(buf: *mut u8) {
    stdio_init(buf);

    println!("Hello from gpu-kernel-std!");
    println!("This uses real Rust std println!, not a custom macro.");
}

/// Test kernel: Vec + String + format! via std allocator.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_vec_format_test(buf: *mut u8) {
    stdio_init(buf);

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
    v2.extend_from_slice(b"gpu-kernel-std works!");
    println!("String from bytes: {}", core::str::from_utf8(&v2).unwrap());
}

/// Test kernel: multiple allocations and drops to verify allocator.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_alloc_stress_test(buf: *mut u8) {
    stdio_init(buf);

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
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_file_io_test(buf: *mut u8) {
    stdio_init(buf);

    // Also initialize gpu-libc I/O for the libc→hostcall bridge
    gpu_libc::gpu_libc_io_init(buf);

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
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_pipeline_test(buf: *mut u8) {
    stdio_init(buf);
    gpu_libc::gpu_libc_io_init(buf);

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
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_stdin_test(buf: *mut u8) {
    stdio_init(buf);

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
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn std_hashmap_test(buf: *mut u8) {
    stdio_init(buf);
    gpu_libc::gpu_libc_io_init(buf);

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
