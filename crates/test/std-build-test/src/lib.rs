// Test: can we compile with -Zbuild-std=std for nvptx64?
// Requires patched std source with cuda target support.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_gpu_kernel)]
#![feature(asm_experimental_arch)]

// Force gpu-libc symbols (open/close/read/write/__errno_location) into the PTX.
// Without this, LTO removes them because std only references them via extern "C" declarations
// which are not visible to the Rust linker as direct dependencies.
//
// We take function pointers to force LLVM to keep these symbols alive through LTO.
extern crate gpu_libc;

// gpu-runtime provides the authoritative #[no_mangle] gpu_stdout_write / gpu_stdin_read
// implementations. We depend on it to avoid symbol collisions at link time.
extern crate gpu_runtime;

/// Force gpu-libc symbols to survive LTO by referencing them in a #[used] array.
/// Without this, LLVM removes the `#[no_mangle]` functions during LTO because
/// std only declares them via `extern "C"` blocks (invisible to the Rust linker).
///
/// We use fn() pointers (which are Sync) wrapped in a newtype.
#[repr(transparent)]
struct FnPtr(*const ());
unsafe impl Sync for FnPtr {}

#[used]
static FORCE_LINK_GPU_LIBC: [FnPtr; 5] = [
    FnPtr(gpu_libc::open as *const ()),
    FnPtr(gpu_libc::close as *const ()),
    FnPtr(gpu_libc::read as *const ()),
    FnPtr(gpu_libc::write as *const ()),
    FnPtr(gpu_libc::__errno_location as *const ()),
];

/// Set the hostcall buffer for stdio and gpu-libc. Must be called at kernel entry.
///
/// Delegates to gpu-runtime for stdout/stdin, and also initializes gpu-libc I/O
/// so that std::fs::File (which uses PAL -> gpu-libc open/read/write/close) works.
fn stdio_init(buf: *mut u8) {
    gpu_runtime::stdio::stdio_init(buf);
    unsafe {
        gpu_libc::gpu_libc_io_init(buf);
    }
}

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_hello_kernel(result: *mut u32) {
    // Test that std types are available
    let v = vec![1u32, 2, 3, 4, 5];
    let sum: u32 = v.iter().sum();

    // Test String from std (uses alloc)
    let s = String::from("Hello from GPU std!");
    let len = s.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, sum + len);
    }
}

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_format_kernel(result: *mut u32) {
    // Test format! macro (uses alloc + fmt)
    let formatted = format!("value = {}", 42u32);
    let len = formatted.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, len);
    }
}

// ============================================================
// product.1: Dynamic allocation stress tests with runtime data
// ============================================================
// These kernels take runtime values as kernel arguments to prevent
// LLVM from constant-folding the allocations away.

/// Test 1: Vec with runtime data — build a Vec from kernel args, sum elements.
///
/// `input` = pointer to array of u32 values (device memory)
/// `input_len` = number of elements to push
/// `result` = output: sum of all elements
///
/// This forces the bump allocator to actually allocate because the Vec
/// contents come from device memory at runtime.
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_dynamic_vec_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut v: Vec<u32> = Vec::new();
    let len = input_len;

    // Push runtime values one by one — forces Vec to grow and reallocate
    let mut i: u32 = 0;
    while i < len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    unsafe {
        core::ptr::write_volatile(result, sum);
    }
}

/// Test 2: format! with runtime value — ensures the allocator is used for String.
///
/// `value` = runtime u32 value to format
/// `result` = output: length of formatted string
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_dynamic_format_kernel(value: u32, result: *mut u32) {
    let formatted = format!("result = {}", value);
    let len = formatted.len() as u32;
    unsafe {
        core::ptr::write_volatile(result, len);
    }
}

/// Test 3: Multiple Vecs alive simultaneously — tests allocator under pressure.
///
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output array of u32[3]:
///   [0] = sum of first Vec (even indices)
///   [1] = sum of second Vec (odd indices)
///   [2] = total elements across both Vecs
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_dynamic_multi_vec_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut evens: Vec<u32> = Vec::new();
    let mut odds: Vec<u32> = Vec::new();

    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        if i % 2 == 0 {
            evens.push(val);
        } else {
            odds.push(val);
        }
        i += 1;
    }

    let even_sum: u32 = evens.iter().sum();
    let odd_sum: u32 = odds.iter().sum();
    let total_len = (evens.len() + odds.len()) as u32;

    unsafe {
        core::ptr::write_volatile(result.add(0), even_sum);
        core::ptr::write_volatile(result.add(1), odd_sum);
        core::ptr::write_volatile(result.add(2), total_len);
    }
}

/// Test 4: Vec with pre-capacity — tests Vec::with_capacity + push.
///
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output: sum of all elements
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_dynamic_vec_capacity_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut v: Vec<u32> = Vec::with_capacity(input_len as usize);

    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    unsafe {
        core::ptr::write_volatile(result, sum);
    }
}

// ============================================================
// std-pal.1: PAL stdout routing — std::io::Write via hostcall
// ============================================================

/// Test: write! to std::io::stdout() routed through hostcall.
///
/// `buf` = hostcall buffer (mapped memory)
/// `value` = runtime u32 value to format and print
/// `result` = output: 1 on success, 0 on failure
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_println_kernel(buf: *mut u8, value: u32, result: *mut u32) {
    use std::io::Write;

    stdio_init(buf);

    // Use writeln! which goes through Write::write_fmt, not the _print path.
    // println! uses _print → print_to which has global capture state that
    // causes LLVM NVPTX "Circular dependency" crashes.
    let ok = writeln!(
        std::io::stdout(),
        "Hello from GPU println! value = {}",
        value
    )
    .is_ok();

    unsafe {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

/// Test: multiple writes to stdout with runtime data.
///
/// `buf` = hostcall buffer
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements to print
/// `result` = output: number of successful writes
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_println_multi_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    let mut count: u32 = 0;
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        if writeln!(std::io::stdout(), "GPU[{}] = {}", i, val).is_ok() {
            count += 1;
        }
        i += 1;
    }

    unsafe {
        core::ptr::write_volatile(result, count);
    }
}

/// Test: writeln! with formatted Vec contents.
///
/// `buf` = hostcall buffer
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output: 1 on success
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_println_vec_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    let mut v: Vec<u32> = Vec::new();
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    let ok = writeln!(
        std::io::stdout(),
        "GPU Vec: {} elements, sum = {}",
        v.len(),
        sum
    )
    .is_ok();

    unsafe {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// std-pal.2: PAL stdin routing — std::io::Read via hostcall
// ============================================================
//
// NOTE: std::io::stdin() wraps the PAL Stdin in OnceLock + ReentrantLock + BufReader.
// These layers don't work correctly on GPU (similar to println! LLVM crash issue).
// The OnceLock/ReentrantLock initialization path returns without doing actual I/O.
//
// Workaround: Call gpu_runtime::stdio::gpu_stdin_read() directly, same pattern
// as writeln! for stdout. This tests the PAL extern function mechanism works
// for both directions.

/// Test: read from stdin via direct PAL extern function call.
/// Bypasses std::io::stdin() wrapper (which uses broken OnceLock/ReentrantLock on GPU).
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[3]:
///   [0] = 1 on success, 0 on failure
///   [1] = bytes read
///   [2] = first byte of data read (for verification)
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_stdin_kernel(buf: *mut u8, result: *mut u32) {
    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
        core::ptr::write_volatile(result.add(2), 0);
    }

    let mut read_buf = [0u8; 56];
    // Call gpu_stdin_read directly via gpu-runtime — the same extern function
    // that our PAL Stdin::read() delegates to. This bypasses the broken
    // std::io::stdin() wrapper layers (OnceLock, ReentrantLock, BufReader).
    let n = unsafe { gpu_runtime::stdio::gpu_stdin_read(read_buf.as_mut_ptr(), read_buf.len()) };
    if n > 0 {
        unsafe {
            core::ptr::write_volatile(result.add(0), 1);
            core::ptr::write_volatile(result.add(1), n as u32);
            core::ptr::write_volatile(result.add(2), read_buf[0] as u32);
        }
    }
}

/// Test: read from stdin and echo to stdout (round-trip I/O).
/// Reads from stdin via extern, then writes to stdout via writeln!.
///
/// `buf` = hostcall buffer
/// `result` = output array of u32[2]:
///   [0] = 1 on success
///   [1] = bytes read from stdin
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_stdin_echo_kernel(buf: *mut u8, result: *mut u32) {
    use std::io::Write;

    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
    }

    let mut read_buf = [0u8; 56];
    let n = unsafe { gpu_runtime::stdio::gpu_stdin_read(read_buf.as_mut_ptr(), read_buf.len()) };
    if n > 0 {
        // Echo what we read back through stdout
        let data = unsafe { core::str::from_utf8_unchecked(&read_buf[..n]) };
        let ok = writeln!(std::io::stdout(), "GPU echo: {}", data).is_ok();
        unsafe {
            core::ptr::write_volatile(result.add(0), if ok { 1 } else { 0 });
            core::ptr::write_volatile(result.add(1), n as u32);
        }
    }
}

// ============================================================
// product.4: Showcase demo kernel
// ============================================================
// Combines all features: Vec, String, format!, writeln!(stdout),
// gpu_stdin_read, runtime kernel arguments — everything running
// on GPU through Rust std with hostcall backend.

/// Showcase demo: Rust std on GPU with hostcall I/O.
///
/// This kernel demonstrates the full stack:
/// 1. Read user name from stdin (via hostcall)
/// 2. Build a Vec from runtime kernel arguments
/// 3. Compute statistics using std iterators
/// 4. Format results using format!() (heap-allocated String)
/// 5. Print everything to stdout via writeln!()
///
/// `buf` = hostcall buffer (mapped memory)
/// `input` = pointer to array of u32 values (device memory)
/// `input_len` = number of elements
/// `result` = output array of u32[2]:
///   [0] = 1 on success
///   [1] = number of stdout messages written
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn showcase_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
    }

    let mut msg_count: u32 = 0;

    // Step 1: Read a name from stdin
    let mut name_buf = [0u8; 56];
    let name_len =
        unsafe { gpu_runtime::stdio::gpu_stdin_read(name_buf.as_mut_ptr(), name_buf.len()) };
    let name = if name_len > 0 {
        // Trim trailing newline if present
        let end = if name_len > 0 && name_buf[name_len - 1] == b'\n' {
            name_len - 1
        } else {
            name_len
        };
        unsafe { core::str::from_utf8_unchecked(&name_buf[..end]) }
    } else {
        "GPU User"
    };

    // Step 2: Greet the user
    if writeln!(
        std::io::stdout(),
        "Hello, {}! Welcome to Rust on GPU.",
        name
    )
    .is_ok()
    {
        msg_count += 1;
    }

    // Step 3: Build a Vec from runtime kernel arguments
    let mut v: Vec<u32> = Vec::new();
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    // Step 4: Compute statistics using std iterators
    let sum: u32 = v.iter().sum();
    let count = v.len();
    let min = v.iter().min().copied().unwrap_or(0);
    let max = v.iter().max().copied().unwrap_or(0);

    // Step 5: Format and print results
    let stats = format!(
        "Data: {} elements, sum={}, min={}, max={}",
        count, sum, min, max
    );
    if writeln!(std::io::stdout(), "{}", stats).is_ok() {
        msg_count += 1;
    }

    // Step 6: Build a filtered Vec and print it
    let evens: Vec<u32> = v.iter().filter(|&&x| x % 2 == 0).copied().collect();
    let odds: Vec<u32> = v.iter().filter(|&&x| x % 2 != 0).copied().collect();
    if writeln!(
        std::io::stdout(),
        "Even count: {}, Odd count: {}",
        evens.len(),
        odds.len()
    )
    .is_ok()
    {
        msg_count += 1;
    }

    // Step 7: Final message
    if writeln!(
        std::io::stdout(),
        "Goodbye, {}! GPU computation complete.",
        name
    )
    .is_ok()
    {
        msg_count += 1;
    }

    unsafe {
        core::ptr::write_volatile(result.add(0), 1);
        core::ptr::write_volatile(result.add(1), msg_count);
    }
}

// ============================================================
// oncelock.2: println!() direct test (bypasses OnceLock)
// ============================================================

/// Test that println!() works directly on GPU (no writeln! workaround needed).
/// This was previously broken due to OnceLock/ReentrantLock in std's Stdout.
/// The fix: _print() on CUDA bypasses OnceLock and writes through PAL directly.
///
/// result[0] = 1 on success
/// result[1] = number of println! calls completed
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn println_direct_test_kernel(
    buf: *mut u8,
    input_val: u32,
    result: *mut u32,
) {
    stdio_init(buf);

    let mut count: u32 = 0;

    // Test 1: Simple string literal
    println!("println test: hello from GPU!");
    count += 1;

    // Test 2: With runtime value formatting
    println!("println test: value = {}", input_val);
    count += 1;

    // Test 3: Multiple format args
    let x = input_val * 2;
    let y = input_val + 10;
    println!("println test: x={}, y={}, sum={}", x, y, x + y);
    count += 1;

    unsafe {
        core::ptr::write_volatile(result.add(0), 1);
        core::ptr::write_volatile(result.add(1), count);
    }
}

// ============================================================
// allocator.2: Slab allocator deallocation test
// ============================================================

/// Test that the slab allocator correctly deallocates memory.
/// Allocates and drops Vec 10 times in a loop. With bump allocator,
/// this would consume 10x the memory. With slab allocator, memory
/// is reused.
///
/// result[0] = 1 on success
/// result[1] = number of successful alloc/dealloc cycles
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn slab_dealloc_test_kernel(_buf: *mut u8, result: *mut u32) {
    let mut cycles: u32 = 0;

    // Run 10 cycles of alloc+dealloc.
    // Each Vec<u32> with 100 elements = 400 bytes + Vec overhead.
    // With bump allocator, 10 cycles = 10x allocation (4KB+ leaked).
    // With slab allocator, memory is reused each cycle.
    for i in 0u32..10 {
        let mut v: Vec<u32> = Vec::new();
        for j in 0..100 {
            v.push(i * 100 + j);
        }
        let sum: u32 = v.iter().sum();
        // Verify correctness.
        let expected = (0..100u32).map(|j| i * 100 + j).sum::<u32>();
        if sum == expected {
            cycles += 1;
        }
        // v is dropped here — dealloc should free the memory.
    }

    // Now test String alloc/dealloc cycles.
    for _ in 0..10 {
        let s = format!("Hello from cycle {}", cycles);
        if !s.is_empty() {
            cycles += 1;
        }
        // s is dropped here — dealloc should free the String's buffer.
    }

    unsafe {
        core::ptr::write_volatile(result.add(0), if cycles == 20 { 1 } else { 0 });
        core::ptr::write_volatile(result.add(1), cycles);
    }
}

// ============================================================
// allocator.3: Concurrent allocator stress test (32 threads)
// ============================================================

/// Test that the slab allocator handles 32 concurrent threads allocating
/// and deallocating simultaneously.
///
/// Each thread performs 5 cycles of: Vec::new() + push N elements + sum + drop.
/// This tests concurrent CAS on bitmap words across threads.
///
/// result[0] = number of threads that completed all cycles successfully
/// result[1] = total successful cycles across all threads
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn slab_concurrent_test_kernel(_buf: *mut u8, result: *mut u32) {
    let tid: u32;
    unsafe {
        core::arch::asm!(
            "mov.u32 {idx}, %tid.x;",
            idx = out(reg32) tid,
            options(nostack, readonly),
        );
    }

    let mut ok_cycles: u32 = 0;

    // Each thread does 5 alloc/dealloc cycles.
    for cycle in 0u32..5 {
        // Allocate a Vec with thread-specific data.
        let count = (tid % 8 + 3) as usize; // 3-10 elements per thread
        let mut v: Vec<u32> = Vec::new();
        for j in 0..count {
            v.push(tid * 100 + cycle * 10 + j as u32);
        }

        // Verify data integrity.
        let sum: u32 = v.iter().sum();
        let expected: u32 = (0..count as u32).map(|j| tid * 100 + cycle * 10 + j).sum();
        if sum == expected && v.len() == count {
            ok_cycles += 1;
        }
        // v dropped here — dealloc frees memory.
    }

    // Each thread writes its result to a unique slot.
    // result[tid] = ok_cycles for this thread
    unsafe {
        core::ptr::write_volatile(result.add(tid as usize), ok_cycles);
    }
}

// ============================================================
// std-sysroot-build.3: std::fs::File on GPU — compile test
// ============================================================

/// Test: File::create + write using std::fs on GPU.
///
/// This verifies that `use std::fs::File` compiles to valid PTX
/// through the patched std PAL (sys_fs_cuda.rs → gpu-libc hostcall).
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[2]:
///   [0] = 1 on success, error code on failure
///   [1] = bytes written
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_file_write_kernel(buf: *mut u8, result: *mut u32) {
    use std::io::Write;

    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
    }

    // Use std::fs::File::create — routes through PAL sys_fs_cuda.rs → gpu-libc open()
    match std::fs::File::create("gpu_test_output.txt") {
        Ok(mut f) => {
            let data = b"Hello from GPU std::fs::File!";
            match f.write_all(data) {
                Ok(()) => unsafe {
                    core::ptr::write_volatile(result.add(0), 1);
                    core::ptr::write_volatile(result.add(1), data.len() as u32);
                },
                Err(_) => unsafe {
                    core::ptr::write_volatile(result.add(0), 0xE002);
                },
            }
        }
        Err(_) => unsafe {
            core::ptr::write_volatile(result.add(0), 0xE001);
        },
    }
}

/// Test: File::open + read using std::fs on GPU.
///
/// `buf` = hostcall buffer
/// `result` = output array of u32[3]:
///   [0] = 1 on success, error code on failure
///   [1] = bytes read
///   [2] = first byte of data
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn std_file_read_kernel(buf: *mut u8, result: *mut u32) {
    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
        core::ptr::write_volatile(result.add(2), 0);
    }

    match std::fs::File::open("gpu_test_input.txt") {
        Ok(mut f) => {
            use std::io::Read;
            let mut buf = [0u8; 64];
            match f.read(&mut buf) {
                Ok(n) => unsafe {
                    core::ptr::write_volatile(result.add(0), 1);
                    core::ptr::write_volatile(result.add(1), n as u32);
                    if n > 0 {
                        core::ptr::write_volatile(result.add(2), buf[0] as u32);
                    }
                },
                Err(_) => unsafe {
                    core::ptr::write_volatile(result.add(0), 0xE002);
                },
            }
        }
        Err(_) => unsafe {
            core::ptr::write_volatile(result.add(0), 0xE001);
        },
    }
}
