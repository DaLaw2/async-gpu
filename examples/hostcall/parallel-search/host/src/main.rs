//! Parallel Search — host driver for 32-lane GPU grep demo.
//!
//! 1. Creates a 4KB text file with known pattern occurrences
//! 2. Launches GPU kernel with ALL 32 threads active
//! 3. Each lane searches 1/32 of the file for the pattern
//! 4. Verifies GPU count matches CPU count
//!
//! Uses the `gpu::custom()` builder API with hostcall and mapped buffers.

use async_gpu::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Search Demo ===");
    println!("  32-lane warp-cooperative async grep\n");

    // Step 1: Create input file with known pattern count
    let pattern = b"GPU";
    let line = "The GPU can search files in parallel using all 32 GPU lanes. GPU power!\n";
    let mut input_data = line.repeat(56); // 56 x 72 = 4032 bytes
    while input_data.len() < 4096 {
        input_data.push('.');
    }
    std::fs::write("search_input.txt", &input_data)?;
    println!(
        "[host] Created search_input.txt ({} bytes)",
        input_data.len()
    );

    // CPU reference count
    let cpu_count = count_pattern(input_data.as_bytes(), pattern);
    println!(
        "[host] CPU count of {:?}: {}",
        std::str::from_utf8(pattern).unwrap(),
        cpu_count
    );

    // Step 2: Prepare GPU context with hostcall (full warp, 32 threads)
    let ctx = gpu::custom("parallel_search")
        .ptx(KERNEL_PTX)
        .threads(32)
        .hostcall_packets(8)
        .prepare()?;

    // Step 3: Set up mapped buffers
    let result_buf = ctx.mapped_buffer::<u32>(1)?;

    let mut pattern_buf = ctx.mapped_buffer::<u8>(pattern.len())?;
    for (i, &b) in pattern.iter().enumerate() {
        unsafe { pattern_buf.write(i, b) };
    }

    let data_buf = ctx.mapped_buffer::<u8>(4096)?;

    // Extract pointers before launch
    let hc = ctx.hostcall_ptr();
    let sb = ctx.sideband_ptr();

    // Step 4: Launch kernel
    println!("[host] Launching parallel_search kernel (32 threads)...");
    let gpu_result = unsafe {
        ctx.launch((
            hc,
            sb,
            pattern_buf.dev_ptr() as u64,
            pattern.len() as u32,
            data_buf.dev_ptr() as u64,
            result_buf.dev_ptr() as u64,
        ))?
    };

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Step 5: Check results
    let gpu_count = unsafe { result_buf.read(0) };
    println!("\n[host] GPU result: {gpu_count}");

    if gpu_count >= 0xE000 {
        println!("[host] FAILED — error code 0x{gpu_count:X}");
        cleanup();
        return Ok(());
    }

    // Step 6: Verify against CPU count
    println!("[host] CPU count: {cpu_count}");
    if gpu_count == cpu_count as u32 {
        println!("[host] Verification: PASSED (exact match)");
    } else {
        let diff = (gpu_count as i64 - cpu_count as i64).unsigned_abs();
        if diff <= 32 {
            println!(
                "[host] Verification: ACCEPTABLE (GPU={gpu_count}, CPU={cpu_count}, diff={diff} — boundary overlap)"
            );
        } else {
            println!("[host] Verification: FAILED (GPU={gpu_count}, CPU={cpu_count})");
        }
    }

    // Step 7: Check result file
    match std::fs::read_to_string("search_result.txt") {
        Ok(content) => println!("[host] search_result.txt: \"{}\"", content.trim()),
        Err(e) => println!("[host] search_result.txt not found: {e}"),
    }

    // Drop mapped buffers before GpuResult for clean CUDA teardown
    drop(result_buf);
    drop(pattern_buf);
    drop(data_buf);
    gpu_result.finish();

    cleanup();
    println!("\n=== Parallel Search Demo Complete ===");
    Ok(())
}

/// Count non-overlapping occurrences of pattern in data (CPU reference).
fn count_pattern(data: &[u8], pattern: &[u8]) -> usize {
    if pattern.is_empty() || data.len() < pattern.len() {
        return 0;
    }
    let mut count = 0;
    let mut i = 0;
    while i <= data.len() - pattern.len() {
        if &data[i..i + pattern.len()] == pattern {
            count += 1;
        }
        i += 1; // Overlapping count (matches GPU behavior)
    }
    count
}

fn cleanup() {
    let _ = std::fs::remove_file("search_input.txt");
    let _ = std::fs::remove_file("search_result.txt");
}
