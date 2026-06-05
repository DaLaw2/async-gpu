//! North Star Demo: read -> compute -> write
//!
//! Reads numerical data from a file, transforms it on GPU (or CPU for small
//! data), and writes the result back to a file. The user never mentions
//! "kernel", "device", "host", "memcpy", "block", "thread", "warp", or "PTX".
//!
//! # What happens under the hood
//!
//! - `AutoScheduler` inspects `data.len()` and routes to CPU (below 4096
//!   elements) or GPU (above) automatically.
//! - The GPU path loads a pre-compiled kernel, uploads data to device memory,
//!   launches with the right grid/block dimensions, downloads results, and
//!   frees device memory — all invisible to the caller.
//! - The CPU path runs the same operation as a plain iterator map.
//!
//! # Running
//!
//! ```bash
//! # Generate sample input (10,000 floats)
//! python3 -c "
//! import struct, random
//! data = [random.uniform(-100, 100) for _ in range(10000)]
//! open('input.bin', 'wb').write(b''.join(struct.pack('<f', x) for x in data))
//! "
//!
//! # Run the demo
//! cargo run -p gpu-host --example unified_pipeline
//!
//! # Verify output
//! python3 -c "
//! import struct
//! data = open('output.bin', 'rb').read()
//! vals = [struct.unpack('<f', data[i:i+4])[0] for i in range(0, len(data), 4)]
//! print(f'{len(vals)} elements, first 5: {vals[:5]}')
//! "
//! ```

use gpu_host::scheduler::AutoScheduler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Step 1: Read data from file ──────────────────────────────
    let raw = std::fs::read("input.bin")?;
    let input: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("Read {} elements from input.bin", input.len());

    // ── Step 2: Transform data (AutoScheduler picks CPU or GPU) ──
    let scheduler = AutoScheduler::new();
    let output = scheduler.par_map(&input, |x| x * 2.0 + 1.0)?;
    println!(
        "Transformed {} elements (threshold: {} — {})",
        output.len(),
        scheduler.threshold(),
        if input.len() < scheduler.threshold() {
            "ran on CPU"
        } else {
            "ran on GPU"
        }
    );

    // ── Step 3: Write result to file ─────────────────────────────
    let bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("output.bin", &bytes)?;
    println!("Wrote {} elements to output.bin", output.len());

    // ── Verify: spot-check a few values ──────────────────────────
    if !input.is_empty() {
        let i = 0;
        let expected = input[i] * 2.0 + 1.0;
        println!(
            "Spot check: input[0]={:.4} -> output[0]={:.4} (expected {:.4})",
            input[i], output[i], expected
        );
    }

    Ok(())
}
