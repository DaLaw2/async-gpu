//! GpuVec zero-copy demo: read -> compute -> write with explicit GPU control.
//!
//! This example shows the GpuVec pattern for users who want GPU execution
//! guaranteed (not auto-routed) but still want zero manual memory management.
//!
//! Compared to `unified_pipeline.rs`:
//! - `AutoScheduler::par_map` hides *everything* — the user never knows GPU exists.
//! - `GpuVec::map_gpu` is explicit "run this on GPU" but still hides transfers.
//!
//! What the user does NOT write:
//! - `cudaMalloc` / `cudaFree`
//! - `cudaMemcpy` (host-to-device or device-to-host)
//! - Kernel launch configuration (grid size, block size calculation)
//! - Device synchronization
//! - Error handling for each CUDA call
//!
//! What the user DOES write:
//! - `GpuVec::from_vec(data)` — wrap existing data for GPU access
//! - `data.map_gpu(ptx, kernel, threads)` — one-liner transform on GPU
//! - `result.as_slice()` — read results back (zero-copy, no download needed)
//!
//! # Running
//!
//! ```bash
//! cargo run -p gpu-host --example gpuvec_pipeline
//! ```

use gpu_host::memory::GpuVec;
use gpu_host::ptx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Step 1: Create data ──────────────────────────────────────
    let n = 8192;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    println!("Created {} input elements", n);

    // ── Step 2: Wrap in GpuVec (pinned memory, GPU-visible) ─────
    let data = GpuVec::from_vec(input_data)?;

    // ── Step 3: One-liner GPU transform ──────────────────────────
    // Kernel: f(x) = x * 2.0 + 1.0  (pre-compiled in PTX)
    let result = data.map_gpu(ptx::KERNEL_TEST, "par_iter_map_collect_multiblock", 256)?;

    // ── Step 4: Read results — zero-copy, no download needed ────
    let output = result.as_slice();
    println!("GPU produced {} output elements", output.len());

    // ── Step 5: Write to file ────────────────────────────────────
    let bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("gpuvec_output.bin", &bytes)?;
    println!("Wrote {} elements to gpuvec_output.bin", output.len());

    // ── Verify correctness ───────────────────────────────────────
    let mut errors = 0;
    for (i, &val) in output.iter().enumerate() {
        let input_val = i as f32 * 0.1;
        let expected = input_val * 2.0 + 1.0;
        if (val - expected).abs() > 1e-4 {
            if errors < 5 {
                eprintln!(
                    "  MISMATCH at [{}]: got {:.6}, expected {:.6}",
                    i, val, expected
                );
            }
            errors += 1;
        }
    }
    if errors == 0 {
        println!("All {} elements verified correct", n);
    } else {
        eprintln!("{} mismatches out of {} elements", errors, n);
    }

    Ok(())
}
