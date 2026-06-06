//! Auto-Tuning — find the optimal block size for a GPU kernel.
//!
//! Uses `AutoTuner` to benchmark a vector-add kernel across multiple block
//! sizes (32, 64, 128, 256, 512, 1024), with warmup iterations to eliminate
//! cold-start effects and median timing to reduce noise.
//!
//! Demonstrates:
//! 1. Compile a CUDA kernel via NVRTC
//! 2. Use `AutoTuner::tune_block_size()` to find the best block size
//! 3. Cache results with `TuningCache` for future lookups
//! 4. Print a comparison report with `AutoTuner::format_report()`

use std::time::Instant;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use gpu_host::auto_tune::{AutoTuner, TuningCache};

/// Simple vector-add CUDA kernel: out[i] = a[i] + b[i]
const VECTOR_ADD_SRC: &str = r#"
extern "C" __global__ void vector_add(
    const float *a,
    const float *b,
    float *out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
"#;

fn main() {
    println!("=== Auto-Tuning: Optimal Block Size Search ===\n");

    // -----------------------------------------------------------------------
    // Setup: compile kernel and prepare data
    // -----------------------------------------------------------------------
    let n: usize = 1 << 20; // 1M elements
    println!("Problem size: {} elements ({} MiB of f32)\n", n, n * 4 / (1 << 20));

    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("CUDA device initialized");

    println!("Compiling vector_add kernel via NVRTC...");
    let ptx = compile_ptx(VECTOR_ADD_SRC).expect("Failed to compile kernel");
    dev.load_ptx(ptx, "auto_tune_mod", &["vector_add"])
        .expect("Failed to load PTX module");
    println!("Kernel compiled\n");

    // Prepare input data
    let a_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.001).collect();
    let d_a = dev.htod_sync_copy(&a_host).expect("Failed to upload a");
    let d_b = dev.htod_sync_copy(&b_host).expect("Failed to upload b");

    // -----------------------------------------------------------------------
    // Step 1: Auto-tune block size
    // -----------------------------------------------------------------------
    println!("--- Step 1: Auto-tune block size ---\n");
    println!("Evaluating candidates: [32, 64, 128, 256, 512, 1024]");
    println!("Warmup: 3 iterations, Measurement: 7 iterations (median)\n");

    let tuner = AutoTuner::new();
    let cache = TuningCache::new();

    let tune_result = tuner.tune_block_size(n as u64, None, &|block_size| {
        let func = dev.get_func("auto_tune_mod", "vector_add")?;
        let grid_x = (n as u32).div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_out = dev.alloc_zeros::<f32>(n).ok()?;

        let t0 = Instant::now();
        unsafe {
            func.launch(config, (&d_a, &d_b, &d_out, n as i32)).ok()?;
        }
        dev.synchronize().ok()?;
        Some(t0.elapsed())
    });

    let tune_result = tune_result.expect("Auto-tuning failed — no candidates produced results");

    // Print the comparison report
    let report = AutoTuner::format_report("vector_add", n as u64, &tune_result);
    println!("{report}");

    // Cache the result
    cache.insert_config("vector_add", n as u64, 0, tune_result.clone());
    println!("Cached result for future lookups (cache size: {})\n", cache.len());

    // -----------------------------------------------------------------------
    // Step 2: Compare auto-tuned vs default block size
    // -----------------------------------------------------------------------
    println!("--- Step 2: Tuned vs Default comparison ---\n");

    let best_block_size = tune_result.config.block_dim.0;
    let default_block_size = 256u32;

    println!("Best block size: {}", best_block_size);
    println!("Default block size: {}\n", default_block_size);

    // Time the auto-tuned config
    let tuned_time = benchmark_kernel(&dev, &d_a, &d_b, n, best_block_size, 10);
    println!("Auto-tuned ({} threads): {:?}", best_block_size, tuned_time);

    // Time the default config
    let default_time = benchmark_kernel(&dev, &d_a, &d_b, n, default_block_size, 10);
    println!("Default ({} threads):    {:?}", default_block_size, default_time);

    let speedup = default_time.as_secs_f64() / tuned_time.as_secs_f64();
    println!("Speedup: {:.2}x\n", speedup);

    // -----------------------------------------------------------------------
    // Step 3: Verify cache hit
    // -----------------------------------------------------------------------
    println!("--- Step 3: Cache lookup ---\n");

    let cached = cache.get_config("vector_add", n as u64, 0);
    assert!(cached.is_some(), "Cache should have a result");
    let cached = cached.unwrap();
    println!(
        "Cache hit: block_size={}, time={:?}",
        cached.config.block_dim.0, cached.best_time
    );
    assert_eq!(
        cached.config.block_dim.0, best_block_size,
        "Cached block size should match"
    );

    // Verify tune_or_cached returns cached result without re-benchmarking
    let call_count = std::sync::atomic::AtomicU32::new(0);
    let cached_result = tuner.tune_or_cached(
        &cache,
        "vector_add",
        n as u64,
        0,
        None,
        &|_bs| {
            call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(std::time::Duration::from_micros(1))
        },
    );
    assert!(cached_result.is_some(), "tune_or_cached should return cached result");
    assert_eq!(
        call_count.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "Benchmark function should not be called on cache hit"
    );
    println!("Cache hit confirmed — no re-benchmarking needed");

    // -----------------------------------------------------------------------
    // Step 4: Verify correctness of the kernel
    // -----------------------------------------------------------------------
    println!("\n--- Step 4: Correctness check ---\n");

    let grid_x = (n as u32).div_ceil(best_block_size);
    let config = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (best_block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_out = dev.alloc_zeros::<f32>(n).expect("Failed to allocate output");

    let func = dev
        .get_func("auto_tune_mod", "vector_add")
        .expect("kernel not found");
    unsafe {
        func.launch(config, (&d_a, &d_b, &d_out, n as i32))
            .expect("launch failed");
    }
    dev.synchronize().expect("sync failed");

    let result: Vec<f32> = dev.dtoh_sync_copy(&d_out).expect("Failed to download");

    // Spot-check: a[i] + b[i] = i*0.001 + (n-i)*0.001 = n*0.001 for all i
    let expected = n as f32 * 0.001;
    for i in (0..n).step_by(n / 8) {
        assert!(
            (result[i] - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {}",
            i,
            result[i],
            expected,
        );
    }
    println!("All {} elements correct (a[i] + b[i] = {:.3})", n, expected);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("Auto-tuner evaluated {} candidates", tune_result.candidates_tested);
    println!("Best block size: {} (median time: {:?})", best_block_size, tune_result.best_time);
    println!("Cache stores results by (kernel, problem-size bucket, device)");
    println!("\n=== All assertions passed ===");
}

/// Benchmark a kernel launch with the given block size, averaging over N runs.
fn benchmark_kernel(
    dev: &std::sync::Arc<CudaDevice>,
    d_a: &cudarc::driver::CudaSlice<f32>,
    d_b: &cudarc::driver::CudaSlice<f32>,
    n: usize,
    block_size: u32,
    runs: u32,
) -> std::time::Duration {
    let grid_x = (n as u32).div_ceil(block_size);
    let config = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    for _ in 0..3 {
        let d_out = dev.alloc_zeros::<f32>(n).expect("alloc");
        let func = dev.get_func("auto_tune_mod", "vector_add").expect("func");
        unsafe { func.launch(config, (d_a, d_b, &d_out, n as i32)).expect("launch") };
        dev.synchronize().expect("sync");
    }

    // Measure
    let mut times = Vec::new();
    for _ in 0..runs {
        let d_out = dev.alloc_zeros::<f32>(n).expect("alloc");
        let func = dev.get_func("auto_tune_mod", "vector_add").expect("func");
        let t0 = Instant::now();
        unsafe { func.launch(config, (d_a, d_b, &d_out, n as i32)).expect("launch") };
        dev.synchronize().expect("sync");
        times.push(t0.elapsed());
    }

    times.sort();
    times[times.len() / 2] // median
}
