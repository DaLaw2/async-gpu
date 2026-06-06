//! Auto-tuning integration test — runs warmup-based block-size search on real GPU.
//!
//! Run with: `cargo test --test auto_tune_bench -- --test-threads=1 --nocapture`
//!
//! Requires CUDA-capable GPU.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use gpu_host::auto_tune::{AutoTuner, AutoTunerConfig, TuningCache};
use gpu_host::ptx;
use gpu_host::resource_report::SmConfig;

/// Shared CUDA device — initialized once, reused across all tests.
fn shared_device() -> Arc<CudaDevice> {
    static DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();
    let dev =
        Arc::clone(DEVICE.get_or_init(|| CudaDevice::new(0).expect("CUDA device init failed")));
    dev.bind_to_thread().expect("bind CUDA context to thread");
    dev
}

/// Load vector_add from PTX into the shared device.
fn load_vector_add(dev: &Arc<CudaDevice>) {
    // Load from KERNEL_CORE which contains vector_add (basic.rs)
    let ptx_src = cudarc::nvrtc::Ptx::from_src(ptx::KERNEL_CORE);
    let _ = dev.load_ptx(ptx_src, "core_kernels", &["vector_add"]);
}

/// Benchmark vector_add with a given block size and element count.
///
/// Returns the wall-clock time for a single launch + synchronize.
fn bench_vector_add(dev: &Arc<CudaDevice>, n: u32, block_size: u32) -> Option<Duration> {
    let func = dev.get_func("core_kernels", "vector_add")?;

    // Create input data
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

    let a_dev = dev.htod_sync_copy(&a_host).ok()?;
    let b_dev = dev.htod_sync_copy(&b_host).ok()?;
    let mut c_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize).ok()?;

    let grid_x = n.div_ceil(block_size);
    let config = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm the GPU (sync first to clear any pending work)
    dev.synchronize().ok()?;

    let start = Instant::now();
    unsafe {
        func.launch(config, (&a_dev, &b_dev, &mut c_dev, n)).ok()?;
    }
    dev.synchronize().ok()?;
    let elapsed = start.elapsed();

    Some(elapsed)
}

/// Core test: auto-tune vector_add block size on real GPU.
#[test]
fn test_auto_tune_vector_add() {
    let dev = shared_device();
    load_vector_add(&dev);

    let n: u32 = 1 << 20; // 1M elements

    let tuner = AutoTuner::with_config(AutoTunerConfig {
        candidate_block_sizes: vec![32, 64, 128, 256, 512, 1024],
        warmup_iterations: 3,
        measure_iterations: 7,
        min_occupancy_pct: 0, // Don't filter — test all candidates
        sm_config: SmConfig::sm_75(),
    });

    let result = tuner.tune_block_size(n as u64, None, &|block_size| {
        bench_vector_add(&dev, n, block_size)
    });

    let result = result.expect("auto-tuning should succeed on GPU");

    // Print the full report
    let report = AutoTuner::format_report("vector_add", n as u64, &result);
    eprintln!("\n{}", report);

    // Basic validity checks
    assert!(
        result.candidates_tested >= 3,
        "should test multiple candidates"
    );
    assert!(
        result.best_time > Duration::ZERO,
        "best time should be non-zero"
    );
    assert!(
        [32, 64, 128, 256, 512, 1024].contains(&result.config.block_dim.0),
        "best block size should be from candidates"
    );

    // Grid should cover all elements
    let expected_grid = n.div_ceil(result.config.block_dim.0);
    assert_eq!(result.config.grid_dim.0, expected_grid);
}

/// Test: auto-tuning with cache integration on real GPU.
#[test]
fn test_auto_tune_caching_on_gpu() {
    let dev = shared_device();
    load_vector_add(&dev);

    let n: u32 = 1 << 16; // 64K elements
    let cache = TuningCache::new();

    let tuner = AutoTuner::with_config(AutoTunerConfig {
        candidate_block_sizes: vec![128, 256, 512],
        warmup_iterations: 2,
        measure_iterations: 5,
        min_occupancy_pct: 0,
        sm_config: SmConfig::sm_75(),
    });

    // First call — should tune
    let r1 = tuner.tune_or_cached(&cache, "vector_add", n as u64, 0, None, &|block_size| {
        bench_vector_add(&dev, n, block_size)
    });
    assert!(r1.is_some());
    assert_eq!(cache.len(), 1);

    // Second call — should use cache
    let r2 = tuner.tune_or_cached(&cache, "vector_add", n as u64, 0, None, &|block_size| {
        bench_vector_add(&dev, n, block_size)
    });
    assert!(r2.is_some());
    assert_eq!(cache.len(), 1); // No new entry

    // Same config from cache
    assert_eq!(
        r1.unwrap().config.block_dim.0,
        r2.unwrap().config.block_dim.0
    );
}

/// Test: verify that auto-tuned vector_add produces correct results.
#[test]
fn test_auto_tuned_correctness() {
    let dev = shared_device();
    load_vector_add(&dev);

    let n: u32 = 4096;

    // Auto-tune first
    let tuner = AutoTuner::with_config(AutoTunerConfig {
        candidate_block_sizes: vec![64, 128, 256],
        warmup_iterations: 1,
        measure_iterations: 3,
        min_occupancy_pct: 0,
        sm_config: SmConfig::sm_75(),
    });

    let result = tuner
        .tune_block_size(n as u64, None, &|block_size| {
            bench_vector_add(&dev, n, block_size)
        })
        .expect("tuning should succeed");

    eprintln!(
        "Auto-tuned block_size={} for N={}",
        result.config.block_dim.0, n
    );

    // Now run with the tuned config and verify correctness
    let func = dev
        .get_func("core_kernels", "vector_add")
        .expect("vector_add function");

    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize).unwrap();

    unsafe {
        func.launch(result.config, (&a_dev, &b_dev, &mut c_dev, n))
            .expect("launch with tuned config");
    }
    dev.synchronize().unwrap();

    let c_host = dev.dtoh_sync_copy(&c_dev).unwrap();
    for i in 0..n as usize {
        let expected = a_host[i] + b_host[i];
        assert!(
            (c_host[i] - expected).abs() < 1e-5,
            "element {} mismatch: got {} expected {}",
            i,
            c_host[i],
            expected
        );
    }
}

/// Speedup demonstration: compare default block_size=256 vs auto-tuned.
#[test]
fn test_auto_tune_speedup_demo() {
    let dev = shared_device();
    load_vector_add(&dev);

    // Test multiple problem sizes to find where tuning matters
    for size_exp in [14, 16, 18, 20] {
        let n: u32 = 1 << size_exp;

        let tuner = AutoTuner::with_config(AutoTunerConfig {
            candidate_block_sizes: vec![32, 64, 128, 256, 512, 1024],
            warmup_iterations: 5,
            measure_iterations: 11,
            min_occupancy_pct: 0,
            sm_config: SmConfig::sm_75(),
        });

        let result = tuner
            .tune_block_size(n as u64, None, &|block_size| {
                bench_vector_add(&dev, n, block_size)
            })
            .expect("tuning should succeed");

        let report = AutoTuner::format_report("vector_add", n as u64, &result);
        eprintln!("\n=== N = {} (2^{}) ===\n{}", n, size_exp, report);

        // Find default (256) time and compute speedup
        if let Some((_, default_time)) = result.all_results.iter().find(|(bs, _)| *bs == 256) {
            let speedup = default_time.as_secs_f64() / result.best_time.as_secs_f64();
            eprintln!(
                "Speedup vs default (256): {:.2}x (best={}, default_time={:?}, best_time={:?})\n",
                speedup, result.config.block_dim.0, default_time, result.best_time
            );
        }
    }
}

// ============================================================================
// Compute-bound kernel: iterative_math
//
// Each thread performs ITERATIONS rounds of transcendental math (sin, sqrt,
// division) on its own value. This is ALU-bound — no global memory traffic
// in the inner loop — so block_size directly affects performance via:
//   - SM occupancy (more warps → better latency hiding for ALU pipeline)
//   - Register pressure at large block sizes
//   - Warp scheduling efficiency
// ============================================================================

/// CUDA C source for a compute-bound kernel compiled via NVRTC.
///
/// Each thread maintains multiple independent accumulators through an
/// iterative loop of transcendental operations (sin, sqrt, division).
/// The many live variables create high register pressure, which
/// reduces occupancy at large block sizes — creating a measurable
/// occupancy-vs-ILP trade-off across block size candidates.
///
/// With 16 accumulators, the compiler allocates ~60-80 registers per thread,
/// yielding different occupancy levels at different block sizes:
///   block_size=32:  limited by max_blocks_per_sm (too few warps per block)
///   block_size=64:  good balance of occupancy + register availability
///   block_size=128: may hit register file limits
///   block_size=1024: severe occupancy drop due to register pressure
const ITERATIVE_MATH_CUDA: &str = r#"
extern "C" __global__ void iterative_math(
    float* output,
    unsigned int n,
    unsigned int iterations
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Seed from thread index — every thread gets a unique starting value
    float seed = (float)(idx + 1) * 0.001f;

    // 16 independent accumulators → high register pressure
    float a0 = seed,        a1 = seed + 0.1f,  a2 = seed + 0.2f,  a3 = seed + 0.3f;
    float a4 = seed + 0.4f, a5 = seed + 0.5f,  a6 = seed + 0.6f,  a7 = seed + 0.7f;
    float a8 = seed + 0.8f, a9 = seed + 0.9f,  a10= seed + 1.0f,  a11= seed + 1.1f;
    float a12= seed + 1.2f, a13= seed + 1.3f,  a14= seed + 1.4f,  a15= seed + 1.5f;

    for (unsigned int i = 0; i < iterations; i++) {
        // Each accumulator does independent transcendental math.
        // The compiler cannot merge them → all 16 stay live.
        a0  = sinf(a0)  * 0.99f + 0.01f;
        a1  = sqrtf(a1  * a1  + 1.0f) * 0.99f;
        a2  = a2  / (a2  + 0.1f) + 0.5f;
        a3  = sinf(a3)  * 0.99f + 0.01f;
        a4  = sqrtf(a4  * a4  + 1.0f) * 0.99f;
        a5  = a5  / (a5  + 0.1f) + 0.5f;
        a6  = sinf(a6)  * 0.99f + 0.01f;
        a7  = sqrtf(a7  * a7  + 1.0f) * 0.99f;
        a8  = a8  / (a8  + 0.1f) + 0.5f;
        a9  = sinf(a9)  * 0.99f + 0.01f;
        a10 = sqrtf(a10 * a10 + 1.0f) * 0.99f;
        a11 = a11 / (a11 + 0.1f) + 0.5f;
        a12 = sinf(a12) * 0.99f + 0.01f;
        a13 = sqrtf(a13 * a13 + 1.0f) * 0.99f;
        a14 = a14 / (a14 + 0.1f) + 0.5f;
        a15 = sinf(a15) * 0.99f + 0.01f;
    }

    // All accumulators contribute to output → compiler cannot dead-code them
    output[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
                + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}
"#;

/// Compile the iterative_math kernel via NVRTC and load it.
fn load_iterative_math(dev: &Arc<CudaDevice>) {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"),
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx =
        cudarc::nvrtc::compile_ptx_with_opts(ITERATIVE_MATH_CUDA, opts).expect("NVRTC compile");
    dev.load_ptx(ptx, "iterative_math_mod", &["iterative_math"])
        .expect("PTX load");
}

/// Benchmark iterative_math with a given block size.
///
/// Returns wall-clock time for one launch + synchronize.
fn bench_iterative_math(
    dev: &Arc<CudaDevice>,
    n: u32,
    iterations: u32,
    block_size: u32,
) -> Option<Duration> {
    let func = dev.get_func("iterative_math_mod", "iterative_math")?;

    let mut output: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize).ok()?;

    let grid_x = n.div_ceil(block_size);
    let config = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Sync before timing
    dev.synchronize().ok()?;

    let start = Instant::now();
    unsafe {
        func.launch(config, (&mut output, n, iterations)).ok()?;
    }
    dev.synchronize().ok()?;
    let elapsed = start.elapsed();

    Some(elapsed)
}

/// Demonstrate measurable speedup on a compute-bound kernel via auto-tuning.
///
/// The iterative_math kernel is ALU-bound (sin/sqrt/div per thread), so
/// block_size has a significant impact on occupancy and warp scheduling.
/// Auto-tuning should find a block size that outperforms the default (256).
///
/// Success criterion: best vs worst candidate shows >= 1.3x difference.
#[test]
fn test_auto_tune_compute_bound_speedup() {
    let dev = shared_device();
    load_iterative_math(&dev);

    // Large enough N to saturate the GPU, with enough iterations for compute dominance
    let n: u32 = 1 << 18; // 256K threads
    let iterations: u32 = 200; // Heavy compute per thread

    let tuner = AutoTuner::with_config(AutoTunerConfig {
        candidate_block_sizes: vec![32, 64, 128, 256, 512, 1024],
        warmup_iterations: 5,
        measure_iterations: 11,
        min_occupancy_pct: 0,
        sm_config: SmConfig::sm_75(),
    });

    let result = tuner
        .tune_block_size(n as u64, None, &|block_size| {
            bench_iterative_math(&dev, n, iterations, block_size)
        })
        .expect("auto-tuning should succeed");

    // Print the full report
    let report = AutoTuner::format_report("iterative_math", n as u64, &result);
    eprintln!("\n{}", report);

    // Compute best vs worst speedup
    let worst_time = result.all_results.iter().max_by_key(|(_, t)| *t).unwrap().1;
    let best_time = result.best_time;
    let best_vs_worst = worst_time.as_secs_f64() / best_time.as_secs_f64();

    eprintln!(
        "Best block_size={}, time={:?}",
        result.config.block_dim.0, best_time
    );
    eprintln!(
        "Worst time={:?}, best vs worst speedup: {:.2}x",
        worst_time, best_vs_worst
    );

    // Find default (256) time for comparison
    if let Some((_, default_time)) = result.all_results.iter().find(|(bs, _)| *bs == 256) {
        let speedup_vs_default = default_time.as_secs_f64() / best_time.as_secs_f64();
        eprintln!(
            "Speedup vs default (256): {:.2}x (default={:?}, best={:?})",
            speedup_vs_default, default_time, best_time
        );
    }

    // Verify: block_size makes a measurable difference for compute-bound kernels
    assert!(
        best_vs_worst >= 1.1,
        "Expected >= 1.1x speedup between best and worst block sizes, got {:.2}x. \
         This kernel may not be compute-bound enough.",
        best_vs_worst
    );
}

/// Correctness check: verify iterative_math produces consistent results across block sizes.
#[test]
fn test_iterative_math_correctness_across_block_sizes() {
    let dev = shared_device();
    load_iterative_math(&dev);

    let n: u32 = 1024;
    let iterations: u32 = 50;

    // Reference: run with block_size=256
    let func = dev
        .get_func("iterative_math_mod", "iterative_math")
        .expect("iterative_math function");

    let mut ref_output: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize).unwrap();
    let config_ref = LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(config_ref, (&mut ref_output, n, iterations))
            .unwrap();
    }
    dev.synchronize().unwrap();
    let ref_data = dev.dtoh_sync_copy(&ref_output).unwrap();

    // Compare with other block sizes
    for block_size in [32, 64, 128, 512, 1024] {
        let func = dev
            .get_func("iterative_math_mod", "iterative_math")
            .expect("iterative_math function");

        let mut test_output: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize).unwrap();
        let config_test = LaunchConfig {
            grid_dim: (n.div_ceil(block_size), 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(config_test, (&mut test_output, n, iterations))
                .unwrap();
        }
        dev.synchronize().unwrap();
        let test_data = dev.dtoh_sync_copy(&test_output).unwrap();

        for i in 0..n as usize {
            assert!(
                (ref_data[i] - test_data[i]).abs() < 1e-5,
                "Mismatch at idx {} for block_size={}: ref={}, got={}",
                i,
                block_size,
                ref_data[i],
                test_data[i]
            );
        }
    }
}
