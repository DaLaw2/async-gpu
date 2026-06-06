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
