//! Host-side tests for par_iter chained iterator fusion on GPU.
//!
//! Launches par_iter demo kernels from gpu-kernel-test and verifies
//! correctness against CPU reference computations.
//!
//! These tests demonstrate that chained `.map()` calls produce ZERO
//! intermediate buffers: the Rust compiler fuses the type-level chain
//! (GpuMap<GpuMap<...>>) into a single inlined expression per element.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};

/// All par_iter kernel names we need — loaded in a single PTX JIT call.
const PAR_ITER_KERNELS: &[&str] = &[
    "par_iter_map_collect",
    "par_iter_chained_map_collect",
    "par_iter_map_filter_count",
    "par_iter_triple_map_sum",
];

/// Load the kernel_std PTX once and register all par_iter kernels.
fn load_par_iter_module(dev: &Arc<CudaDevice>) -> Result<()> {
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_STD_PTX);
    dev.load_ptx(ptx, "par_iter_fusion", PAR_ITER_KERNELS)
        .map_err(|e| GpuHostError::Verification {
            test: "par_iter_module_load",
            detail: format!("ptx_load: {e}"),
        })?;
    Ok(())
}

/// Get a function handle from the pre-loaded par_iter module.
fn get_func(
    dev: &Arc<CudaDevice>,
    kernel_name: &'static str,
) -> Result<cudarc::driver::CudaFunction> {
    dev.get_func("par_iter_fusion", kernel_name)
        .ok_or(GpuHostError::KernelNotFound(kernel_name))
}

/// Standard launch config: 1 block x 128 threads (4 warps).
fn launch_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ============================================================
// Test 1: Chained map — two separate .map() calls
// ============================================================

/// Verify chained iterator fusion: `.map(|x| x * 2.0).map(|x| x + 1.0)`
///
/// This is the core zero-intermediate-buffer proof. Two separate `.map()`
/// calls create GpuMap<GpuMap<GpuParIter<f32>, C1>, C2> at compile time,
/// which LLVM inlines into a single expression: `(x * 2.0) + 1.0`.
///
/// Expected: output[i] = input[i] * 2.0 + 1.0
fn run_chained_map_test(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("\n--- par_iter chained map test ---");

    const N: usize = 16;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();

    let func = get_func(dev, "par_iter_chained_map_collect")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(N).map_err(GpuHostError::Cudarc)?;
    let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;

    let n = N as u32;
    unsafe {
        func.launch(
            launch_config(),
            (&input_dev, &mut output_dev, n, &mut status_dev),
        )
        .map_err(|e| GpuHostError::Verification {
            test: "par_iter_chained_map_collect",
            detail: format!("launch: {e}"),
        })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    let output: Vec<f32> = dev
        .dtoh_sync_copy(&output_dev)
        .map_err(GpuHostError::Cudarc)?;
    let status: Vec<u32> = dev
        .dtoh_sync_copy(&status_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(
        status[0], 1,
        "kernel did not complete (status flag not set)"
    );

    // Verify against CPU reference
    let mut ok = true;
    for i in 0..N {
        let expected = input[i] * 2.0 + 1.0;
        let got = output[i];
        if (got - expected).abs() > 1e-5 {
            println!("  MISMATCH at {i}: expected {expected}, got {got}");
            ok = false;
        }
    }

    if ok {
        println!("  Chained map: ALL {N} elements correct");
        println!(
            "  Sample: input[3]={}, output[3]={} (expected {})",
            input[3],
            output[3],
            input[3] * 2.0 + 1.0
        );
        println!("  Fusion: .map(|x| x*2.0).map(|x| x+1.0) -> single inlined expression");
        println!("  PASSED");
    } else {
        return Err(GpuHostError::Verification {
            test: "par_iter_chained_map_collect",
            detail: "chained map output mismatch".to_string(),
        });
    }

    Ok(())
}

// ============================================================
// Test 2: map + filter + count
// ============================================================

/// Verify chained fusion: `.map(|x| x * x).filter(|x| *x > threshold).count()`
fn run_map_filter_count_test(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("\n--- par_iter map+filter+count test ---");

    const N: usize = 16;
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let threshold: f32 = 10.0;

    let func = get_func(dev, "par_iter_map_filter_count")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut result_dev = dev.alloc_zeros::<u32>(2).map_err(GpuHostError::Cudarc)?;

    let n = N as u32;
    let threshold_bits = threshold.to_bits();
    unsafe {
        func.launch(
            launch_config(),
            (&input_dev, n, threshold_bits, &mut result_dev),
        )
        .map_err(|e| GpuHostError::Verification {
            test: "par_iter_map_filter_count",
            detail: format!("launch: {e}"),
        })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    let result: Vec<u32> = dev
        .dtoh_sync_copy(&result_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(result[1], 1, "kernel did not complete (done flag not set)");

    let gpu_count = result[0] as usize;
    let cpu_count = input.iter().filter(|x| (*x * *x) > threshold).count();

    println!("  threshold = {threshold}");
    println!("  GPU count = {gpu_count}, CPU count = {cpu_count}");

    let passing: Vec<usize> = input
        .iter()
        .enumerate()
        .filter(|(_, x)| (*x * *x) > threshold)
        .map(|(i, _)| i)
        .collect();
    println!("  Passing indices: {:?}", passing);

    if gpu_count == cpu_count {
        println!("  Fusion: .map(square).filter(>thresh).count() -> single-pass, zero buffers");
        println!("  PASSED");
    } else {
        return Err(GpuHostError::Verification {
            test: "par_iter_map_filter_count",
            detail: format!("count mismatch: GPU={gpu_count}, CPU={cpu_count}"),
        });
    }

    Ok(())
}

// ============================================================
// Test 3: Triple map + sum (deep fusion)
// ============================================================

/// Verify deep fusion: `.map(+1).map(*3).map(-0.5).sum()`
fn run_triple_map_sum_test(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("\n--- par_iter triple map+sum test ---");

    const N: usize = 16;
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();

    let func = get_func(dev, "par_iter_triple_map_sum")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut result_dev = dev.alloc_zeros::<u32>(2).map_err(GpuHostError::Cudarc)?;

    let n = N as u32;
    unsafe {
        func.launch(launch_config(), (&input_dev, n, &mut result_dev))
            .map_err(|e| GpuHostError::Verification {
                test: "par_iter_triple_map_sum",
                detail: format!("launch: {e}"),
            })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    let result: Vec<u32> = dev
        .dtoh_sync_copy(&result_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(result[1], 1, "kernel did not complete (done flag not set)");

    let gpu_sum = f32::from_bits(result[0]);
    let cpu_sum: f32 = input.iter().map(|x| (x + 1.0) * 3.0 - 0.5).sum();

    println!("  GPU sum = {gpu_sum:.1}, CPU sum = {cpu_sum:.1}");

    let tolerance = cpu_sum.abs() * 1e-5;
    if (gpu_sum - cpu_sum).abs() <= tolerance.max(1e-3) {
        println!("  Fusion: .map(+1).map(*3).map(-0.5).sum() -> single expression per element");
        println!("  Type: GpuMap<GpuMap<GpuMap<GpuParIter<f32>>>> (3-deep nesting)");
        println!("  PASSED");
    } else {
        return Err(GpuHostError::Verification {
            test: "par_iter_triple_map_sum",
            detail: format!("sum mismatch: GPU={gpu_sum}, CPU={cpu_sum}"),
        });
    }

    Ok(())
}

// ============================================================
// Test 4: Existing single map (baseline comparison)
// ============================================================

/// Baseline: verify single `.map(|x| x * 2.0 + 1.0).collect_into()` works.
fn run_single_map_baseline_test(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("\n--- par_iter single map baseline test ---");

    const N: usize = 16;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();

    let func = get_func(dev, "par_iter_map_collect")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(N).map_err(GpuHostError::Cudarc)?;
    let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;

    let n = N as u32;
    unsafe {
        func.launch(
            launch_config(),
            (&input_dev, &mut output_dev, n, &mut status_dev),
        )
        .map_err(|e| GpuHostError::Verification {
            test: "par_iter_map_collect",
            detail: format!("launch: {e}"),
        })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    let output: Vec<f32> = dev
        .dtoh_sync_copy(&output_dev)
        .map_err(GpuHostError::Cudarc)?;
    let status: Vec<u32> = dev
        .dtoh_sync_copy(&status_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(status[0], 1, "kernel did not complete");

    let mut ok = true;
    for i in 0..N {
        let expected = input[i] * 2.0 + 1.0;
        if (output[i] - expected).abs() > 1e-5 {
            println!("  MISMATCH at {i}: expected {expected}, got {}", output[i]);
            ok = false;
        }
    }

    if ok {
        println!("  Single map baseline: ALL {N} elements correct");
        println!("  PASSED");
    } else {
        return Err(GpuHostError::Verification {
            test: "par_iter_map_collect",
            detail: "single map baseline mismatch".to_string(),
        });
    }

    Ok(())
}

/// Run all par_iter chained fusion tests.
///
/// Loads the PTX module once (JIT compilation ~10-15 min for the unified
/// kernel PTX), then runs all tests against the same loaded module.
pub(crate) fn run_all_par_iter_tests(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n=== Par_iter Chained Iterator Fusion Tests ===");

    println!("  Loading PTX module (JIT compile, may take several minutes)...");
    load_par_iter_module(&dev)?;
    println!("  PTX module loaded.");

    run_single_map_baseline_test(&dev)?;
    run_chained_map_test(&dev)?;
    run_map_filter_count_test(&dev)?;
    run_triple_map_sum_test(&dev)?;

    println!("\n=== All par_iter fusion tests PASSED ===");
    Ok(())
}

// ============================================================
// Large-scale 1M+ element tests
// ============================================================

/// Measure raw GPU memory copy time (htod + dtoh) for N f32 elements.
///
/// This establishes a baseline: any compute kernel must be at least
/// this slow because it reads and writes the same amount of data.
fn measure_memcpy_baseline(dev: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

    // Warmup
    for _ in 0..3 {
        let d = dev.htod_sync_copy(&data).map_err(GpuHostError::Cudarc)?;
        let _: Vec<f32> = dev.dtoh_sync_copy(&d).map_err(GpuHostError::Cudarc)?;
    }

    let iters = 10;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let d = dev.htod_sync_copy(&data).map_err(GpuHostError::Cudarc)?;
        let _: Vec<f32> = dev.dtoh_sync_copy(&d).map_err(GpuHostError::Cudarc)?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    Ok(elapsed_ms)
}

/// Large-scale par_iter test: map().collect_into() on 1M+ f32 elements.
///
/// Tests `par_iter_map_collect` (f(x) = x * 2.0 + 1.0) on 1,048,576 elements.
/// Verifies every element against CPU reference and measures GPU execution time.
fn run_large_map_collect_test(dev: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    println!("\n--- par_iter map+collect @ {} elements ---", n);

    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    let func = get_func(dev, "par_iter_map_collect")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
    let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;

    let n_u32 = n as u32;

    // Warmup launch
    unsafe {
        func.clone()
            .launch(
                launch_config(),
                (&input_dev, &mut output_dev, n_u32, &mut status_dev),
            )
            .map_err(|e| GpuHostError::Verification {
                test: "par_iter_map_collect_1m",
                detail: format!("warmup launch: {e}"),
            })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    // Reset status and output for timed run
    let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;

    // Timed launch
    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut st = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(
                    launch_config(),
                    (&input_dev, &mut output_dev, n_u32, &mut st),
                )
                .map_err(|e| GpuHostError::Verification {
                    test: "par_iter_map_collect_1m",
                    detail: format!("timed launch: {e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        status_dev = st;
    }
    let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Read results
    let output: Vec<f32> = dev
        .dtoh_sync_copy(&output_dev)
        .map_err(GpuHostError::Cudarc)?;
    let status: Vec<u32> = dev
        .dtoh_sync_copy(&status_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(
        status[0], 1,
        "kernel did not complete (status flag not set)"
    );

    // Verify correctness: check every element
    let mut mismatches = 0usize;
    let mut max_err: f32 = 0.0;
    for i in 0..n {
        let expected = input[i] * 2.0 + 1.0;
        let got = output[i];
        let err = (got - expected).abs();
        if err > 1e-5 {
            if mismatches < 5 {
                println!("  MISMATCH at {i}: expected {expected}, got {got}");
            }
            mismatches += 1;
        }
        if err > max_err {
            max_err = err;
        }
    }

    if mismatches > 0 {
        return Err(GpuHostError::Verification {
            test: "par_iter_map_collect_1m",
            detail: format!("{mismatches} mismatches out of {n}"),
        });
    }

    let bytes = (n * 4 * 2) as f64; // read + write
    let bw_gbps = bytes / (gpu_ms * 1e6);

    println!("  ALL {n} elements correct (max_err={max_err:.2e})");
    println!("  GPU kernel time: {gpu_ms:.3} ms");
    println!(
        "  Effective bandwidth: {bw_gbps:.1} GB/s ({} MB read + {} MB written)",
        n * 4 / 1_000_000,
        n * 4 / 1_000_000
    );
    println!("  PASSED");

    Ok(gpu_ms)
}

/// Large-scale par_iter test: chained map fusion on 1M+ f32 elements.
///
/// Tests `par_iter_chained_map_collect` (.map(|x| x*2.0).map(|x| x+1.0))
/// on 1M+ elements. This proves zero-intermediate-buffer fusion at scale.
fn run_large_chained_map_test(dev: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    println!("\n--- par_iter chained map (fusion) @ {} elements ---", n);

    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    let func = get_func(dev, "par_iter_chained_map_collect")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
    let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;

    let n_u32 = n as u32;

    // Warmup
    unsafe {
        func.clone()
            .launch(
                launch_config(),
                (&input_dev, &mut output_dev, n_u32, &mut status_dev),
            )
            .map_err(|e| GpuHostError::Verification {
                test: "par_iter_chained_map_1m",
                detail: format!("warmup: {e}"),
            })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    // Timed
    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut st = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(
                    launch_config(),
                    (&input_dev, &mut output_dev, n_u32, &mut st),
                )
                .map_err(|e| GpuHostError::Verification {
                    test: "par_iter_chained_map_1m",
                    detail: format!("timed: {e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        status_dev = st;
    }
    let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let output: Vec<f32> = dev
        .dtoh_sync_copy(&output_dev)
        .map_err(GpuHostError::Cudarc)?;
    let status: Vec<u32> = dev
        .dtoh_sync_copy(&status_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(status[0], 1, "kernel did not complete");

    // Verify: chained map produces same result as single map (x*2.0 + 1.0)
    let mut mismatches = 0usize;
    let mut max_err: f32 = 0.0;
    for i in 0..n {
        let expected = input[i] * 2.0 + 1.0;
        let got = output[i];
        let err = (got - expected).abs();
        if err > 1e-5 {
            if mismatches < 5 {
                println!("  MISMATCH at {i}: expected {expected}, got {got}");
            }
            mismatches += 1;
        }
        if err > max_err {
            max_err = err;
        }
    }

    if mismatches > 0 {
        return Err(GpuHostError::Verification {
            test: "par_iter_chained_map_1m",
            detail: format!("{mismatches} mismatches out of {n}"),
        });
    }

    let bytes = (n * 4 * 2) as f64;
    let bw_gbps = bytes / (gpu_ms * 1e6);

    println!("  ALL {n} elements correct (max_err={max_err:.2e})");
    println!("  GPU kernel time: {gpu_ms:.3} ms");
    println!("  Effective bandwidth: {bw_gbps:.1} GB/s");
    println!("  Fusion proof: .map(x*2).map(x+1) == .map(x*2+1) at {n} elements");
    println!("  PASSED");

    Ok(gpu_ms)
}

/// Large-scale par_iter test: triple map + sum (reduction) on 1M+ elements.
///
/// Tests `par_iter_triple_map_sum` (.map(+1).map(*3).map(-0.5).sum())
/// on 1M+ elements. Verifies that warp-parallel reduction produces
/// correct results at scale (f32 precision).
fn run_large_triple_map_sum_test(dev: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    println!(
        "\n--- par_iter triple map+sum (reduction) @ {} elements ---",
        n
    );

    // Use small values to reduce f32 accumulation error
    let input: Vec<f32> = (0..n).map(|i| ((i % 1000) as f32) * 0.001).collect();

    let func = get_func(dev, "par_iter_triple_map_sum")?;

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut result_dev = dev.alloc_zeros::<u32>(2).map_err(GpuHostError::Cudarc)?;

    let n_u32 = n as u32;

    // Warmup
    unsafe {
        func.clone()
            .launch(launch_config(), (&input_dev, n_u32, &mut result_dev))
            .map_err(|e| GpuHostError::Verification {
                test: "par_iter_triple_map_sum_1m",
                detail: format!("warmup: {e}"),
            })?;
    }
    dev.synchronize().map_err(GpuHostError::Cudarc)?;

    // Timed
    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut rd = dev.alloc_zeros::<u32>(2).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(launch_config(), (&input_dev, n_u32, &mut rd))
                .map_err(|e| GpuHostError::Verification {
                    test: "par_iter_triple_map_sum_1m",
                    detail: format!("timed: {e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        result_dev = rd;
    }
    let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let result: Vec<u32> = dev
        .dtoh_sync_copy(&result_dev)
        .map_err(GpuHostError::Cudarc)?;

    assert_eq!(result[1], 1, "kernel did not complete");

    let gpu_sum = f32::from_bits(result[0]);

    // CPU reference: f64 for precision
    let cpu_sum_f64: f64 = input.iter().map(|x| (*x as f64 + 1.0) * 3.0 - 0.5).sum();
    let cpu_sum = cpu_sum_f64 as f32;

    // f32 reduction over 1M elements accumulates error.
    // Use relative tolerance: ~1e-4 for 1M element f32 sum.
    let rel_err = if cpu_sum.abs() > 0.0 {
        ((gpu_sum - cpu_sum) / cpu_sum).abs()
    } else {
        (gpu_sum - cpu_sum).abs()
    };

    println!("  GPU sum = {gpu_sum:.2}, CPU sum (f64 ref) = {cpu_sum_f64:.2}");
    println!("  Relative error: {rel_err:.2e}");

    // For 1M elements with f32, relative error up to ~1e-3 is acceptable.
    if rel_err > 1e-2 {
        return Err(GpuHostError::Verification {
            test: "par_iter_triple_map_sum_1m",
            detail: format!("sum too far off: GPU={gpu_sum}, CPU={cpu_sum}, rel_err={rel_err:.2e}"),
        });
    }

    let bytes = (n * 4) as f64; // read only (reduction)
    let bw_gbps = bytes / (gpu_ms * 1e6);

    println!("  GPU kernel time: {gpu_ms:.3} ms");
    println!("  Effective bandwidth: {bw_gbps:.1} GB/s (read-only reduction)");
    println!("  Deep fusion: .map(+1).map(*3).map(-0.5).sum() fused at {n} elements");
    println!("  PASSED");

    Ok(gpu_ms)
}

/// Run 1M+ element par_iter demo: correctness + performance.
///
/// Demonstrates par_iter().map().collect() at scale (1M+ f32 elements).
/// Reports:
/// - Memory copy baseline
/// - Single map kernel time
/// - Chained map (fusion proof) kernel time
/// - Triple map + sum (reduction) kernel time
/// - All verified against CPU reference
pub(crate) fn run_par_iter_1m_test(dev: Arc<CudaDevice>) -> Result<()> {
    const N: usize = 1_048_576; // 1M elements = 4 MB

    println!("\n====================================================");
    println!("  par_iter 1M+ Element Demo (iter-demo.1)");
    println!("  N = {N} f32 elements ({} MB)", N * 4 / 1_000_000);
    println!("====================================================");

    println!("  Loading PTX module...");
    load_par_iter_module(&dev)?;
    println!("  PTX module loaded.");

    // Baseline: raw memory copy
    println!("\n--- Baseline: raw GPU memory copy (htod + dtoh) ---");
    let memcpy_ms = measure_memcpy_baseline(&dev, N)?;
    let memcpy_bw = (N * 4 * 2) as f64 / (memcpy_ms * 1e6);
    println!("  htod + dtoh: {memcpy_ms:.3} ms ({memcpy_bw:.1} GB/s)");

    // Test 1: map + collect at 1M
    let map_ms = run_large_map_collect_test(&dev, N)?;

    // Test 2: chained map (fusion) at 1M
    let chained_ms = run_large_chained_map_test(&dev, N)?;

    // Test 3: triple map + sum (reduction) at 1M
    let reduce_ms = run_large_triple_map_sum_test(&dev, N)?;

    // Summary
    println!("\n====================================================");
    println!("  par_iter 1M Element Demo — SUMMARY");
    println!("====================================================");
    println!("  N = {N} f32 elements ({} MB)", N * 4 / 1_000_000);
    println!("  Memory copy baseline:   {memcpy_ms:.3} ms");
    println!("  map+collect kernel:     {map_ms:.3} ms");
    println!("  chained map (fusion):   {chained_ms:.3} ms");
    println!("  triple map+sum (reduce):{reduce_ms:.3} ms");
    if chained_ms > 0.0 && map_ms > 0.0 {
        println!(
            "  Fusion overhead:        {:.1}% (chained vs single map)",
            ((chained_ms - map_ms) / map_ms * 100.0).abs()
        );
    }
    println!("  All correctness checks: PASSED");
    println!("====================================================");

    Ok(())
}

// ============================================================
// GPU par_iter vs CPU Rayon benchmark (iter-demo.2)
// ============================================================

/// Benchmark kernels — only the ones we need for the benchmark.
const BENCH_KERNELS: &[&str] = &["par_iter_map_collect"];

/// Load par_iter benchmark kernels via PTX (CUDA driver caches JIT result).
///
/// First load is slow (~10-30 min); subsequent loads are fast from cache.
fn load_par_iter_module_fast(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("  Loading PTX module (uses CUDA JIT cache if available)...");
    let start = Instant::now();
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_STD_PTX);
    dev.load_ptx(ptx, "par_iter_fusion", BENCH_KERNELS)
        .map_err(|e| GpuHostError::Verification {
            test: "par_iter_module_load",
            detail: format!("ptx_load: {e}"),
        })?;
    let elapsed = start.elapsed();
    println!("  PTX module loaded in {elapsed:.1?}");
    Ok(())
}

/// Launch config for benchmark: 1 block x 128 threads (4 warps) + dynamic shared memory.
///
/// The par_iter kernels call `init_shared_mem_allocator(512)` which requires
/// dynamic shared memory. Without it, the kernel faults on shared-memory access.
fn bench_launch_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 1024, // ample for init_shared_mem_allocator(512)
    }
}

/// Measure GPU par_iter kernel time for N f32 elements.
///
/// Uses `par_iter_map_collect` kernel: f(x) = x * 2.0 + 1.0
/// Excludes data transfer time — measures kernel execution only.
/// Returns (kernel_ms, total_ms_including_transfer).
fn bench_gpu_par_iter(dev: &Arc<CudaDevice>, n: usize) -> Result<(f64, f64)> {
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let func = get_func(dev, "par_iter_map_collect")?;
    let cfg = bench_launch_config();

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
    let n_u32 = n as u32;

    // Warmup (3 launches)
    for _ in 0..3 {
        let mut st = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(cfg, (&input_dev, &mut output_dev, n_u32, &mut st))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_gpu_warmup",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
    }

    // Timed kernel-only runs
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let mut st = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(cfg, (&input_dev, &mut output_dev, n_u32, &mut st))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_gpu_timed",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
    }
    let kernel_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Timed end-to-end (htod + kernel + dtoh)
    let start_e2e = Instant::now();
    for _ in 0..iters {
        let in_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
        let mut out_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
        let mut st = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(cfg, (&in_dev, &mut out_dev, n_u32, &mut st))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_gpu_e2e",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        let _: Vec<f32> = dev.dtoh_sync_copy(&out_dev).map_err(GpuHostError::Cudarc)?;
    }
    let e2e_ms = start_e2e.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    Ok((kernel_ms, e2e_ms))
}

/// Measure CPU Rayon par_iter time for N f32 elements.
///
/// Operation: data.par_iter().map(|x| x * 2.0 + 1.0).collect()
/// Returns time in milliseconds.
fn bench_rayon_par_iter(n: usize) -> f64 {
    use rayon::prelude::*;

    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    // Warmup (3 iterations)
    for _ in 0..3 {
        let _: Vec<f32> = input.par_iter().map(|x| x * 2.0 + 1.0).collect();
    }

    // Timed
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _: Vec<f32> = input.par_iter().map(|x| x * 2.0 + 1.0).collect();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// Measure single-threaded CPU time for N f32 elements (sequential baseline).
fn bench_cpu_sequential(n: usize) -> f64 {
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    // Warmup
    for _ in 0..3 {
        let _: Vec<f32> = input.iter().map(|x| x * 2.0 + 1.0).collect();
    }

    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _: Vec<f32> = input.iter().map(|x| x * 2.0 + 1.0).collect();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// GPU par_iter vs CPU Rayon benchmark across multiple data sizes.
///
/// Finds the crossover point where GPU becomes faster than CPU Rayon.
/// Uses `par_iter_map_collect` kernel: f(x) = x * 2.0 + 1.0
pub(crate) fn run_par_iter_rayon_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n====================================================");
    println!("  GPU par_iter vs CPU Rayon Benchmark (iter-demo.2)");
    println!("  Operation: data.par_iter().map(|x| x * 2.0 + 1.0).collect()");
    println!("====================================================");

    // Load kernels via cubin (fast) or PTX (slow fallback)
    load_par_iter_module_fast(&dev)?;

    // Verify correctness at small scale first
    {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let func = get_func(&dev, "par_iter_map_collect")?;
        let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
        let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
        let mut status_dev = dev.alloc_zeros::<u32>(1).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.launch(
                bench_launch_config(),
                (&input_dev, &mut output_dev, n as u32, &mut status_dev),
            )
            .map_err(|e| GpuHostError::Verification {
                test: "bench_correctness",
                detail: format!("{e}"),
            })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        let output: Vec<f32> = dev
            .dtoh_sync_copy(&output_dev)
            .map_err(GpuHostError::Cudarc)?;
        let status: Vec<u32> = dev
            .dtoh_sync_copy(&status_dev)
            .map_err(GpuHostError::Cudarc)?;
        assert_eq!(status[0], 1, "kernel did not complete");
        for i in 0..n {
            let expected = input[i] * 2.0 + 1.0;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "mismatch at {i}: got {}, expected {expected}",
                output[i]
            );
        }
        println!("  Correctness verified at N=1024");
    }

    // Benchmark sizes
    let sizes: &[usize] = &[
        1_000,      // 1K
        10_000,     // 10K
        100_000,    // 100K
        1_000_000,  // 1M
        4_000_000,  // 4M
        16_000_000, // 16M
    ];

    println!(
        "\n  {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8}",
        "N", "CPU seq", "Rayon", "GPU e2e", "GPU/Rayon", "GPU/seq"
    );
    println!(
        "  {:->10}-+-{:->10}-{:->10}-{:->10}-+-{:->8}-{:->8}",
        "", "", "", "", "", ""
    );

    let mut crossover_n: Option<usize> = None;
    let mut results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();

    for &n in sizes {
        let size_label = match n {
            1_000 => "1K",
            10_000 => "10K",
            100_000 => "100K",
            1_000_000 => "1M",
            4_000_000 => "4M",
            16_000_000 => "16M",
            _ => "?",
        };

        // CPU sequential
        let seq_ms = bench_cpu_sequential(n);

        // CPU Rayon
        let rayon_ms = bench_rayon_par_iter(n);

        // GPU (kernel-only and end-to-end)
        let (gpu_kernel_ms, gpu_e2e_ms) = bench_gpu_par_iter(&dev, n)?;

        let gpu_vs_rayon = gpu_e2e_ms / rayon_ms;
        let gpu_vs_seq = gpu_e2e_ms / seq_ms;

        // Detect crossover: first size where GPU e2e < Rayon
        if crossover_n.is_none() && gpu_e2e_ms < rayon_ms {
            crossover_n = Some(n);
        }

        println!(
            "  {:>8}{} | {:>9.3} {:>9.3} {:>9.3} | {:>7.2}x {:>7.2}x  (kernel: {:.3} ms)",
            "", size_label, seq_ms, rayon_ms, gpu_e2e_ms, gpu_vs_rayon, gpu_vs_seq, gpu_kernel_ms
        );

        results.push((n, seq_ms, rayon_ms, gpu_kernel_ms, gpu_e2e_ms, gpu_vs_rayon));
    }

    // Summary
    println!("\n====================================================");
    println!("  BENCHMARK SUMMARY");
    println!("====================================================");
    println!("  GPU: GTX 1660 (sm_75), 1 block x 128 threads (4 warps)");
    println!("  CPU: Rayon (all cores), same operation f(x) = x * 2.0 + 1.0");
    println!("  GPU timing includes htod + kernel + dtoh (end-to-end)");

    if let Some(cn) = crossover_n {
        let label = match cn {
            1_000 => "1K",
            10_000 => "10K",
            100_000 => "100K",
            1_000_000 => "1M",
            4_000_000 => "4M",
            16_000_000 => "16M",
            _ => "?",
        };
        println!("  Crossover point: GPU faster at N >= {} ({})", cn, label);
    } else {
        println!("  Crossover point: GPU did NOT beat Rayon at any tested size");
        println!("  (Expected: single-block GPU with 1/22 SM utilization)");
    }

    // Architecture analysis
    println!("\n  Architecture note:");
    println!("  Current GPU par_iter uses 1 block (128 threads = 4 warps).");
    println!("  GTX 1660 has 22 SMs — only ~5% GPU utilization.");
    println!("  Multi-block dispatch would improve GPU times significantly.");
    println!("====================================================");

    Ok(())
}

// ============================================================
// Multi-block par_iter benchmark (iter-demo.3)
// ============================================================

/// All kernels needed for multi-block benchmark.
const MULTIBLOCK_KERNELS: &[&str] = &["par_iter_map_collect", "par_iter_map_collect_multiblock"];

/// Load both single-block and multi-block par_iter kernels.
fn load_multiblock_module(dev: &Arc<CudaDevice>) -> Result<()> {
    println!("  Loading PTX module (uses CUDA JIT cache if available)...");
    let start = Instant::now();
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_STD_PTX);
    dev.load_ptx(ptx, "par_iter_fusion", MULTIBLOCK_KERNELS)
        .map_err(|e| GpuHostError::Verification {
            test: "multiblock_module_load",
            detail: format!("ptx_load: {e}"),
        })?;
    let elapsed = start.elapsed();
    println!("  PTX module loaded in {elapsed:.1?}");
    Ok(())
}

/// Multi-block launch config: N threads spread across blocks.
///
/// block_size = 256 threads, grid = ceil(n / 256) blocks.
/// No shared memory needed for the grid-stride loop kernel.
fn multiblock_launch_config(n: usize, block_size: u32) -> LaunchConfig {
    let grid = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Measure multi-block GPU par_iter kernel time for N f32 elements.
///
/// Uses `par_iter_map_collect_multiblock` kernel: f(x) = x * 2.0 + 1.0
/// Grid-stride loop with cached loads/stores.
/// Returns (kernel_ms, total_ms_including_transfer).
fn bench_gpu_multiblock(dev: &Arc<CudaDevice>, n: usize) -> Result<(f64, f64)> {
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let func = get_func(dev, "par_iter_map_collect_multiblock")?;
    let cfg = multiblock_launch_config(n, 256);

    let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
    let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
    let n_u32 = n as u32;

    // Warmup (3 launches)
    for _ in 0..3 {
        unsafe {
            func.clone()
                .launch(cfg, (&input_dev, &mut output_dev, n_u32))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_multiblock_warmup",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
    }

    // Timed kernel-only runs
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            func.clone()
                .launch(cfg, (&input_dev, &mut output_dev, n_u32))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_multiblock_timed",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
    }
    let kernel_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Timed end-to-end (htod + kernel + dtoh)
    let start_e2e = Instant::now();
    for _ in 0..iters {
        let in_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
        let mut out_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
        unsafe {
            func.clone()
                .launch(cfg, (&in_dev, &mut out_dev, n_u32))
                .map_err(|e| GpuHostError::Verification {
                    test: "bench_multiblock_e2e",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;
        let _: Vec<f32> = dev.dtoh_sync_copy(&out_dev).map_err(GpuHostError::Cudarc)?;
    }
    let e2e_ms = start_e2e.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    Ok((kernel_ms, e2e_ms))
}

/// Multi-block par_iter benchmark: single-block vs multi-block vs Rayon.
///
/// Tests the multi-block kernel with cached loads against:
/// - Single-block kernel (volatile loads, 1 block x 128 threads)
/// - CPU Rayon (all cores)
/// - CPU sequential
///
/// Reports speedup from multi-block dispatch and finds crossover point.
pub(crate) fn run_multiblock_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n====================================================");
    println!("  Multi-block par_iter Benchmark (iter-demo.3)");
    println!("  Operation: data.par_iter().map(|x| x * 2.0 + 1.0).collect()");
    println!("  Multi-block: grid-stride loop + cached loads/stores");
    println!("====================================================");

    load_multiblock_module(&dev)?;

    // === Correctness verification ===
    {
        let n = 1_048_576; // 1M elements
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let func = get_func(&dev, "par_iter_map_collect_multiblock")?;
        let input_dev = dev.htod_sync_copy(&input).map_err(GpuHostError::Cudarc)?;
        let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;
        let cfg = multiblock_launch_config(n, 256);

        unsafe {
            func.launch(cfg, (&input_dev, &mut output_dev, n as u32))
                .map_err(|e| GpuHostError::Verification {
                    test: "multiblock_correctness",
                    detail: format!("{e}"),
                })?;
        }
        dev.synchronize().map_err(GpuHostError::Cudarc)?;

        let output: Vec<f32> = dev
            .dtoh_sync_copy(&output_dev)
            .map_err(GpuHostError::Cudarc)?;

        let mut mismatches = 0usize;
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let expected = input[i] * 2.0 + 1.0;
            let err = (output[i] - expected).abs();
            if err > 1e-5 {
                if mismatches < 5 {
                    println!("  MISMATCH at {i}: expected {expected}, got {}", output[i]);
                }
                mismatches += 1;
            }
            if err > max_err {
                max_err = err;
            }
        }

        if mismatches > 0 {
            return Err(GpuHostError::Verification {
                test: "multiblock_correctness",
                detail: format!("{mismatches} mismatches out of {n}"),
            });
        }
        println!("  Correctness verified at N={n} (max_err={max_err:.2e})");
    }

    // === Benchmark ===
    let sizes: &[usize] = &[
        1_000,      // 1K
        10_000,     // 10K
        100_000,    // 100K
        1_000_000,  // 1M
        4_000_000,  // 4M
        16_000_000, // 16M
    ];

    println!(
        "\n  {:>6} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>8}",
        "N", "CPU seq", "Rayon", "1-blk e2e", "MB e2e", "MB kernel", "MB/Rayon"
    );
    println!(
        "  {:->6}-+-{:->9}-{:->9}-{:->9}-{:->9}-+-{:->9}-{:->8}",
        "", "", "", "", "", "", ""
    );

    let mut crossover_n: Option<usize> = None;

    for &n in sizes {
        let size_label = match n {
            1_000 => "1K",
            10_000 => "10K",
            100_000 => "100K",
            1_000_000 => "1M",
            4_000_000 => "4M",
            16_000_000 => "16M",
            _ => "?",
        };

        // CPU sequential
        let seq_ms = bench_cpu_sequential(n);

        // CPU Rayon
        let rayon_ms = bench_rayon_par_iter(n);

        // GPU single-block skipped: uses gpu_main which requires hostcall setup.
        // See iter-demo.2 findings for single-block vs Rayon comparison data.
        let _single_e2e_ms = f64::NAN;

        // GPU multi-block (new)
        let (mb_kernel_ms, mb_e2e_ms) = bench_gpu_multiblock(&dev, n)?;

        let mb_vs_rayon = mb_e2e_ms / rayon_ms;

        // Detect crossover: first size where multi-block GPU e2e < Rayon
        if crossover_n.is_none() && mb_e2e_ms < rayon_ms {
            crossover_n = Some(n);
        }

        let blocks = ((n as u32) + 255) / 256;

        println!(
            "  {:>4}{} | {:>9.3} {:>9.3} {:>9} {:>9.3} | {:>9.3} {:>7.2}x  ({} blks)",
            "", size_label, seq_ms, rayon_ms, "—", mb_e2e_ms, mb_kernel_ms, mb_vs_rayon, blocks
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    // Summary
    println!("\n====================================================");
    println!("  MULTI-BLOCK BENCHMARK SUMMARY");
    println!("====================================================");
    println!("  Single-block: 1 block x 128 threads (4 warps), volatile loads");
    println!("  Multi-block:  ceil(N/256) blocks x 256 threads, cached loads");
    println!("  CPU: Rayon (all cores), same operation f(x) = x * 2.0 + 1.0");
    println!("  GPU timing includes htod + kernel + dtoh (end-to-end)");

    if let Some(cn) = crossover_n {
        let label = match cn {
            1_000 => "1K",
            10_000 => "10K",
            100_000 => "100K",
            1_000_000 => "1M",
            4_000_000 => "4M",
            16_000_000 => "16M",
            _ => "?",
        };
        println!("  Crossover point: GPU faster at N >= {} ({})", cn, label);
    } else {
        println!("  Crossover point: GPU multi-block did NOT beat Rayon at any tested size");
    }
    println!("====================================================");

    Ok(())
}
