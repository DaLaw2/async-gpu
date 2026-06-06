//! Par-Iter — GPU parallel iterators with Rayon-like API.
//!
//! Demonstrates the par_iter combinator chain API for GPU kernels:
//! 1. map + collect — transform elements in parallel
//! 2. map + sum (fold) — parallel reduction
//! 3. enumerate + map + collect — index-aware transforms
//! 4. zip + map + collect — element-wise operations on two arrays
//! 5. filter + map + sum — conditional reduction
//! 6. chained map + collect — fusion proof (zero intermediate buffers)
//!
//! The kernel-side code uses `GpuSlice::par_iter()` chains that compile
//! to a single fused loop per warp — no intermediate buffers, all
//! register-to-register via Rust monomorphization.
//!
//! The host side loads the pre-compiled kernel PTX from gpu-host and
//! launches each demo kernel via cudarc.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::error::Result;

/// Load the kernel PTX containing par_iter demo kernels.
fn load_kernels(dev: &Arc<CudaDevice>) -> Result<()> {
    let ptx = cudarc::nvrtc::Ptx::from_src(gpu_host::ptx::KERNEL_STD);
    dev.load_ptx(
        ptx,
        "par_iter_demo",
        &[
            "par_iter_map_collect",
            "par_iter_map_sum",
            "par_iter_enumerate_collect",
            "par_iter_zip_collect",
            "par_iter_filter_map_sum",
            "par_iter_chained_map_collect",
        ],
    )
    .map_err(|e| gpu_host::error::GpuHostError::Verification {
        test: "par_iter_load",
        detail: format!("ptx_load: {e}"),
    })?;
    Ok(())
}

/// Standard launch config: 1 block x 128 threads (4 warps).
fn launch_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Par-Iter Example: GPU Parallel Iterators ===\n");

    let dev = Arc::new(CudaDevice::new(0)?);
    println!("[host] CUDA device initialized");

    println!("[host] Loading par_iter kernels from PTX...");
    load_kernels(&dev)?;
    println!("[host] Kernels loaded\n");

    let cfg = launch_config();

    // ---- Demo 1: map + collect_into ----
    // Kernel: par_iter_map_collect
    // Chain: data.par_iter().map(|x| x * 2.0 + 1.0).collect_into(output)
    println!("--- Demo 1: map + collect ---");
    {
        let n: u32 = 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

        let d_input = dev.htod_sync_copy(&input)?;
        let mut d_output = dev.alloc_zeros::<f32>(n as usize)?;
        let d_status = dev.htod_sync_copy(&[0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_map_collect")
            .expect("par_iter_map_collect not found");
        unsafe {
            func.launch(cfg, (&d_input, &mut d_output, n, &d_status))?;
        }
        dev.synchronize()?;

        let result: Vec<f32> = dev.dtoh_sync_copy(&d_output)?;
        let ok = (0..n as usize).all(|i| {
            let expected = input[i] * 2.0 + 1.0;
            (result[i] - expected).abs() < 0.001
        });
        println!("  Chain: .map(|x| x * 2.0 + 1.0).collect_into()");
        println!("  First 5: {:?}", &result[..5]);
        println!("  Expected: {:?}", (0..5).map(|i| (i as f32) * 0.5 * 2.0 + 1.0).collect::<Vec<_>>());
        println!(
            "  Result: {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
        assert!(ok, "map+collect mismatch");
    }

    // ---- Demo 2: map + sum (reduction) ----
    // Kernel: par_iter_map_sum
    // Chain: data.par_iter().map(|x| x * x).sum()
    println!("--- Demo 2: map + sum (reduction) ---");
    {
        let n: u32 = 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();

        let d_input = dev.htod_sync_copy(&input)?;
        // Result: [sum_bits, done_flag]
        let d_result = dev.htod_sync_copy(&[0u32, 0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_map_sum")
            .expect("par_iter_map_sum not found");
        unsafe {
            func.launch(cfg, (&d_input, n, &d_result))?;
        }
        dev.synchronize()?;

        let result: Vec<u32> = dev.dtoh_sync_copy(&d_result)?;
        let gpu_sum = f32::from_bits(result[0]);
        let cpu_sum: f32 = input.iter().map(|x| x * x).sum();

        let rel_err = ((gpu_sum - cpu_sum) / cpu_sum).abs();
        println!("  Chain: .map(|x| x * x).sum()");
        println!("  GPU sum of squares: {gpu_sum:.4}");
        println!("  CPU sum of squares: {cpu_sum:.4}");
        println!("  Relative error: {rel_err:.6e}");
        println!(
            "  Result: {}\n",
            if rel_err < 0.01 { "PASSED" } else { "FAILED" }
        );
        assert!(rel_err < 0.01, "sum reduction mismatch");
    }

    // ---- Demo 3: enumerate + map + collect_into ----
    // Kernel: par_iter_enumerate_collect
    // Chain: data.par_iter().enumerate().map(|(i, x)| x + i as f32).collect_into()
    println!("--- Demo 3: enumerate + map + collect ---");
    {
        let n: u32 = 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();

        let d_input = dev.htod_sync_copy(&input)?;
        let mut d_output = dev.alloc_zeros::<f32>(n as usize)?;
        let d_status = dev.htod_sync_copy(&[0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_enumerate_collect")
            .expect("par_iter_enumerate_collect not found");
        unsafe {
            func.launch(cfg, (&d_input, &mut d_output, n, &d_status))?;
        }
        dev.synchronize()?;

        let result: Vec<f32> = dev.dtoh_sync_copy(&d_output)?;
        let ok = (0..n as usize).all(|i| {
            let expected = input[i] + i as f32;
            (result[i] - expected).abs() < 0.001
        });
        println!("  Chain: .enumerate().map(|(i, x)| x + i as f32).collect_into()");
        println!("  First 5: {:?}", &result[..5]);
        println!(
            "  Result: {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
        assert!(ok, "enumerate+map+collect mismatch");
    }

    // ---- Demo 4: zip + map + collect_into ----
    // Kernel: par_iter_zip_collect
    // Chain: a.par_iter().zip(b.par_iter()).map(|(x,y)| x + y).collect_into()
    println!("--- Demo 4: zip + map + collect ---");
    {
        let n: u32 = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let d_a = dev.htod_sync_copy(&a)?;
        let d_b = dev.htod_sync_copy(&b)?;
        let mut d_output = dev.alloc_zeros::<f32>(n as usize)?;
        let d_status = dev.htod_sync_copy(&[0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_zip_collect")
            .expect("par_iter_zip_collect not found");
        unsafe {
            func.launch(cfg, (&d_a, &d_b, &mut d_output, n, &d_status))?;
        }
        dev.synchronize()?;

        let result: Vec<f32> = dev.dtoh_sync_copy(&d_output)?;
        let ok = (0..n as usize).all(|i| {
            let expected = a[i] + b[i];
            (result[i] - expected).abs() < 0.001
        });
        println!("  Chain: a.par_iter().zip(b.par_iter()).map(|(x,y)| x + y).collect_into()");
        println!("  All elements should equal {}: {}", n, if ok { "YES" } else { "NO" });
        println!(
            "  Result: {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
        assert!(ok, "zip+map+collect mismatch");
    }

    // ---- Demo 5: filter + map + sum ----
    // Kernel: par_iter_filter_map_sum
    // Chain: data.par_iter().filter(|x| *x > threshold).map(|x| x * x).sum()
    println!("--- Demo 5: filter + map + sum ---");
    {
        let n: u32 = 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 1.0).collect();
        let threshold: f32 = 2.0;

        let d_input = dev.htod_sync_copy(&input)?;
        let d_result = dev.htod_sync_copy(&[0u32, 0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_filter_map_sum")
            .expect("par_iter_filter_map_sum not found");
        unsafe {
            func.launch(cfg, (&d_input, n, threshold.to_bits(), &d_result))?;
        }
        dev.synchronize()?;

        let result: Vec<u32> = dev.dtoh_sync_copy(&d_result)?;
        let gpu_sum = f32::from_bits(result[0]);
        let cpu_sum: f32 = input
            .iter()
            .filter(|x| **x > threshold)
            .map(|x| x * x)
            .sum();

        let rel_err = if cpu_sum.abs() > 0.001 {
            ((gpu_sum - cpu_sum) / cpu_sum).abs()
        } else {
            (gpu_sum - cpu_sum).abs()
        };
        println!("  Chain: .filter(|x| *x > {threshold}).map(|x| x * x).sum()");
        println!("  GPU filtered sum: {gpu_sum:.4}");
        println!("  CPU filtered sum: {cpu_sum:.4}");
        println!("  Relative error: {rel_err:.6e}");
        println!(
            "  Result: {}\n",
            if rel_err < 0.01 { "PASSED" } else { "FAILED" }
        );
        assert!(rel_err < 0.01, "filter+map+sum mismatch");
    }

    // ---- Demo 6: chained map + collect (fusion proof) ----
    // Kernel: par_iter_chained_map_collect
    // Chain: data.par_iter().map(|x| x * 2.0).map(|x| x + 1.0).collect_into()
    // Key: Two SEPARATE .map() calls, proving zero-intermediate-buffer fusion.
    println!("--- Demo 6: chained map + collect (fusion proof) ---");
    {
        let n: u32 = 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();

        let d_input = dev.htod_sync_copy(&input)?;
        let mut d_output = dev.alloc_zeros::<f32>(n as usize)?;
        let d_status = dev.htod_sync_copy(&[0u32])?;

        let func = dev
            .get_func("par_iter_demo", "par_iter_chained_map_collect")
            .expect("par_iter_chained_map_collect not found");
        unsafe {
            func.launch(cfg, (&d_input, &mut d_output, n, &d_status))?;
        }
        dev.synchronize()?;

        let result: Vec<f32> = dev.dtoh_sync_copy(&d_output)?;
        let ok = (0..n as usize).all(|i| {
            let expected = input[i] * 2.0 + 1.0;
            (result[i] - expected).abs() < 0.001
        });
        println!("  Chain: .map(|x| x * 2.0).map(|x| x + 1.0).collect_into()");
        println!("  Two separate .map() calls — fused at compile time");
        println!("  Zero intermediate buffers (register-to-register)");
        println!("  First 5: {:?}", &result[..5]);
        println!(
            "  Result: {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
        assert!(ok, "chained map fusion mismatch");
    }

    println!("=== All demos complete! ===");
    Ok(())
}
