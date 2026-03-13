//! Vector Math — host binary demonstrating pure GPU compute.
//!
//! Three demos showing CPU-GPU cooperation:
//! 1. SAXPY — y = a*x + y (element-wise GPU compute)
//! 2. Dot product — GPU does element-wise multiply, CPU sums
//! 3. Softmax — GPU exp + normalize, CPU finds max and sum
//!
//! Uses [`GpuRuntime`] for device management, PTX loading, and data transfer.
//! No hostcall needed — pure compute kernels.

use cudarc::driver::LaunchAsync;
use gpu_host::{GpuHostError, GpuRuntime};

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    println!("=== Vector Math Example ===\n");

    let rt = GpuRuntime::new(0)?;
    println!("[host] CUDA device initialized.");

    rt.load_ptx(
        KERNEL_PTX,
        "vecmath",
        &[
            "saxpy",
            "elementwise_mul",
            "softmax_exp",
            "softmax_normalize",
        ],
    )?;
    println!("[host] PTX module loaded.\n");

    // ---- Demo 1: SAXPY ----
    println!("--- Demo 1: SAXPY (y = 2.0 * x + y) ---");
    {
        const N: usize = 1024;
        let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let y_orig: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let a = 2.0f32;

        let x_dev = rt.htod_sync_copy(&x)?;
        let mut y_dev = rt.htod_sync_copy(&y_orig)?;

        let f = rt
            .get_func("vecmath", "saxpy")
            .ok_or(GpuHostError::KernelNotFound("saxpy"))?;
        let cfg = GpuRuntime::launch_config(((N as u32).div_ceil(256), 1, 1), (256, 1, 1), 0);
        unsafe { f.launch(cfg, (&x_dev, &mut y_dev, a, N as u32))? };
        let result = rt.dtoh_sync_copy(&y_dev)?;

        let ok = (0..N).all(|i| {
            let expected = a * x[i] + y_orig[i];
            (result[i] - expected).abs() < 0.001
        });
        println!(
            "[host] SAXPY ({N} elements): {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
    }

    // ---- Demo 2: Dot Product (GPU mul + CPU sum) ----
    println!("--- Demo 2: Dot Product (GPU multiply, CPU reduce) ---");
    {
        const N: usize = 1024;
        let x: Vec<f32> = (0..N).map(|i| (i % 10) as f32).collect();
        let y: Vec<f32> = (0..N).map(|i| ((i + 3) % 7) as f32).collect();

        let x_dev = rt.htod_sync_copy(&x)?;
        let y_dev = rt.htod_sync_copy(&y)?;
        let mut products_dev = rt.alloc_zeros::<f32>(N)?;

        let f = rt
            .get_func("vecmath", "elementwise_mul")
            .ok_or(GpuHostError::KernelNotFound("elementwise_mul"))?;
        let cfg = GpuRuntime::launch_config(((N as u32).div_ceil(256), 1, 1), (256, 1, 1), 0);
        unsafe { f.launch(cfg, (&x_dev, &y_dev, &mut products_dev, N as u32))? };
        let products = rt.dtoh_sync_copy(&products_dev)?;

        let gpu_dot: f32 = products.iter().sum();
        let cpu_dot: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let ok = (gpu_dot - cpu_dot).abs() < 0.01;
        println!("[host] dot(x,y) GPU = {gpu_dot:.2}, CPU = {cpu_dot:.2}");
        println!(
            "[host] Dot Product ({N} elements): {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
    }

    // ---- Demo 3: Softmax (GPU exp + normalize, CPU max + sum) ----
    println!("--- Demo 3: Softmax (GPU-CPU cooperative) ---");
    {
        const N: usize = 256;
        let input: Vec<f32> = (0..N).map(|i| (i as f32 - 128.0) * 0.1).collect();

        // Step 1: CPU finds max for numerical stability
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let input_dev = rt.htod_sync_copy(&input)?;
        let mut exp_dev = rt.alloc_zeros::<f32>(N)?;

        // Step 2: GPU computes exp(x - max)
        let f_exp = rt
            .get_func("vecmath", "softmax_exp")
            .ok_or(GpuHostError::KernelNotFound("softmax_exp"))?;
        let cfg = GpuRuntime::launch_config(((N as u32).div_ceil(256), 1, 1), (256, 1, 1), 0);
        unsafe { f_exp.launch(cfg, (&input_dev, &mut exp_dev, max_val, N as u32))? };
        let exp_vals = rt.dtoh_sync_copy(&exp_dev)?;

        // Step 3: CPU sums exp values
        let exp_sum: f32 = exp_vals.iter().sum();

        // Step 4: GPU normalizes
        let mut result_dev = rt.htod_sync_copy(&exp_vals)?;
        let f_norm = rt
            .get_func("vecmath", "softmax_normalize")
            .ok_or(GpuHostError::KernelNotFound("softmax_normalize"))?;
        unsafe { f_norm.launch(cfg, (&mut result_dev, exp_sum, N as u32))? };
        let result = rt.dtoh_sync_copy(&result_dev)?;

        // Verify
        let gpu_sum: f32 = result.iter().sum();
        let cpu_softmax: Vec<f32> = {
            let exps: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
            exps.iter().map(|e| e / exp_sum).collect()
        };
        let max_err = result
            .iter()
            .zip(cpu_softmax.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        println!("[host] softmax sum = {gpu_sum:.6} (expected 1.0)");
        println!("[host] max |GPU - CPU| = {max_err:.6}");

        let ok = (gpu_sum - 1.0).abs() < 0.01 && max_err < 0.001;
        println!(
            "[host] Softmax ({N} elements): {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
    }

    println!("=== All demos complete! ===");
    Ok(())
}
