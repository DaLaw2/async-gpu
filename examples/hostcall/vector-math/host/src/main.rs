//! Vector Math — host binary demonstrating pure GPU compute.
//!
//! Three demos showing CPU-GPU cooperation:
//! 1. SAXPY — y = a*x + y (element-wise GPU compute)
//! 2. Dot product — GPU does element-wise multiply, CPU sums
//! 3. Softmax — GPU exp + normalize, CPU finds max and sum
//!
//! Uses the `gpu::custom()` builder API for clean, minimal boilerplate.

use async_gpu::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> async_gpu::Result<()> {
    println!("=== Vector Math Example ===\n");

    // ---- Demo 1: SAXPY ----
    println!("--- Demo 1: SAXPY (y = 2.0 * x + y) ---");
    {
        const N: usize = 1024;
        let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let y_orig: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let a = 2.0f32;

        let ctx = gpu::custom("saxpy")
            .ptx(KERNEL_PTX)
            .threads(256)
            .elements(N as u32)
            .prepare()?;

        let x_dev = ctx.upload(&x)?;
        let mut y_dev = ctx.upload(&y_orig)?;

        let result = unsafe { ctx.launch((&x_dev, &mut y_dev, a, N as u32))? };
        let y_result = result.download(&y_dev)?;

        let ok = (0..N).all(|i| {
            let expected = a * x[i] + y_orig[i];
            (y_result[i] - expected).abs() < 0.001
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

        let ctx = gpu::custom("elementwise_mul")
            .ptx(KERNEL_PTX)
            .threads(256)
            .elements(N as u32)
            .prepare()?;

        let x_dev = ctx.upload(&x)?;
        let y_dev = ctx.upload(&y)?;
        let mut products_dev = ctx.alloc_zeros::<f32>(N)?;

        let result = unsafe { ctx.launch((&x_dev, &y_dev, &mut products_dev, N as u32))? };
        let products = result.download(&products_dev)?;

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

        // Step 2: GPU computes exp(x - max)
        let ctx_exp = gpu::custom("softmax_exp")
            .ptx(KERNEL_PTX)
            .threads(256)
            .elements(N as u32)
            .prepare()?;

        let input_dev = ctx_exp.upload(&input)?;
        let mut exp_dev = ctx_exp.alloc_zeros::<f32>(N)?;

        let result_exp = unsafe { ctx_exp.launch((&input_dev, &mut exp_dev, max_val, N as u32))? };
        let exp_vals = result_exp.download(&exp_dev)?;

        // Step 3: CPU sums exp values
        let exp_sum: f32 = exp_vals.iter().sum();

        // Step 4: GPU normalizes
        let ctx_norm = gpu::custom("softmax_normalize")
            .ptx(KERNEL_PTX)
            .threads(256)
            .elements(N as u32)
            .prepare()?;

        let mut result_dev = ctx_norm.upload(&exp_vals)?;
        let result_norm = unsafe { ctx_norm.launch((&mut result_dev, exp_sum, N as u32))? };
        let softmax = result_norm.download(&result_dev)?;

        // Verify
        let gpu_sum: f32 = softmax.iter().sum();
        let cpu_softmax: Vec<f32> = {
            let exps: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
            exps.iter().map(|e| e / exp_sum).collect()
        };
        let max_err = softmax
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
