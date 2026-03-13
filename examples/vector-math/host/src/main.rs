//! Vector Math — host binary demonstrating pure GPU compute.
//!
//! Three demos showing CPU-GPU cooperation:
//! 1. SAXPY — y = a*x + y (element-wise GPU compute)
//! 2. Dot product — GPU does element-wise multiply, CPU sums
//! 3. Softmax — GPU exp + normalize, CPU finds max and sum

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() {
    println!("=== Vector Math Example ===\n");

    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("[host] CUDA device initialized.");

    let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_PTX);
    dev.load_ptx(
        ptx,
        "vecmath",
        &[
            "saxpy",
            "elementwise_mul",
            "softmax_exp",
            "softmax_normalize",
        ],
    )
    .expect("Failed to load PTX module");
    println!("[host] PTX module loaded.\n");

    // ---- Demo 1: SAXPY ----
    println!("--- Demo 1: SAXPY (y = 2.0 * x + y) ---");
    {
        const N: usize = 1024;
        let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let y_orig: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let a = 2.0f32;

        let x_dev = dev.htod_sync_copy(&x).unwrap();
        let mut y_dev = dev.htod_sync_copy(&y_orig).unwrap();

        let f = dev.get_func("vecmath", "saxpy").unwrap();
        let cfg = LaunchConfig {
            grid_dim: ((N as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&x_dev, &mut y_dev, a, N as u32)).unwrap() };
        let result = dev.dtoh_sync_copy(&y_dev).unwrap();

        // Verify: y[i] = 2.0 * i + 2*i = 4*i
        let mut ok = true;
        for i in 0..N {
            let expected = a * x[i] + y_orig[i];
            if (result[i] - expected).abs() > 0.001 {
                println!(
                    "[host]   MISMATCH at {i}: got {}, expected {}",
                    result[i], expected
                );
                ok = false;
                break;
            }
        }
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

        let x_dev = dev.htod_sync_copy(&x).unwrap();
        let y_dev = dev.htod_sync_copy(&y).unwrap();
        let mut products_dev = dev.alloc_zeros::<f32>(N).unwrap();

        let f = dev.get_func("vecmath", "elementwise_mul").unwrap();
        let cfg = LaunchConfig {
            grid_dim: ((N as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&x_dev, &y_dev, &mut products_dev, N as u32))
                .unwrap()
        };
        let products = dev.dtoh_sync_copy(&products_dev).unwrap();

        // CPU reduction
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

        let input_dev = dev.htod_sync_copy(&input).unwrap();
        let mut exp_dev = dev.alloc_zeros::<f32>(N).unwrap();

        // Step 2: GPU computes exp(x - max)
        let f_exp = dev.get_func("vecmath", "softmax_exp").unwrap();
        let cfg = LaunchConfig {
            grid_dim: ((N as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f_exp
                .launch(cfg, (&input_dev, &mut exp_dev, max_val, N as u32))
                .unwrap()
        };
        let exp_vals = dev.dtoh_sync_copy(&exp_dev).unwrap();

        // Step 3: CPU sums exp values
        let exp_sum: f32 = exp_vals.iter().sum();

        // Step 4: GPU normalizes
        let mut result_dev = dev.htod_sync_copy(&exp_vals).unwrap();
        let f_norm = dev.get_func("vecmath", "softmax_normalize").unwrap();
        unsafe {
            f_norm
                .launch(cfg, (&mut result_dev, exp_sum, N as u32))
                .unwrap()
        };
        let result = dev.dtoh_sync_copy(&result_dev).unwrap();

        // Verify
        let gpu_sum: f32 = result.iter().sum();
        let cpu_softmax: Vec<f32> = input
            .iter()
            .map(|x| (x - max_val).exp())
            .collect::<Vec<_>>()
            .iter()
            .map(|e| e / exp_sum)
            .collect();
        let max_err = result
            .iter()
            .zip(cpu_softmax.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        println!("[host] softmax sum = {gpu_sum:.6} (expected 1.0)");
        println!("[host] max |GPU - CPU| = {max_err:.6}");
        println!(
            "[host] first 3: [{:.6}, {:.6}, {:.6}]",
            result[0], result[1], result[2]
        );
        println!(
            "[host] last 3:  [{:.6}, {:.6}, {:.6}]",
            result[N - 3],
            result[N - 2],
            result[N - 1]
        );

        let ok = (gpu_sum - 1.0).abs() < 0.01 && max_err < 0.001;
        println!(
            "[host] Softmax ({N} elements): {}\n",
            if ok { "PASSED" } else { "FAILED" }
        );
    }

    println!("=== All demos complete! ===");
}
