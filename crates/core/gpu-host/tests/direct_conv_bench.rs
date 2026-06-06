//! Direct conv benchmark — measure GFLOPS for non-3x3 kernel sizes (5x5, 7x7).
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test direct_conv_bench -- --nocapture

use std::time::Instant;

#[test]
fn bench_direct_conv_non3x3() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    println!("\n=== Direct Conv2D Benchmark (non-3x3 kernels) ===");
    println!(
        "{:>5} {:>5} {:>5}x{:<5} {:>2} {:>3} | {:>10} {:>10} {:>8}",
        "Cin", "Cout", "H", "W", "K", "S", "Time(ms)", "GFLOPS", "%Peak"
    );
    println!("{}", "-".repeat(75));

    let warmup = 5;
    let iters = 20;
    let peak_gflops = 5000.0_f64; // GTX 1660 FP32 peak

    // Non-3x3 shapes covering real use cases:
    // 5x5 kernels (used in some ResNet variants, EfficientNet)
    // 7x7 kernels (ResNet stem, some attention-based models)
    // Also 3x3 stride=2 for comparison (direct conv path)
    let configs: Vec<(usize, usize, usize, usize, usize, usize, &str)> = vec![
        // 5x5 kernels
        (3, 32, 224, 224, 5, 1, "5x5 s1 stem"),
        (32, 64, 112, 112, 5, 1, "5x5 s1 mid"),
        (64, 128, 56, 56, 5, 1, "5x5 s1 deep"),
        (3, 32, 224, 224, 5, 2, "5x5 s2 stem"),
        (32, 64, 112, 112, 5, 2, "5x5 s2 mid"),
        (64, 128, 56, 56, 5, 2, "5x5 s2 deep"),
        // 7x7 kernels
        (3, 64, 224, 224, 7, 2, "7x7 s2 ResNet stem"),
        (3, 32, 224, 224, 7, 1, "7x7 s1 stem"),
        (32, 64, 56, 56, 7, 1, "7x7 s1 mid"),
        // 3x3 stride=2 (direct conv path, for comparison)
        (3, 16, 640, 640, 3, 2, "3x3 s2 YOLO stem"),
        (16, 32, 320, 320, 3, 2, "3x3 s2 BB down"),
        (64, 128, 80, 80, 3, 2, "3x3 s2 BB deep"),
    ];

    for &(c_in, c_out, h, w, kh, stride, label) in &configs {
        let padding = kh / 2;
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kh) / stride + 1;
        let flops =
            2.0 * c_out as f64 * h_out as f64 * w_out as f64 * c_in as f64 * kh as f64 * kh as f64;

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * kh * kh)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)
                .expect("input");
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, kh, kh], &dev)
                .expect("weight");

        // Warmup
        for _ in 0..warmup {
            let _ = gpu_host::nn::ops::conv2d(
                &input_tensor,
                &weight_tensor,
                None,
                stride,
                padding,
                &registry,
            )
            .expect("conv2d");
            dev.synchronize().expect("sync");
        }

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = gpu_host::nn::ops::conv2d(
                &input_tensor,
                &weight_tensor,
                None,
                stride,
                padding,
                &registry,
            )
            .expect("conv2d");
            dev.synchronize().expect("sync");
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let gflops = flops / (ms / 1000.0) / 1e9;
        let pct_peak = gflops / peak_gflops * 100.0;

        println!(
            "{:>5} {:>5} {:>5}x{:<5} {:>2} {:>3} | {:>9.3} {:>10.1} {:>7.1}%  {}",
            c_in, c_out, h, w, kh, stride, ms, gflops, pct_peak, label
        );
    }

    println!("\nGTX 1660 theoretical peak: 5000 GFLOPS FP32");
}
