//! Conv2D benchmark — measure GFLOPS for common shapes.
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test conv2d_bench -- --nocapture

use std::time::Instant;

#[test]
fn bench_conv2d_shapes() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    println!("\n=== Conv2D Benchmark (im2col + GEMM) ===");
    println!(
        "{:>5} {:>5} {:>5}x{:<5} {:>2} {:>3} | {:>10} {:>10}",
        "Cin", "Cout", "H", "W", "K", "S", "Time(ms)", "GFLOPS"
    );
    println!("{}", "-".repeat(70));

    let warmup = 3;
    let iters = 10;

    // Common ResNet/YOLO conv shapes with 3x3 kernel
    let configs: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
        // (c_in, c_out, h, w, kh, stride)
        (3, 64, 224, 224, 3, 1),  // ResNet first conv
        (64, 64, 56, 56, 3, 1),   // ResNet layer1
        (128, 128, 28, 28, 3, 1), // ResNet layer2
        (256, 256, 14, 14, 3, 1), // ResNet layer3
        (512, 512, 7, 7, 3, 1),   // ResNet layer4
        (32, 32, 32, 32, 3, 1),   // CIFAR-10 style
        (16, 32, 32, 32, 3, 2),   // YOLO downsample
        (3, 16, 640, 640, 3, 2),  // YOLO first conv
        (64, 64, 80, 80, 3, 1),   // YOLO P3
    ];

    for &(c_in, c_out, h, w, kh, stride) in &configs {
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

        println!(
            "{:>5} {:>5} {:>5}x{:<5} {:>2} {:>3} | {:>9.3} {:>10.1}",
            c_in, c_out, h, w, kh, stride, ms, gflops
        );
    }

    // Reference: GTX 1660 theoretical peak = 5 TFLOPS FP32
    // cuDNN typically achieves 60-80% of peak for conv2d
    // Target: >= 70% of cuDNN (i.e., >= 42-56% of peak = 2100-2800 GFLOPS)
    println!("\nGTX 1660 theoretical peak: 5000 GFLOPS FP32");
    println!("Target: >= 70% of cuDNN (~2100-2800 GFLOPS for common shapes)");
}
