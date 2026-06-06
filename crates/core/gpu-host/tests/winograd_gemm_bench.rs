//! Benchmark: Winograd batched-GEMM pipeline vs old fused kernel.
//!
//! Measures GFLOPS for the new Winograd F(2x2,3x3) implementation that
//! uses cuBLAS strided batched GEMM for the transform-domain multiply.
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test winograd_gemm_bench -- --nocapture

use std::time::Instant;

#[cfg(feature = "cublas")]
#[test]
fn bench_winograd_gemm() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    println!("\n=== Winograd Batched-GEMM Benchmark ===");
    println!(
        "{:>5} {:>5} {:>5}x{:<5} {:>2} | {:>10} {:>10} {:>6}",
        "Cin", "Cout", "H", "W", "P", "Time(ms)", "GFLOPS", "%Peak"
    );
    println!("{}", "-".repeat(62));

    let warmup = 5;
    let iters = 20;

    // ResNet-style 3x3 stride=1 shapes (these use Winograd)
    let configs: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        (3, 64, 224, 224, 1, "ResNet conv1"),
        (64, 64, 56, 56, 1, "ResNet L1"),
        (128, 128, 28, 28, 1, "ResNet L2"),
        (256, 256, 14, 14, 1, "ResNet L3"),
        (512, 512, 7, 7, 1, "ResNet L4"),
        (32, 32, 32, 32, 1, "CIFAR-10"),
        (64, 64, 80, 80, 1, "YOLO P3"),
        (128, 128, 40, 40, 1, "YOLO P4"),
        (256, 256, 20, 20, 1, "YOLO P5"),
    ];

    // GTX 1660: 5027 GFLOPS FP32 peak
    let peak_gflops = 5027.0;

    for &(c_in, c_out, h, w, padding, desc) in &configs {
        let kh = 3;
        let stride = 1;
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
            "{:>5} {:>5} {:>5}x{:<5} {:>2} | {:>9.3} {:>10.1} {:>5.1}%  {}",
            c_in, c_out, h, w, padding, ms, gflops, pct_peak, desc
        );
    }

    println!("\nGTX 1660 theoretical peak: {peak_gflops:.0} GFLOPS FP32");
    println!("cuDNN target: ~50% peak = {:.0} GFLOPS", peak_gflops * 0.5);

    // --- YOLOv8-nano synthetic e2e benchmark ---
    // All 3x3 stride=1 convolution layers from YOLOv8-nano backbone + neck.
    // This simulates the conv-heavy portion of inference.
    println!("\n=== YOLOv8-nano 3x3 Conv Layers (Synthetic) ===");
    println!(
        "{:>5} {:>5} {:>5}x{:<5} | {:>9} {:>10}",
        "Cin", "Cout", "H", "W", "Time(ms)", "GFLOPS"
    );
    println!("{}", "-".repeat(50));

    let yolo_layers: Vec<(usize, usize, usize, usize, &str)> = vec![
        // Backbone C2f blocks (3x3 bottleneck convs, stride=1)
        (32, 32, 160, 160, "BB-P2"),
        (64, 64, 80, 80, "BB-P3"),
        (64, 64, 80, 80, "BB-P3b"),
        (128, 128, 40, 40, "BB-P4"),
        (128, 128, 40, 40, "BB-P4b"),
        (256, 256, 20, 20, "BB-P5"),
        // Neck upsampled concat + C2f 3x3 convs
        (128, 128, 40, 40, "Neck-P4"),
        (64, 64, 80, 80, "Neck-P3"),
        // Neck downsample + C2f 3x3 convs
        (128, 128, 40, 40, "Neck-P4d"),
        (256, 256, 20, 20, "Neck-P5d"),
    ];

    let yolo_warmup = 3;
    let yolo_iters = 10;
    let mut total_yolo_ms = 0.0f64;
    let mut total_yolo_flops = 0.0f64;

    for &(c_in, c_out, h, w, label) in &yolo_layers {
        let kh = 3;
        let stride = 1;
        let padding = 1;
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

        for _ in 0..yolo_warmup {
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

        let t0 = Instant::now();
        for _ in 0..yolo_iters {
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
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / yolo_iters as f64;
        let gflops = flops / (ms / 1000.0) / 1e9;
        total_yolo_ms += ms;
        total_yolo_flops += flops;

        println!(
            "{:>5} {:>5} {:>5}x{:<5} | {:>8.3} {:>10.1}  {}",
            c_in, c_out, h, w, ms, gflops, label
        );
    }

    let total_gflops = total_yolo_flops / (total_yolo_ms / 1000.0) / 1e9;
    println!("{}", "-".repeat(50));
    println!(
        "Total 3x3 conv time: {:.2}ms, aggregate GFLOPS: {:.0}",
        total_yolo_ms, total_gflops
    );
    // Previous baseline (conv-wino-gemm.2): ~0.5-0.6ms per layer = ~5-6ms total
    // Target: >= 2x improvement = <= 2.5-3ms total
    println!("Previous baseline estimate: ~5.0ms total (from conv-wino-gemm.2 numbers)");
    println!("Speedup vs baseline: ~{:.1}x", 5.0 / total_yolo_ms);
}

/// Correctness test for batched Winograd with bias, through the conv2d API.
#[cfg(feature = "cublas")]
#[test]
fn winograd_gemm_correctness_with_bias() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    let tests = vec![
        (1, 1, 5, 5, 0),
        (3, 4, 8, 8, 1),
        (3, 8, 32, 32, 1),
        (64, 64, 14, 14, 1),
    ];

    for (c_in, c_out, h, w, padding) in tests {
        eprint!(
            "  bias test (c_in={}, c_out={}, {}x{}, pad={}) ... ",
            c_in, c_out, h, w, padding
        );

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * 9)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..c_out)
            .map(|i| ((i * 7 + 11) % 100) as f32 / 100.0 - 0.5)
            .collect();

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)
                .expect("input");
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, 3, 3], &dev)
                .expect("weight");
        let bias_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&bias_data, &[c_out], &dev).expect("bias");

        let result = gpu_host::nn::ops::conv2d(
            &input_tensor,
            &weight_tensor,
            Some(&bias_tensor),
            1,
            padding,
            &registry,
        )
        .expect("conv2d");

        let gpu_output = result.to_host().expect("download");

        // CPU reference
        let expected = cpu_conv2d_bias(
            &input_data,
            &weight_data,
            &bias_data,
            c_in,
            h,
            w,
            c_out,
            padding,
        );

        let max_err: f32 = gpu_output
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);

        eprintln!("max_err={max_err:.6}");
        assert!(
            max_err < 0.01,
            "max_err={max_err} exceeds threshold for {}x{} c_in={} c_out={} pad={}",
            h,
            w,
            c_in,
            c_out,
            padding
        );
    }
    eprintln!("All bias tests passed!");
}

/// Correctness test for batched (N > 1) Winograd through the conv2d API.
#[cfg(feature = "cublas")]
#[test]
fn winograd_gemm_batched_correctness() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    let (batch, c_in, c_out, h, w, padding) = (4, 3, 8, 16, 16, 1);
    eprint!(
        "  batched test (N={}, c_in={}, c_out={}, {}x{}, pad={}) ... ",
        batch, c_in, c_out, h, w, padding
    );

    let input_data: Vec<f32> = (0..batch * c_in * h * w)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let weight_data: Vec<f32> = (0..c_out * c_in * 9)
        .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let input_tensor =
        gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[batch, c_in, h, w], &dev)
            .expect("input");
    let weight_tensor =
        gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, 3, 3], &dev)
            .expect("weight");

    let result =
        gpu_host::nn::ops::conv2d(&input_tensor, &weight_tensor, None, 1, padding, &registry)
            .expect("conv2d");

    let gpu_output = result.to_host().expect("download");

    // CPU reference per sample
    let h_out = h + 2 * padding - 2;
    let w_out = w + 2 * padding - 2;
    let sample_in_size = c_in * h * w;
    let sample_out_size = c_out * h_out * w_out;

    let mut max_err = 0.0f32;
    for b in 0..batch {
        let sample_input = &input_data[b * sample_in_size..(b + 1) * sample_in_size];
        let expected = cpu_conv2d_nobias(sample_input, &weight_data, c_in, h, w, c_out, padding);
        let gpu_sample = &gpu_output[b * sample_out_size..(b + 1) * sample_out_size];

        let sample_err: f32 = gpu_sample
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        max_err = max_err.max(sample_err);
    }

    eprintln!("max_err={max_err:.6}");
    assert!(max_err < 0.01, "max_err={max_err} exceeds threshold");
    eprintln!("Batched test passed!");
}

#[cfg(feature = "cublas")]
fn cpu_conv2d_nobias(
    input: &[f32],
    filter: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    padding: usize,
) -> Vec<f32> {
    let h_out = h + 2 * padding - 2;
    let w_out = w + 2 * padding - 2;
    let mut out = vec![0.0f32; c_out * h_out * w_out];

    for co in 0..c_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0f64;
                for ci in 0..c_in {
                    for fh in 0..3 {
                        for fw in 0..3 {
                            let ih = oh as isize + fh as isize - padding as isize;
                            let iw = ow as isize + fw as isize - padding as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_val = input[ci * h * w + ih as usize * w + iw as usize];
                                let w_val = filter[co * (c_in * 9) + ci * 9 + fh * 3 + fw];
                                sum += in_val as f64 * w_val as f64;
                            }
                        }
                    }
                }
                out[co * h_out * w_out + oh * w_out + ow] = sum as f32;
            }
        }
    }
    out
}

#[cfg(feature = "cublas")]
fn cpu_conv2d_bias(
    input: &[f32],
    filter: &[f32],
    bias: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    padding: usize,
) -> Vec<f32> {
    let h_out = h + 2 * padding - 2;
    let w_out = w + 2 * padding - 2;
    let mut out = cpu_conv2d_nobias(input, filter, c_in, h, w, c_out, padding);

    for co in 0..c_out {
        let b = bias[co];
        for i in 0..h_out * w_out {
            out[co * h_out * w_out + i] += b;
        }
    }
    out
}
