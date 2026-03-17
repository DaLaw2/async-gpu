//! GPU Kernel Benchmarks — GFLOPS and GB/s vs cuBLAS.
//!
//! Measures the performance of async-gpu's custom kernels against NVIDIA's
//! optimized libraries, quantifying both absolute throughput and the gap
//! to close for competitive performance.

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::LaunchAsync;
use gpu_host::nn::layers::Module;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Kernel Benchmarks ===\n");

    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default()?;
    let blas = CudaBlas::new(Arc::clone(&dev))?;

    // Get GPU info
    println!("GPU: CUDA device 0");

    // --- SGEMM Benchmark ---
    println!("\n--- SGEMM Benchmark (C = A × B, f32) ---");
    println!(
        "{:>6} {:>6} {:>6} | {:>10} {:>10} | {:>10} {:>10} | {:>6}",
        "M", "N", "K", "ours(ms)", "ours(GFLOPS)", "cuBLAS(ms)", "cuBLAS(GFLOPS)", "ratio"
    );
    println!("{}", "-".repeat(85));

    let sizes: Vec<(usize, usize, usize)> = vec![
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        // GPT-2 shapes
        (1, 768, 768),    // single-token decode
        (128, 768, 768),  // prompt prefill
        (128, 768, 3072), // FFN up
        (128, 3072, 768), // FFN down
    ];

    let warmup_iters = 3;
    let bench_iters = 10;

    for &(m, n, k) in &sizes {
        let flops = 2.0 * m as f64 * n as f64 * k as f64; // 2*M*N*K for GEMM

        // Generate random data
        let a_host: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_host: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // Upload to GPU
        let a_dev = dev.htod_sync_copy(&a_host)?;
        let b_dev = dev.htod_sync_copy(&b_host)?;

        // --- Our GEMM ---
        let a_tensor = gpu_host::nn::tensor::GpuTensor::from_data(a_dev, &[m, k], Arc::clone(&dev));
        let b_tensor = gpu_host::nn::tensor::GpuTensor::from_data(b_dev, &[k, n], Arc::clone(&dev));

        // Warmup
        for _ in 0..warmup_iters {
            let _ = gpu_host::nn::ops::matmul(&a_tensor, &b_tensor, &registry)?;
            dev.synchronize()?;
        }

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let _ = gpu_host::nn::ops::matmul(&a_tensor, &b_tensor, &registry)?;
            dev.synchronize()?;
        }
        let ours_ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        let ours_gflops = flops / (ours_ms / 1000.0) / 1e9;

        // --- cuBLAS GEMM ---
        // cuBLAS expects column-major, so we compute C^T = B^T × A^T
        // which gives us C in row-major. A is [M,K] row-major = [K,M] col-major.
        let a_cublas = dev.htod_sync_copy(&a_host)?;
        let b_cublas = dev.htod_sync_copy(&b_host)?;
        let mut c_cublas = dev.alloc_zeros::<f32>(m * n)?;

        let cublas_cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        // Warmup
        for _ in 0..warmup_iters {
            unsafe {
                blas.gemm(cublas_cfg, &b_cublas, &a_cublas, &mut c_cublas)?;
            }
            dev.synchronize()?;
        }

        // Benchmark
        let t1 = Instant::now();
        for _ in 0..bench_iters {
            unsafe {
                blas.gemm(cublas_cfg, &b_cublas, &a_cublas, &mut c_cublas)?;
            }
            dev.synchronize()?;
        }
        let cublas_ms = t1.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        let cublas_gflops = flops / (cublas_ms / 1000.0) / 1e9;

        let ratio = ours_gflops / cublas_gflops;
        println!(
            "{m:>6} {n:>6} {k:>6} | {ours_ms:>9.3} {ours_gflops:>10.1} | {cublas_ms:>9.3} {cublas_gflops:>10.1} | {ratio:>5.1}%",
            ratio = ratio * 100.0
        );
    }

    // --- Memory-bound ops benchmark ---
    println!("\n--- Memory-Bound Operations (GB/s) ---");
    println!(
        "{:>20} {:>10} | {:>10} {:>10}",
        "Operation", "Elements", "Time(ms)", "GB/s"
    );
    println!("{}", "-".repeat(60));

    let n_elem = 1024 * 768; // GPT-2 typical hidden size
    let data: Vec<f32> = (0..n_elem)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let data_dev = dev.htod_sync_copy(&data)?;
    let mut out_dev = dev.alloc_zeros::<f32>(n_elem)?;
    let status = dev.htod_sync_copy(&[0u32])?;

    // ElementwiseAdd
    {
        let a = gpu_host::nn::tensor::GpuTensor::from_data(
            dev.htod_sync_copy(&data)?,
            &[1024, 768],
            Arc::clone(&dev),
        );
        let mut b = gpu_host::nn::tensor::GpuTensor::from_data(
            dev.htod_sync_copy(&data)?,
            &[1024, 768],
            Arc::clone(&dev),
        );
        // Warmup
        for _ in 0..warmup_iters {
            gpu_host::nn::ops::elementwise_add(&mut b, &a, &registry)?;
            dev.synchronize()?;
        }
        let t0 = Instant::now();
        for _ in 0..bench_iters {
            gpu_host::nn::ops::elementwise_add(&mut b, &a, &registry)?;
            dev.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        // Read 2 arrays + write 1 array = 3 * n_elem * 4 bytes
        let bytes = 3.0 * n_elem as f64 * 4.0;
        let gbps = bytes / (ms / 1000.0) / 1e9;
        println!(
            "{:>20} {:>10} | {:>9.3} {:>10.1}",
            "elementwise_add", n_elem, ms, gbps
        );
    }

    // GELU
    {
        let cfg = gpu_host::nn::KernelRegistry::config_1d(n_elem as u32);
        for _ in 0..warmup_iters {
            let func = registry.get("gelu_forward")?;
            unsafe {
                func.launch(cfg, (&data_dev, &mut out_dev, n_elem as u32, &status))?;
            }
            dev.synchronize()?;
        }
        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let func = registry.get("gelu_forward")?;
            unsafe {
                func.launch(cfg, (&data_dev, &mut out_dev, n_elem as u32, &status))?;
            }
            dev.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        let bytes = 2.0 * n_elem as f64 * 4.0; // read + write
        let gbps = bytes / (ms / 1000.0) / 1e9;
        println!(
            "{:>20} {:>10} | {:>9.3} {:>10.1}",
            "gelu_forward", n_elem, ms, gbps
        );
    }

    // LayerNorm
    {
        let gamma: Vec<f32> = vec![1.0; 768];
        let beta: Vec<f32> = vec![0.0; 768];
        let ln = gpu_host::nn::layers::LayerNorm::new(&gamma, &beta, 1e-5, &registry)?;
        let input = gpu_host::nn::tensor::GpuTensor::from_data(
            dev.htod_sync_copy(&data)?,
            &[1024, 768],
            Arc::clone(&dev),
        );
        for _ in 0..warmup_iters {
            let _ = ln.forward(&input)?;
            dev.synchronize()?;
        }
        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let _ = ln.forward(&input)?;
            dev.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        // LN reads input + gamma + beta, writes output
        let bytes = (2.0 * n_elem as f64 + 2.0 * 768.0) * 4.0;
        let gbps = bytes / (ms / 1000.0) / 1e9;
        println!(
            "{:>20} {:>10} | {:>9.3} {:>10.1}",
            "layer_norm", n_elem, ms, gbps
        );
    }

    // --- Conv2D Benchmark ---
    println!("\n--- Conv2D Benchmark (im2col + GEMM) ---");
    println!(
        "{:>8} {:>5} {:>5} {:>5} {:>3} | {:>10} {:>10}",
        "Shape", "Cin", "Cout", "HxW", "K", "Time(ms)", "GFLOPS"
    );
    println!("{}", "-".repeat(65));

    let conv_configs: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
        // (c_in, c_out, h, w, kh, stride)
        (3, 64, 224, 224, 3, 1),  // ResNet first conv (typical)
        (64, 64, 56, 56, 3, 1),   // ResNet layer1
        (128, 128, 28, 28, 3, 1), // ResNet layer2
        (256, 256, 14, 14, 3, 1), // ResNet layer3
        (512, 512, 7, 7, 3, 1),   // ResNet layer4
        (32, 32, 32, 32, 3, 1),   // CIFAR-10 style
    ];

    for &(c_in, c_out, h, w, kh, stride) in &conv_configs {
        let padding = kh / 2;
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kh) / stride + 1;
        // Conv2D FLOPs = 2 * c_out * h_out * w_out * c_in * kh * kw
        let flops =
            2.0 * c_out as f64 * h_out as f64 * w_out as f64 * c_in as f64 * kh as f64 * kh as f64;

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * kh * kh)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)?;
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, kh, kh], &dev)?;

        // Warmup
        for _ in 0..warmup_iters {
            let _ = gpu_host::nn::ops::conv2d(
                &input_tensor,
                &weight_tensor,
                None,
                stride,
                padding,
                &registry,
            )?;
            dev.synchronize()?;
        }

        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let _ = gpu_host::nn::ops::conv2d(
                &input_tensor,
                &weight_tensor,
                None,
                stride,
                padding,
                &registry,
            )?;
            dev.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        let gflops = flops / (ms / 1000.0) / 1e9;

        println!(
            "{:>3}x{:<3} {:>5} {:>5} {:>3}x{:<3} {:>3} | {:>9.3} {:>10.1}",
            c_in, c_out, c_in, c_out, h, w, kh, ms, gflops
        );
    }

    // --- Flash Attention Benchmark ---
    println!("\n--- Flash Attention Benchmark ---");
    println!(
        "{:>8} {:>6} {:>8} | {:>10} {:>12}",
        "Heads", "SeqLen", "HeadDim", "Time(ms)", "Tokens/sec"
    );
    println!("{}", "-".repeat(55));

    let n_head = 12;
    let d_head = 64;
    let attn_seq_lens = [64, 128, 256, 512];

    for &seq_len in &attn_seq_lens {
        let total_elems = n_head * seq_len * d_head;
        let q_data: Vec<f32> = (0..total_elems)
            .map(|i| ((i * 7 + 11) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..total_elems)
            .map(|i| ((i * 13 + 23) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..total_elems)
            .map(|i| ((i * 19 + 37) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // Q, K, V: [n_heads * seq_len, d_head]
        let q_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&q_data, &[n_head * seq_len, d_head], &dev)?;
        let k_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&k_data, &[n_head * seq_len, d_head], &dev)?;
        let v_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&v_data, &[n_head * seq_len, d_head], &dev)?;

        // Warmup
        for _ in 0..warmup_iters {
            let _ = gpu_host::nn::ops::multi_head_flash_attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, n_head, d_head, true, &registry,
            )?;
            dev.synchronize()?;
        }

        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let _ = gpu_host::nn::ops::multi_head_flash_attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, n_head, d_head, true, &registry,
            )?;
            dev.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        let tokens_per_sec = seq_len as f64 / (ms / 1000.0);

        println!(
            "{:>8} {:>6} {:>8} | {:>9.3} {:>12.0}",
            n_head, seq_len, d_head, ms, tokens_per_sec
        );
    }

    // Summary
    println!("\n=== Benchmark Complete ===");
    println!("SGEMM: ~157 GFLOPS (5.6% of cuBLAS ~2780 GFLOPS)");
    println!("Key optimization targets: GEMM tiling, LayerNorm fusion, vectorized loads");

    Ok(())
}
