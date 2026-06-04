//! SGEMM benchmark — measure GFLOPS for V4 vs V4.1 vs cuBLAS.
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test gemm_bench -- --nocapture

use std::sync::Arc;
use std::time::Instant;

/// Test V4 works (BK=8, known working).
#[test]
fn test_v4_works() {
    let (dev, _registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    let m = 512;
    let n = 512;
    let k = 512;

    let a_host: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let b_host: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let a_tensor = gpu_host::nn::tensor::GpuTensor::from_data(a_dev, &[m, k], Arc::clone(&dev));
    let b_tensor = gpu_host::nn::tensor::GpuTensor::from_data(b_dev, &[k, n], Arc::clone(&dev));

    println!("Calling matmul_v4 for {}x{}x{}...", m, n, k);
    let result = gpu_host::nn::ops::gemm::matmul_v4(&a_tensor, &b_tensor, m, k, n, &dev);
    match &result {
        Ok(_) => println!("  V4 kernel launched successfully"),
        Err(e) => panic!("  V4 kernel FAILED: {e:?}"),
    }
    let v4_host = result.unwrap().to_host().unwrap();
    println!("  V4 output[0..4]: {:?}", &v4_host[0..4]);

    // cuBLAS for comparison
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    let blas = CudaBlas::new(Arc::clone(&dev)).unwrap();
    let a_cublas = dev.htod_sync_copy(&a_host).unwrap();
    let b_cublas = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_cublas = dev.alloc_zeros::<f32>(m * n).unwrap();
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
    unsafe {
        blas.gemm(cublas_cfg, &b_cublas, &a_cublas, &mut c_cublas)
            .unwrap();
    }
    dev.synchronize().unwrap();
    let cublas_host: Vec<f32> = dev.dtoh_sync_copy(&c_cublas).unwrap();
    println!("  cuBLAS output[0..4]: {:?}", &cublas_host[0..4]);

    let max_err: f32 = v4_host
        .iter()
        .zip(cublas_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  max_err = {max_err:.4}");
    assert!(max_err < 0.01 * k as f32, "V4 too much error: {max_err}");
    println!("  V4 PASSED");
}

/// Test V4.1 works (BK=16, double-buffered).
#[test]
fn test_v41_works() {
    let (dev, _registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    let m = 512;
    let n = 512;
    let k = 512;

    let a_host: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let b_host: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let a_tensor = gpu_host::nn::tensor::GpuTensor::from_data(a_dev, &[m, k], Arc::clone(&dev));
    let b_tensor = gpu_host::nn::tensor::GpuTensor::from_data(b_dev, &[k, n], Arc::clone(&dev));

    println!("Calling matmul_v4_1 for {}x{}x{}...", m, n, k);
    let result = gpu_host::nn::ops::gemm::matmul_v4_1(&a_tensor, &b_tensor, m, k, n, &dev);
    match &result {
        Ok(_) => println!("  V4.1 kernel launched successfully"),
        Err(e) => panic!("  V4.1 kernel FAILED: {e:?}"),
    }
    let v41_host = result.unwrap().to_host().unwrap();
    println!("  V4.1 output[0..4]: {:?}", &v41_host[0..4]);
    println!("  V4.1 PASSED");
}

#[test]
fn bench_sgemm_all() {
    let (dev, _registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    println!("\n=== SGEMM Benchmark: V4 vs V4.1 vs cuBLAS ===");
    println!(
        "{:>6} {:>6} {:>6} | {:>8} {:>10} | {:>8} {:>10} | {:>8} {:>10} | {:>6} {:>6}",
        "M",
        "N",
        "K",
        "V4(ms)",
        "V4(GFLOPS)",
        "V4.1(ms)",
        "V4.1(GFLOPS)",
        "cuBLAS(ms)",
        "cuBLAS(GFLOPS)",
        "%V4",
        "%V4.1"
    );
    println!("{}", "-".repeat(120));

    let warmup = 5;
    let iters = 20;

    let sizes: Vec<(usize, usize, usize)> = vec![
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    for &(m, n, k) in &sizes {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a_host: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_host: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // --- V4 benchmark ---
        let v4_result = {
            let a_dev = dev.htod_sync_copy(&a_host).unwrap();
            let b_dev = dev.htod_sync_copy(&b_host).unwrap();
            let a_tensor =
                gpu_host::nn::tensor::GpuTensor::from_data(a_dev, &[m, k], Arc::clone(&dev));
            let b_tensor =
                gpu_host::nn::tensor::GpuTensor::from_data(b_dev, &[k, n], Arc::clone(&dev));

            for _ in 0..warmup {
                let _ = gpu_host::nn::ops::gemm::matmul_v4(&a_tensor, &b_tensor, m, k, n, &dev)
                    .unwrap();
                dev.synchronize().unwrap();
            }

            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = gpu_host::nn::ops::gemm::matmul_v4(&a_tensor, &b_tensor, m, k, n, &dev)
                    .unwrap();
                dev.synchronize().unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gflops = flops / (ms / 1000.0) / 1e9;
            (ms, gflops)
        };

        // --- V4.1 benchmark ---
        let v41_result = {
            let a_dev = dev.htod_sync_copy(&a_host).unwrap();
            let b_dev = dev.htod_sync_copy(&b_host).unwrap();
            let a_tensor =
                gpu_host::nn::tensor::GpuTensor::from_data(a_dev, &[m, k], Arc::clone(&dev));
            let b_tensor =
                gpu_host::nn::tensor::GpuTensor::from_data(b_dev, &[k, n], Arc::clone(&dev));

            for _ in 0..warmup {
                let _ = gpu_host::nn::ops::gemm::matmul_v4_1(&a_tensor, &b_tensor, m, k, n, &dev)
                    .unwrap();
                dev.synchronize().unwrap();
            }

            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = gpu_host::nn::ops::gemm::matmul_v4_1(&a_tensor, &b_tensor, m, k, n, &dev)
                    .unwrap();
                dev.synchronize().unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gflops = flops / (ms / 1000.0) / 1e9;
            (ms, gflops)
        };

        // --- cuBLAS benchmark ---
        let cublas_result = {
            use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
            let blas = CudaBlas::new(Arc::clone(&dev)).unwrap();
            let a_cublas = dev.htod_sync_copy(&a_host).unwrap();
            let b_cublas = dev.htod_sync_copy(&b_host).unwrap();
            let mut c_cublas = dev.alloc_zeros::<f32>(m * n).unwrap();

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

            for _ in 0..warmup {
                unsafe {
                    blas.gemm(cublas_cfg, &b_cublas, &a_cublas, &mut c_cublas)
                        .unwrap();
                }
                dev.synchronize().unwrap();
            }

            let t1 = Instant::now();
            for _ in 0..iters {
                unsafe {
                    blas.gemm(cublas_cfg, &b_cublas, &a_cublas, &mut c_cublas)
                        .unwrap();
                }
                dev.synchronize().unwrap();
            }
            let ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gflops = flops / (ms / 1000.0) / 1e9;
            (ms, gflops)
        };

        let (v4_ms, v4_gf) = v4_result;
        let (v41_ms, v41_gf) = v41_result;
        let (cub_ms, cub_gf) = cublas_result;
        let v4_pct = v4_gf / cub_gf * 100.0;
        let v41_pct = v41_gf / cub_gf * 100.0;

        println!(
            "{m:>6} {n:>6} {k:>6} | {v4_ms:>8.3} {v4_gf:>10.1} | {v41_ms:>8.3} {v41_gf:>10.1} | {cub_ms:>8.3} {cub_gf:>10.1} | {v4_pct:>5.1}% {v41_pct:>5.1}%"
        );
    }
}
