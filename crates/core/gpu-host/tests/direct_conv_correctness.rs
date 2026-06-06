//! Correctness test for direct_conv2d_warp_reduce kernel.
//!
//! Verifies against CPU f64 reference for 5x5 and 7x7 kernel sizes
//! with various C_in values to exercise the warp-level reduction.
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test direct_conv_correctness -- --nocapture

#[test]
fn test_direct_conv_warp_reduce_5x5() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    // 5x5 kernel, C_in=32 (forces warp-reduce path: C_in >= CI_WARPS=8)
    let configs: Vec<(usize, usize, usize, usize, usize, usize, &str)> = vec![
        (8, 16, 28, 28, 5, 1, "5x5 s1 Cin=8"),
        (16, 32, 28, 28, 5, 1, "5x5 s1 Cin=16"),
        (32, 64, 14, 14, 5, 1, "5x5 s1 Cin=32"),
        (64, 128, 14, 14, 5, 2, "5x5 s2 Cin=64"),
        (3, 32, 56, 56, 5, 2, "5x5 s2 Cin=3 (tiled path)"),
    ];

    for &(c_in, c_out, h, w, kh, stride, label) in &configs {
        let padding = kh / 2;

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * kh * kh)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let cpu_ref = gpu_host::nn::cpu_ref::cpu_conv2d_f64(
            &input_data,
            &weight_data,
            None,
            c_in,
            h,
            w,
            c_out,
            kh,
            kh,
            stride,
            padding,
        );

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)
                .expect("input");
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, kh, kh], &dev)
                .expect("weight");

        let gpu_out = gpu_host::nn::ops::conv2d(
            &input_tensor,
            &weight_tensor,
            None,
            stride,
            padding,
            &registry,
        )
        .expect("conv2d");
        let gpu_host = gpu_out.to_host().expect("to_host");

        assert_eq!(
            cpu_ref.len(),
            gpu_host.len(),
            "{}: output length mismatch",
            label
        );

        let max_err = cpu_ref
            .iter()
            .zip(gpu_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = cpu_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_val > 0.0 {
            max_err / max_val
        } else {
            max_err
        };

        println!(
            "{}: max_abs_err={:.6}, max_val={:.2}, rel_err={:.6}",
            label, max_err, max_val, rel_err
        );

        assert!(
            rel_err < 1e-3,
            "{}: relative error too large: {:.6}",
            label,
            rel_err
        );
    }
}

#[test]
fn test_direct_conv_warp_reduce_7x7() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    let configs: Vec<(usize, usize, usize, usize, usize, usize, &str)> = vec![
        (3, 64, 224, 224, 7, 2, "7x7 s2 ResNet stem"),
        (8, 32, 56, 56, 7, 1, "7x7 s1 Cin=8"),
        (32, 64, 28, 28, 7, 1, "7x7 s1 Cin=32"),
        (16, 32, 28, 28, 7, 2, "7x7 s2 Cin=16"),
    ];

    for &(c_in, c_out, h, w, kh, stride, label) in &configs {
        let padding = kh / 2;

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * kh * kh)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let cpu_ref = gpu_host::nn::cpu_ref::cpu_conv2d_f64(
            &input_data,
            &weight_data,
            None,
            c_in,
            h,
            w,
            c_out,
            kh,
            kh,
            stride,
            padding,
        );

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)
                .expect("input");
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, kh, kh], &dev)
                .expect("weight");

        let gpu_out = gpu_host::nn::ops::conv2d(
            &input_tensor,
            &weight_tensor,
            None,
            stride,
            padding,
            &registry,
        )
        .expect("conv2d");
        let gpu_host_data = gpu_out.to_host().expect("to_host");

        assert_eq!(
            cpu_ref.len(),
            gpu_host_data.len(),
            "{}: output length mismatch",
            label
        );

        let max_err = cpu_ref
            .iter()
            .zip(gpu_host_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = cpu_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_val > 0.0 {
            max_err / max_val
        } else {
            max_err
        };

        println!(
            "{}: max_abs_err={:.6}, max_val={:.2}, rel_err={:.6}",
            label, max_err, max_val, rel_err
        );

        assert!(
            rel_err < 1e-3,
            "{}: relative error too large: {:.6}",
            label,
            rel_err
        );
    }
}

#[test]
fn test_direct_conv_warp_reduce_3x3_stride2() {
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default().expect("init");

    // 3x3 stride=2 goes through direct conv path (not Winograd which requires stride=1)
    let configs: Vec<(usize, usize, usize, usize, usize, usize, &str)> = vec![
        (16, 32, 320, 320, 3, 2, "3x3 s2 Cin=16"),
        (32, 64, 160, 160, 3, 2, "3x3 s2 Cin=32"),
        (64, 128, 80, 80, 3, 2, "3x3 s2 Cin=64"),
    ];

    for &(c_in, c_out, h, w, kh, stride, label) in &configs {
        let padding = kh / 2;

        let input_data: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..c_out * c_in * kh * kh)
            .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let cpu_ref = gpu_host::nn::cpu_ref::cpu_conv2d_f64(
            &input_data,
            &weight_data,
            None,
            c_in,
            h,
            w,
            c_out,
            kh,
            kh,
            stride,
            padding,
        );

        let input_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&input_data, &[c_in, h, w], &dev)
                .expect("input");
        let weight_tensor =
            gpu_host::nn::tensor::GpuTensor::from_host(&weight_data, &[c_out, c_in, kh, kh], &dev)
                .expect("weight");

        let gpu_out = gpu_host::nn::ops::conv2d(
            &input_tensor,
            &weight_tensor,
            None,
            stride,
            padding,
            &registry,
        )
        .expect("conv2d");
        let gpu_host_data = gpu_out.to_host().expect("to_host");

        assert_eq!(
            cpu_ref.len(),
            gpu_host_data.len(),
            "{}: output length mismatch",
            label
        );

        let max_err = cpu_ref
            .iter()
            .zip(gpu_host_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = cpu_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_val > 0.0 {
            max_err / max_val
        } else {
            max_err
        };

        println!(
            "{}: max_abs_err={:.6}, max_val={:.2}, rel_err={:.6}",
            label, max_err, max_val, rel_err
        );

        assert!(
            rel_err < 1e-3,
            "{}: relative error too large: {:.6}",
            label,
            rel_err
        );
    }
}
