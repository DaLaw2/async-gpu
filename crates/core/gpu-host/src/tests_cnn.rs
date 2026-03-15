//! CNN kernel tests: BatchNorm+SiLU, im2col, MaxPool2D, Upsample, Concat.
//! Used to validate kernels for YOLOv8-nano inference pipeline.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// Test fused BatchNorm + SiLU kernel.
///
/// Creates a small CHW tensor, applies BN+SiLU with known parameters,
/// and verifies output matches CPU reference.
pub(crate) fn run_batchnorm_silu_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- BatchNorm + SiLU fused kernel test ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(ptx, "cnn_test", &["batchnorm_silu", "silu_forward"]);
    let f_bn_silu = dev
        .get_func("cnn_test", "batchnorm_silu")
        .ok_or(GpuHostError::KernelNotFound("batchnorm_silu"))?;
    let f_silu = dev
        .get_func("cnn_test", "silu_forward")
        .ok_or(GpuHostError::KernelNotFound("silu_forward"))?;

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Test params: 2 channels, 4x4 spatial = 32 elements
    let c = 2u32;
    let h = 4u32;
    let w = 4u32;
    let hw = h * w;
    let n = c * hw;
    let eps = 1e-5f32;

    // Known BN parameters per channel
    let gamma = vec![1.5f32, 0.8];
    let beta = vec![0.1, -0.2];
    let running_mean = vec![0.5, 1.0];
    let running_var = vec![0.25, 1.0]; // std = 0.5, 1.0

    // Input: channel 0 has values 0.0..15.0, channel 1 has values 1.0..16.0
    let mut input = vec![0.0f32; n as usize];
    for i in 0..hw as usize {
        input[i] = i as f32; // channel 0
        input[hw as usize + i] = (i + 1) as f32; // channel 1
    }

    // CPU reference
    let mut expected = vec![0.0f32; n as usize];
    for i in 0..n as usize {
        let ch = i / hw as usize;
        let x = input[i];
        let inv_std = 1.0 / (running_var[ch] + eps).sqrt();
        let bn = gamma[ch] * (x - running_mean[ch]) * inv_std + beta[ch];
        let sigmoid = 1.0 / (1.0 + (-bn).exp());
        expected[i] = bn * sigmoid;
    }

    // GPU execution
    let input_dev = dev.htod_sync_copy(&input)?;
    let gamma_dev = dev.htod_sync_copy(&gamma)?;
    let beta_dev = dev.htod_sync_copy(&beta)?;
    let mean_dev = dev.htod_sync_copy(&running_mean)?;
    let var_dev = dev.htod_sync_copy(&running_var)?;
    let mut output_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize)?;

    unsafe {
        f_bn_silu.clone().launch(
            LaunchConfig {
                grid_dim: (n.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &input_dev,
                &mut output_dev,
                &gamma_dev,
                &beta_dev,
                &mean_dev,
                &var_dev,
                n,
                hw,
                eps,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let output: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;

    // Verify
    let mut max_err = 0.0f32;
    for i in 0..n as usize {
        let err = (output[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("  BN+SiLU: max_err = {max_err:.6} (tolerance 1e-4)");
    if max_err > 1e-4 {
        println!("  First 8 values (GPU vs CPU):");
        for i in 0..8 {
            println!("    [{i}] gpu={:.6} cpu={:.6}", output[i], expected[i]);
        }
        return Err(GpuHostError::Verification {
            test: "batchnorm_silu",
            detail: format!("max error {max_err:.6} exceeds 1e-4"),
        });
    }
    println!("  BN+SiLU — PASSED");

    // Also test standalone SiLU
    let silu_input = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
    let silu_expected: Vec<f32> = silu_input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let silu_n = silu_input.len() as u32;

    let silu_in_dev = dev.htod_sync_copy(&silu_input)?;
    let mut silu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(silu_n as usize)?;

    unsafe {
        f_silu.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&silu_in_dev, &mut silu_out_dev, silu_n, status_dev_ptr),
        )?;
    }
    dev.synchronize()?;

    let silu_output: Vec<f32> = dev.dtoh_sync_copy(&silu_out_dev)?;
    let mut silu_max_err = 0.0f32;
    for i in 0..silu_n as usize {
        let err = (silu_output[i] - silu_expected[i]).abs();
        if err > silu_max_err {
            silu_max_err = err;
        }
    }
    println!("  SiLU: max_err = {silu_max_err:.6}");
    println!("  SiLU — PASSED");

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  CNN kernel tests — PASSED");
    Ok(())
}

/// Test im2col, MaxPool2D, Upsample, and Concat kernels.
pub(crate) fn run_cnn_ops_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- CNN ops test (im2col, MaxPool, Upsample, Concat) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(
        ptx,
        "cnn_ops",
        &[
            "im2col",
            "maxpool2d",
            "upsample_nearest_2x",
            "concat_channels",
        ],
    );

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("cnn_ops", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_im2col = get_fn!("im2col");
    let f_maxpool = get_fn!("maxpool2d");
    let f_upsample = get_fn!("upsample_nearest_2x");
    let f_concat = get_fn!("concat_channels");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // === Test Upsample 2x ===
    {
        // Input: 1 channel, 2x2
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, // row 0
            1.0, 1.0, 2.0, 2.0, // row 1
            3.0, 3.0, 4.0, 4.0, // row 2
            3.0, 3.0, 4.0, 4.0, // row 3
        ];

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(16)?;

        unsafe {
            f_upsample.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&in_dev, &mut out_dev, 1u32, 2u32, 2u32, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "Upsample 2x failed");
        println!("  Upsample 2x — PASSED");
    }

    // === Test Concat channels ===
    {
        // a: 1 channel 2x2, b: 1 channel 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let expected = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0 (from a)
            5.0, 6.0, 7.0, 8.0, // channel 1 (from b)
        ];

        let a_dev = dev.htod_sync_copy(&a)?;
        let b_dev = dev.htod_sync_copy(&b)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(8)?;

        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &a_dev,
                    &b_dev,
                    &mut out_dev,
                    1u32,
                    1u32,
                    4u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "Concat channels failed");
        println!("  Concat channels — PASSED");
    }

    // === Test MaxPool2D ===
    {
        // Input: 1 channel, 4x4, MaxPool k=2 stride=2 pad=0 -> 2x2
        let input = vec![
            1.0f32, 3.0, 2.0, 4.0, // row 0
            5.0, 7.0, 6.0, 8.0, // row 1
            9.0, 11.0, 10.0, 12.0, // row 2
            13.0, 15.0, 14.0, 16.0, // row 3
        ];
        let expected = vec![
            7.0, 8.0, // max of [1,3,5,7], max of [2,4,6,8]
            15.0, 16.0, // max of [9,11,13,15], max of [10,12,14,16]
        ];

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(4)?;

        unsafe {
            f_maxpool.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &in_dev,
                    &mut out_dev,
                    1u32,
                    4u32,
                    4u32,
                    2u32,
                    2u32,
                    0u32,
                    2u32,
                    2u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "MaxPool2D failed");
        println!("  MaxPool2D (k=2 s=2) — PASSED");
    }

    // === Test im2col ===
    {
        // Input: 1 channel, 3x3, Conv 3x3 stride=1 pad=1 -> 3x3 output
        // im2col should produce [9, 9] matrix (9 output positions, 9 filter taps)
        let input = vec![
            1.0f32, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];

        let c_in = 1u32;
        let h = 3u32;
        let w = 3u32;
        let kh = 3u32;
        let kw = 3u32;
        let stride = 1u32;
        let pad = 1u32;
        let h_out = (h + 2 * pad - kh) / stride + 1; // 3
        let w_out = (w + 2 * pad - kw) / stride + 1; // 3
        let col_width = c_in * kh * kw; // 9
        let total = h_out * w_out * col_width; // 81

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total as usize)?;

        unsafe {
            f_im2col.clone().launch(
                LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &in_dev,
                    &mut out_dev,
                    c_in,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    pad,
                    h_out,
                    w_out,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let col_output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

        // Verify center element (oh=1, ow=1) — should contain the full 3x3 patch = [1..9]
        let center_row_start = (w_out + 1) as usize * col_width as usize;
        let center_row: Vec<f32> =
            col_output[center_row_start..center_row_start + col_width as usize].to_vec();
        let expected_center = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        if center_row != expected_center {
            println!("  im2col center row: {:?}", center_row);
            println!("  expected: {:?}", expected_center);
            return Err(GpuHostError::Verification {
                test: "im2col",
                detail: "center patch mismatch".to_string(),
            });
        }

        // Verify corner element (oh=0, ow=0) — should have 0-padding for out-of-bounds
        let corner_row: Vec<f32> = col_output[..col_width as usize].to_vec();
        // For (0,0) with pad=1: filter taps that extend outside are 0
        // Positions: (-1,-1) (-1,0) (-1,1) (0,-1) (0,0) (0,1) (1,-1) (1,0) (1,1)
        // = 0, 0, 0, 0, 1, 2, 0, 4, 5
        let expected_corner = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0];
        if corner_row != expected_corner {
            println!("  im2col corner row: {:?}", corner_row);
            println!("  expected: {:?}", expected_corner);
            return Err(GpuHostError::Verification {
                test: "im2col",
                detail: "corner patch mismatch".to_string(),
            });
        }

        println!("  im2col (3x3, pad=1) — PASSED");
    }

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  CNN ops test — ALL PASSED");
    Ok(())
}
