//! Winograd F(2×2, 3×3) correctness test.
//!
//! Uses NVRTC to compile the Winograd CUDA kernel directly, avoiding the
//! slow PTX JIT of the 254K-line main kernel.
//!
//! Run with: cargo test -p gpu-host --features nn,cublas --test winograd_test -- --nocapture

#[cfg(feature = "cublas")]
#[test]
fn winograd_f2x2_correctness() {
    let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");

    // Compile Winograd kernels via NVRTC
    let src = include_str!("../src/nn/ops/winograd_f2x2.cu");
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"),
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts).expect("NVRTC compile");
    dev.load_ptx(
        ptx,
        "winograd_f2x2",
        &["winograd_filter_transform", "winograd_conv2d_f2x2"],
    )
    .expect("PTX load");

    // Test cases: (c_in, c_out, h, w, padding, filter_desc)
    let tests = vec![
        // Single channel identity filter, no padding
        (1, 1, 5, 5, 0, "identity"),
        // Single channel averaging filter, no padding
        (1, 1, 5, 5, 0, "averaging"),
        // Single channel averaging filter, padding=1
        (1, 1, 5, 5, 1, "averaging+pad1"),
        // Multi-channel
        (3, 4, 8, 8, 1, "multichannel"),
        // Larger spatial
        (3, 8, 32, 32, 1, "cifar10"),
        // Odd output size (h_out=4, only 2 full tiles)
        (1, 1, 6, 6, 0, "6x6"),
    ];

    for (c_in, c_out, h, w, padding, desc) in tests {
        eprint!(
            "  test {} (c_in={}, c_out={}, {}x{}, pad={}) ... ",
            desc, c_in, c_out, h, w, padding
        );

        let h_out = h + 2 * padding - 2; // stride=1, kh=3
        let w_out = w + 2 * padding - 2;
        let n_tile_y = (h_out + 1) / 2;
        let n_tile_x = (w_out + 1) / 2;
        let total_tiles = n_tile_x * n_tile_y;

        // Generate input
        let input: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // Generate filter
        let filter: Vec<f32> = match desc {
            "identity" => {
                let mut f = vec![0.0f32; c_out * c_in * 9];
                for co in 0..c_out.min(c_in) {
                    f[co * c_in * 9 + co * 9 + 4] = 1.0; // center
                }
                f
            }
            d if d.starts_with("averaging") => {
                vec![1.0 / (9 * c_in) as f32; c_out * c_in * 9]
            }
            _ => (0..c_out * c_in * 9)
                .map(|i| ((i * 13 + 47) % 1000) as f32 / 1000.0 - 0.5)
                .collect(),
        };

        // Filter transform on GPU
        let filter_plane = c_out * c_in;
        let filter_dev = dev.htod_sync_copy(&filter).expect("filter upload");
        let mut filter_wino = dev.alloc_zeros::<f32>(16 * filter_plane).expect("alloc");

        let ft_func = dev
            .get_func("winograd_f2x2", "winograd_filter_transform")
            .unwrap();
        let ft_config = cudarc::driver::LaunchConfig {
            grid_dim: ((filter_plane as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            cudarc::driver::LaunchAsync::launch(
                ft_func,
                ft_config,
                (&filter_dev, &mut filter_wino, c_out as u32, c_in as u32),
            )
            .expect("filter transform");
        }

        // Winograd conv on GPU
        let input_dev = dev.htod_sync_copy(&input).expect("input upload");
        let mut output_dev = dev
            .alloc_zeros::<f32>(c_out * h_out * w_out)
            .expect("alloc");

        let tile_c_out: u32 = 32;
        let conv_func = dev
            .get_func("winograd_f2x2", "winograd_conv2d_f2x2")
            .unwrap();
        let conv_config = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles as u32, (c_out as u32).div_ceil(tile_c_out), 1),
            block_dim: (tile_c_out, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            cudarc::driver::LaunchAsync::launch(
                conv_func,
                conv_config,
                (
                    &input_dev,
                    &filter_wino,
                    &mut output_dev,
                    c_in as u32,
                    c_out as u32,
                    h as u32,
                    w as u32,
                    h_out as u32,
                    w_out as u32,
                    n_tile_x as u32,
                    n_tile_y as u32,
                    padding as u32,
                ),
            )
            .expect("winograd conv");
        }

        let gpu_output = dev.dtoh_sync_copy(&output_dev).expect("download");

        // CPU reference
        let expected = cpu_conv2d(&input, &filter, c_in, h, w, c_out, padding);

        // Compare
        let max_err: f32 = gpu_output
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);

        if max_err > 0.01 {
            eprintln!("FAIL (max_err={max_err})");
            if gpu_output.len() <= 25 {
                eprintln!("    GPU:      {:?}", gpu_output);
                eprintln!("    Expected: {:?}", expected);
            }
        } else {
            eprintln!("ok (max_err={max_err:.6})");
        }
        assert!(
            max_err < 0.01,
            "{}: Winograd max_err={max_err} exceeds threshold",
            desc
        );
    }

    eprintln!("\nAll Winograd F(2×2,3×3) GPU tests passed!");
}

/// CPU reference conv2d: input [C_in, H, W], filter [C_out, C_in, 3, 3], stride=1.
#[cfg(feature = "cublas")]
fn cpu_conv2d(
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
