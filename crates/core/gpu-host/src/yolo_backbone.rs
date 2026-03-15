//! YOLOv8-nano backbone inference on GPU.
//!
//! Implements the full YOLOv8-nano forward pass (layers 0-21) using GPU kernels:
//! im2col + gemm_f32 for Conv2D, batchnorm_silu for BN+SiLU, maxpool2d,
//! upsample_nearest_2x, concat_channels.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};
use crate::model_yolo::{ConvBnSiluWeights, ConvWeights, YoloWeights, NUM_CLASSES, REG_MAX};

/// GPU tensor: device memory + shape metadata.
pub struct GpuTensor {
    /// Device memory holding f32 data.
    pub data: CudaSlice<f32>,
    /// Number of channels.
    pub c: u32,
    /// Height.
    pub h: u32,
    /// Width.
    pub w: u32,
}

impl GpuTensor {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        (self.c * self.h * self.w) as usize
    }
}

/// Holds GPU kernel functions and device reference for running YOLO inference.
pub struct YoloRunner {
    dev: Arc<CudaDevice>,
    f_im2col: CudaFunction,
    f_gemm: CudaFunction,
    f_bn_silu: CudaFunction,
    f_maxpool: CudaFunction,
    f_upsample: CudaFunction,
    f_concat: CudaFunction,
    f_sigmoid: CudaFunction,
    f_bias_add: CudaFunction,
    status_host_ptr: *mut u32,
    status_dev_ptr: CUdeviceptr,
}

impl YoloRunner {
    /// Create a new YoloRunner by loading kernel functions from PTX.
    pub fn new(dev: Arc<CudaDevice>, ptx_src: &'static str) -> Result<Self> {
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        let _ = dev.load_ptx(
            ptx,
            "yolo",
            &[
                "im2col",
                "gemm_f32",
                "batchnorm_silu",
                "maxpool2d",
                "upsample_nearest_2x",
                "concat_channels",
                "sigmoid_forward",
                "bias_add_chw",
            ],
        );

        macro_rules! get_fn {
            ($name:expr) => {
                dev.get_func("yolo", $name)
                    .ok_or(GpuHostError::KernelNotFound($name))?
            };
        }

        let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

        let f_im2col = get_fn!("im2col");
        let f_gemm = get_fn!("gemm_f32");
        let f_bn_silu = get_fn!("batchnorm_silu");
        let f_maxpool = get_fn!("maxpool2d");
        let f_upsample = get_fn!("upsample_nearest_2x");
        let f_concat = get_fn!("concat_channels");
        let f_sigmoid = get_fn!("sigmoid_forward");
        let f_bias_add = get_fn!("bias_add_chw");

        Ok(Self {
            dev,
            f_im2col,
            f_gemm,
            f_bn_silu,
            f_maxpool,
            f_upsample,
            f_concat,
            f_sigmoid,
            f_bias_add,
            status_host_ptr,
            status_dev_ptr,
        })
    }

    /// Upload f32 data to GPU.
    pub fn upload(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        Ok(self.dev.htod_sync_copy(data)?)
    }

    /// Download f32 data from GPU.
    pub fn download(&self, data: &CudaSlice<f32>) -> Result<Vec<f32>> {
        Ok(self.dev.dtoh_sync_copy(data)?)
    }

    /// Allocate zeroed GPU memory.
    pub fn alloc_zeros(&self, n: usize) -> Result<CudaSlice<f32>> {
        Ok(self.dev.alloc_zeros::<f32>(n)?)
    }

    /// Synchronize device.
    pub fn sync(&self) -> Result<()> {
        Ok(self.dev.synchronize()?)
    }

    // ---------------------------------------------------------------
    // Primitive operations
    // ---------------------------------------------------------------

    /// Run Conv2D via im2col + GEMM.
    ///
    /// - `input`: [C_in, H, W] tensor on GPU
    /// - `weight`: [C_out, C_in*kH*kW] in column-major layout on GPU
    /// - Returns: [C_out, H_out, W_out] tensor on GPU
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        input: &GpuTensor,
        weight_cm: &CudaSlice<f32>,
        c_out: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
    ) -> Result<GpuTensor> {
        let c_in = input.c;
        let h = input.h;
        let w = input.w;
        let h_out = (h + 2 * pad - kh) / stride + 1;
        let w_out = (w + 2 * pad - kw) / stride + 1;

        let k_gemm = c_in * kh * kw;
        let m_gemm = h_out * w_out;
        let n_gemm = c_out;
        let n_padded = n_gemm.next_multiple_of(16);

        // Step 1: im2col
        let im2col_size = (m_gemm * k_gemm) as usize;
        let mut col_dev = self.alloc_zeros(im2col_size)?;

        unsafe {
            self.f_im2col.clone().launch(
                LaunchConfig {
                    grid_dim: ((im2col_size as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &input.data,
                    &mut col_dev,
                    c_in,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    pad,
                    h_out,
                    w_out,
                    self.status_dev_ptr,
                ),
            )?;
        }

        // Step 2: GEMM with N-padding
        let mut output_dev = self.alloc_zeros((m_gemm * n_padded) as usize)?;
        let num_blocks_m = m_gemm.div_ceil(32);
        let num_blocks_n = n_padded.div_ceil(16);
        let gemm_shared = (32 * 16 + 16 * 16) * 4;

        unsafe {
            self.f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks_m, num_blocks_n, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &col_dev,
                    weight_cm,
                    &mut output_dev,
                    k_gemm,
                    n_padded,
                    self.status_dev_ptr,
                ),
            )?;
        }

        self.sync()?;

        // Step 3: Reshape [M, N_padded] → [C_out, H_out, W_out]
        // If n_padded == n_gemm, output is already in the right layout (after transpose)
        // We need to transpose from [M, N_padded] row-major to [C_out, H_out*W_out]
        // i.e., [C_out, H_out, W_out] in CHW format
        if n_padded == n_gemm {
            // Output is [M, N] row-major = [HW, C_out]
            // We need [C_out, HW] = transpose
            let raw = self.download(&output_dev)?;
            let mut chw = vec![0.0f32; (c_out * h_out * w_out) as usize];
            for pos in 0..m_gemm as usize {
                for co in 0..n_gemm as usize {
                    chw[co * (h_out * w_out) as usize + pos] = raw[pos * n_gemm as usize + co];
                }
            }
            let chw_dev = self.upload(&chw)?;
            Ok(GpuTensor {
                data: chw_dev,
                c: c_out,
                h: h_out,
                w: w_out,
            })
        } else {
            // Need to extract valid columns and transpose
            let raw = self.download(&output_dev)?;
            let mut chw = vec![0.0f32; (c_out * h_out * w_out) as usize];
            for pos in 0..m_gemm as usize {
                for co in 0..n_gemm as usize {
                    chw[co * (h_out * w_out) as usize + pos] = raw[pos * n_padded as usize + co];
                }
            }
            let chw_dev = self.upload(&chw)?;
            Ok(GpuTensor {
                data: chw_dev,
                c: c_out,
                h: h_out,
                w: w_out,
            })
        }
    }

    /// Run fused BatchNorm + SiLU on a CHW tensor.
    pub fn bn_silu(
        &self,
        input: &GpuTensor,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        running_mean: &CudaSlice<f32>,
        running_var: &CudaSlice<f32>,
    ) -> Result<GpuTensor> {
        let n = input.numel() as u32;
        let hw = input.h * input.w;
        let mut output = self.alloc_zeros(input.numel())?;

        unsafe {
            self.f_bn_silu.clone().launch(
                LaunchConfig {
                    grid_dim: (n.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &input.data,
                    &mut output,
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                    n,
                    hw,
                    1e-5f32,
                    self.status_dev_ptr,
                ),
            )?;
        }

        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: input.c,
            h: input.h,
            w: input.w,
        })
    }

    /// Conv2D + BN + SiLU (the most common YOLO building block).
    ///
    /// `weight_cm` is column-major [K, C_out] where K = C_in*kH*kW.
    /// If c_out < 16, weight_cm must be padded to next_multiple_of(16) columns with zeros.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_bn_silu(
        &self,
        input: &GpuTensor,
        weight_cm: &CudaSlice<f32>,
        c_out: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        running_mean: &CudaSlice<f32>,
        running_var: &CudaSlice<f32>,
    ) -> Result<GpuTensor> {
        let conv_out = self.conv2d(input, weight_cm, c_out, kh, kw, stride, pad)?;
        self.bn_silu(&conv_out, gamma, beta, running_mean, running_var)
    }

    /// MaxPool2D.
    pub fn maxpool2d(&self, input: &GpuTensor, k: u32, stride: u32, pad: u32) -> Result<GpuTensor> {
        let h_out = (input.h + 2 * pad - k) / stride + 1;
        let w_out = (input.w + 2 * pad - k) / stride + 1;
        let out_size = (input.c * h_out * w_out) as usize;
        let mut output = self.alloc_zeros(out_size)?;

        unsafe {
            self.f_maxpool.clone().launch(
                LaunchConfig {
                    grid_dim: ((out_size as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &input.data,
                    &mut output,
                    input.c,
                    input.h,
                    input.w,
                    k,
                    stride,
                    pad,
                    h_out,
                    w_out,
                    self.status_dev_ptr,
                ),
            )?;
        }

        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: input.c,
            h: h_out,
            w: w_out,
        })
    }

    /// Upsample 2x nearest-neighbor.
    pub fn upsample_2x(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let h_out = input.h * 2;
        let w_out = input.w * 2;
        let out_size = (input.c * h_out * w_out) as usize;
        let mut output = self.alloc_zeros(out_size)?;

        unsafe {
            self.f_upsample.clone().launch(
                LaunchConfig {
                    grid_dim: ((out_size as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &input.data,
                    &mut output,
                    input.c,
                    input.h,
                    input.w,
                    self.status_dev_ptr,
                ),
            )?;
        }

        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: input.c,
            h: h_out,
            w: w_out,
        })
    }

    /// Concatenate two tensors along the channel dimension.
    pub fn concat(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        assert_eq!(a.h, b.h, "concat: height mismatch");
        assert_eq!(a.w, b.w, "concat: width mismatch");

        let c_out = a.c + b.c;
        let hw = a.h * a.w;
        let out_size = (c_out * hw) as usize;
        let mut output = self.alloc_zeros(out_size)?;

        unsafe {
            self.f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: ((out_size as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &a.data,
                    &b.data,
                    &mut output,
                    a.c,
                    b.c,
                    hw,
                    self.status_dev_ptr,
                ),
            )?;
        }

        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: c_out,
            h: a.h,
            w: a.w,
        })
    }

    /// Chunk-split a tensor along the channel dimension into two halves.
    pub fn chunk_split(&self, input: &GpuTensor) -> Result<(GpuTensor, GpuTensor)> {
        let half_c = input.c / 2;
        let hw = (input.h * input.w) as usize;
        let raw = self.download(&input.data)?;

        let mut first = vec![0.0f32; (half_c as usize) * hw];
        let mut second = vec![0.0f32; (half_c as usize) * hw];

        for c in 0..half_c as usize {
            first[c * hw..(c + 1) * hw].copy_from_slice(&raw[c * hw..(c + 1) * hw]);
        }
        for c in 0..half_c as usize {
            let src_c = c + half_c as usize;
            second[c * hw..(c + 1) * hw].copy_from_slice(&raw[src_c * hw..(src_c + 1) * hw]);
        }

        Ok((
            GpuTensor {
                data: self.upload(&first)?,
                c: half_c,
                h: input.h,
                w: input.w,
            },
            GpuTensor {
                data: self.upload(&second)?,
                c: half_c,
                h: input.h,
                w: input.w,
            },
        ))
    }

    /// Element-wise add two tensors of the same shape (residual connection).
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        assert_eq!(a.c, b.c);
        assert_eq!(a.h, b.h);
        assert_eq!(a.w, b.w);

        // CPU-side add for now (no dedicated add kernel)
        let a_data = self.download(&a.data)?;
        let b_data = self.download(&b.data)?;
        let sum: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();

        Ok(GpuTensor {
            data: self.upload(&sum)?,
            c: a.c,
            h: a.h,
            w: a.w,
        })
    }

    /// Element-wise sigmoid activation.
    pub fn sigmoid(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let n = input.numel() as u32;
        let mut output = self.alloc_zeros(input.numel())?;

        unsafe {
            self.f_sigmoid.clone().launch(
                LaunchConfig {
                    grid_dim: (n.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&input.data, &mut output, n, self.status_dev_ptr),
            )?;
        }
        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: input.c,
            h: input.h,
            w: input.w,
        })
    }

    /// Add per-channel bias to a CHW tensor.
    pub fn bias_add(&self, input: &GpuTensor, bias: &CudaSlice<f32>) -> Result<GpuTensor> {
        let n = input.numel() as u32;
        let hw = input.h * input.w;
        let mut output = self.alloc_zeros(input.numel())?;

        unsafe {
            self.f_bias_add.clone().launch(
                LaunchConfig {
                    grid_dim: (n.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&input.data, &mut output, bias, n, hw, self.status_dev_ptr),
            )?;
        }
        self.sync()?;

        Ok(GpuTensor {
            data: output,
            c: input.c,
            h: input.h,
            w: input.w,
        })
    }

    /// Conv2D + bias (no BN, no activation) — used in detect head final layers.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_bias(
        &self,
        input: &GpuTensor,
        weight_cm: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        c_out: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
    ) -> Result<GpuTensor> {
        let conv_out = self.conv2d(input, weight_cm, c_out, kh, kw, stride, pad)?;
        self.bias_add(&conv_out, bias)
    }

    /// Clean up mapped memory.
    pub fn cleanup(self) -> Result<()> {
        unsafe {
            free_mapped_mem(self.status_host_ptr)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // High-level inference helpers (upload weights on-the-fly)
    // ---------------------------------------------------------------

    /// Create a GpuTensor from host data.
    pub fn make_tensor(&self, data: &[f32], c: u32, h: u32, w: u32) -> Result<GpuTensor> {
        Ok(GpuTensor {
            data: self.upload(data)?,
            c,
            h,
            w,
        })
    }

    /// Run Conv+BN+SiLU from weight struct (uploads weights to GPU).
    #[allow(clippy::too_many_arguments)]
    fn run_conv_bn_silu_w(
        &self,
        input: &GpuTensor,
        w: &ConvBnSiluWeights,
        c_out: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
    ) -> Result<GpuTensor> {
        let k = w.conv_shape[1] * w.conv_shape[2] * w.conv_shape[3];
        let weight_cm = prepare_conv_weight(&w.conv_weight, c_out as usize, k);
        let wt = self.upload(&weight_cm)?;
        let g = self.upload(&w.bn_weight)?;
        let b = self.upload(&w.bn_bias)?;
        let m = self.upload(&w.bn_running_mean)?;
        let v = self.upload(&w.bn_running_var)?;
        self.conv_bn_silu(input, &wt, c_out, kh, kw, stride, pad, &g, &b, &m, &v)
    }

    /// Run bare Conv+bias from weight struct (uploads weights to GPU).
    #[allow(clippy::too_many_arguments)]
    fn run_conv_bias_w(
        &self,
        input: &GpuTensor,
        w: &ConvWeights,
        c_out: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
    ) -> Result<GpuTensor> {
        let k = w.shape[1] * w.shape[2] * w.shape[3];
        let weight_cm = prepare_conv_weight(&w.weight, c_out as usize, k);
        let wt = self.upload(&weight_cm)?;
        let b = self.upload(&w.bias)?;
        self.conv_bias(input, &wt, &b, c_out, kh, kw, stride, pad)
    }

    /// Run a C2f block.
    ///
    /// C2f structure: cv1 (1x1) → chunk split → N bottlenecks → concat all → cv2 (1x1)
    #[allow(clippy::too_many_arguments)]
    fn run_c2f(
        &self,
        input: &GpuTensor,
        weights: &YoloWeights,
        layer_idx: usize,
        c_out: u32,
        n_bottleneck: usize,
        shortcut: bool,
    ) -> Result<GpuTensor> {
        let hidden = c_out / 2;

        // cv1: Conv 1x1, c_in -> 2*hidden
        let cv1_w = weights
            .sub_conv_bn_silu(layer_idx, "cv1")
            .map_err(map_model_err)?;
        let cv1_out = self.run_conv_bn_silu_w(input, &cv1_w, 2 * hidden, 1, 1, 1, 0)?;

        // Chunk split into two halves
        let (branch_0, mut prev) = self.chunk_split(&cv1_out)?;

        // Collect all branches for final concat: [branch_0, branch_1, bn0_out, bn1_out, ...]
        let mut branches: Vec<GpuTensor> = vec![branch_0];
        // branch_1 (the second chunk) is the input to the first bottleneck
        // We also concat branch_1 itself
        let branch_1_for_concat = GpuTensor {
            data: self.upload(&self.download(&prev.data)?)?,
            c: prev.c,
            h: prev.h,
            w: prev.w,
        };
        branches.push(branch_1_for_concat);

        for j in 0..n_bottleneck {
            let bn_cv1 = weights
                .bottleneck_conv_bn_silu(layer_idx, j, "cv1")
                .map_err(map_model_err)?;
            let bn1 = self.run_conv_bn_silu_w(&prev, &bn_cv1, hidden, 3, 3, 1, 1)?;

            let bn_cv2 = weights
                .bottleneck_conv_bn_silu(layer_idx, j, "cv2")
                .map_err(map_model_err)?;
            let bn2 = self.run_conv_bn_silu_w(&bn1, &bn_cv2, hidden, 3, 3, 1, 1)?;

            let bn_out = if shortcut {
                self.add(&prev, &bn2)?
            } else {
                bn2
            };

            // Save for concat and as input to next bottleneck
            let bn_out_copy = GpuTensor {
                data: self.upload(&self.download(&bn_out.data)?)?,
                c: bn_out.c,
                h: bn_out.h,
                w: bn_out.w,
            };
            branches.push(bn_out_copy);
            prev = bn_out;
        }

        // Concat all branches along channel dimension
        let mut cat = branches.remove(0);
        for b in branches {
            cat = self.concat(&cat, &b)?;
        }

        // cv2: Conv 1x1, (2+n)*hidden -> c_out
        let cv2_w = weights
            .sub_conv_bn_silu(layer_idx, "cv2")
            .map_err(map_model_err)?;
        self.run_conv_bn_silu_w(&cat, &cv2_w, c_out, 1, 1, 1, 0)
    }

    /// Run SPPF block.
    ///
    /// SPPF: cv1 (1x1) → 3× MaxPool5x5 → concat [x, p1, p2, p3] → cv2 (1x1)
    fn run_sppf(
        &self,
        input: &GpuTensor,
        weights: &YoloWeights,
        layer_idx: usize,
        c_out: u32,
    ) -> Result<GpuTensor> {
        let c_hidden = input.c / 2;

        let cv1_w = weights
            .sub_conv_bn_silu(layer_idx, "cv1")
            .map_err(map_model_err)?;
        let x = self.run_conv_bn_silu_w(input, &cv1_w, c_hidden, 1, 1, 1, 0)?;

        let p1 = self.maxpool2d(&x, 5, 1, 2)?;
        let p2 = self.maxpool2d(&p1, 5, 1, 2)?;
        let p3 = self.maxpool2d(&p2, 5, 1, 2)?;

        let cat1 = self.concat(&x, &p1)?;
        let cat2 = self.concat(&cat1, &p2)?;
        let cat3 = self.concat(&cat2, &p3)?;

        let cv2_w = weights
            .sub_conv_bn_silu(layer_idx, "cv2")
            .map_err(map_model_err)?;
        self.run_conv_bn_silu_w(&cat3, &cv2_w, c_out, 1, 1, 1, 0)
    }

    /// Run detect head for one scale, returning (box_output, cls_output).
    ///
    /// box_output: [64, H, W] (4*reg_max channels)
    /// cls_output: [80, H, W] (num_classes channels, after sigmoid)
    fn run_detect_scale(
        &self,
        input: &GpuTensor,
        weights: &YoloWeights,
        scale: usize,
    ) -> Result<(GpuTensor, GpuTensor)> {
        // Box branch (cv2): Conv3x3+BN+SiLU → Conv3x3+BN+SiLU → Conv1x1(bare)
        let cv2_0 = weights
            .detect_conv_bn_silu("cv2", scale, 0)
            .map_err(map_model_err)?;
        let cv2_1 = weights
            .detect_conv_bn_silu("cv2", scale, 1)
            .map_err(map_model_err)?;
        let cv2_2 = weights
            .detect_conv("cv2", scale, 2)
            .map_err(map_model_err)?;

        let b0 = self.run_conv_bn_silu_w(input, &cv2_0, 64, 3, 3, 1, 1)?;
        let b1 = self.run_conv_bn_silu_w(&b0, &cv2_1, 64, 3, 3, 1, 1)?;
        let box_out = self.run_conv_bias_w(&b1, &cv2_2, 64, 1, 1, 1, 0)?;

        // Class branch (cv3): Conv3x3+BN+SiLU → Conv3x3+BN+SiLU → Conv1x1(bare) → sigmoid
        let cv3_0 = weights
            .detect_conv_bn_silu("cv3", scale, 0)
            .map_err(map_model_err)?;
        let cv3_1 = weights
            .detect_conv_bn_silu("cv3", scale, 1)
            .map_err(map_model_err)?;
        let cv3_2 = weights
            .detect_conv("cv3", scale, 2)
            .map_err(map_model_err)?;

        let c0 = self.run_conv_bn_silu_w(input, &cv3_0, 80, 3, 3, 1, 1)?;
        let c1 = self.run_conv_bn_silu_w(&c0, &cv3_1, 80, 3, 3, 1, 1)?;
        let cls_logits = self.run_conv_bias_w(&c1, &cv3_2, 80, 1, 1, 1, 0)?;
        let cls_out = self.sigmoid(&cls_logits)?;

        Ok((box_out, cls_out))
    }

    /// Run full YOLOv8-nano inference: backbone + neck + detect head + NMS.
    ///
    /// Input: CHW f32 data [3, 640, 640] normalized to [0, 1].
    /// Returns: list of detections (x1, y1, x2, y2 in pixel coords, class_id, confidence).
    #[allow(clippy::too_many_lines)]
    pub fn yolo_inference(
        &self,
        weights: &YoloWeights,
        input_data: &[f32],
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let input = self.make_tensor(input_data, 3, 640, 640)?;
        println!("  Running YOLOv8-nano inference (640x640)...");

        // === Backbone ===
        // L0: Conv 3x3 s2, 3->16, out 320x320
        let w0 = weights.conv_bn_silu(0).map_err(map_model_err)?;
        let l0 = self.run_conv_bn_silu_w(&input, &w0, 16, 3, 3, 2, 1)?;
        println!("    L0:  Conv 3->16 s2 → {}x{}x{}", l0.c, l0.h, l0.w);

        // L1: Conv 3x3 s2, 16->32, out 160x160
        let w1 = weights.conv_bn_silu(1).map_err(map_model_err)?;
        let l1 = self.run_conv_bn_silu_w(&l0, &w1, 32, 3, 3, 2, 1)?;
        println!("    L1:  Conv 16->32 s2 → {}x{}x{}", l1.c, l1.h, l1.w);

        // L2: C2f 32->32, 1 bottleneck, shortcut=true
        let l2 = self.run_c2f(&l1, weights, 2, 32, 1, true)?;
        println!("    L2:  C2f 32->32 → {}x{}x{}", l2.c, l2.h, l2.w);

        // L3: Conv 3x3 s2, 32->64, out 80x80
        let w3 = weights.conv_bn_silu(3).map_err(map_model_err)?;
        let l3 = self.run_conv_bn_silu_w(&l2, &w3, 64, 3, 3, 2, 1)?;
        println!("    L3:  Conv 32->64 s2 → {}x{}x{}", l3.c, l3.h, l3.w);

        // L4: C2f 64->64, 2 bottlenecks, shortcut=true  (*** KEEP for skip ***)
        let l4 = self.run_c2f(&l3, weights, 4, 64, 2, true)?;
        println!("    L4:  C2f 64->64 → {}x{}x{}", l4.c, l4.h, l4.w);

        // L5: Conv 3x3 s2, 64->128, out 40x40
        let w5 = weights.conv_bn_silu(5).map_err(map_model_err)?;
        let l5 = self.run_conv_bn_silu_w(&l4, &w5, 128, 3, 3, 2, 1)?;
        println!("    L5:  Conv 64->128 s2 → {}x{}x{}", l5.c, l5.h, l5.w);

        // L6: C2f 128->128, 2 bottlenecks, shortcut=true  (*** KEEP for skip ***)
        let l6 = self.run_c2f(&l5, weights, 6, 128, 2, true)?;
        println!("    L6:  C2f 128->128 → {}x{}x{}", l6.c, l6.h, l6.w);

        // L7: Conv 3x3 s2, 128->256, out 20x20
        let w7 = weights.conv_bn_silu(7).map_err(map_model_err)?;
        let l7 = self.run_conv_bn_silu_w(&l6, &w7, 256, 3, 3, 2, 1)?;
        println!("    L7:  Conv 128->256 s2 → {}x{}x{}", l7.c, l7.h, l7.w);

        // L8: C2f 256->256, 1 bottleneck, shortcut=true
        let l8 = self.run_c2f(&l7, weights, 8, 256, 1, true)?;
        println!("    L8:  C2f 256->256 → {}x{}x{}", l8.c, l8.h, l8.w);

        // L9: SPPF 256->256  (*** KEEP for skip ***)
        let l9 = self.run_sppf(&l8, weights, 9, 256)?;
        println!("    L9:  SPPF 256->256 → {}x{}x{}", l9.c, l9.h, l9.w);

        // === Neck (FPN/PAN) ===
        // L10: Upsample 2x → 40x40
        let l10 = self.upsample_2x(&l9)?;
        println!("    L10: Upsample → {}x{}x{}", l10.c, l10.h, l10.w);

        // L11: Concat [L10, L6] → 40x40x(256+128)=384
        let l11 = self.concat(&l10, &l6)?;
        println!("    L11: Concat → {}x{}x{}", l11.c, l11.h, l11.w);

        // L12: C2f 384->128, 1 bottleneck, shortcut=false  (*** KEEP for skip ***)
        let l12 = self.run_c2f(&l11, weights, 12, 128, 1, false)?;
        println!("    L12: C2f 384->128 → {}x{}x{}", l12.c, l12.h, l12.w);

        // L13: Upsample 2x → 80x80
        let l13 = self.upsample_2x(&l12)?;
        println!("    L13: Upsample → {}x{}x{}", l13.c, l13.h, l13.w);

        // L14: Concat [L13, L4] → 80x80x(128+64)=192
        let l14 = self.concat(&l13, &l4)?;
        println!("    L14: Concat → {}x{}x{}", l14.c, l14.h, l14.w);

        // L15: C2f 192->64, 1 bottleneck, shortcut=false → P3 output
        let l15 = self.run_c2f(&l14, weights, 15, 64, 1, false)?;
        println!("    L15: C2f 192->64 (P3) → {}x{}x{}", l15.c, l15.h, l15.w);

        // L16: Conv 3x3 s2, 64->64, out 40x40
        let w16 = weights.conv_bn_silu(16).map_err(map_model_err)?;
        let l16 = self.run_conv_bn_silu_w(&l15, &w16, 64, 3, 3, 2, 1)?;
        println!("    L16: Conv 64->64 s2 → {}x{}x{}", l16.c, l16.h, l16.w);

        // L17: Concat [L16, L12] → 40x40x(64+128)=192
        let l17 = self.concat(&l16, &l12)?;
        println!("    L17: Concat → {}x{}x{}", l17.c, l17.h, l17.w);

        // L18: C2f 192->128, 1 bottleneck, shortcut=false → P4 output
        let l18 = self.run_c2f(&l17, weights, 18, 128, 1, false)?;
        println!("    L18: C2f 192->128 (P4) → {}x{}x{}", l18.c, l18.h, l18.w);

        // L19: Conv 3x3 s2, 128->128, out 20x20
        let w19 = weights.conv_bn_silu(19).map_err(map_model_err)?;
        let l19 = self.run_conv_bn_silu_w(&l18, &w19, 128, 3, 3, 2, 1)?;
        println!("    L19: Conv 128->128 s2 → {}x{}x{}", l19.c, l19.h, l19.w);

        // L20: Concat [L19, L9] → 20x20x(128+256)=384
        let l20 = self.concat(&l19, &l9)?;
        println!("    L20: Concat → {}x{}x{}", l20.c, l20.h, l20.w);

        // L21: C2f 384->256, 1 bottleneck, shortcut=false → P5 output
        let l21 = self.run_c2f(&l20, weights, 21, 256, 1, false)?;
        println!("    L21: C2f 384->256 (P5) → {}x{}x{}", l21.c, l21.h, l21.w);

        // === Detect Head (Layer 22) ===
        println!("    Running detect head...");
        let (box_p3, cls_p3) = self.run_detect_scale(&l15, weights, 0)?;
        println!(
            "    P3: box {}x{}, cls {}x{}",
            box_p3.c,
            box_p3.h * box_p3.w,
            cls_p3.c,
            cls_p3.h * cls_p3.w
        );
        let (box_p4, cls_p4) = self.run_detect_scale(&l18, weights, 1)?;
        println!(
            "    P4: box {}x{}, cls {}x{}",
            box_p4.c,
            box_p4.h * box_p4.w,
            cls_p4.c,
            cls_p4.h * cls_p4.w
        );
        let (box_p5, cls_p5) = self.run_detect_scale(&l21, weights, 2)?;
        println!(
            "    P5: box {}x{}, cls {}x{}",
            box_p5.c,
            box_p5.h * box_p5.w,
            cls_p5.c,
            cls_p5.h * cls_p5.w
        );

        // === Post-processing ===
        // Concatenate outputs from all scales: [channels, total_anchors]
        let total_anchors =
            (box_p3.h * box_p3.w + box_p4.h * box_p4.w + box_p5.h * box_p5.w) as usize;
        println!("    Total anchors: {total_anchors}");

        // Download and flatten box outputs [64, H*W] for each scale → [64, total_anchors]
        let box_p3_data = self.download(&box_p3.data)?;
        let box_p4_data = self.download(&box_p4.data)?;
        let box_p5_data = self.download(&box_p5.data)?;
        let cls_p3_data = self.download(&cls_p3.data)?;
        let cls_p4_data = self.download(&cls_p4.data)?;
        let cls_p5_data = self.download(&cls_p5.data)?;

        let box_ch = 4 * REG_MAX; // 64
        let mut box_concat = vec![0.0f32; box_ch * total_anchors];
        let mut cls_concat = vec![0.0f32; NUM_CLASSES * total_anchors];

        let n_p3 = (box_p3.h * box_p3.w) as usize;
        let n_p4 = (box_p4.h * box_p4.w) as usize;
        let n_p5 = (box_p5.h * box_p5.w) as usize;

        // Interleave: for each channel, copy [P3 anchors | P4 anchors | P5 anchors]
        for ch in 0..box_ch {
            let dst_base = ch * total_anchors;
            let src_p3 = ch * n_p3;
            let src_p4 = ch * n_p4;
            let src_p5 = ch * n_p5;
            box_concat[dst_base..dst_base + n_p3]
                .copy_from_slice(&box_p3_data[src_p3..src_p3 + n_p3]);
            box_concat[dst_base + n_p3..dst_base + n_p3 + n_p4]
                .copy_from_slice(&box_p4_data[src_p4..src_p4 + n_p4]);
            box_concat[dst_base + n_p3 + n_p4..dst_base + n_p3 + n_p4 + n_p5]
                .copy_from_slice(&box_p5_data[src_p5..src_p5 + n_p5]);
        }
        for ch in 0..NUM_CLASSES {
            let dst_base = ch * total_anchors;
            let src_p3 = ch * n_p3;
            let src_p4 = ch * n_p4;
            let src_p5 = ch * n_p5;
            cls_concat[dst_base..dst_base + n_p3]
                .copy_from_slice(&cls_p3_data[src_p3..src_p3 + n_p3]);
            cls_concat[dst_base + n_p3..dst_base + n_p3 + n_p4]
                .copy_from_slice(&cls_p4_data[src_p4..src_p4 + n_p4]);
            cls_concat[dst_base + n_p3 + n_p4..dst_base + n_p3 + n_p4 + n_p5]
                .copy_from_slice(&cls_p5_data[src_p5..src_p5 + n_p5]);
        }

        // Generate anchors and decode
        let anchors = generate_anchors(640);
        assert_eq!(anchors.len(), total_anchors);

        let mut detections = decode_detections(
            &box_concat,
            &cls_concat,
            &anchors,
            NUM_CLASSES,
            REG_MAX,
            conf_threshold,
        );

        // NMS
        nms(&mut detections, iou_threshold);

        println!("    Detections after NMS: {}", detections.len());
        Ok(detections)
    }
}

// ---------------------------------------------------------------------------
// Weight preparation helpers
// ---------------------------------------------------------------------------

/// Prepare Conv2D weight for GEMM: [C_out, C_in, kH, kW] → column-major with N-padding.
///
/// PyTorch stores weight as [C_out, C_in*kH*kW] row-major, which is the same as
/// [C_in*kH*kW, C_out] column-major. We just need to pad N to a multiple of 16.
pub fn prepare_conv_weight(weight: &[f32], c_out: usize, k: usize) -> Vec<f32> {
    let n_padded = c_out.next_multiple_of(16);
    if n_padded == c_out {
        // No padding needed — weight is already in the right layout
        return weight.to_vec();
    }

    // Pad: copy each column (C_out values per K row in column-major)
    // Column-major [K, N_padded]: for each col co in 0..c_out, copy K values
    let mut padded = vec![0.0f32; k * n_padded];
    for co in 0..c_out {
        for ki in 0..k {
            padded[co * k + ki] = weight[co * k + ki];
        }
    }
    // Columns c_out..n_padded are already zero
    padded
}

// ---------------------------------------------------------------------------
// Detection post-processing (host-side)
// ---------------------------------------------------------------------------

/// A single detected object.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Left edge (pixels).
    pub x1: f32,
    /// Top edge (pixels).
    pub y1: f32,
    /// Right edge (pixels).
    pub x2: f32,
    /// Bottom edge (pixels).
    pub y2: f32,
    /// Class ID (0-79 for COCO).
    pub class_id: usize,
    /// Confidence score (class probability * objectness).
    pub confidence: f32,
}

/// Decode DFL distribution to box coordinate offset (public for testing).
pub fn dfl_decode_pub(logits: &[f32], reg_max: usize) -> f32 {
    dfl_decode(logits, reg_max)
}

/// Decode DFL distribution to box coordinate offset.
///
/// DFL: softmax over reg_max bins, then weighted sum with bin indices [0, 1, ..., reg_max-1].
fn dfl_decode(logits: &[f32], reg_max: usize) -> f32 {
    // Softmax
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let mut exps = vec![0.0f32; reg_max];
    for i in 0..reg_max {
        exps[i] = (logits[i] - max_val).exp();
        exp_sum += exps[i];
    }
    // Weighted sum: sum(i * softmax(logits[i]))
    let mut val = 0.0f32;
    for i in 0..reg_max {
        val += (i as f32) * exps[i] / exp_sum;
    }
    val
}

/// Decode raw detection outputs to bounding boxes.
///
/// `box_output`: [4 * reg_max, total_anchors] — DFL logits for (left, top, right, bottom)
/// `cls_output`: [num_classes, total_anchors] — sigmoid class scores
/// `strides_and_grids`: for each anchor, (stride, grid_x, grid_y)
/// `conf_threshold`: minimum confidence to keep a detection
///
/// Returns: list of detections above the confidence threshold.
pub fn decode_detections(
    box_output: &[f32],
    cls_output: &[f32],
    strides_and_grids: &[(f32, f32, f32)],
    num_classes: usize,
    reg_max: usize,
    conf_threshold: f32,
) -> Vec<Detection> {
    let total_anchors = strides_and_grids.len();
    let mut detections = Vec::new();

    for anchor_idx in 0..total_anchors {
        // Find best class
        let mut best_class = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let score = cls_output[c * total_anchors + anchor_idx];
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        // Decode DFL box
        let (stride, gx, gy) = strides_and_grids[anchor_idx];
        let mut offsets = [0.0f32; 4]; // left, top, right, bottom
        for d in 0..4 {
            let start = (d * reg_max) * total_anchors + anchor_idx;
            let logits: Vec<f32> = (0..reg_max)
                .map(|r| box_output[start + r * total_anchors])
                .collect();
            offsets[d] = dfl_decode(&logits, reg_max);
        }

        // Convert from (left, top, right, bottom) offsets to (x1, y1, x2, y2)
        let cx = (gx + 0.5) * stride;
        let cy = (gy + 0.5) * stride;
        let x1 = cx - offsets[0] * stride;
        let y1 = cy - offsets[1] * stride;
        let x2 = cx + offsets[2] * stride;
        let y2 = cy + offsets[3] * stride;

        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            class_id: best_class,
            confidence: best_score,
        });
    }

    detections
}

/// Non-Maximum Suppression: filter overlapping detections.
///
/// Keeps only the highest-confidence detection among overlapping boxes
/// of the same class (IoU > `iou_threshold`).
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    // Sort by confidence descending
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] || detections[i].class_id != detections[j].class_id {
                continue;
            }
            if iou(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Compute Intersection over Union between two boxes.
fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Convert ModelError to GpuHostError.
fn map_model_err(e: crate::model::ModelError) -> GpuHostError {
    GpuHostError::Verification {
        test: "yolo_inference",
        detail: format!("weight load: {e}"),
    }
}

/// Generate anchor grid positions and strides for YOLOv8 multi-scale detection.
///
/// Returns: Vec of (stride, grid_x, grid_y) for each anchor point.
pub fn generate_anchors(input_size: u32) -> Vec<(f32, f32, f32)> {
    let strides = [8u32, 16, 32]; // P3, P4, P5
    let mut anchors = Vec::new();

    for &stride in &strides {
        let grid_size = input_size / stride;
        for gy in 0..grid_size {
            for gx in 0..grid_size {
                anchors.push((stride as f32, gx as f32, gy as f32));
            }
        }
    }

    anchors
}
