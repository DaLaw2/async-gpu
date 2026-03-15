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
                    input.c,
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
