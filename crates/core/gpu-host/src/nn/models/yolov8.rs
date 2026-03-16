//! YOLOv8-nano model configuration and architecture.
//!
//! Provides [`YoloV8Nano`] for end-to-end object detection using composable
//! nn layers. All inference runs through the [`Module`] trait — no raw kernel
//! launches needed.
//!
//! Architecture: Backbone (Conv + C2f + SPPF) → Neck (FPN + PAN) → DetectHead.

use std::sync::Arc;

use crate::nn::error::{NnError, Result};
use crate::nn::layers::{BatchNorm2d, Conv2d, MaxPool2d, Module, Sigmoid};
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

#[cfg(feature = "gpt2")]
use crate::model_yolo::{ConvBnSiluWeights, ConvWeights, YoloWeights};

/// Number of COCO detection classes.
pub const NUM_CLASSES: usize = 80;
/// DFL bins per coordinate.
pub const REG_MAX: usize = 16;

// ============================================================
// Composite layers
// ============================================================

/// Conv2d + BatchNorm2d + SiLU fused block.
pub struct ConvBnSilu {
    conv: Conv2d,
    bn: BatchNorm2d,
}

impl ConvBnSilu {
    /// Create from raw weight slices.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conv_weight: &[f32],
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        bn_weight: &[f32],
        bn_bias: &[f32],
        bn_mean: &[f32],
        bn_var: &[f32],
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(
                conv_weight,
                None,
                c_out,
                c_in,
                kh,
                kw,
                stride,
                padding,
                registry,
            )?,
            bn: BatchNorm2d::new(bn_weight, bn_bias, bn_mean, bn_var, 1e-5, true, registry)?,
        })
    }

    /// Create from pre-loaded [`ConvBnSiluWeights`].
    #[cfg(feature = "gpt2")]
    pub fn from_weights(
        w: &ConvBnSiluWeights,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let c_out = w.conv_shape[0];
        let c_in = w.conv_shape[1];
        let kh = w.conv_shape[2];
        let kw = w.conv_shape[3];
        Self::new(
            &w.conv_weight,
            c_out,
            c_in,
            kh,
            kw,
            stride,
            padding,
            &w.bn_weight,
            &w.bn_bias,
            &w.bn_running_mean,
            &w.bn_running_var,
            registry,
        )
    }
}

impl Module for ConvBnSilu {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let conv_out = self.conv.forward(input)?;
        self.bn.forward(&conv_out)
    }
}

/// Conv2d with bias (no BN, no activation) — used in detect head final layers.
pub struct ConvBias {
    conv: Conv2d,
}

impl ConvBias {
    /// Create from raw weight + bias slices.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: &[f32],
        bias: &[f32],
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(
                weight,
                Some(bias),
                c_out,
                c_in,
                kh,
                kw,
                stride,
                padding,
                registry,
            )?,
        })
    }

    /// Create from pre-loaded [`ConvWeights`].
    #[cfg(feature = "gpt2")]
    pub fn from_weights(
        w: &ConvWeights,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let c_out = w.shape[0];
        let c_in = w.shape[1];
        let kh = w.shape[2];
        let kw = w.shape[3];
        Self::new(
            &w.weight, &w.bias, c_out, c_in, kh, kw, stride, padding, registry,
        )
    }
}

impl Module for ConvBias {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        self.conv.forward(input)
    }
}

/// C2f block (YOLOv8 bottleneck with channel split + concat).
///
/// Architecture: cv1(1x1) → split → N × Bottleneck → concat all → cv2(1x1).
pub struct C2f {
    cv1: ConvBnSilu,
    bottlenecks: Vec<(ConvBnSilu, ConvBnSilu)>,
    cv2: ConvBnSilu,
    shortcut: bool,
    hidden: usize,
    registry: Arc<KernelRegistry>,
}

impl C2f {
    /// Create a C2f block from pre-loaded weights.
    #[cfg(feature = "gpt2")]
    pub fn from_yolo_weights(
        weights: &YoloWeights,
        layer_idx: usize,
        c_out: usize,
        n_bottleneck: usize,
        shortcut: bool,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let hidden = c_out / 2;

        let cv1_w =
            weights
                .sub_conv_bn_silu(layer_idx, "cv1")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("c2f cv1: {e}").into_boxed_str()),
                })?;
        let cv1 = ConvBnSilu::from_weights(&cv1_w, 1, 0, registry)?;

        let mut bottlenecks = Vec::with_capacity(n_bottleneck);
        for j in 0..n_bottleneck {
            let bn_cv1_w = weights
                .bottleneck_conv_bn_silu(layer_idx, j, "cv1")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("c2f bn{j}.cv1: {e}").into_boxed_str()),
                })?;
            let bn_cv2_w = weights
                .bottleneck_conv_bn_silu(layer_idx, j, "cv2")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("c2f bn{j}.cv2: {e}").into_boxed_str()),
                })?;
            bottlenecks.push((
                ConvBnSilu::from_weights(&bn_cv1_w, 1, 1, registry)?,
                ConvBnSilu::from_weights(&bn_cv2_w, 1, 1, registry)?,
            ));
        }

        let cv2_w =
            weights
                .sub_conv_bn_silu(layer_idx, "cv2")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("c2f cv2: {e}").into_boxed_str()),
                })?;
        let cv2 = ConvBnSilu::from_weights(&cv2_w, 1, 0, registry)?;

        Ok(Self {
            cv1,
            bottlenecks,
            cv2,
            shortcut,
            hidden,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for C2f {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // cv1: 1x1 conv, c_in → 2*hidden
        let cv1_out = self.cv1.forward(input)?;

        // Channel split: first half (branch_0) and second half (prev)
        let cv1_host = cv1_out.to_host()?;
        let c = cv1_out.shape()[0];
        let h = cv1_out.shape()[1];
        let w = cv1_out.shape()[2];
        let hw = h * w;
        let half_c = c / 2;
        let dev = self.registry.device();

        let branch_0_data: Vec<f32> = cv1_host[..half_c * hw].to_vec();
        let mut prev_data: Vec<f32> = cv1_host[half_c * hw..].to_vec();

        let mut all_branches: Vec<Vec<f32>> = vec![branch_0_data];

        // Bottlenecks
        for (bn_cv1, bn_cv2) in &self.bottlenecks {
            let prev_tensor = GpuTensor::from_host(&prev_data, &[half_c, h, w], dev)?;
            let bn1_out = bn_cv1.forward(&prev_tensor)?;
            let bn2_out = bn_cv2.forward(&bn1_out)?;

            let bn2_host = bn2_out.to_host()?;

            if self.shortcut {
                // Residual add
                let mut residual = prev_data.clone();
                for (r, &b) in residual.iter_mut().zip(bn2_host.iter()) {
                    *r += b;
                }
                all_branches.push(residual.clone());
                prev_data = residual;
            } else {
                all_branches.push(bn2_host.clone());
                prev_data = bn2_host;
            }
        }

        // Concat all branches along channel dimension
        let total_c = all_branches.len() * self.hidden;
        let mut cat_data = Vec::with_capacity(total_c * hw);
        for branch in &all_branches {
            cat_data.extend_from_slice(branch);
        }
        let cat_tensor = GpuTensor::from_host(&cat_data, &[total_c, h, w], dev)?;

        // cv2: 1x1 conv
        self.cv2.forward(&cat_tensor)
    }
}

/// SPPF (Spatial Pyramid Pooling Fast).
///
/// Architecture: cv1(1x1) → 3× MaxPool5x5(same) → concat(x, p1, p2, p3) → cv2(1x1).
pub struct Sppf {
    cv1: ConvBnSilu,
    pool: MaxPool2d,
    cv2: ConvBnSilu,
    registry: Arc<KernelRegistry>,
}

impl Sppf {
    /// Create from pre-loaded weights.
    #[cfg(feature = "gpt2")]
    pub fn from_yolo_weights(
        weights: &YoloWeights,
        layer_idx: usize,
        _c_out: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let cv1_w =
            weights
                .sub_conv_bn_silu(layer_idx, "cv1")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("sppf cv1: {e}").into_boxed_str()),
                })?;
        let cv2_w =
            weights
                .sub_conv_bn_silu(layer_idx, "cv2")
                .map_err(|e| NnError::KernelNotFound {
                    name: Box::leak(format!("sppf cv2: {e}").into_boxed_str()),
                })?;

        Ok(Self {
            cv1: ConvBnSilu::from_weights(&cv1_w, 1, 0, registry)?,
            pool: MaxPool2d::new(5, 1, 2, registry),
            cv2: ConvBnSilu::from_weights(&cv2_w, 1, 0, registry)?,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for Sppf {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let x = self.cv1.forward(input)?;
        let p1 = self.pool.forward(&x)?;
        let p2 = self.pool.forward(&p1)?;
        let p3 = self.pool.forward(&p2)?;

        // Concat [x, p1, p2, p3] along channel dim
        let cat1 = ops::concat_channels(&x, &p1, &self.registry)?;
        let cat2 = ops::concat_channels(&cat1, &p2, &self.registry)?;
        let cat3 = ops::concat_channels(&cat2, &p3, &self.registry)?;

        self.cv2.forward(&cat3)
    }
}

// ============================================================
// Detect Head
// ============================================================

/// Per-scale detect head branch (box or class).
struct DetectBranch {
    cv0: ConvBnSilu,
    cv1: ConvBnSilu,
    cv2: ConvBias,
}

impl DetectBranch {
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let x = self.cv0.forward(input)?;
        let x = self.cv1.forward(&x)?;
        self.cv2.forward(&x)
    }
}

/// Detection head for all 3 scales.
pub struct DetectHead {
    box_branches: Vec<DetectBranch>,
    cls_branches: Vec<DetectBranch>,
    sigmoid: Sigmoid,
}

impl DetectHead {
    /// Create from pre-loaded weights.
    #[cfg(feature = "gpt2")]
    pub fn from_yolo_weights(
        weights: &YoloWeights,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let mut box_branches = Vec::with_capacity(3);
        let mut cls_branches = Vec::with_capacity(3);

        for scale in 0..3 {
            // Box branch (cv2)
            let b0_w = weights.detect_conv_bn_silu("cv2", scale, 0).map_err(|e| {
                NnError::KernelNotFound {
                    name: Box::leak(format!("det cv2.{scale}.0: {e}").into_boxed_str()),
                }
            })?;
            let b1_w = weights.detect_conv_bn_silu("cv2", scale, 1).map_err(|e| {
                NnError::KernelNotFound {
                    name: Box::leak(format!("det cv2.{scale}.1: {e}").into_boxed_str()),
                }
            })?;
            let b2_w =
                weights
                    .detect_conv("cv2", scale, 2)
                    .map_err(|e| NnError::KernelNotFound {
                        name: Box::leak(format!("det cv2.{scale}.2: {e}").into_boxed_str()),
                    })?;
            box_branches.push(DetectBranch {
                cv0: ConvBnSilu::from_weights(&b0_w, 1, 1, registry)?,
                cv1: ConvBnSilu::from_weights(&b1_w, 1, 1, registry)?,
                cv2: ConvBias::from_weights(&b2_w, 1, 0, registry)?,
            });

            // Class branch (cv3)
            let c0_w = weights.detect_conv_bn_silu("cv3", scale, 0).map_err(|e| {
                NnError::KernelNotFound {
                    name: Box::leak(format!("det cv3.{scale}.0: {e}").into_boxed_str()),
                }
            })?;
            let c1_w = weights.detect_conv_bn_silu("cv3", scale, 1).map_err(|e| {
                NnError::KernelNotFound {
                    name: Box::leak(format!("det cv3.{scale}.1: {e}").into_boxed_str()),
                }
            })?;
            let c2_w =
                weights
                    .detect_conv("cv3", scale, 2)
                    .map_err(|e| NnError::KernelNotFound {
                        name: Box::leak(format!("det cv3.{scale}.2: {e}").into_boxed_str()),
                    })?;
            cls_branches.push(DetectBranch {
                cv0: ConvBnSilu::from_weights(&c0_w, 1, 1, registry)?,
                cv1: ConvBnSilu::from_weights(&c1_w, 1, 1, registry)?,
                cv2: ConvBias::from_weights(&c2_w, 1, 0, registry)?,
            });
        }

        Ok(Self {
            box_branches,
            cls_branches,
            sigmoid: Sigmoid::new(registry),
        })
    }

    /// Forward pass for one scale.
    ///
    /// Returns `(box_output [64, H, W], cls_output [80, H, W])`.
    pub fn forward_scale(&self, input: &GpuTensor, scale: usize) -> Result<(GpuTensor, GpuTensor)> {
        let box_out = self.box_branches[scale].forward(input)?;
        let cls_logits = self.cls_branches[scale].forward(input)?;
        let cls_out = self.sigmoid.forward(&cls_logits)?;
        Ok((box_out, cls_out))
    }
}

// ============================================================
// Full YOLOv8-Nano Model
// ============================================================

/// Complete YOLOv8-nano model.
///
/// Architecture: Backbone (layers 0-9) → Neck (layers 10-21) → DetectHead (layer 22).
pub struct YoloV8Nano {
    // Backbone
    l0: ConvBnSilu, // stem, stride 2
    l1: ConvBnSilu, // stride 2
    l2: C2f,
    l3: ConvBnSilu, // stride 2
    l4: C2f,
    l5: ConvBnSilu, // stride 2
    l6: C2f,
    l7: ConvBnSilu, // stride 2
    l8: C2f,
    l9: Sppf,

    // Neck
    l12: C2f,
    l15: C2f,
    l16: ConvBnSilu, // stride 2
    l18: C2f,
    l19: ConvBnSilu, // stride 2
    l21: C2f,

    // Detect
    detect: DetectHead,

    registry: Arc<KernelRegistry>,
}

impl YoloV8Nano {
    /// Build the full model from pre-loaded [`YoloWeights`].
    #[cfg(feature = "gpt2")]
    pub fn from_weights(weights: &YoloWeights, registry: &Arc<KernelRegistry>) -> Result<Self> {
        // Backbone
        let l0 =
            ConvBnSilu::from_weights(&weights.conv_bn_silu(0).map_err(model_err)?, 2, 1, registry)?;
        let l1 =
            ConvBnSilu::from_weights(&weights.conv_bn_silu(1).map_err(model_err)?, 2, 1, registry)?;
        let l2 = C2f::from_yolo_weights(weights, 2, 32, 1, true, registry)?;
        let l3 =
            ConvBnSilu::from_weights(&weights.conv_bn_silu(3).map_err(model_err)?, 2, 1, registry)?;
        let l4 = C2f::from_yolo_weights(weights, 4, 64, 2, true, registry)?;
        let l5 =
            ConvBnSilu::from_weights(&weights.conv_bn_silu(5).map_err(model_err)?, 2, 1, registry)?;
        let l6 = C2f::from_yolo_weights(weights, 6, 128, 2, true, registry)?;
        let l7 =
            ConvBnSilu::from_weights(&weights.conv_bn_silu(7).map_err(model_err)?, 2, 1, registry)?;
        let l8 = C2f::from_yolo_weights(weights, 8, 256, 1, true, registry)?;
        let l9 = Sppf::from_yolo_weights(weights, 9, 256, registry)?;

        // Neck
        let l12 = C2f::from_yolo_weights(weights, 12, 128, 1, false, registry)?;
        let l15 = C2f::from_yolo_weights(weights, 15, 64, 1, false, registry)?;
        let l16 = ConvBnSilu::from_weights(
            &weights.conv_bn_silu(16).map_err(model_err)?,
            2,
            1,
            registry,
        )?;
        let l18 = C2f::from_yolo_weights(weights, 18, 128, 1, false, registry)?;
        let l19 = ConvBnSilu::from_weights(
            &weights.conv_bn_silu(19).map_err(model_err)?,
            2,
            1,
            registry,
        )?;
        let l21 = C2f::from_yolo_weights(weights, 21, 256, 1, false, registry)?;

        // Detect head
        let detect = DetectHead::from_yolo_weights(weights, registry)?;

        Ok(Self {
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l12,
            l15,
            l16,
            l18,
            l19,
            l21,
            detect,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass: input `[3, 640, 640]` → 3 scales of (box, cls) outputs.
    ///
    /// Returns `[(box_p3, cls_p3), (box_p4, cls_p4), (box_p5, cls_p5)]`.
    pub fn forward_multi_scale(&self, input: &GpuTensor) -> Result<Vec<(GpuTensor, GpuTensor)>> {
        // === Backbone ===
        let x = self.l0.forward(input)?; // [16, 320, 320]
        let x = self.l1.forward(&x)?; // [32, 160, 160]
        let x = self.l2.forward(&x)?; // [32, 160, 160]
        let x = self.l3.forward(&x)?; // [64, 80, 80]
        let l4_out = self.l4.forward(&x)?; // [64, 80, 80] — saved for neck
        let x = self.l5.forward(&l4_out)?; // [128, 40, 40]
        let l6_out = self.l6.forward(&x)?; // [128, 40, 40] — saved for neck
        let x = self.l7.forward(&l6_out)?; // [256, 20, 20]
        let l8_out = self.l8.forward(&x)?; // [256, 20, 20]
        let l9_out = self.l9.forward(&l8_out)?; // [256, 20, 20]

        // === Neck (FPN top-down) ===
        let l10 = ops::upsample_nearest_2x(&l9_out, &self.registry)?; // [256, 40, 40]
        let l11 = ops::concat_channels(&l10, &l6_out, &self.registry)?; // [384, 40, 40]
        let l12_out = self.l12.forward(&l11)?; // [128, 40, 40]

        let l13 = ops::upsample_nearest_2x(&l12_out, &self.registry)?; // [128, 80, 80]
        let l14 = ops::concat_channels(&l13, &l4_out, &self.registry)?; // [192, 80, 80]
        let l15_out = self.l15.forward(&l14)?; // [64, 80, 80] — P3

        // === Neck (PAN bottom-up) ===
        let l16_out = self.l16.forward(&l15_out)?; // [64, 40, 40]
        let l17 = ops::concat_channels(&l16_out, &l12_out, &self.registry)?; // [192, 40, 40]
        let l18_out = self.l18.forward(&l17)?; // [128, 40, 40] — P4

        let l19_out = self.l19.forward(&l18_out)?; // [128, 20, 20]
        let l20 = ops::concat_channels(&l19_out, &l9_out, &self.registry)?; // [384, 20, 20]
        let l21_out = self.l21.forward(&l20)?; // [256, 20, 20] — P5

        // === Detect Head ===
        let scale_p3 = self.detect.forward_scale(&l15_out, 0)?;
        let scale_p4 = self.detect.forward_scale(&l18_out, 1)?;
        let scale_p5 = self.detect.forward_scale(&l21_out, 2)?;

        Ok(vec![scale_p3, scale_p4, scale_p5])
    }

    /// Run full detection pipeline: input image → list of detections.
    ///
    /// `input_data`: `[3, 640, 640]` normalized to `[0, 1]`.
    /// Returns detections as `(x1, y1, x2, y2, confidence, class_id)`.
    pub fn detect(
        &self,
        input_data: &[f32],
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let dev = self.registry.device();
        let input = GpuTensor::from_host(input_data, &[3, 640, 640], dev)?;

        let scales = self.forward_multi_scale(&input)?;

        // Post-process: decode boxes, apply NMS
        let mut all_detections = Vec::new();
        let strides = [8, 16, 32];

        for (scale_idx, (box_out, cls_out)) in scales.iter().enumerate() {
            let stride = strides[scale_idx] as f32;
            let box_host = box_out.to_host()?;
            let cls_host = cls_out.to_host()?;

            let h = box_out.shape()[1];
            let w = box_out.shape()[2];
            let hw = h * w;

            for y in 0..h {
                for x in 0..w {
                    // Find best class
                    let mut best_class = 0usize;
                    let mut best_conf = 0.0f32;
                    for c in 0..NUM_CLASSES {
                        let val = cls_host[c * hw + y * w + x];
                        if val > best_conf {
                            best_conf = val;
                            best_class = c;
                        }
                    }

                    if best_conf < conf_threshold {
                        continue;
                    }

                    // Decode DFL box (simplified: use raw regression values)
                    let cx = (x as f32 + 0.5) * stride;
                    let cy = (y as f32 + 0.5) * stride;

                    // DFL decode: 4 coords × 16 bins → 4 distances
                    let mut dists = [0.0f32; 4];
                    for d in 0..4 {
                        let mut softmax_sum = 0.0f32;
                        let mut weighted_sum = 0.0f32;
                        for bin in 0..REG_MAX {
                            let val = box_host[(d * REG_MAX + bin) * hw + y * w + x];
                            let exp_val = val.exp();
                            softmax_sum += exp_val;
                            weighted_sum += exp_val * bin as f32;
                        }
                        dists[d] = weighted_sum / softmax_sum * stride;
                    }

                    let x1 = cx - dists[0];
                    let y1 = cy - dists[1];
                    let x2 = cx + dists[2];
                    let y2 = cy + dists[3];

                    all_detections.push(Detection {
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence: best_conf,
                        class_id: best_class,
                    });
                }
            }
        }

        // NMS
        nms(&mut all_detections, iou_threshold);

        Ok(all_detections)
    }
}

// ============================================================
// Detection output
// ============================================================

/// A single object detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Top-left x coordinate (in input image space).
    pub x1: f32,
    /// Top-left y coordinate.
    pub y1: f32,
    /// Bottom-right x coordinate.
    pub x2: f32,
    /// Bottom-right y coordinate.
    pub y2: f32,
    /// Detection confidence (0..1).
    pub confidence: f32,
    /// COCO class ID (0..79).
    pub class_id: usize,
}

/// Non-maximum suppression (greedy, per-class).
fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
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

/// Intersection over Union.
fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;

    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Convert a [`crate::model::ModelError`] to [`NnError`].
#[cfg(feature = "gpt2")]
fn model_err(e: crate::model::ModelError) -> NnError {
    NnError::KernelNotFound {
        name: Box::leak(format!("weight loading: {e}").into_boxed_str()),
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::test_utils::{GoldenEntry, Tolerance};
    use std::sync::Arc;

    /// Capture or verify YOLO golden detections.
    #[test]
    fn test_yolo_golden_regression() {
        let models = crate::model_dir(Some(env!("CARGO_MANIFEST_DIR")));
        let weights_path = models.join("yolov8n.safetensors");
        let image_path = models.join("bus.ppm");

        if !weights_path.exists() || !image_path.exists() {
            println!(
                "SKIP: YOLO files not found ({}, {})",
                weights_path.display(),
                image_path.display()
            );
            return;
        }

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry = Arc::new(
            crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"),
        );
        let weights = crate::model_yolo::load_yolo_weights(&weights_path).expect("weights");
        let model = super::YoloV8Nano::from_weights(&weights, &registry).expect("model");

        // Load and preprocess image (letterbox to 640×640, normalize to [0,1])
        let img = crate::model_yolo::load_ppm(&image_path).expect("load ppm");
        let (letterboxed, _scale, _pad_x, _pad_y) =
            img.letterbox(crate::model_yolo::YOLO_INPUT_SIZE);
        let input: Vec<f32> = letterboxed.data.iter().map(|&v| v as f32 / 255.0).collect();

        let detections = model.detect(&input, 0.25, 0.45).expect("detect");
        let n_det = detections.len();

        // Golden: save/check detection count + top-5 class IDs
        let golden_dir = crate::nn::test_utils::golden_dir();
        std::fs::create_dir_all(&golden_dir).ok();
        let golden_path = golden_dir.join("yolo_bus_detections.golden");

        let top_n = n_det.min(5);
        let mut golden_data: Vec<f32> = vec![n_det as f32];
        for d in detections.iter().take(top_n) {
            golden_data.push(d.class_id as f32);
            golden_data.push(d.confidence);
        }

        if golden_path.exists() {
            let golden = GoldenEntry::load(&golden_path).expect("load golden");
            // Check detection count is within ±2
            let expected_count = golden.data[0] as usize;
            assert!(
                (n_det as isize - expected_count as isize).unsigned_abs() <= 2,
                "Detection count changed: expected ~{expected_count}, got {n_det}"
            );
            // Check top-5 class IDs match
            for i in 0..top_n.min((golden.data.len() - 1) / 2) {
                let expected_class = golden.data[1 + i * 2] as usize;
                let actual_class = detections[i].class_id;
                assert_eq!(
                    actual_class, expected_class,
                    "Detection {i} class changed: expected {expected_class}, got {actual_class}"
                );
            }
            println!("REGRESSION OK: YOLO {n_det} detections match golden");
        } else {
            let entry = GoldenEntry {
                label: format!("yolo_bus_{n_det}_detections"),
                shape: vec![1 + top_n * 2],
                data: golden_data,
                tolerance: Tolerance::f32_loose(),
            };
            entry.save(&golden_path).expect("save golden");
            println!("CAPTURED: YOLO {n_det} detections, top-5 classes:");
            for (i, d) in detections.iter().take(top_n).enumerate() {
                println!(
                    "  [{i}] class={} conf={:.1}%",
                    d.class_id,
                    d.confidence * 100.0
                );
            }
        }
    }
}
