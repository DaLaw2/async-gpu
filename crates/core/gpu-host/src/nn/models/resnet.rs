//! ResNet-18 model for CIFAR-10 classification.
//!
//! Modified ResNet-18 for 32×32 images (CIFAR variant):
//! - conv1: 3×3, stride=1, no maxpool (vs. 7×7 stride=2 + maxpool for ImageNet)
//! - 4 stages with [2, 2, 2, 2] BasicBlocks
//! - Global average pooling → Linear(512, num_classes)

use std::sync::Arc;

use crate::nn::error::{NnError, Result};
use crate::nn::layers::{BatchNorm2d, Conv2d, Linear, Module};
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// BasicBlock: two 3×3 convolutions with batch norm and residual connection.
///
/// ```text
/// x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU
/// ```
///
/// When stride > 1 or channels change, the shortcut uses Conv(1×1) + BN.
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    shortcut_conv: Option<Conv2d>,
    shortcut_bn: Option<BatchNorm2d>,
    registry: Arc<KernelRegistry>,
}

impl BasicBlock {
    /// Create a BasicBlock.
    ///
    /// `weights` slice order: conv1_w, bn1_gamma, bn1_beta, bn1_mean, bn1_var,
    ///                        conv2_w, bn2_gamma, bn2_beta, bn2_mean, bn2_var,
    ///                        [shortcut_conv_w, shortcut_bn_gamma, shortcut_bn_beta, shortcut_bn_mean, shortcut_bn_var]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        weights: &BasicBlockWeights,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(
            &weights.conv1_w,
            None,
            out_channels,
            in_channels,
            3,
            3,
            stride,
            1,
            registry,
        )?;
        let bn1 = BatchNorm2d::new(
            &weights.bn1_gamma,
            &weights.bn1_beta,
            &weights.bn1_mean,
            &weights.bn1_var,
            1e-5,
            false,
            registry,
        )?;
        let conv2 = Conv2d::new(
            &weights.conv2_w,
            None,
            out_channels,
            out_channels,
            3,
            3,
            1,
            1,
            registry,
        )?;
        let bn2 = BatchNorm2d::new(
            &weights.bn2_gamma,
            &weights.bn2_beta,
            &weights.bn2_mean,
            &weights.bn2_var,
            1e-5,
            false,
            registry,
        )?;

        let need_shortcut = stride != 1 || in_channels != out_channels;
        let (shortcut_conv, shortcut_bn) = if need_shortcut {
            let missing = |name: &str| {
                NnError::ShapeMismatch {
                expected: format!("shortcut weights present (stride={stride}, in={in_channels}, out={out_channels})"),
                actual: format!("{name} is None"),
            }
            };
            let sc = Conv2d::new(
                weights
                    .shortcut_conv_w
                    .as_ref()
                    .ok_or_else(|| missing("shortcut_conv_w"))?,
                None,
                out_channels,
                in_channels,
                1,
                1,
                stride,
                0,
                registry,
            )?;
            let sbn = BatchNorm2d::new(
                weights
                    .shortcut_bn_gamma
                    .as_ref()
                    .ok_or_else(|| missing("shortcut_bn_gamma"))?,
                weights
                    .shortcut_bn_beta
                    .as_ref()
                    .ok_or_else(|| missing("shortcut_bn_beta"))?,
                weights
                    .shortcut_bn_mean
                    .as_ref()
                    .ok_or_else(|| missing("shortcut_bn_mean"))?,
                weights
                    .shortcut_bn_var
                    .as_ref()
                    .ok_or_else(|| missing("shortcut_bn_var"))?,
                1e-5,
                false,
                registry,
            )?;
            (Some(sc), Some(sbn))
        } else {
            (None, None)
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut_conv,
            shortcut_bn,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // Main path
        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = ops::relu(&out, &self.registry)?;
        let out = self.conv2.forward(&out)?;
        let mut out = self.bn2.forward(&out)?;

        // Shortcut
        let shortcut = if let (Some(sc), Some(sbn)) = (&self.shortcut_conv, &self.shortcut_bn) {
            sbn.forward(&sc.forward(input)?)?
        } else {
            input.clone_tensor()?
        };

        // Residual addition: out += shortcut
        ops::elementwise_add(&mut out, &shortcut, &self.registry)?;
        ops::relu(&out, &self.registry)
    }
}

/// Weight container for a BasicBlock.
#[allow(missing_docs)]
pub struct BasicBlockWeights {
    pub conv1_w: Vec<f32>,   // [out, in, 3, 3]
    pub bn1_gamma: Vec<f32>, // [out]
    pub bn1_beta: Vec<f32>,  // [out]
    pub bn1_mean: Vec<f32>,  // [out]
    pub bn1_var: Vec<f32>,   // [out]
    pub conv2_w: Vec<f32>,   // [out, out, 3, 3]
    pub bn2_gamma: Vec<f32>, // [out]
    pub bn2_beta: Vec<f32>,  // [out]
    pub bn2_mean: Vec<f32>,  // [out]
    pub bn2_var: Vec<f32>,   // [out]
    // Only present when stride > 1 or in_channels != out_channels
    pub shortcut_conv_w: Option<Vec<f32>>, // [out, in, 1, 1]
    pub shortcut_bn_gamma: Option<Vec<f32>>,
    pub shortcut_bn_beta: Option<Vec<f32>>,
    pub shortcut_bn_mean: Option<Vec<f32>>,
    pub shortcut_bn_var: Option<Vec<f32>>,
}

impl BasicBlockWeights {
    /// Generate random weights (He initialization) for testing.
    pub fn random(in_ch: usize, out_ch: usize, stride: usize, seed: u64) -> Self {
        let he = |fan_in: usize, n: usize, s: u64| -> Vec<f32> {
            let scale = (2.0 / fan_in as f64).sqrt() as f32;
            (0..n)
                .map(|i| {
                    let v = ((i as u64).wrapping_mul(s).wrapping_add(0x9E3779B97F4A7C15) % 10007)
                        as f32;
                    (v / 10007.0 - 0.5) * 2.0 * scale
                })
                .collect()
        };

        let need_shortcut = stride != 1 || in_ch != out_ch;
        Self {
            conv1_w: he(in_ch * 9, out_ch * in_ch * 3 * 3, seed),
            bn1_gamma: vec![1.0; out_ch],
            bn1_beta: vec![0.0; out_ch],
            bn1_mean: vec![0.0; out_ch],
            bn1_var: vec![1.0; out_ch],
            conv2_w: he(out_ch * 9, out_ch * out_ch * 3 * 3, seed.wrapping_add(1)),
            bn2_gamma: vec![1.0; out_ch],
            bn2_beta: vec![0.0; out_ch],
            bn2_mean: vec![0.0; out_ch],
            bn2_var: vec![1.0; out_ch],
            shortcut_conv_w: if need_shortcut {
                Some(he(in_ch, out_ch * in_ch, seed.wrapping_add(2)))
            } else {
                None
            },
            shortcut_bn_gamma: if need_shortcut {
                Some(vec![1.0; out_ch])
            } else {
                None
            },
            shortcut_bn_beta: if need_shortcut {
                Some(vec![0.0; out_ch])
            } else {
                None
            },
            shortcut_bn_mean: if need_shortcut {
                Some(vec![0.0; out_ch])
            } else {
                None
            },
            shortcut_bn_var: if need_shortcut {
                Some(vec![1.0; out_ch])
            } else {
                None
            },
        }
    }
}

/// ResNet-18 model (CIFAR variant).
///
/// ```text
/// conv1(3→64, 3×3) → BN → ReLU
/// → layer1: 2×BasicBlock(64→64)
/// → layer2: 2×BasicBlock(64→128, stride=2)
/// → layer3: 2×BasicBlock(128→256, stride=2)
/// → layer4: 2×BasicBlock(256→512, stride=2)
/// → global_avg_pool → Linear(512, num_classes)
/// ```
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
    fc: Linear,
    registry: Arc<KernelRegistry>,
}

/// Complete weight set for ResNet-18.
#[allow(missing_docs)]
pub struct ResNet18Weights {
    pub conv1_w: Vec<f32>,              // [64, 3, 3, 3]
    pub bn1_gamma: Vec<f32>,            // [64]
    pub bn1_beta: Vec<f32>,             // [64]
    pub bn1_mean: Vec<f32>,             // [64]
    pub bn1_var: Vec<f32>,              // [64]
    pub layer1: Vec<BasicBlockWeights>, // 2 blocks
    pub layer2: Vec<BasicBlockWeights>, // 2 blocks
    pub layer3: Vec<BasicBlockWeights>, // 2 blocks
    pub layer4: Vec<BasicBlockWeights>, // 2 blocks
    pub fc_w: Vec<f32>,                 // [num_classes, 512]
    pub fc_b: Vec<f32>,                 // [num_classes]
}

impl ResNet18Weights {
    /// Generate random weights for testing.
    pub fn random(num_classes: usize) -> Self {
        let mut seed = 42u64;
        let mut next_seed = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };

        Self {
            conv1_w: BasicBlockWeights::random(3, 64, 1, next_seed()).conv1_w[..64 * 3 * 3 * 3]
                .to_vec(),
            bn1_gamma: vec![1.0; 64],
            bn1_beta: vec![0.0; 64],
            bn1_mean: vec![0.0; 64],
            bn1_var: vec![1.0; 64],
            layer1: vec![
                BasicBlockWeights::random(64, 64, 1, next_seed()),
                BasicBlockWeights::random(64, 64, 1, next_seed()),
            ],
            layer2: vec![
                BasicBlockWeights::random(64, 128, 2, next_seed()),
                BasicBlockWeights::random(128, 128, 1, next_seed()),
            ],
            layer3: vec![
                BasicBlockWeights::random(128, 256, 2, next_seed()),
                BasicBlockWeights::random(256, 256, 1, next_seed()),
            ],
            layer4: vec![
                BasicBlockWeights::random(256, 512, 2, next_seed()),
                BasicBlockWeights::random(512, 512, 1, next_seed()),
            ],
            fc_w: {
                let scale = (2.0 / 512.0f64).sqrt() as f32;
                (0..num_classes * 512)
                    .map(|i| {
                        let v = ((i as u64).wrapping_mul(next_seed()) % 10007) as f32;
                        (v / 10007.0 - 0.5) * 2.0 * scale
                    })
                    .collect()
            },
            fc_b: vec![0.0; num_classes],
        }
    }

    /// Load weights from a SafeTensors file (requires `gpt2` feature for safetensors support).
    #[cfg(feature = "gpt2")]
    ///
    /// Expected key naming convention (matches PyTorch ResNet-18 CIFAR variant):
    /// - `conv1.weight`, `bn1.weight`, `bn1.bias`, `bn1.running_mean`, `bn1.running_var`
    /// - `layer{1-4}.{0-1}.conv{1-2}.weight`, `layer{1-4}.{0-1}.bn{1-2}.*`
    /// - `layer{1-4}.{0-1}.shortcut.conv.weight`, `layer{1-4}.{0-1}.shortcut.bn.*`
    /// - `fc.weight`, `fc.bias`
    pub fn from_safetensors(
        path: impl AsRef<std::path::Path>,
        num_classes: usize,
    ) -> std::result::Result<Self, crate::model::ModelError> {
        let raw = crate::model_generic::load_safetensors_raw(path)?;

        let get = |key: &str| -> std::result::Result<Vec<f32>, crate::model::ModelError> {
            raw.get(key)
                .map(|t| t.data.clone())
                .ok_or_else(|| crate::model::ModelError::MissingTensor(key.to_string()))
        };

        let load_block =
            |layer: usize,
             block: usize,
             has_shortcut: bool|
             -> std::result::Result<BasicBlockWeights, crate::model::ModelError> {
                let p = format!("layer{layer}.{block}");
                Ok(BasicBlockWeights {
                    conv1_w: get(&format!("{p}.conv1.weight"))?,
                    bn1_gamma: get(&format!("{p}.bn1.weight"))?,
                    bn1_beta: get(&format!("{p}.bn1.bias"))?,
                    bn1_mean: get(&format!("{p}.bn1.running_mean"))?,
                    bn1_var: get(&format!("{p}.bn1.running_var"))?,
                    conv2_w: get(&format!("{p}.conv2.weight"))?,
                    bn2_gamma: get(&format!("{p}.bn2.weight"))?,
                    bn2_beta: get(&format!("{p}.bn2.bias"))?,
                    bn2_mean: get(&format!("{p}.bn2.running_mean"))?,
                    bn2_var: get(&format!("{p}.bn2.running_var"))?,
                    shortcut_conv_w: if has_shortcut {
                        Some(get(&format!("{p}.shortcut.conv.weight"))?)
                    } else {
                        None
                    },
                    shortcut_bn_gamma: if has_shortcut {
                        Some(get(&format!("{p}.shortcut.bn.weight"))?)
                    } else {
                        None
                    },
                    shortcut_bn_beta: if has_shortcut {
                        Some(get(&format!("{p}.shortcut.bn.bias"))?)
                    } else {
                        None
                    },
                    shortcut_bn_mean: if has_shortcut {
                        Some(get(&format!("{p}.shortcut.bn.running_mean"))?)
                    } else {
                        None
                    },
                    shortcut_bn_var: if has_shortcut {
                        Some(get(&format!("{p}.shortcut.bn.running_var"))?)
                    } else {
                        None
                    },
                })
            };

        let _ = num_classes; // FC size inferred from weight shape
        Ok(Self {
            conv1_w: get("conv1.weight")?,
            bn1_gamma: get("bn1.weight")?,
            bn1_beta: get("bn1.bias")?,
            bn1_mean: get("bn1.running_mean")?,
            bn1_var: get("bn1.running_var")?,
            layer1: vec![
                load_block(1, 0, false)?, // 64→64, no shortcut
                load_block(1, 1, false)?, // 64→64, no shortcut
            ],
            layer2: vec![
                load_block(2, 0, true)?,  // 64→128, stride=2, shortcut
                load_block(2, 1, false)?, // 128→128, no shortcut
            ],
            layer3: vec![
                load_block(3, 0, true)?,  // 128→256, stride=2, shortcut
                load_block(3, 1, false)?, // 256→256, no shortcut
            ],
            layer4: vec![
                load_block(4, 0, true)?,  // 256→512, stride=2, shortcut
                load_block(4, 1, false)?, // 512→512, no shortcut
            ],
            fc_w: get("fc.weight")?,
            fc_b: get("fc.bias")?,
        })
    }
}

impl ResNet18 {
    /// Build ResNet-18 from weights.
    pub fn from_weights(
        weights: &ResNet18Weights,
        num_classes: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(&weights.conv1_w, None, 64, 3, 3, 3, 1, 1, registry)?;
        let bn1 = BatchNorm2d::new(
            &weights.bn1_gamma,
            &weights.bn1_beta,
            &weights.bn1_mean,
            &weights.bn1_var,
            1e-5,
            false,
            registry,
        )?;

        let make_layer = |block_weights: &[BasicBlockWeights],
                          in_ch,
                          out_ch,
                          stride|
         -> Result<Vec<BasicBlock>> {
            let mut blocks = Vec::new();
            blocks.push(BasicBlock::new(
                in_ch,
                out_ch,
                stride,
                &block_weights[0],
                registry,
            )?);
            for w in &block_weights[1..] {
                blocks.push(BasicBlock::new(out_ch, out_ch, 1, w, registry)?);
            }
            Ok(blocks)
        };

        let layer1 = make_layer(&weights.layer1, 64, 64, 1)?;
        let layer2 = make_layer(&weights.layer2, 64, 128, 2)?;
        let layer3 = make_layer(&weights.layer3, 128, 256, 2)?;
        let layer4 = make_layer(&weights.layer4, 256, 512, 2)?;

        let fc = Linear::new(
            &weights.fc_w,
            Some(&weights.fc_b),
            512,
            num_classes,
            registry,
        )?;

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass: input [C, H, W] or [N, C, H, W] → logits [N, num_classes].
    ///
    /// For CIFAR-10: input [3, 32, 32] → logits [10].
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // conv1 → bn1 → relu
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = ops::relu(&x, &self.registry)?;

        // Residual layers
        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward(&x)?;
        }
        for block in &self.layer3 {
            x = block.forward(&x)?;
        }
        for block in &self.layer4 {
            x = block.forward(&x)?;
        }

        // Global average pooling: [C, H, W] → [C]
        x = global_avg_pool(&x)?;

        // FC
        self.fc.forward(&x)
    }
}

/// Global average pooling: [C, H, W] → [C] or [N, C, H, W] → [N, C].
///
/// Averages over spatial dimensions (last 2 dims).
fn global_avg_pool(input: &GpuTensor) -> Result<GpuTensor> {
    let shape = input.shape();
    let host = input.to_host()?;
    let dev = input.device();

    match shape.len() {
        3 => {
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            let hw = (h * w) as f32;
            let mut out = vec![0.0f32; c];
            for ch in 0..c {
                let sum: f32 = host[ch * h * w..(ch + 1) * h * w].iter().sum();
                out[ch] = sum / hw;
            }
            GpuTensor::from_host(&out, &[c], dev)
        }
        4 => {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let hw = (h * w) as f32;
            let chw = c * h * w;
            let mut out = vec![0.0f32; n * c];
            for b in 0..n {
                for ch in 0..c {
                    let start = b * chw + ch * h * w;
                    let sum: f32 = host[start..start + h * w].iter().sum();
                    out[b * c + ch] = sum / hw;
                }
            }
            GpuTensor::from_host(&out, &[n, c], dev)
        }
        _ => Err(crate::nn::error::NnError::ShapeMismatch {
            expected: "3D [C,H,W] or 4D [N,C,H,W]".to_string(),
            actual: format!("{shape:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_registry() -> Arc<KernelRegistry> {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX load"))
    }

    #[test]
    fn test_resnet18_forward_cifar10() {
        let registry = test_registry();
        let dev = registry.device();

        let weights = ResNet18Weights::random(10);
        let model = ResNet18::from_weights(&weights, 10, &registry).unwrap();

        // CIFAR-10 image: [3, 32, 32]
        let input: Vec<f32> = (0..3 * 32 * 32)
            .map(|i| (i as f32 / (3.0 * 32.0 * 32.0)) - 0.5)
            .collect();
        let input_tensor = GpuTensor::from_host(&input, &[3, 32, 32], dev).unwrap();

        let logits = model.forward(&input_tensor).unwrap();
        let logits_host = logits.to_host().unwrap();

        // Linear outputs [1, 10] for single-sample input
        assert_eq!(logits_host.len(), 10);
        // Check no NaN
        assert!(
            logits_host.iter().all(|x| x.is_finite()),
            "logits contain NaN/Inf: {logits_host:?}"
        );
        eprintln!("ResNet-18 CIFAR-10 logits: {logits_host:?}");
    }
}
