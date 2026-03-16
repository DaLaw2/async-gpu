//! Conv2d layer: 2D convolution with weight ownership.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::Module;

/// 2D convolution layer.
///
/// Weight: `[C_out, C_in, kH, kW]`, bias: `[C_out]` (optional).
pub struct Conv2d {
    weight: GpuTensor,
    bias: Option<GpuTensor>,
    stride: usize,
    padding: usize,
    registry: Arc<KernelRegistry>,
}

impl Conv2d {
    /// Create a new Conv2d layer.
    ///
    /// `weight`: `[c_out, c_in, kh, kw]`, `bias`: `[c_out]` (optional).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        let weight_tensor = GpuTensor::from_host(weight, &[c_out, c_in, kh, kw], dev)?;
        let bias_tensor = if let Some(b) = bias {
            Some(GpuTensor::from_host(b, &[c_out], dev)?)
        } else {
            None
        };

        Ok(Self {
            weight: weight_tensor,
            bias: bias_tensor,
            stride,
            padding,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for Conv2d {
    /// Forward pass: input `[C_in, H, W]` → output `[C_out, H_out, W_out]`.
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        ops::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            &self.registry,
        )
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

    /// CPU f64 reference for Conv2d (im2col + matmul approach, but computed directly).
    fn cpu_conv2d(
        input: &[f32],  // [C_in, H, W]
        weight: &[f32], // [C_out, C_in, kH, kW]
        bias: Option<&[f32]>,
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
    ) -> Vec<f32> {
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kw) / stride + 1;
        let mut out = vec![0.0f32; c_out * h_out * w_out];

        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f64;
                    for ci in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * stride + fh;
                                let iw = ow * stride + fw;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let in_val = input[ci * h * w + ih * w + iw] as f64;
                                    let w_val = weight
                                        [co * (c_in * kh * kw) + ci * (kh * kw) + fh * kw + fw]
                                        as f64;
                                    sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[co] as f64;
                    }
                    out[co * h_out * w_out + oh * w_out + ow] = sum as f32;
                }
            }
        }
        out
    }

    #[test]
    fn test_conv2d_1x1_identity() {
        let registry = test_registry();
        let dev = registry.device();
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2, 1, 1] identity
        let input: Vec<f32> = (0..2 * 4 * 4).map(|i| i as f32 * 0.1).collect();
        let expected = cpu_conv2d(&input, &weight, None, 2, 4, 4, 2, 1, 1, 1, 0);
        let layer = Conv2d::new(&weight, None, 2, 2, 1, 1, 1, 0, &registry).unwrap();
        let input_t = GpuTensor::from_host(&input, &[2, 4, 4], dev).unwrap();
        let output_t = layer.forward(&input_t).unwrap();
        let gpu = output_t.to_host().unwrap();
        let max_err: f32 = expected
            .iter()
            .zip(gpu.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "1x1 identity conv max_err={max_err}");
    }

    #[test]
    fn test_conv2d_multichannel() {
        let registry = test_registry();
        let dev = registry.device();
        let (c_in, c_out, h, w, kh, kw) = (3, 4, 8, 8, 3, 3);
        let weight: Vec<f32> = (0..c_out * c_in * kh * kw)
            .map(|i| ((i as f32) - 54.0) * 0.01)
            .collect();
        let input: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i as f32) - 96.0) * 0.01)
            .collect();
        let expected = cpu_conv2d(&input, &weight, None, c_in, h, w, c_out, kh, kw, 1, 1);
        let layer = Conv2d::new(&weight, None, c_out, c_in, kh, kw, 1, 1, &registry).unwrap();
        let input_t = GpuTensor::from_host(&input, &[c_in, h, w], dev).unwrap();
        let output_t = layer.forward(&input_t).unwrap();
        let gpu = output_t.to_host().unwrap();
        let max_err: f32 = expected
            .iter()
            .zip(gpu.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "multichannel conv max_err={max_err}");
    }

    /// CIFAR-10 dimensions: input [3,32,32], weight [8,3,3,3], stride=1, pad=1.
    /// Output should be [8,32,32]. Compare GPU vs CPU reference.
    #[test]
    fn test_conv2d_cifar10_dims() {
        let registry = test_registry();
        let dev = registry.device();

        let c_in = 3;
        let c_out = 8;
        let h = 32;
        let w = 32;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;

        // Deterministic pseudo-random weights and input
        let weight: Vec<f32> = (0..c_out * c_in * kh * kw)
            .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
            .collect();
        let input: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
            .collect();
        let bias: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.1 - 0.4).collect();

        let expected = cpu_conv2d(
            &input,
            &weight,
            Some(&bias),
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            stride,
            padding,
        );

        let layer = Conv2d::new(
            &weight,
            Some(&bias),
            c_out,
            c_in,
            kh,
            kw,
            stride,
            padding,
            &registry,
        )
        .unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[c_in, h, w], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kw) / stride + 1;
        assert_eq!(output_tensor.shape(), &[c_out, h_out, w_out]);
        assert_eq!(h_out, 32);
        assert_eq!(w_out, 32);
        assert_eq!(expected.len(), gpu_result.len());

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("CIFAR-10 conv2d max_err = {max_err}");
        assert!(
            max_err < 0.1,
            "CIFAR-10 conv2d max_err={max_err} exceeds threshold"
        );
    }

    /// Batched conv2d: input [2, 3, 8, 8], weight [4, 3, 3, 3], stride=1, pad=1.
    /// Verify each sample in the batch matches the single-sample CPU reference.
    #[test]
    fn test_conv2d_batched_matches_cpu() {
        let registry = test_registry();
        let dev = registry.device();

        let batch = 2;
        let c_in = 3;
        let c_out = 4;
        let h = 8;
        let w = 8;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kw) / stride + 1;

        let weight: Vec<f32> = (0..c_out * c_in * kh * kw)
            .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
            .collect();
        // Two different samples
        let input: Vec<f32> = (0..batch * c_in * h * w)
            .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
            .collect();

        // CPU reference per sample
        let sample_size = c_in * h * w;
        let mut expected = vec![0.0f32; batch * c_out * h_out * w_out];
        for b in 0..batch {
            let sample = &input[b * sample_size..(b + 1) * sample_size];
            let ref_out = cpu_conv2d(
                sample, &weight, None, c_in, h, w, c_out, kh, kw, stride, padding,
            );
            let out_size = c_out * h_out * w_out;
            expected[b * out_size..(b + 1) * out_size].copy_from_slice(&ref_out);
        }

        let weight_t = GpuTensor::from_host(&weight, &[c_out, c_in, kh, kw], dev).unwrap();
        let input_t = GpuTensor::from_host(&input, &[batch, c_in, h, w], dev).unwrap();
        let output_t =
            crate::nn::ops::conv2d(&input_t, &weight_t, None, stride, padding, &registry).unwrap();
        let gpu_result = output_t.to_host().unwrap();

        assert_eq!(output_t.shape(), &[batch, c_out, h_out, w_out]);
        assert_eq!(expected.len(), gpu_result.len());

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Batched conv2d max_err = {max_err}");
        assert!(
            max_err < 0.1,
            "Batched conv2d max_err={max_err} exceeds threshold"
        );
    }

    #[test]
    fn test_conv2d_3x3_matches_cpu() {
        let registry = test_registry();
        let dev = registry.device();

        let c_in = 1;
        let c_out = 1;
        let h = 5;
        let w = 5;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 0;

        // Simple 3x3 averaging filter
        let weight = vec![1.0 / 9.0; 9]; // [1, 1, 3, 3]
        let input: Vec<f32> = (0..h * w).map(|i| i as f32).collect();

        let expected = cpu_conv2d(
            &input, &weight, None, c_in, h, w, c_out, kh, kw, stride, padding,
        );

        let layer = Conv2d::new(
            &weight, None, c_out, c_in, kh, kw, stride, padding, &registry,
        )
        .unwrap();
        let input_tensor = GpuTensor::from_host(&input, &[c_in, h, w], dev).unwrap();
        let output_tensor = layer.forward(&input_tensor).unwrap();
        let gpu_result = output_tensor.to_host().unwrap();

        let h_out = (h - kh) / stride + 1;
        let w_out = (w - kw) / stride + 1;
        assert_eq!(output_tensor.shape(), &[c_out, h_out, w_out]);

        let max_err: f32 = expected
            .iter()
            .zip(gpu_result.iter())
            .map(|(e, g)| (e - g).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-2,
            "max absolute error {max_err} exceeds 1e-2 (cpu={expected:?} gpu={gpu_result:?})"
        );
    }
}
