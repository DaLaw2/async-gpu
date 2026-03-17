//! INT4 quantized Linear layer (W4A16 — 4-bit weights, f32 activations).
//!
//! Uses pre-quantized packed u32 weights + per-group scale factors.
//! Forward pass: dequantize on-the-fly via `int4_gemm_w4a16` GPU kernel.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// INT4 quantized Linear layer.
///
/// Stores weights as packed INT4 (8 values per u32) with per-group scale factors.
/// Forward pass dequantizes on-the-fly during GEMM — no f32 weight buffer needed.
pub struct Int4Linear {
    /// Packed INT4 weights: [K/8, N] as u32 on GPU.
    packed_weights: cudarc::driver::CudaSlice<u32>,
    /// Per-group scale factors: [K/group_size, N] as f32 on GPU.
    scales: cudarc::driver::CudaSlice<f32>,
    /// Optional bias: [N] as f32 on GPU.
    bias: Option<GpuTensor>,
    /// Input features (K dimension).
    in_features: usize,
    /// Output features (N dimension).
    out_features: usize,
    /// Quantization group size.
    group_size: usize,
    registry: Arc<KernelRegistry>,
}

impl Int4Linear {
    /// Create from pre-quantized data.
    ///
    /// - `packed`: [K/8, N] packed u32 (8 INT4 values per u32, unsigned [0,15] zero_point=8)
    /// - `scales`: [n_groups, N] per-group scale factors
    /// - `bias`: optional [N] bias vector
    pub fn new(
        packed: &[u32],
        scales: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        let packed_dev = dev.htod_copy(packed.to_vec()).map_err(NnError::Cuda)?;
        let scales_dev = dev.htod_copy(scales.to_vec()).map_err(NnError::Cuda)?;

        let bias_dev = if let Some(b) = bias {
            Some(GpuTensor::from_host(b, &[out_features], dev)?)
        } else {
            None
        };

        Ok(Self {
            packed_weights: packed_dev,
            scales: scales_dev,
            bias: bias_dev,
            in_features,
            out_features,
            group_size,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass: input [*, K] → output [*, N].
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let ndim = input.ndim();
        let k = input.shape()[ndim - 1];
        if k != self.in_features {
            return Err(NnError::ShapeMismatch {
                expected: format!("last dim = {}", self.in_features),
                actual: format!("last dim = {k}"),
            });
        }

        let batch: usize = input.shape()[..ndim - 1].iter().product::<usize>().max(1);
        let m = batch;
        let n = self.out_features;

        // Reshape input to [batch, K]
        let input_2d = if ndim == 2 {
            input.clone_tensor()?
        } else {
            input.reshape(&[m, k])?
        };

        // Launch int4_gemm_w4a16 kernel
        let dev = self.registry.device();
        let c_dev = dev.alloc_zeros::<f32>(m * n).map_err(NnError::Cuda)?;
        let status = dev.htod_sync_copy(&[0u32]).map_err(NnError::Cuda)?;

        let func = self.registry.get("int4_gemm_w4a16")?;
        let total = (m * n) as u32;
        let config = KernelRegistry::config_1d(total);
        unsafe {
            func.launch(
                config,
                (
                    input_2d.data(),
                    &self.packed_weights,
                    &self.scales,
                    &c_dev,
                    m as u32,
                    n as u32,
                    k as u32,
                    self.group_size as u32,
                    &status,
                ),
            )
            .map_err(NnError::Cuda)?;
        }

        let mut output = GpuTensor::from_data(c_dev, &[m, n], Arc::clone(dev));

        // Add bias
        if let Some(ref bias) = self.bias {
            crate::nn::ops::bias_add(&mut output, bias, &self.registry)?;
        }

        // Reshape back
        if ndim > 2 {
            let mut out_shape: Vec<usize> = input.shape()[..ndim - 1].to_vec();
            out_shape.push(n);
            output.reshape(&out_shape)
        } else {
            Ok(output)
        }
    }

    /// Memory usage in bytes (packed weights + scales).
    pub fn memory_bytes(&self) -> usize {
        let k_packed = self.in_features / 8;
        let n_groups = self.in_features.div_ceil(self.group_size);
        k_packed * self.out_features * 4 // packed u32
            + n_groups * self.out_features * 4 // scales f32
            + if self.bias.is_some() {
                self.out_features * 4
            } else {
                0
            }
    }
}
