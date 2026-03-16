//! LoRA (Low-Rank Adaptation) layer wrapper.
//!
//! Wraps a frozen Linear layer with trainable low-rank matrices A and B.
//! Forward: `y = Linear(x) + x @ A @ B` where A=[in, rank], B=[rank, out].

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

use super::{Linear, Module};

/// LoRA adapter wrapping a frozen Linear layer.
///
/// Only `lora_a` and `lora_b` are trainable. The base `linear` is frozen.
pub struct LoraLinear {
    /// Frozen base linear layer.
    pub linear: Linear,
    /// Low-rank down-projection: [in_features, rank].
    pub lora_a: GpuTensor,
    /// Low-rank up-projection: [rank, out_features].
    pub lora_b: GpuTensor,
    /// Scaling factor (alpha / rank).
    pub scaling: f32,
    registry: Arc<KernelRegistry>,
}

impl LoraLinear {
    /// Create a LoRA adapter around an existing Linear layer.
    ///
    /// `rank`: LoRA rank (typically 4-16).
    /// `alpha`: scaling factor (typically equal to rank).
    pub fn new(
        linear: Linear,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();

        // Initialize A with small random values, B with zeros (standard LoRA init)
        let a_data: Vec<f32> = (0..in_features * rank)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 0.01)
            .collect();
        let b_data = vec![0.0f32; rank * out_features];

        let mut lora_a = GpuTensor::from_host(&a_data, &[in_features, rank], dev)?;
        lora_a.set_requires_grad(true);

        let mut lora_b = GpuTensor::from_host(&b_data, &[rank, out_features], dev)?;
        lora_b.set_requires_grad(true);

        Ok(Self {
            linear,
            lora_a,
            lora_b,
            scaling: alpha / rank as f32,
            registry: Arc::clone(registry),
        })
    }
}

impl Module for LoraLinear {
    /// Forward: y = Linear(x) + scaling * (x @ A @ B)
    fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // Base forward (frozen weights, pre-padded)
        let mut base_out = self.linear.forward(input)?;

        // LoRA path: x @ A → [batch, rank], then [batch, rank] @ B → [batch, out]
        let xa = ops::matmul(input, &self.lora_a, &self.registry)?;
        let lora_out = ops::matmul(&xa, &self.lora_b, &self.registry)?;

        // Scale and add: base_out += scaling * lora_out
        // For now, do on host (GPU fused scale+add kernel would be better)
        let mut base_host = base_out.to_host()?;
        let lora_host = lora_out.to_host()?;
        for (b, l) in base_host.iter_mut().zip(lora_host.iter()) {
            *b += self.scaling * l;
        }

        GpuTensor::from_host(&base_host, base_out.shape(), self.registry.device())
    }
}
