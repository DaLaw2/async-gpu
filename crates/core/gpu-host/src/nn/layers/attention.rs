//! Multi-head attention layer.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::layers::Module;
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Multi-head self-attention.
///
/// Implements: QKV projection → split heads → attention → concat → output projection.
pub struct MultiHeadAttention {
    /// QKV combined projection: [3 * n_embd, n_embd]
    qkv_proj: super::Linear,
    /// Output projection: [n_embd, n_embd]
    out_proj: super::Linear,
    n_heads: usize,
    d_head: usize,
    registry: Arc<KernelRegistry>,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention layer.
    ///
    /// `qkv_weight`: `[3*n_embd, n_embd]`, `qkv_bias`: `[3*n_embd]`.
    /// `out_weight`: `[n_embd, n_embd]`, `out_bias`: `[n_embd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv_weight: &[f32],
        qkv_bias: &[f32],
        out_weight: &[f32],
        out_bias: &[f32],
        n_embd: usize,
        n_heads: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let d_head = n_embd / n_heads;
        let qkv_proj =
            super::Linear::new(qkv_weight, Some(qkv_bias), n_embd, 3 * n_embd, registry)?;
        let out_proj = super::Linear::new(out_weight, Some(out_bias), n_embd, n_embd, registry)?;

        Ok(Self {
            qkv_proj,
            out_proj,
            n_heads,
            d_head,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass with causal attention.
    ///
    /// Input: `[seq_len, n_embd]` → output: `[seq_len, n_embd]`.
    pub fn forward_causal(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let seq_len = input.shape()[0];

        // 1. QKV projection: [seq, n_embd] → [seq, 3*n_embd]
        let qkv = self.qkv_proj.forward(input)?;

        // 2. Split into Q, K, V and reshape for multi-head
        let qkv_host = qkv.to_host()?;
        let n_embd = self.n_heads * self.d_head;
        let dev = self.registry.device();

        // For each head, run attention separately
        let mut head_outputs = vec![0.0f32; seq_len * n_embd];

        for h in 0..self.n_heads {
            // Extract Q, K, V for this head
            let mut q_head = vec![0.0f32; seq_len * self.d_head];
            let mut k_head = vec![0.0f32; seq_len * self.d_head];
            let mut v_head = vec![0.0f32; seq_len * self.d_head];

            for s in 0..seq_len {
                for d in 0..self.d_head {
                    let qkv_idx = s * (3 * n_embd);
                    q_head[s * self.d_head + d] = qkv_host[qkv_idx + h * self.d_head + d];
                    k_head[s * self.d_head + d] = qkv_host[qkv_idx + n_embd + h * self.d_head + d];
                    v_head[s * self.d_head + d] =
                        qkv_host[qkv_idx + 2 * n_embd + h * self.d_head + d];
                }
            }

            let q_tensor = GpuTensor::from_host(&q_head, &[seq_len, self.d_head], dev)?;
            let k_tensor = GpuTensor::from_host(&k_head, &[seq_len, self.d_head], dev)?;
            let v_tensor = GpuTensor::from_host(&v_head, &[seq_len, self.d_head], dev)?;

            // 3. Attention per head
            let attn_out = ops::scaled_dot_product_attention(
                &q_tensor,
                &k_tensor,
                &v_tensor,
                true, // causal
                &self.registry,
            )?;

            // Collect head output
            let head_host = attn_out.to_host()?;
            for s in 0..seq_len {
                for d in 0..self.d_head {
                    head_outputs[s * n_embd + h * self.d_head + d] = head_host[s * self.d_head + d];
                }
            }
        }

        // 4. Concat heads (already done by interleaving above)
        let concat = GpuTensor::from_host(&head_outputs, &[seq_len, n_embd], dev)?;

        // 5. Output projection
        self.out_proj.forward(&concat)
    }
}
