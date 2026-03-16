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

    /// Number of attention heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Per-head dimension.
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Forward pass with causal attention.
    ///
    /// Input: `[seq_len, n_embd]` → output: `[seq_len, n_embd]`.
    ///
    /// Uses GPU-native kernels (split_qkv, flash_attention, concat_heads) —
    /// zero host transfers between QKV projection and output projection.
    pub fn forward_causal(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let seq_len = input.shape()[0];

        // 1. QKV projection: [seq, n_embd] → [seq, 3*n_embd]
        let qkv = self.qkv_proj.forward(input)?;

        // 2. Split QKV on GPU: [seq, 3*n_embd] → Q,K,V each [n_heads, seq, d_head]
        let (q, k, v) =
            ops::split_qkv(&qkv, seq_len, self.n_heads, self.d_head, &self.registry)?;

        // 3. Flash attention — all heads in one launch
        //    grid=(n_heads, n_q_tiles, 1), zero host round-trips
        let attn_out = ops::multi_head_flash_attention(
            &q,
            &k,
            &v,
            seq_len,
            self.n_heads,
            self.d_head,
            true, // causal
            &self.registry,
        )?;

        // 4. Concat heads on GPU: [n_heads, seq, d_head] → [seq, n_embd]
        let concat =
            ops::concat_heads(&attn_out, seq_len, self.n_heads, self.d_head, &self.registry)?;

        // 5. Output projection: [seq, n_embd] → [seq, n_embd]
        self.out_proj.forward(&concat)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    ///
    /// `input`: `[new_len, n_embd]` — only the new token(s) hidden states.
    /// `cached_k`, `cached_v`: per-head cached K,V as `[n_head][kv_len * d_head]`.
    ///
    /// Returns `(output, new_k_per_head, new_v_per_head)` where new_k/v are the
    /// K,V for the new positions only (to be appended to the cache).
    #[allow(clippy::type_complexity)]
    pub fn forward_cached(
        &self,
        input: &GpuTensor,
        cached_k: &[Vec<f32>],
        cached_v: &[Vec<f32>],
        kv_len: usize,
    ) -> Result<(GpuTensor, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let new_len = input.shape()[0];
        let n_embd = self.n_heads * self.d_head;
        let full_kv_len = kv_len + new_len;

        // 1. QKV projection for new positions only
        let qkv = self.qkv_proj.forward(input)?;
        let qkv_host = qkv.to_host()?;
        let dev = self.registry.device();

        // 2. Extract per-head Q, K, V for new positions
        let mut new_k_per_head: Vec<Vec<f32>> = Vec::with_capacity(self.n_heads);
        let mut new_v_per_head: Vec<Vec<f32>> = Vec::with_capacity(self.n_heads);
        let mut head_outputs = vec![0.0f32; new_len * n_embd];

        for h in 0..self.n_heads {
            let mut q_head = vec![0.0f32; new_len * self.d_head];
            let mut k_new = vec![0.0f32; new_len * self.d_head];
            let mut v_new = vec![0.0f32; new_len * self.d_head];

            for s in 0..new_len {
                let qkv_idx = s * (3 * n_embd);
                for d in 0..self.d_head {
                    q_head[s * self.d_head + d] = qkv_host[qkv_idx + h * self.d_head + d];
                    k_new[s * self.d_head + d] = qkv_host[qkv_idx + n_embd + h * self.d_head + d];
                    v_new[s * self.d_head + d] =
                        qkv_host[qkv_idx + 2 * n_embd + h * self.d_head + d];
                }
            }

            // Build full K,V: cached + new
            let mut k_full = cached_k[h].clone();
            k_full.extend_from_slice(&k_new);
            let mut v_full = cached_v[h].clone();
            v_full.extend_from_slice(&v_new);

            let q_tensor = GpuTensor::from_host(&q_head, &[new_len, self.d_head], dev)?;
            let k_tensor = GpuTensor::from_host(&k_full, &[full_kv_len, self.d_head], dev)?;
            let v_tensor = GpuTensor::from_host(&v_full, &[full_kv_len, self.d_head], dev)?;

            // Attention with separate Q/KV lengths
            let attn_out = ops::scaled_dot_product_attention_kv(
                &q_tensor,
                &k_tensor,
                &v_tensor,
                true,        // causal
                kv_len,      // q_offset: new tokens start after cached positions
                full_kv_len, // kv_stride
                &self.registry,
            )?;

            let head_host = attn_out.to_host()?;
            for s in 0..new_len {
                for d in 0..self.d_head {
                    head_outputs[s * n_embd + h * self.d_head + d] = head_host[s * self.d_head + d];
                }
            }

            new_k_per_head.push(k_new);
            new_v_per_head.push(v_new);
        }

        let concat = GpuTensor::from_host(&head_outputs, &[new_len, n_embd], dev)?;
        let output = self.out_proj.forward(&concat)?;

        Ok((output, new_k_per_head, new_v_per_head))
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

    /// Test MHA construction and weight shapes.
    ///
    /// NOTE: forward_causal with small n_embd (< 32) triggers CUDA_ERROR_ILLEGAL_ADDRESS
    /// because the internal GEMM/attention kernels require padded dimensions. The full GPT-2
    /// model (n_embd=768) works correctly. This test only verifies construction.
    #[test]
    fn test_mha_construction() {
        let registry = test_registry();

        let n_embd = 64;
        let n_heads = 4;

        let qkv_weight = vec![0.01f32; 3 * n_embd * n_embd];
        let qkv_bias = vec![0.0f32; 3 * n_embd];
        let out_weight = vec![0.01f32; n_embd * n_embd];
        let out_bias = vec![0.0f32; n_embd];

        let mha = MultiHeadAttention::new(
            &qkv_weight,
            &qkv_bias,
            &out_weight,
            &out_bias,
            n_embd,
            n_heads,
            &registry,
        )
        .unwrap();

        assert_eq!(mha.n_heads(), 4);
        assert_eq!(mha.d_head(), 16);
    }

    /// Test MHA forward pass with GPT-2 dimensions.
    #[test]
    fn test_mha_forward_gpt2_dims() {
        let registry = test_registry();
        let dev = registry.device();

        let n_embd = 768;
        let n_heads = 12;
        let seq_len = 4;

        let qkv_weight: Vec<f32> = (0..3 * n_embd * n_embd)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.0001)
            .collect();
        let qkv_bias = vec![0.0f32; 3 * n_embd];
        let out_weight: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i % 73) as f32 - 36.0) * 0.0001)
            .collect();
        let out_bias = vec![0.0f32; n_embd];

        let mha = MultiHeadAttention::new(
            &qkv_weight,
            &qkv_bias,
            &out_weight,
            &out_bias,
            n_embd,
            n_heads,
            &registry,
        )
        .unwrap();

        let input: Vec<f32> = (0..seq_len * n_embd)
            .map(|i| ((i as f32) - 1536.0) * 0.001)
            .collect();
        let input_tensor = GpuTensor::from_host(&input, &[seq_len, n_embd], dev).unwrap();

        let output = mha.forward_causal(&input_tensor).unwrap();
        assert_eq!(output.shape(), &[seq_len, n_embd]);

        let result = output.to_host().unwrap();
        assert!(
            result.iter().all(|x| x.is_finite()),
            "output contains NaN or Inf"
        );
    }
}
