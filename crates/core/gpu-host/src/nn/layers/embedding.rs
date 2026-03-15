//! Embedding lookup layer.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Embedding table lookup.
///
/// wte: `[vocab_size, embed_dim]`, wpe: `[max_seq, embed_dim]`.
/// Combines token embeddings + position embeddings.
pub struct Embedding {
    /// Token embedding weights.
    pub wte: GpuTensor,
    /// Position embedding weights.
    pub wpe: GpuTensor,
    /// Embedding dimension.
    pub embed_dim: usize,
    registry: Arc<KernelRegistry>,
}

impl Embedding {
    /// Create a new Embedding layer.
    pub fn new(
        wte: &[f32],
        wpe: &[f32],
        vocab_size: usize,
        max_seq: usize,
        embed_dim: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let dev = registry.device();
        Ok(Self {
            wte: GpuTensor::from_host(wte, &[vocab_size, embed_dim], dev)?,
            wpe: GpuTensor::from_host(wpe, &[max_seq, embed_dim], dev)?,
            embed_dim,
            registry: Arc::clone(registry),
        })
    }

    /// Look up token embeddings + position embeddings.
    ///
    /// `token_ids`: device buffer of `u32` with `seq_len` elements.
    /// Returns `[seq_len, embed_dim]`.
    pub fn forward_tokens(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<GpuTensor> {
        crate::nn::ops::embedding_lookup(&self.wte, &self.wpe, token_ids, seq_len, &self.registry)
    }

    /// Look up token embeddings + position embeddings with a position offset.
    ///
    /// Like [`forward_tokens`](Self::forward_tokens) but uses positions
    /// `[pos_offset .. pos_offset + seq_len]` instead of `[0 .. seq_len]`.
    /// Needed for KV-cached decoding where new tokens start at a later position.
    pub fn forward_tokens_with_offset(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
        pos_offset: usize,
    ) -> Result<GpuTensor> {
        if pos_offset == 0 {
            return self.forward_tokens(token_ids, seq_len);
        }

        // The kernel always uses position 0..seq_len for wpe lookup.
        // Fix by subtracting wrong wpe and adding correct wpe on host.
        let output = crate::nn::ops::embedding_lookup(
            &self.wte,
            &self.wpe,
            token_ids,
            seq_len,
            &self.registry,
        )?;

        let wpe_host = self.wpe.to_host()?;
        let mut out_host = output.to_host()?;
        let d = self.embed_dim;

        for s in 0..seq_len {
            let wrong_pos = s;
            let right_pos = s + pos_offset;
            for i in 0..d {
                out_host[s * d + i] += wpe_host[right_pos * d + i] - wpe_host[wrong_pos * d + i];
            }
        }

        let dev = self.registry.device();
        GpuTensor::from_host(&out_host, &[seq_len, d], dev)
    }
}
