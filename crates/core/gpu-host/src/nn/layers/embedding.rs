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
}
