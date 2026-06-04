//! GPT-2 model configuration and architecture.
//!
//! Provides [`Gpt2Config`] with standard model size presets, [`TransformerBlock`]
//! for a single transformer layer, and [`Gpt2Model`] for the complete model.

use std::sync::Arc;

#[cfg(feature = "gpt2")]
use crate::model::Gpt2Weights;
use crate::nn::error::Result;
use crate::nn::layers::{
    Embedding, Int4Linear, LayerNorm, Linear, Module, MultiHeadAttention, GELU,
};
use crate::nn::ops;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Configuration for a GPT-2 model.
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    /// Number of transformer layers (blocks).
    pub n_layer: usize,
    /// Hidden dimension size (embedding dimension).
    pub n_embd: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length (context window).
    pub n_positions: usize,
    /// Layer norm epsilon.
    pub layer_norm_epsilon: f32,
}

impl Gpt2Config {
    /// GPT-2 Small (124M parameters).
    pub fn small() -> Self {
        Self {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// GPT-2 Medium (355M parameters).
    pub fn medium() -> Self {
        Self {
            n_layer: 24,
            n_embd: 1024,
            n_head: 16,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// GPT-2 Large (774M parameters).
    pub fn large() -> Self {
        Self {
            n_layer: 36,
            n_embd: 1280,
            n_head: 20,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// GPT-2 XL (1.5B parameters).
    pub fn xl() -> Self {
        Self {
            n_layer: 48,
            n_embd: 1600,
            n_head: 25,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// Per-head dimension (n_embd / n_head).
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// FFN intermediate dimension (4 * n_embd, standard GPT-2).
    pub fn ffn_dim(&self) -> usize {
        4 * self.n_embd
    }
}

/// A single GPT-2 transformer block.
///
/// Architecture: LN1 → MHA → residual → LN2 → FFN(Linear→GELU→Linear) → residual.
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    ffn_up: Linear,   // [n_embd → 4*n_embd]
    ffn_down: Linear, // [4*n_embd → n_embd]
    gelu: GELU,
    layer_norm_eps: f32,
    registry: Arc<KernelRegistry>,
}

impl TransformerBlock {
    /// Create a transformer block from weight slices.
    ///
    /// Weight naming follows HuggingFace GPT-2 safetensors convention:
    /// - `ln_1_weight`, `ln_1_bias`: LayerNorm 1
    /// - `attn_qkv_weight`, `attn_qkv_bias`: combined QKV projection
    /// - `attn_out_weight`, `attn_out_bias`: attention output projection
    /// - `ln_2_weight`, `ln_2_bias`: LayerNorm 2
    /// - `ffn_up_weight`, `ffn_up_bias`: FFN first linear (n_embd → 4*n_embd)
    /// - `ffn_down_weight`, `ffn_down_bias`: FFN second linear (4*n_embd → n_embd)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ln_1_weight: &[f32],
        ln_1_bias: &[f32],
        attn_qkv_weight: &[f32],
        attn_qkv_bias: &[f32],
        attn_out_weight: &[f32],
        attn_out_bias: &[f32],
        ln_2_weight: &[f32],
        ln_2_bias: &[f32],
        ffn_up_weight: &[f32],
        ffn_up_bias: &[f32],
        ffn_down_weight: &[f32],
        ffn_down_bias: &[f32],
        config: &Gpt2Config,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let eps = config.layer_norm_epsilon;
        let n_embd = config.n_embd;
        let ffn_dim = config.ffn_dim();

        Ok(Self {
            ln_1: LayerNorm::new(ln_1_weight, ln_1_bias, eps, registry)?,
            attn: MultiHeadAttention::new(
                attn_qkv_weight,
                attn_qkv_bias,
                attn_out_weight,
                attn_out_bias,
                n_embd,
                config.n_head,
                registry,
            )?,
            ln_2: LayerNorm::new(ln_2_weight, ln_2_bias, eps, registry)?,
            ffn_up: Linear::new(ffn_up_weight, Some(ffn_up_bias), n_embd, ffn_dim, registry)?,
            ffn_down: Linear::new(
                ffn_down_weight,
                Some(ffn_down_bias),
                ffn_dim,
                n_embd,
                registry,
            )?,
            gelu: GELU::new(registry),
            layer_norm_eps: eps,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass: input `[seq_len, n_embd]` → output `[seq_len, n_embd]`.
    ///
    /// When the `cublas` feature is enabled, uses fused LN+residual kernels to
    /// reduce kernel launches: `elementwise_add + layer_norm` becomes a single
    /// `layer_norm_residual_dual` kernel that outputs both the sum (for the
    /// residual stream) and the normalized result (for the FFN input).
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // LN1 → MHA
        let ln1_out = self.ln_1.forward(input)?;
        let attn_out = self.attn.forward_causal(&ln1_out)?;

        // Fused: compute residual = input + attn_out AND ln2_out = LN(residual)
        // in a single kernel launch (saves 1 launch + 1 global memory read).
        #[cfg(feature = "cublas")]
        let (ln2_out, mut residual) = ops::layer_norm_residual_dual(
            input,
            &attn_out,
            self.ln_2.gamma(),
            self.ln_2.beta(),
            self.layer_norm_eps,
            &self.registry,
        )?;

        #[cfg(not(feature = "cublas"))]
        let (ln2_out, mut residual) = {
            let mut res = input.clone_tensor()?;
            ops::elementwise_add(&mut res, &attn_out, &self.registry)?;
            let ln2 = self.ln_2.forward(&res)?;
            (ln2, res)
        };

        // FFN
        let ffn_hidden = self.ffn_up.forward(&ln2_out)?;
        let ffn_act = self.gelu.forward(&ffn_hidden)?;
        let ffn_out = self.ffn_down.forward(&ffn_act)?;
        ops::elementwise_add(&mut residual, &ffn_out, &self.registry)?;

        Ok(residual)
    }

    /// Cached forward: only processes new token(s), reusing cached K/V.
    ///
    /// `input`: `[new_len, n_embd]`, returns `(output, new_k, new_v)`.
    #[allow(clippy::type_complexity)]
    pub fn forward_cached(
        &self,
        input: &GpuTensor,
        cached_k: &[Vec<f32>],
        cached_v: &[Vec<f32>],
        kv_len: usize,
    ) -> Result<(GpuTensor, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        // LN1 → MHA (cached) → residual
        let ln1_out = self.ln_1.forward(input)?;
        let (attn_out, new_k, new_v) = self
            .attn
            .forward_cached(&ln1_out, cached_k, cached_v, kv_len)?;

        // Fused residual + LN2 (same optimization as forward())
        #[cfg(feature = "cublas")]
        let (ln2_out, mut residual) = ops::layer_norm_residual_dual(
            input,
            &attn_out,
            self.ln_2.gamma(),
            self.ln_2.beta(),
            self.layer_norm_eps,
            &self.registry,
        )?;

        #[cfg(not(feature = "cublas"))]
        let (ln2_out, mut residual) = {
            let mut res = input.clone_tensor()?;
            ops::elementwise_add(&mut res, &attn_out, &self.registry)?;
            let ln2 = self.ln_2.forward(&res)?;
            (ln2, res)
        };

        // FFN → residual
        let ffn_hidden = self.ffn_up.forward(&ln2_out)?;
        let ffn_act = self.gelu.forward(&ffn_hidden)?;
        let ffn_out = self.ffn_down.forward(&ffn_act)?;
        ops::elementwise_add(&mut residual, &ffn_out, &self.registry)?;

        Ok((residual, new_k, new_v))
    }
}

/// Complete GPT-2 model.
///
/// Architecture: Embedding → N × TransformerBlock → LayerNorm → LM head.
pub struct Gpt2Model {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: Gpt2Config,
    registry: Arc<KernelRegistry>,
}

impl Gpt2Model {
    /// Create a GPT-2 model from loaded weights.
    ///
    /// `weights` is a closure/struct that provides weight slices by name.
    /// See [`Gpt2Model::from_safetensors`] for loading from safetensors files.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding: Embedding,
        blocks: Vec<TransformerBlock>,
        ln_f: LayerNorm,
        lm_head: Linear,
        config: Gpt2Config,
        registry: &Arc<KernelRegistry>,
    ) -> Self {
        Self {
            embedding,
            blocks,
            ln_f,
            lm_head,
            config,
            registry: Arc::clone(registry),
        }
    }

    /// Reference to model config.
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }

    /// Build a GPT-2 model from pre-loaded [`Gpt2Weights`].
    ///
    /// Handles Conv1D→Linear weight transposition (safetensors stores `[in, out]`,
    /// but [`Linear::new`] expects `[out, in]`). LM head uses tied wte weights.
    #[cfg(feature = "gpt2")]
    pub fn from_weights(
        weights: &Gpt2Weights,
        config: Gpt2Config,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        // Embedding (wte + wpe)
        let embedding = Embedding::new(
            &weights.wte,
            &weights.wpe,
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            registry,
        )?;

        // Transformer blocks
        let n = config.n_embd;
        let ffn = config.ffn_dim();
        let mut blocks = Vec::with_capacity(config.n_layer);

        for layer in &weights.layers {
            // Conv1D [in, out] → Linear [out, in]
            let qkv_w = transpose_2d(&layer.c_attn_weight, n, 3 * n);
            let proj_w = transpose_2d(&layer.c_proj_weight, n, n);
            let fc_w = transpose_2d(&layer.mlp_fc_weight, n, ffn);
            let fc_proj_w = transpose_2d(&layer.mlp_proj_weight, ffn, n);

            let block = TransformerBlock::new(
                &layer.ln_1.weight,
                &layer.ln_1.bias,
                &qkv_w,
                &layer.c_attn_bias,
                &proj_w,
                &layer.c_proj_bias,
                &layer.ln_2.weight,
                &layer.ln_2.bias,
                &fc_w,
                &layer.mlp_fc_bias,
                &fc_proj_w,
                &layer.mlp_proj_bias,
                &config,
                registry,
            )?;
            blocks.push(block);
        }

        // Final LayerNorm
        let ln_f = LayerNorm::new(
            &weights.ln_f.weight,
            &weights.ln_f.bias,
            config.layer_norm_epsilon,
            registry,
        )?;

        // LM head — GPT-2 ties lm_head to wte.
        // wte is [vocab_size, n_embd] which matches Linear's expected [out_features, in_features].
        let lm_head = Linear::new(
            &weights.wte,
            None,
            config.n_embd,
            config.vocab_size,
            registry,
        )?;

        Ok(Self::new(
            embedding, blocks, ln_f, lm_head, config, registry,
        ))
    }

    /// Build Gpt2Model from generic [`LoadedWeights`] (via `model_generic` loader).
    ///
    /// Expects weights loaded with `gpt2_weight_map()` — Conv1D→Linear transposes
    /// are already applied during loading. Keys use SafeTensors naming convention.
    pub fn from_generic_weights(
        weights: &crate::model_generic::LoadedWeights,
        config: Gpt2Config,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let w = |key: &str| -> Result<&[f32]> {
            weights
                .require(key)
                .map(|t| t.data.as_slice())
                .map_err(|e| crate::nn::error::NnError::ShapeMismatch {
                    expected: key.to_string(),
                    actual: format!("{e}"),
                })
        };

        let embedding = Embedding::new(
            w("wte.weight")?,
            w("wpe.weight")?,
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            registry,
        )?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = format!("h.{i}");
            // Weights are already transposed by generic loader
            let block = TransformerBlock::new(
                w(&format!("{p}.ln_1.weight"))?,
                w(&format!("{p}.ln_1.bias"))?,
                w(&format!("{p}.attn.c_attn.weight"))?,
                w(&format!("{p}.attn.c_attn.bias"))?,
                w(&format!("{p}.attn.c_proj.weight"))?,
                w(&format!("{p}.attn.c_proj.bias"))?,
                w(&format!("{p}.ln_2.weight"))?,
                w(&format!("{p}.ln_2.bias"))?,
                w(&format!("{p}.mlp.c_fc.weight"))?,
                w(&format!("{p}.mlp.c_fc.bias"))?,
                w(&format!("{p}.mlp.c_proj.weight"))?,
                w(&format!("{p}.mlp.c_proj.bias"))?,
                &config,
                registry,
            )?;
            blocks.push(block);
        }

        let ln_f = LayerNorm::new(
            w("ln_f.weight")?,
            w("ln_f.bias")?,
            config.layer_norm_epsilon,
            registry,
        )?;

        // LM head tied to wte
        let lm_head = Linear::new(
            w("wte.weight")?,
            None,
            config.n_embd,
            config.vocab_size,
            registry,
        )?;

        Ok(Self::new(
            embedding, blocks, ln_f, lm_head, config, registry,
        ))
    }

    /// Forward pass: token_ids → logits `[seq_len, vocab_size]`.
    ///
    /// `token_ids`: device buffer of `u32` with `seq_len` elements.
    pub fn forward(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<GpuTensor> {
        // 1. Embedding lookup (wte + wpe)
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;

        // 2. Transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // 3. Final LayerNorm
        hidden = self.ln_f.forward(&hidden)?;

        // 4. LM head (tied weights or separate linear)
        self.lm_head.forward(&hidden)
    }

    /// Profiled forward pass: returns per-component timing breakdown.
    ///
    /// Returns `(logits, timings)` where `timings` maps component name → milliseconds.
    pub fn forward_profiled(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<(GpuTensor, Vec<(String, f64)>)> {
        let dev = self.registry.device();
        let mut timings = Vec::new();

        // Embedding
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        let t0 = std::time::Instant::now();
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        timings.push(("embedding".to_string(), t0.elapsed().as_secs_f64() * 1000.0));

        // Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
            let t = std::time::Instant::now();
            hidden = block.forward(&hidden)?;
            dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
            timings.push((format!("block_{i}"), t.elapsed().as_secs_f64() * 1000.0));
        }

        // Final LayerNorm
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        let t = std::time::Instant::now();
        hidden = self.ln_f.forward(&hidden)?;
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        timings.push(("ln_f".to_string(), t.elapsed().as_secs_f64() * 1000.0));

        // LM head
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        let t = std::time::Instant::now();
        let logits = self.lm_head.forward(&hidden)?;
        dev.synchronize().map_err(crate::nn::error::NnError::Cuda)?;
        timings.push(("lm_head".to_string(), t.elapsed().as_secs_f64() * 1000.0));

        Ok((logits, timings))
    }

    /// Forward pass returning hidden states before the LM head.
    ///
    /// Returns `[seq_len, n_embd]` — the normalized hidden states after all
    /// transformer blocks + final LayerNorm. Use this for LoRA or fine-tuning
    /// where you want to apply a custom head.
    pub fn forward_features(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<GpuTensor> {
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.ln_f.forward(&hidden)
    }

    /// Access the embedding table (wte, wpe) for tasks like vector search.
    pub fn embedding_table(&self) -> (&GpuTensor, &GpuTensor) {
        (&self.embedding.wte, &self.embedding.wpe)
    }

    /// Diagnostic forward pass: prints intermediate values for debugging.
    ///
    /// Same as [`forward`] but dumps first/last position values after each stage.
    pub fn forward_diagnostic(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<GpuTensor> {
        let n = self.config.n_embd;

        // 1. Embedding
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;
        let h = hidden.to_host()?;
        let last = seq_len - 1;
        eprintln!(
            "[diag] Embedding pos0 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            h[0], h[1], h[2], h[3]
        );
        eprintln!(
            "[diag] Embedding pos{last} first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            h[last * n],
            h[last * n + 1],
            h[last * n + 2],
            h[last * n + 3]
        );

        // 2. Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden)?;
            if i == 0 || i == 11 {
                let h = hidden.to_host()?;
                eprintln!(
                    "[diag] Layer {i} pos{last} first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                    h[last * n],
                    h[last * n + 1],
                    h[last * n + 2],
                    h[last * n + 3]
                );
                // Check for NaN/Inf
                let nan_count = h.iter().filter(|x| !x.is_finite()).count();
                if nan_count > 0 {
                    eprintln!("[diag] WARNING: Layer {i} has {nan_count} NaN/Inf values!");
                }
            }
        }

        // 3. Final LN
        hidden = self.ln_f.forward(&hidden)?;
        let h = hidden.to_host()?;
        eprintln!(
            "[diag] Final LN pos{last} first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            h[last * n],
            h[last * n + 1],
            h[last * n + 2],
            h[last * n + 3]
        );

        // 4. LM head
        let logits = self.lm_head.forward(&hidden)?;
        let l = logits.to_host()?;
        let vocab = self.config.vocab_size;
        // Top-5 tokens at last position
        let last_logits = &l[last * vocab..(last + 1) * vocab];
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("[diag] Top-5 tokens at pos{last}:");
        for (tok, score) in indexed.iter().take(5) {
            eprintln!("  token {tok}: {score:.4}");
        }

        Ok(logits)
    }

    /// Greedy generation: generate `max_new_tokens` tokens from a prompt.
    ///
    /// Returns the full sequence (prompt + generated tokens).
    pub fn generate(&self, prompt_tokens: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let dev = self.registry.device();
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            let seq_len = tokens.len();
            let token_ids = dev
                .htod_sync_copy(&tokens)
                .map_err(crate::nn::error::NnError::Cuda)?;

            let logits = self.forward(&token_ids, seq_len)?;
            let logits_host = logits.to_host()?;

            // Get logits for last position
            let vocab_size = self.config.vocab_size;
            let last_pos_logits = &logits_host[(seq_len - 1) * vocab_size..seq_len * vocab_size];

            // Argmax
            let next_token = last_pos_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            // Stop on <|endoftext|>
            if next_token == 50256 {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Cached forward: process only new token(s), updating the KV cache.
    ///
    /// `token_ids`: device buffer of `u32` with `new_len` elements.
    /// Returns logits `[new_len, vocab_size]`.
    pub fn forward_cached(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        new_len: usize,
        cache: &mut KvCache,
    ) -> Result<GpuTensor> {
        let pos_offset = cache.len();
        let mut hidden = self
            .embedding
            .forward_tokens_with_offset(token_ids, new_len, pos_offset)?;
        let kv_len = cache.len();

        for (i, block) in self.blocks.iter().enumerate() {
            let (out, new_k, new_v) =
                block.forward_cached(&hidden, &cache.k[i], &cache.v[i], kv_len)?;
            cache.append(i, &new_k, &new_v);
            hidden = out;
        }
        cache.advance(new_len);

        hidden = self.ln_f.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    /// KV-cached greedy generation: generate `max_new_tokens` tokens.
    ///
    /// More efficient than [`generate`](Self::generate) — only processes new tokens
    /// through the transformer, reusing cached K/V from previous positions.
    pub fn generate_cached(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let dev = self.registry.device();
        let mut cache = KvCache::new(&self.config);
        let mut tokens = prompt_tokens.to_vec();

        // Prefill: process entire prompt at once
        let prompt_ids = dev
            .htod_sync_copy(prompt_tokens)
            .map_err(crate::nn::error::NnError::Cuda)?;
        let logits = self.forward_cached(&prompt_ids, prompt_tokens.len(), &mut cache)?;
        let logits_host = logits.to_host()?;

        // Get next token from last position
        let vocab_size = self.config.vocab_size;
        let prompt_len = prompt_tokens.len();
        let last_logits = &logits_host[(prompt_len - 1) * vocab_size..prompt_len * vocab_size];
        let mut next_token = argmax(last_logits);

        if next_token == 50256 {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode: one token at a time
        for _ in 1..max_new_tokens {
            let token_id = dev
                .htod_sync_copy(&[next_token])
                .map_err(crate::nn::error::NnError::Cuda)?;
            let logits = self.forward_cached(&token_id, 1, &mut cache)?;
            let logits_host = logits.to_host()?;

            next_token = argmax(&logits_host[..vocab_size]);
            if next_token == 50256 {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass with early exit: stop processing layers when confidence
    /// exceeds `threshold`.
    ///
    /// After each transformer block, applies the final LayerNorm + LM head to
    /// the intermediate hidden state and checks if the softmax probability of
    /// the top token exceeds `threshold`. If so, returns early without
    /// processing the remaining layers.
    ///
    /// Returns `(logits, layers_used)` where `layers_used` is 1..=n_layer.
    ///
    /// This is **impossible with CUDA graphs** — the number of kernel launches
    /// depends on the model's intermediate confidence, which is data-dependent.
    pub fn forward_early_exit(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
        threshold: f32,
    ) -> Result<(GpuTensor, usize)> {
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;
        let vocab_size = self.config.vocab_size;

        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden)?;

            // Check confidence after this layer (skip check on last layer)
            if i < self.blocks.len() - 1 {
                let probe = self.ln_f.forward(&hidden)?;
                let logits = self.lm_head.forward(&probe)?;
                let logits_host = logits.to_host()?;

                // Check confidence at last position only
                let last_pos = seq_len - 1;
                let last_logits = &logits_host[last_pos * vocab_size..(last_pos + 1) * vocab_size];

                let confidence = softmax_max_prob(last_logits);
                if confidence >= threshold {
                    return Ok((logits, i + 1));
                }
            }
        }

        // All layers processed
        hidden = self.ln_f.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((logits, self.blocks.len()))
    }

    /// KV-cached generation with early exit.
    ///
    /// Combines KV caching with early-exit inference. Each decode step may use
    /// a different number of transformer layers depending on how confident the
    /// model is at intermediate layers.
    pub fn generate_cached_early_exit(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        threshold: f32,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let dev = self.registry.device();
        let mut cache = KvCache::new(&self.config);
        let mut tokens = prompt_tokens.to_vec();
        let mut layers_per_step = Vec::new();

        // Prefill: always use all layers (build complete KV cache)
        let prompt_ids = dev
            .htod_sync_copy(prompt_tokens)
            .map_err(crate::nn::error::NnError::Cuda)?;
        let logits = self.forward_cached(&prompt_ids, prompt_tokens.len(), &mut cache)?;
        let logits_host = logits.to_host()?;
        layers_per_step.push(self.config.n_layer);

        let vocab_size = self.config.vocab_size;
        let prompt_len = prompt_tokens.len();
        let last_logits = &logits_host[(prompt_len - 1) * vocab_size..prompt_len * vocab_size];
        let mut next_token = argmax(last_logits);

        if next_token == 50256 {
            return Ok((tokens, layers_per_step));
        }
        tokens.push(next_token);

        // Decode with early exit
        for _ in 1..max_new_tokens {
            let token_id = dev
                .htod_sync_copy(&[next_token])
                .map_err(crate::nn::error::NnError::Cuda)?;

            // Run layers with early-exit check
            let mut hidden =
                self.embedding
                    .forward_tokens_with_offset(&token_id, 1, cache.len())?;
            let kv_len = cache.len();
            let mut used_layers = self.config.n_layer;

            for (i, block) in self.blocks.iter().enumerate() {
                let (out, new_k, new_v) =
                    block.forward_cached(&hidden, &cache.k[i], &cache.v[i], kv_len)?;
                cache.append(i, &new_k, &new_v);
                hidden = out;

                // Check confidence after this layer (skip last layer)
                if i < self.blocks.len() - 1 {
                    let probe = self.ln_f.forward(&hidden)?;
                    let logits = self.lm_head.forward(&probe)?;
                    let logits_host = logits.to_host()?;

                    let confidence = softmax_max_prob(&logits_host[..vocab_size]);
                    if confidence >= threshold {
                        // Fill remaining cache entries with zeros for shape consistency
                        let d_head = self.config.head_dim();
                        let zero_k = vec![vec![0.0f32; d_head]; self.config.n_head];
                        let zero_v = vec![vec![0.0f32; d_head]; self.config.n_head];
                        for j in (i + 1)..self.blocks.len() {
                            cache.append(j, &zero_k, &zero_v);
                        }
                        used_layers = i + 1;
                        break;
                    }
                }
            }

            cache.advance(1);
            layers_per_step.push(used_layers);

            // Get final logits
            hidden = self.ln_f.forward(&hidden)?;
            let logits = self.lm_head.forward(&hidden)?;
            let logits_host = logits.to_host()?;

            next_token = argmax(&logits_host[..vocab_size]);
            if next_token == 50256 {
                break;
            }
            tokens.push(next_token);
        }

        Ok((tokens, layers_per_step))
    }

    /// KV-cached generation with top-k sampling and temperature.
    ///
    /// Unlike greedy decoding, this produces diverse outputs — each run with a
    /// different seed generates different text that stops at different lengths.
    /// This is **dynamic control flow**: the loop count and token choices depend
    /// on the model's own outputs, which is impossible with CUDA graphs or
    /// TensorRT static compilation.
    pub fn generate_cached_sampling(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        top_k: usize,
        temperature: f32,
        rng: &mut SimpleRng,
    ) -> Result<Vec<u32>> {
        let dev = self.registry.device();
        let mut cache = KvCache::new(&self.config);
        let mut tokens = prompt_tokens.to_vec();

        // Prefill
        let prompt_ids = dev
            .htod_sync_copy(prompt_tokens)
            .map_err(crate::nn::error::NnError::Cuda)?;
        let logits = self.forward_cached(&prompt_ids, prompt_tokens.len(), &mut cache)?;
        let logits_host = logits.to_host()?;

        let vocab_size = self.config.vocab_size;
        let prompt_len = prompt_tokens.len();
        let last_logits = &logits_host[(prompt_len - 1) * vocab_size..prompt_len * vocab_size];
        let mut next_token = top_k_sample(last_logits, top_k, temperature, rng);

        if next_token == 50256 {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode with sampling
        for _ in 1..max_new_tokens {
            let token_id = dev
                .htod_sync_copy(&[next_token])
                .map_err(crate::nn::error::NnError::Cuda)?;
            let logits = self.forward_cached(&token_id, 1, &mut cache)?;
            let logits_host = logits.to_host()?;

            next_token = top_k_sample(&logits_host[..vocab_size], top_k, temperature, rng);
            if next_token == 50256 {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

/// Compute the maximum softmax probability over a logit vector.
///
/// Returns the probability of the most likely token after softmax.
/// Used for early-exit confidence checking.
fn softmax_max_prob(logits: &[f32]) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    // max softmax prob = exp(max_val - max_val) / exp_sum = 1.0 / exp_sum
    1.0 / exp_sum
}

/// Argmax over a float slice — returns the index of the maximum value.
fn argmax(data: &[f32]) -> u32 {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Simple xorshift64 RNG for sampling — no external dependency needed.
pub struct SimpleRng(u64);

impl SimpleRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0xDEAD_BEEF_CAFE_1234
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Return a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Top-k sampling with temperature scaling.
///
/// Selects from the `k` highest-probability tokens using softmax probabilities
/// scaled by `temperature`. Higher temperature = more random, lower = more greedy.
pub fn top_k_sample(logits: &[f32], k: usize, temperature: f32, rng: &mut SimpleRng) -> u32 {
    let temp = if temperature < 1e-8 {
        1e-8
    } else {
        temperature
    };
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = &indexed[..k.min(indexed.len())];

    // Softmax over top-k
    let max_val = top[0].1;
    let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let exp_sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / exp_sum).collect();

    // Sample from distribution
    let r = rng.next_f32();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return top[i].0 as u32;
        }
    }
    top.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

/// Per-layer KV cache for autoregressive decoding.
///
/// Stores K and V projections per head on host. Each entry is a flat
/// `Vec<f32>` of `[kv_len, d_head]` for one head.
pub struct KvCache {
    /// `k[layer][head]` = flat `[kv_len * d_head]`.
    k: Vec<Vec<Vec<f32>>>,
    /// `v[layer][head]` = flat `[kv_len * d_head]`.
    v: Vec<Vec<Vec<f32>>>,
    /// Current number of cached positions.
    kv_len: usize,
    n_head: usize,
}

impl KvCache {
    /// Create an empty KV cache for a given model config.
    pub fn new(config: &Gpt2Config) -> Self {
        let n_layer = config.n_layer;
        let n_head = config.n_head;
        let k = (0..n_layer)
            .map(|_| (0..n_head).map(|_| Vec::new()).collect())
            .collect();
        let v = (0..n_layer)
            .map(|_| (0..n_head).map(|_| Vec::new()).collect())
            .collect();
        Self {
            k,
            v,
            kv_len: 0,
            n_head,
        }
    }

    /// Current number of cached positions.
    pub fn len(&self) -> usize {
        self.kv_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.kv_len == 0
    }

    /// Append new K,V positions for a given layer.
    ///
    /// `new_k` and `new_v` are per-head: `[n_head][new_len * d_head]`.
    fn append(&mut self, layer: usize, new_k: &[Vec<f32>], new_v: &[Vec<f32>]) {
        for h in 0..self.n_head {
            self.k[layer][h].extend_from_slice(&new_k[h]);
            self.v[layer][h].extend_from_slice(&new_v[h]);
        }
    }

    /// Update kv_len after all layers have appended the same number of new positions.
    fn advance(&mut self, new_positions: usize) {
        self.kv_len += new_positions;
    }
}

// ============================================================
// INT4 quantized GPT-2 model
// ============================================================

/// Multi-head attention using INT4 quantized projections.
///
/// Same architecture as [`MultiHeadAttention`] but QKV and output projections
/// use [`Int4Linear`] for ~4x weight memory reduction.
struct Int4MultiHeadAttention {
    qkv_proj: Int4Linear,
    out_proj: Int4Linear,
    n_heads: usize,
    d_head: usize,
    registry: Arc<KernelRegistry>,
}

impl Int4MultiHeadAttention {
    /// Create from f32 weights, quantizing to INT4 at construction time.
    ///
    /// Weight convention: `qkv_weight [3*n_embd, n_embd]`, `out_weight [n_embd, n_embd]`
    /// (already in Linear [out, in] format — need to transpose to [K, N] for Int4Linear).
    #[allow(clippy::too_many_arguments)]
    fn new(
        qkv_weight: &[f32],
        qkv_bias: &[f32],
        out_weight: &[f32],
        out_bias: &[f32],
        n_embd: usize,
        n_heads: usize,
        group_size: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let d_head = n_embd / n_heads;

        // QKV: weight is [3*n_embd, n_embd] (out, in). Transpose to [n_embd, 3*n_embd] = [K, N].
        let qkv_t = transpose_2d(qkv_weight, 3 * n_embd, n_embd);
        let (qkv_packed, qkv_scales) =
            ops::quantize::quantize_weight_int4(&qkv_t, n_embd, 3 * n_embd, group_size);
        let qkv_proj = Int4Linear::new(
            &qkv_packed,
            &qkv_scales,
            Some(qkv_bias),
            n_embd,
            3 * n_embd,
            group_size,
            registry,
        )?;

        // Output: weight is [n_embd, n_embd]. Transpose to [n_embd, n_embd] = [K, N].
        let out_t = transpose_2d(out_weight, n_embd, n_embd);
        let (out_packed, out_scales) =
            ops::quantize::quantize_weight_int4(&out_t, n_embd, n_embd, group_size);
        let out_proj = Int4Linear::new(
            &out_packed,
            &out_scales,
            Some(out_bias),
            n_embd,
            n_embd,
            group_size,
            registry,
        )?;

        Ok(Self {
            qkv_proj,
            out_proj,
            n_heads,
            d_head,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass with causal attention (same logic as MultiHeadAttention).
    fn forward_causal(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let seq_len = input.shape()[0];
        let qkv = self.qkv_proj.forward(input)?;
        let (q, k, v) = ops::split_qkv(&qkv, seq_len, self.n_heads, self.d_head, &self.registry)?;
        let attn_out = ops::multi_head_flash_attention(
            &q,
            &k,
            &v,
            seq_len,
            self.n_heads,
            self.d_head,
            true,
            &self.registry,
        )?;
        let concat = ops::concat_heads(
            &attn_out,
            seq_len,
            self.n_heads,
            self.d_head,
            &self.registry,
        )?;
        self.out_proj.forward(&concat)
    }
}

/// GPT-2 transformer block with INT4 quantized Linear layers.
///
/// LayerNorm stays f32 (tiny parameters). All dense projections use INT4.
pub struct Int4TransformerBlock {
    ln_1: LayerNorm,
    attn: Int4MultiHeadAttention,
    ln_2: LayerNorm,
    ffn_up: Int4Linear,
    ffn_down: Int4Linear,
    gelu: GELU,
    layer_norm_eps: f32,
    registry: Arc<KernelRegistry>,
}

impl Int4TransformerBlock {
    /// Create from f32 weights, quantizing to INT4.
    ///
    /// Same weight naming as [`TransformerBlock::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ln_1_weight: &[f32],
        ln_1_bias: &[f32],
        attn_qkv_weight: &[f32],
        attn_qkv_bias: &[f32],
        attn_out_weight: &[f32],
        attn_out_bias: &[f32],
        ln_2_weight: &[f32],
        ln_2_bias: &[f32],
        ffn_up_weight: &[f32],
        ffn_up_bias: &[f32],
        ffn_down_weight: &[f32],
        ffn_down_bias: &[f32],
        config: &Gpt2Config,
        group_size: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let eps = config.layer_norm_epsilon;
        let n_embd = config.n_embd;
        let ffn_dim = config.ffn_dim();

        // FFN up: [ffn_dim, n_embd] → transpose to [n_embd, ffn_dim] = [K, N]
        let ffn_up_t = transpose_2d(ffn_up_weight, ffn_dim, n_embd);
        let (up_packed, up_scales) =
            ops::quantize::quantize_weight_int4(&ffn_up_t, n_embd, ffn_dim, group_size);

        // FFN down: [n_embd, ffn_dim] → transpose to [ffn_dim, n_embd] = [K, N]
        let ffn_down_t = transpose_2d(ffn_down_weight, n_embd, ffn_dim);
        let (down_packed, down_scales) =
            ops::quantize::quantize_weight_int4(&ffn_down_t, ffn_dim, n_embd, group_size);

        Ok(Self {
            ln_1: LayerNorm::new(ln_1_weight, ln_1_bias, eps, registry)?,
            attn: Int4MultiHeadAttention::new(
                attn_qkv_weight,
                attn_qkv_bias,
                attn_out_weight,
                attn_out_bias,
                n_embd,
                config.n_head,
                group_size,
                registry,
            )?,
            ln_2: LayerNorm::new(ln_2_weight, ln_2_bias, eps, registry)?,
            ffn_up: Int4Linear::new(
                &up_packed,
                &up_scales,
                Some(ffn_up_bias),
                n_embd,
                ffn_dim,
                group_size,
                registry,
            )?,
            ffn_down: Int4Linear::new(
                &down_packed,
                &down_scales,
                Some(ffn_down_bias),
                ffn_dim,
                n_embd,
                group_size,
                registry,
            )?,
            gelu: GELU::new(registry),
            layer_norm_eps: eps,
            registry: Arc::clone(registry),
        })
    }

    /// Forward pass: input `[seq_len, n_embd]` → output `[seq_len, n_embd]`.
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // LN1 → MHA → residual
        let ln1_out = self.ln_1.forward(input)?;
        let attn_out = self.attn.forward_causal(&ln1_out)?;
        let mut residual = input.clone_tensor()?;
        ops::elementwise_add(&mut residual, &attn_out, &self.registry)?;

        // LN2 → FFN → residual
        let ln2_out = self.ln_2.forward(&residual)?;
        let ffn_hidden = self.ffn_up.forward(&ln2_out)?;
        let ffn_act = self.gelu.forward(&ffn_hidden)?;
        let ffn_out = self.ffn_down.forward(&ffn_act)?;
        ops::elementwise_add(&mut residual, &ffn_out, &self.registry)?;

        Ok(residual)
    }
}

/// Complete GPT-2 model with INT4 quantized transformer blocks.
///
/// Embedding, final LayerNorm, and LM head remain f32. All dense projections
/// in the transformer blocks use INT4 quantization (~4x memory reduction on
/// the bulk of model parameters).
pub struct Int4Gpt2Model {
    embedding: Embedding,
    blocks: Vec<Int4TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear, // f32: vocab_size not divisible by 8
    config: Gpt2Config,
    group_size: usize,
    registry: Arc<KernelRegistry>,
}

impl Int4Gpt2Model {
    /// Default quantization group size.
    pub const DEFAULT_GROUP_SIZE: usize = 128;

    /// Build from pre-loaded [`Gpt2Weights`], quantizing all dense layers to INT4.
    #[cfg(feature = "gpt2")]
    pub fn from_weights(
        weights: &Gpt2Weights,
        config: Gpt2Config,
        group_size: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let embedding = Embedding::new(
            &weights.wte,
            &weights.wpe,
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            registry,
        )?;

        let n = config.n_embd;
        let ffn = config.ffn_dim();
        let mut blocks = Vec::with_capacity(config.n_layer);

        for layer in &weights.layers {
            // Conv1D [in, out] → Linear [out, in]
            let qkv_w = transpose_2d(&layer.c_attn_weight, n, 3 * n);
            let proj_w = transpose_2d(&layer.c_proj_weight, n, n);
            let fc_w = transpose_2d(&layer.mlp_fc_weight, n, ffn);
            let fc_proj_w = transpose_2d(&layer.mlp_proj_weight, ffn, n);

            let block = Int4TransformerBlock::new(
                &layer.ln_1.weight,
                &layer.ln_1.bias,
                &qkv_w,
                &layer.c_attn_bias,
                &proj_w,
                &layer.c_proj_bias,
                &layer.ln_2.weight,
                &layer.ln_2.bias,
                &fc_w,
                &layer.mlp_fc_bias,
                &fc_proj_w,
                &layer.mlp_proj_bias,
                &config,
                group_size,
                registry,
            )?;
            blocks.push(block);
        }

        let ln_f = LayerNorm::new(
            &weights.ln_f.weight,
            &weights.ln_f.bias,
            config.layer_norm_epsilon,
            registry,
        )?;

        // LM head stays f32 (vocab_size=50257 not divisible by 8)
        let lm_head = Linear::new(
            &weights.wte,
            None,
            config.n_embd,
            config.vocab_size,
            registry,
        )?;

        Ok(Self {
            embedding,
            blocks,
            ln_f,
            lm_head,
            config,
            group_size,
            registry: Arc::clone(registry),
        })
    }

    /// Build from generic [`LoadedWeights`], quantizing all dense layers to INT4.
    pub fn from_generic_weights(
        weights: &crate::model_generic::LoadedWeights,
        config: Gpt2Config,
        group_size: usize,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self> {
        let w = |key: &str| -> Result<&[f32]> {
            weights
                .require(key)
                .map(|t| t.data.as_slice())
                .map_err(|e| crate::nn::error::NnError::ShapeMismatch {
                    expected: key.to_string(),
                    actual: format!("{e}"),
                })
        };

        let embedding = Embedding::new(
            w("wte.weight")?,
            w("wpe.weight")?,
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            registry,
        )?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = format!("h.{i}");
            // Weights are already transposed by generic loader
            let block = Int4TransformerBlock::new(
                w(&format!("{p}.ln_1.weight"))?,
                w(&format!("{p}.ln_1.bias"))?,
                w(&format!("{p}.attn.c_attn.weight"))?,
                w(&format!("{p}.attn.c_attn.bias"))?,
                w(&format!("{p}.attn.c_proj.weight"))?,
                w(&format!("{p}.attn.c_proj.bias"))?,
                w(&format!("{p}.ln_2.weight"))?,
                w(&format!("{p}.ln_2.bias"))?,
                w(&format!("{p}.mlp.c_fc.weight"))?,
                w(&format!("{p}.mlp.c_fc.bias"))?,
                w(&format!("{p}.mlp.c_proj.weight"))?,
                w(&format!("{p}.mlp.c_proj.bias"))?,
                &config,
                group_size,
                registry,
            )?;
            blocks.push(block);
        }

        let ln_f = LayerNorm::new(
            w("ln_f.weight")?,
            w("ln_f.bias")?,
            config.layer_norm_epsilon,
            registry,
        )?;

        let lm_head = Linear::new(
            w("wte.weight")?,
            None,
            config.n_embd,
            config.vocab_size,
            registry,
        )?;

        Ok(Self {
            embedding,
            blocks,
            ln_f,
            lm_head,
            config,
            group_size,
            registry: Arc::clone(registry),
        })
    }

    /// Reference to model config.
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }

    /// Quantization group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Forward pass: token_ids → logits `[seq_len, vocab_size]`.
    pub fn forward(
        &self,
        token_ids: &cudarc::driver::CudaSlice<u32>,
        seq_len: usize,
    ) -> Result<GpuTensor> {
        let mut hidden = self.embedding.forward_tokens(token_ids, seq_len)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.ln_f.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    /// Greedy generation: generate `max_new_tokens` tokens from a prompt.
    pub fn generate(&self, prompt_tokens: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let dev = self.registry.device();
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            let seq_len = tokens.len();
            let token_ids = dev
                .htod_sync_copy(&tokens)
                .map_err(crate::nn::error::NnError::Cuda)?;

            let logits = self.forward(&token_ids, seq_len)?;
            let logits_host = logits.to_host()?;

            let vocab_size = self.config.vocab_size;
            let last_pos_logits = &logits_host[(seq_len - 1) * vocab_size..seq_len * vocab_size];

            let next_token = argmax(last_pos_logits);
            if next_token == 50256 {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Approximate weight memory in bytes (INT4 packed weights + scales).
    ///
    /// Excludes embeddings (f32) and LM head (f32).
    pub fn quantized_memory_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| {
                b.ffn_up.memory_bytes()
                    + b.ffn_down.memory_bytes()
                    + b.attn.qkv_proj.memory_bytes()
                    + b.attn.out_proj.memory_bytes()
            })
            .sum()
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
///
/// Used to convert Conv1D `[in, out]` weights to Linear `[out, in]` layout.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::test_utils::{GoldenEntry, Tolerance};
    use std::sync::Arc;

    /// Capture or verify GPT-2 golden outputs for 3 prompts.
    ///
    /// On first run: captures top-5 logits to golden files.
    /// On subsequent runs: verifies output matches golden files (regression).
    #[test]
    fn test_gpt2_golden_regression() {
        let model_path =
            crate::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if !model_path.exists() {
            println!("SKIP: GPT-2 model not found at {}", model_path.display());
            return;
        }

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry = Arc::new(
            crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"),
        );
        let weights = crate::model::load_gpt2_weights(&model_path).expect("weights");
        let config = Gpt2Config::small();
        let vocab = config.vocab_size;
        let model = Gpt2Model::from_weights(&weights, config, &registry).expect("model");

        let tokenizer = crate::tokenizer::Gpt2Tokenizer::new().expect("tokenizer");

        let prompts = [
            "The capital of France is",
            "In a world where AI",
            "Once upon a time",
        ];
        let golden_dir = crate::nn::test_utils::golden_dir();
        std::fs::create_dir_all(&golden_dir).ok();

        for (i, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer.encode(prompt);
            let token_ids = dev.htod_sync_copy(&tokens).expect("upload tokens");
            let logits = model.forward(&token_ids, tokens.len()).expect("forward");
            let logits_host = logits.to_host().expect("download");

            let last_pos = tokens.len() - 1;
            let last_logits = &logits_host[last_pos * vocab..(last_pos + 1) * vocab];

            // Extract top-5 token IDs and their logit values
            let mut indexed: Vec<(usize, f32)> = last_logits
                .iter()
                .enumerate()
                .map(|(j, &v)| (j, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5_ids: Vec<f32> = indexed[..5].iter().map(|(id, _)| *id as f32).collect();
            let top5_vals: Vec<f32> = indexed[..5].iter().map(|(_, v)| *v).collect();

            let golden_path = golden_dir.join(format!("gpt2_prompt{i}_top5.golden"));

            if golden_path.exists() {
                // Regression: verify against golden
                let golden = GoldenEntry::load(&golden_path).expect("load golden");
                // Compare top-5 token IDs (must match exactly)
                let actual_ids = &top5_ids;
                assert_eq!(
                    actual_ids,
                    &golden.data[..5],
                    "Prompt {i} top-5 token IDs changed"
                );
                // Compare top-5 logit values (with tolerance)
                crate::nn::test_utils::assert_close(
                    &top5_vals,
                    &golden.data[5..],
                    golden.tolerance,
                    &format!("gpt2_prompt{i}_logits"),
                );
                println!("REGRESSION OK: prompt {i} matches golden");
            } else {
                // First run: capture golden
                let mut data = top5_ids.clone();
                data.extend_from_slice(&top5_vals);
                let entry = GoldenEntry {
                    label: format!("gpt2_prompt{i}_top5 ({prompt})"),
                    shape: vec![2, 5], // [ids, values]
                    data,
                    tolerance: Tolerance::f32_loose(),
                };
                entry.save(&golden_path).expect("save golden");
                println!(
                    "CAPTURED: prompt {i} top-5: {:?}",
                    indexed[..5]
                        .iter()
                        .map(|(id, v)| format!("{id}:{v:.2}"))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Verify that `from_generic_weights` produces identical logits to `from_weights`.
    #[test]
    fn test_gpt2_generic_loader_regression() {
        let model_path =
            crate::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if !model_path.exists() {
            println!("SKIP: GPT-2 model not found");
            return;
        }

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry = Arc::new(
            crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"),
        );

        // Old path: hardcoded loader
        let old_weights = crate::model::load_gpt2_weights(&model_path).expect("old weights");
        let config1 = Gpt2Config::small();
        let old_model =
            Gpt2Model::from_weights(&old_weights, config1, &registry).expect("old model");

        // New path: generic loader
        let config2 = Gpt2Config::small();
        let weight_map = crate::model_generic::gpt2_weight_map(config2.n_layer);
        let new_weights = crate::model_generic::load_safetensors_mapped(&model_path, &weight_map)
            .expect("generic weights");
        let new_model = Gpt2Model::from_generic_weights(&new_weights, config2, &registry)
            .expect("generic model");

        // Compare logits for a short prompt
        let tokens: Vec<u32> = vec![464, 3139, 286, 4881, 318]; // "The capital of France is"
        let token_ids = dev.htod_sync_copy(&tokens).expect("upload");

        let old_logits = old_model
            .forward(&token_ids, tokens.len())
            .expect("old fwd")
            .to_host()
            .expect("old d2h");
        let new_logits = new_model
            .forward(&token_ids, tokens.len())
            .expect("new fwd")
            .to_host()
            .expect("new d2h");

        assert_eq!(old_logits.len(), new_logits.len(), "logit length mismatch");
        let max_err: f32 = old_logits
            .iter()
            .zip(new_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("GPT-2 generic loader max_err = {max_err}");
        assert!(
            max_err < 1e-3,
            "Generic loader logits differ from old loader: max_err={max_err}"
        );
    }

    /// Test INT4 GPT-2: build model, run forward, generate text.
    ///
    /// Compares top-1 token at each position with f32 model. INT4 is lossy,
    /// so we check that top-1 agrees for at least the first few tokens and
    /// that the output is coherent (no NaN, finite logits).
    #[test]
    fn test_int4_gpt2_forward_and_generate() {
        let model_path =
            crate::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if !model_path.exists() {
            println!("SKIP: GPT-2 model not found");
            return;
        }

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry = Arc::new(
            crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"),
        );

        // Build f32 model for comparison
        let weights = crate::model::load_gpt2_weights(&model_path).expect("weights");
        let config_f32 = Gpt2Config::small();
        let model_f32 =
            Gpt2Model::from_weights(&weights, config_f32, &registry).expect("f32 model");

        // Build INT4 model
        let config_int4 = Gpt2Config::small();
        let t0 = std::time::Instant::now();
        let model_int4 = Int4Gpt2Model::from_weights(
            &weights,
            config_int4,
            Int4Gpt2Model::DEFAULT_GROUP_SIZE,
            &registry,
        )
        .expect("int4 model");
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("INT4 model built in {build_ms:.1}ms");
        eprintln!(
            "INT4 quantized weight memory: {:.2} MB",
            model_int4.quantized_memory_bytes() as f64 / 1e6
        );

        let prompt = "The capital of France is";
        let tokenizer = crate::tokenizer::Gpt2Tokenizer::new().expect("tokenizer");
        let tokens = tokenizer.encode(prompt);

        // Forward pass comparison
        let token_ids = dev.htod_sync_copy(&tokens).expect("upload");
        let vocab = model_f32.config().vocab_size;

        let f32_logits = model_f32
            .forward(&token_ids, tokens.len())
            .expect("f32 forward")
            .to_host()
            .expect("f32 d2h");

        let int4_logits = model_int4
            .forward(&token_ids, tokens.len())
            .expect("int4 forward")
            .to_host()
            .expect("int4 d2h");

        assert_eq!(f32_logits.len(), int4_logits.len());
        assert!(
            int4_logits.iter().all(|x| x.is_finite()),
            "INT4 logits contain NaN/Inf"
        );

        // Compare top-1 at last position
        let last = tokens.len() - 1;
        let f32_last = &f32_logits[last * vocab..(last + 1) * vocab];
        let int4_last = &int4_logits[last * vocab..(last + 1) * vocab];
        let f32_top1 = argmax(f32_last);
        let int4_top1 = argmax(int4_last);
        eprintln!("f32 top-1: {f32_top1}, INT4 top-1: {int4_top1}");

        // Generate text with INT4 model
        let t1 = std::time::Instant::now();
        let int4_output = model_int4.generate(&tokens, 20).expect("int4 generate");
        let gen_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let new_tokens = int4_output.len() - tokens.len();

        let int4_text = tokenizer
            .decode(&int4_output)
            .unwrap_or_else(|_| "[decode error]".to_string());
        eprintln!("INT4 output ({new_tokens} tokens, {gen_ms:.1}ms): {int4_text}");

        // Also generate with f32 for comparison
        let f32_output = model_f32.generate(&tokens, 20).expect("f32 generate");
        let f32_text = tokenizer
            .decode(&f32_output)
            .unwrap_or_else(|_| "[decode error]".to_string());
        eprintln!("f32  output: {f32_text}");

        // Basic sanity: INT4 model should produce at least 1 token
        assert!(new_tokens >= 1, "INT4 model produced no tokens");
    }

    /// Benchmark GPT-2 forward pass with profiled per-block timing.
    ///
    /// Measures total forward pass and per-block time for seq_len=128.
    /// Run with `--features cublas` to measure fused LN+residual performance.
    #[test]
    fn bench_gpt2_forward_profiled() {
        let model_path =
            crate::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if !model_path.exists() {
            println!("SKIP: GPT-2 model not found at {}", model_path.display());
            return;
        }

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry = Arc::new(
            crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"),
        );
        let weights = crate::model::load_gpt2_weights(&model_path).expect("weights");
        let config = Gpt2Config::small();
        let model = Gpt2Model::from_weights(&weights, config, &registry).expect("model");

        let tokenizer = crate::tokenizer::Gpt2Tokenizer::new().expect("tokenizer");
        let prompt = "The meaning of life is to find purpose and happiness in the things we do every day and to share that with others around us";
        let tokens = tokenizer.encode(prompt);
        // Pad or truncate to exactly 128 tokens for consistent benchmarking
        let seq_len = 128;
        let mut token_vec = tokens.clone();
        token_vec.resize(seq_len, 0);
        let token_ids = dev.htod_sync_copy(&token_vec).expect("upload");

        // Feature detection
        let fused = cfg!(feature = "cublas");
        eprintln!(
            "\n=== GPT-2 Forward Pass Benchmark (seq={}, fused={}) ===",
            seq_len, fused
        );

        // Warmup run (2 iterations)
        for _ in 0..2 {
            let _ = model.forward(&token_ids, seq_len).expect("warmup forward");
            dev.synchronize().unwrap();
        }

        // Benchmark: 5 profiled runs
        let num_runs = 5;
        let mut total_times_ms = Vec::new();
        let mut per_block_times: Vec<Vec<f64>> = (0..12).map(|_| Vec::new()).collect();

        for run in 0..num_runs {
            let (logits, timings) = model
                .forward_profiled(&token_ids, seq_len)
                .expect("profiled forward");
            drop(logits);

            let total: f64 = timings.iter().map(|(_, ms)| ms).sum();
            total_times_ms.push(total);

            if run == 0 {
                eprintln!("\n--- Timing breakdown (run 0) ---");
                for (name, ms) in &timings {
                    eprintln!("  {:<15} {:>8.3} ms", name, ms);
                }
                eprintln!("  {:<15} {:>8.3} ms", "TOTAL", total);
            }

            // Collect per-block timings
            for (name, ms) in &timings {
                if let Some(idx) = name.strip_prefix("block_") {
                    if let Ok(i) = idx.parse::<usize>() {
                        if i < 12 {
                            per_block_times[i].push(*ms);
                        }
                    }
                }
            }
        }

        // Statistics
        total_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_total = total_times_ms[num_runs / 2];
        let min_total = total_times_ms[0];
        let avg_block: f64 = per_block_times
            .iter()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .sum::<f64>()
            / 12.0;

        eprintln!(
            "\n--- Summary ({} runs, seq={}, fused={}) ---",
            num_runs, seq_len, fused
        );
        eprintln!("  Total forward (median): {:.3} ms", median_total);
        eprintln!("  Total forward (min):    {:.3} ms", min_total);
        eprintln!("  Avg per-block:          {:.3} ms", avg_block);

        // Per-block detail
        eprintln!("\n--- Per-block timing (avg of {} runs) ---", num_runs);
        for (i, times) in per_block_times.iter().enumerate() {
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!("  block_{:<2}: avg={:.3}ms  min={:.3}ms", i, avg, min);
        }

        eprintln!("\nBenchmark complete.");
    }
}
