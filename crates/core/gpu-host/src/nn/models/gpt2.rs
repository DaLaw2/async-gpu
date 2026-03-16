//! GPT-2 model configuration and architecture.
//!
//! Provides [`Gpt2Config`] with standard model size presets, [`TransformerBlock`]
//! for a single transformer layer, and [`Gpt2Model`] for the complete model.

use std::sync::Arc;

#[cfg(feature = "gpt2")]
use crate::model::Gpt2Weights;
use crate::nn::error::Result;
use crate::nn::layers::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention, GELU};
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
        let mut residual = input.clone_tensor()?;
        ops::elementwise_add(&mut residual, &attn_out, &self.registry)?;

        // LN2 → FFN → residual
        let ln2_out = self.ln_2.forward(&residual)?;
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
}

/// Argmax over a float slice — returns the index of the maximum value.
fn argmax(data: &[f32]) -> u32 {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
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
}
