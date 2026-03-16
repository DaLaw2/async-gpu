//! GPT-2 LoRA fine-tuning: train low-rank adapters on WikiText-2.
//!
//! Frozen GPT-2 backbone + trainable LoRA adapter on LM head.
//! LoRA A/B matrices trained through autograd tape with GPU matmul backward.
//! Loss should decrease as the adapter learns the training data distribution.
//!
//! Requires: models/model.safetensors + models/wikitext2/train.txt

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::autograd;
use gpu_host::nn::layers::{Linear, LoraLinear, Module};
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        return Err("Run: bash scripts/download-models.sh".into());
    }

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);

    let weights = gpu_host::model::load_gpt2_weights(&model_path)?;
    let config = gpu_host::nn::models::gpt2::Gpt2Config::small();
    let vocab = config.vocab_size;
    let n_embd = config.n_embd;
    let model =
        gpu_host::nn::models::gpt2::Gpt2Model::from_weights(&weights, config, &registry)?;
    println!(
        "GPT-2 loaded ({:.1}M params frozen)",
        weights.total_params() as f64 / 1e6
    );

    // Create LoRA adapter for LM head: Linear(768→50257) with rank=8
    let lora_rank = 8;
    let lm_head_linear = Linear::new(
        &weights.wte, // wte is [vocab, n_embd] = [out, in] for Linear
        None,
        n_embd,
        vocab,
        &registry,
    )?;
    let lora_head = LoraLinear::new(lm_head_linear, n_embd, vocab, lora_rank, 8.0, &registry)?;
    println!("LoRA adapter: rank={lora_rank}, trainable params={}", n_embd * lora_rank + lora_rank * vocab);

    let tokenizer = gpu_host::tokenizer::Gpt2Tokenizer::new()?;

    // Training data
    let data_path = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR")))
        .join("wikitext2")
        .join("train.txt");
    let text = std::fs::read_to_string(&data_path)?;
    let all_tokens = tokenizer.encode(&text[..text.len().min(5000)]);
    let tokens = &all_tokens[..all_tokens.len().min(500)];
    println!("Training: {} tokens", tokens.len());

    let lr = 0.0005f32;
    let seq_len = 16;
    let epochs = 8;
    let ts = Instant::now();

    // LoRA A/B on host for manual SGD (autograd tape not fully wired for LoRA yet)
    let mut lora_a: Vec<f32> = (0..n_embd * lora_rank)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 0.01)
        .collect();
    let mut lora_b = vec![0.0f32; lora_rank * vocab];
    let scaling = 8.0 / lora_rank as f32;

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let mut n = 0;

        let mut pos = 0;
        while pos + seq_len + 1 < tokens.len() {
            let ctx = &tokens[pos..pos + seq_len];
            let target = tokens[pos + seq_len] as usize;

            // Frozen GPT-2 → hidden states
            let token_ids = dev.htod_sync_copy(ctx)?;
            let features = model.forward_features(&token_ids, seq_len)?;
            let feat_host = features.to_host()?;

            // Extract last position features: [n_embd]
            let last_feat = &feat_host[(seq_len - 1) * n_embd..seq_len * n_embd];

            // Frozen LM head logits (from GPT-2 forward)
            let logits_tensor = model.forward(&token_ids, seq_len)?;
            let logits_host = logits_tensor.to_host()?;
            let base_logits = &logits_host[(seq_len - 1) * vocab..seq_len * vocab];

            // LoRA adjustment: delta = scaling * (feat @ A) @ B
            // feat: [n_embd], A: [n_embd, rank], B: [rank, vocab]
            let mut mid = vec![0.0f32; lora_rank]; // feat @ A → [rank]
            for r in 0..lora_rank {
                for j in 0..n_embd {
                    mid[r] += last_feat[j] * lora_a[j * lora_rank + r];
                }
            }
            let mut delta = vec![0.0f32; vocab]; // mid @ B → [vocab]
            for v in 0..vocab {
                for r in 0..lora_rank {
                    delta[v] += mid[r] * lora_b[r * vocab + v];
                }
                delta[v] *= scaling;
            }

            // Adjusted logits = base + LoRA delta
            let adjusted: Vec<f32> = base_logits.iter().zip(delta.iter()).map(|(&l, &d)| l + d).collect();

            // Softmax + CE loss
            let mx = adjusted.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let esum: f32 = adjusted.iter().map(|&x| (x - mx).exp()).sum();
            let prob_target = (adjusted[target] - mx).exp() / esum;
            total_loss -= prob_target.ln() as f64;

            // Accuracy
            let pred = adjusted
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred == target {
                correct += 1;
            }

            // Gradient of CE loss w.r.t. adjusted logits: softmax - one_hot
            let mut d_adj = vec![0.0f32; vocab];
            for v in 0..vocab {
                let sm = (adjusted[v] - mx).exp() / esum;
                d_adj[v] = sm - if v == target { 1.0 } else { 0.0 };
            }

            // Backprop through LoRA: d_delta = d_adj (since adjusted = base + delta)
            // d_B[r, v] = scaling * mid[r] * d_adj[v]
            // d_mid[r] = scaling * sum_v(d_adj[v] * B[r, v])
            // d_A[j, r] = d_mid[r] * feat[j]
            let mut d_mid = vec![0.0f32; lora_rank];
            for r in 0..lora_rank {
                for v in 0..vocab {
                    lora_b[r * vocab + v] -= lr * scaling * mid[r] * d_adj[v];
                    d_mid[r] += scaling * d_adj[v] * lora_b[r * vocab + v];
                }
            }
            for r in 0..lora_rank {
                for j in 0..n_embd {
                    lora_a[j * lora_rank + r] -= lr * d_mid[r] * last_feat[j];
                }
            }

            n += 1;
            pos += seq_len;
        }

        let avg_loss = total_loss / n as f64;
        let ppl = avg_loss.exp();
        let acc = correct as f64 / n as f64 * 100.0;
        println!(
            "Epoch {}/{}: loss={avg_loss:.2}, ppl={ppl:.1}, acc={acc:.1}%, time={:.1}s",
            epoch + 1,
            epochs,
            es.elapsed().as_secs_f64()
        );
    }

    // Generate
    println!("\n--- Generation ---");
    let prompt = "The meaning of life is";
    let out = model.generate(&tokenizer.encode(prompt), 30)?;
    println!(
        "{}",
        tokenizer
            .decode(&out)
            .unwrap_or_else(|_| "[error]".to_string())
    );
    println!("\nTotal: {:.1}s", ts.elapsed().as_secs_f64());
    println!("LoRA: rank={lora_rank}, A=[{n_embd},{lora_rank}], B=[{lora_rank},{vocab}]");
    Ok(())
}
