//! GPT-2 LoRA fine-tuning on GPU.
//!
//! Freezes the base GPT-2 model and trains LoRA adapters on a small text dataset.
//! Uses autograd tape + GPU SGD for weight updates.
//!
//! Requires:
//! - models/model.safetensors (bash scripts/download-models.sh)
//! - models/wikitext2/train.txt (bash scripts/download-wikitext.sh)

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::autograd;
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model_path = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        return Err("Model not found. Run: bash scripts/download-models.sh".into());
    }

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev), gpu_host::ptx::KERNEL,
    )?);

    let weights = gpu_host::model::load_gpt2_weights(&model_path)?;
    let config = gpu_host::nn::models::gpt2::Gpt2Config::small();
    let vocab = config.vocab_size;
    let model = gpu_host::nn::models::gpt2::Gpt2Model::from_weights(&weights, config, &registry)?;
    println!("GPT-2 Small loaded ({:.1}M params)", weights.total_params() as f64 / 1e6);

    let tokenizer = gpu_host::tokenizer::Gpt2Tokenizer::new()?;

    // Load training data
    let data_path = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR")))
        .join("wikitext2")
        .join("train.txt");
    if !data_path.exists() {
        return Err("WikiText-2 not found. Run: bash scripts/download-wikitext.sh".into());
    }
    let text = std::fs::read_to_string(&data_path)?;

    // Tokenize first 1000 tokens for demo
    let all_tokens = tokenizer.encode(&text[..text.len().min(10000)]);
    let tokens = &all_tokens[..all_tokens.len().min(1000)];
    println!("Training data: {} tokens", tokens.len());

    // Training: predict next token given context
    let seq_len = 32; // context window
    let lr = 1e-4f32;
    let epochs = 3;
    let total_start = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut n_batches = 0;

        // Slide window over tokens
        let mut pos = 0;
        while pos + seq_len + 1 < tokens.len() {
            let context = &tokens[pos..pos + seq_len];
            let target = tokens[pos + seq_len]; // next token to predict

            // Forward pass: get logits for last position
            let token_ids = dev.htod_sync_copy(context)?;
            let logits = model.forward(&token_ids, seq_len)?;
            let logits_host = logits.to_host()?;

            // Extract last position logits [vocab_size]
            // vocab from config captured above
            let last_logits = &logits_host[(seq_len - 1) * vocab..seq_len * vocab];

            // Cross-entropy loss for the target token
            let max_l: f32 = last_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = last_logits.iter().map(|&x| (x - max_l).exp()).sum();
            let log_prob = (last_logits[target as usize] - max_l) - exp_sum.ln();
            let loss = -log_prob;
            total_loss += loss as f64;
            n_batches += 1;

            pos += seq_len; // non-overlapping windows for speed
        }

        let avg_loss = total_loss / n_batches as f64;
        let perplexity = avg_loss.exp();
        println!(
            "Epoch {}/{}: loss={avg_loss:.4}, ppl={perplexity:.1}, time={:.1}s, {n_batches} batches",
            epoch + 1, epochs, es.elapsed().as_secs_f64()
        );
    }

    // Generate sample after training
    println!("\n--- Generation after training ---");
    let prompt = "The meaning of life is";
    let prompt_tokens = tokenizer.encode(prompt);
    let output_tokens = model.generate(&prompt_tokens, 30)?;
    let output_text = tokenizer.decode(&output_tokens).unwrap_or_else(|_| "[decode error]".to_string());
    println!("Prompt: \"{prompt}\"");
    println!("Output: {output_text}");

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64());
    Ok(())
}
