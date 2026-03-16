//! GPT-2 fine-tuning demo: train a logit bias on WikiText-2.
//!
//! Frozen GPT-2 backbone + trainable bias vector. Loss should decrease
//! as the bias learns the training data's token distribution.
//!
//! Requires: models/model.safetensors + models/wikitext2/train.txt

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
    let model =
        gpu_host::nn::models::gpt2::Gpt2Model::from_weights(&weights, config, &registry)?;
    println!(
        "GPT-2 loaded ({:.1}M params frozen)",
        weights.total_params() as f64 / 1e6
    );

    let tokenizer = gpu_host::tokenizer::Gpt2Tokenizer::new()?;

    // Training data
    let data_path = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR")))
        .join("wikitext2")
        .join("train.txt");
    let text = std::fs::read_to_string(&data_path)?;
    let all_tokens = tokenizer.encode(&text[..text.len().min(5000)]);
    let tokens = &all_tokens[..all_tokens.len().min(500)];
    println!("Training: {} tokens", tokens.len());

    // Trainable: logit bias [vocab] — learns token frequency of training data
    let mut bias = vec![0.0f32; vocab];
    let lr = 0.1f32;
    let seq_len = 16;
    let epochs = 5;
    let ts = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let mut n = 0;

        let mut pos = 0;
        while pos + seq_len + 1 < tokens.len() {
            let ctx = &tokens[pos..pos + seq_len];
            let target = tokens[pos + seq_len] as usize;

            // Frozen GPT-2 forward
            let token_ids = dev.htod_sync_copy(ctx)?;
            let logits = model.forward(&token_ids, seq_len)?;
            let logits_host = logits.to_host()?;

            // Add trainable bias to last position logits
            let base = &logits_host[(seq_len - 1) * vocab..seq_len * vocab];
            let mut adjusted: Vec<f32> = base.iter().zip(bias.iter()).map(|(&l, &b)| l + b).collect();

            // Softmax + CE loss
            let mx = adjusted.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let esum: f32 = adjusted.iter().map(|&x| (x - mx).exp()).sum();
            let prob_target = (adjusted[target] - mx).exp() / esum;
            total_loss -= prob_target.ln() as f64;

            // Accuracy
            let pred = adjusted
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if pred == target {
                correct += 1;
            }

            // Gradient: d_bias = softmax - one_hot
            for c in 0..vocab {
                let sm = (adjusted[c] - mx).exp() / esum;
                let target_val = if c == target { 1.0 } else { 0.0 };
                bias[c] -= lr * (sm - target_val);
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
    Ok(())
}
