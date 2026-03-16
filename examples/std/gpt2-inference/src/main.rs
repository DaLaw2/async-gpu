//! GPT-2 inference example using the gpu-host nn API.
//!
//! Loads GPT-2 Small (124M params) from a safetensors file and generates text
//! using the composable nn module — no raw kernel launches needed.
//!
//! Usage:
//!   cargo run --release
//!
//! Requires `models/model.safetensors` in the repository root.

use std::sync::Arc;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize CUDA device
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // 2. Load PTX kernels via KernelRegistry
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);
    println!("Kernel registry loaded ({} ML kernels)", 23);

    // 3. Load model weights from safetensors
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        return Err(format!(
            "Model file not found: {}\nDownload GPT-2 Small safetensors to models/model.safetensors",
            model_path.display()
        )
        .into());
    }

    let t0 = Instant::now();
    let weights = gpu_host::model::load_gpt2_weights(&model_path)
        .map_err(|e| format!("Failed to load weights: {e}"))?;
    println!(
        "Loaded {:.1}M params ({:.1} MB) in {:.1}ms",
        weights.total_params() as f64 / 1e6,
        weights.memory_bytes() as f64 / 1e6,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    // 4. Build model from weights
    let config = gpu_host::nn::models::gpt2::Gpt2Config::small();
    let t1 = Instant::now();
    let model = gpu_host::nn::models::gpt2::Gpt2Model::from_weights(&weights, config, &registry)?;
    println!(
        "Model built on GPU in {:.1}ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // 5. Tokenize prompts
    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| format!("Tokenizer error: {e}"))?;

    let prompts = [
        "The capital of France is",
        "In a world where AI",
        "Once upon a time",
    ];

    // 6. Generate text for each prompt
    let max_new_tokens = 50;
    for prompt in &prompts {
        let tokens = tokenizer.encode(prompt);
        println!("\n--- Prompt: \"{prompt}\" ({} tokens) ---", tokens.len());

        // Greedy generation (no KV cache)
        let t2 = Instant::now();
        let output_tokens = model.generate(&tokens, max_new_tokens)?;
        let gen_time = t2.elapsed();
        let new_tokens = output_tokens.len() - tokens.len();

        let output_text = tokenizer
            .decode(&output_tokens)
            .unwrap_or_else(|_| "[decode error]".to_string());
        println!("Output: {output_text}");
        println!(
            "Generated {} tokens in {:.1}ms ({:.1}ms/token)",
            new_tokens,
            gen_time.as_secs_f64() * 1000.0,
            if new_tokens > 0 {
                gen_time.as_secs_f64() * 1000.0 / new_tokens as f64
            } else {
                0.0
            },
        );

        // KV-cached generation for comparison
        let t3 = Instant::now();
        let cached_tokens = model.generate_cached(&tokens, max_new_tokens)?;
        let cached_time = t3.elapsed();
        let cached_new = cached_tokens.len() - tokens.len();

        let cached_text = tokenizer
            .decode(&cached_tokens)
            .unwrap_or_else(|_| "[decode error]".to_string());
        println!("Cached: {cached_text}");
        println!(
            "Cached: {} tokens in {:.1}ms ({:.1}ms/token)",
            cached_new,
            cached_time.as_secs_f64() * 1000.0,
            if cached_new > 0 {
                cached_time.as_secs_f64() * 1000.0 / cached_new as f64
            } else {
                0.0
            },
        );

        // Verify outputs match
        if output_tokens == cached_tokens {
            println!("MATCH: cached and non-cached outputs agree");
        } else {
            println!(
                "MISMATCH: cached={} tokens, non-cached={} tokens",
                cached_tokens.len(),
                output_tokens.len()
            );
        }
    }

    println!("\nDone.");
    Ok(())
}
