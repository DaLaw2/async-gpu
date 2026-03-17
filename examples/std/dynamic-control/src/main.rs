//! Dynamic Control Flow on GPU — the advantage over CUDA graphs.
//!
//! This example demonstrates GPU compute patterns that are **impossible** with
//! CUDA graphs, TensorRT, or any static graph compilation framework:
//!
//! 1. **Variable-length generation**: Each prompt generates a different number
//!    of tokens, stopping when the model produces EOS. The loop count is
//!    data-dependent — it cannot be known at graph capture time.
//!
//! 2. **Top-k sampling**: Token selection depends on the model's own output
//!    distribution, introducing stochastic, data-dependent branching at every
//!    step.
//!
//! 3. **Per-sample early stopping**: Different seeds produce different outputs
//!    with different lengths — the compute graph is different for every run.
//!
//! ## Why CUDA Graphs Cannot Do This
//!
//! CUDA graphs capture a fixed sequence of kernel launches at "capture time"
//! and replay them identically. They cannot:
//! - Change the number of loop iterations based on output
//! - Skip kernel launches when a sample reaches EOS
//! - Select different code paths based on generated token values
//!
//! async-gpu runs real Rust on GPU with full control flow — loops, branches,
//! and early exits all work naturally because the code is executing, not
//! replaying a captured trace.

use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dynamic Control Flow Demo ===");
    println!("Demonstrating data-dependent GPU compute that CUDA graphs cannot do.\n");

    // Initialize
    let (_dev, registry) = gpu_host::nn::KernelRegistry::init_default()?;
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        return Err(format!("Model not found: {}", model_path.display()).into());
    }
    let weights = gpu_host::model::load_gpt2_weights(&model_path)?;
    let config = gpu_host::nn::models::gpt2::Gpt2Config::small();
    let model = gpu_host::nn::models::gpt2::Gpt2Model::from_weights(&weights, config, &registry)?;
    let tokenizer = gpu_host::tokenizer::Gpt2Tokenizer::new()?;

    // --- Demo 1: Variable-length generation with sampling ---
    println!("--- Demo 1: Variable-Length Generation (top-k sampling) ---");
    println!("Each prompt generates a DIFFERENT number of tokens.");
    println!("The model decides when to stop — loop count is data-dependent.\n");

    let prompts = [
        ("The end.", 1u64),
        ("The capital of France is", 42),
        ("In the beginning, there was nothing but darkness.", 100),
        ("Once upon a time in a land far away,", 200),
        ("def fibonacci(n):", 300),
        ("Breaking news:", 500),
        ("Q: What is 2+2? A:", 777),
        ("To summarize:", 999),
    ];

    let max_tokens = 100;
    let mut results = Vec::new();

    for (prompt, seed) in &prompts {
        let tokens = tokenizer.encode(prompt);
        let mut rng = gpu_host::nn::models::gpt2::SimpleRng::new(*seed);
        let t0 = Instant::now();
        let output = model.generate_cached_sampling(&tokens, max_tokens, 40, 1.0, &mut rng)?;
        let elapsed = t0.elapsed();
        let new_count = output.len() - tokens.len();
        let text = tokenizer.decode(&output).unwrap_or_else(|_| "[decode error]".into());

        let stopped_early = new_count < max_tokens;
        let status = if stopped_early { "EOS" } else { "MAX" };
        println!("  [{status:3}] {:50} | {:3} tokens | {:.1}ms",
            format!("\"{prompt}\""), new_count, elapsed.as_secs_f64() * 1000.0);
        results.push(new_count);

        let preview: String = text.chars().take(120).collect();
        println!("         {preview}...\n");
    }

    let min_len = results.iter().min().unwrap();
    let max_len = results.iter().max().unwrap();
    let eos_count = results.iter().filter(|&&n| n < max_tokens).count();
    println!("  Length range: {min_len} - {max_len} tokens");
    println!("  {eos_count}/{} prompts hit EOS early (data-dependent stopping)", results.len());
    println!("  A CUDA graph would execute {max_tokens} iterations for ALL prompts.\n");

    // --- Demo 2: Top-k sampling — same prompt, different seeds → different outputs ---
    println!("--- Demo 2: Stochastic Sampling — Same Prompt, Different Outputs ---");
    println!("Top-k=40, temperature=0.8. Each seed produces different text + length.\n");

    let sample_prompt = "The future of artificial intelligence";
    let sample_tokens = tokenizer.encode(sample_prompt);
    let seeds = [42u64, 123, 7777, 31415, 99999];

    for &seed in &seeds {
        let mut rng = gpu_host::nn::models::gpt2::SimpleRng::new(seed);
        let t0 = Instant::now();
        let output = model.generate_cached_sampling(&sample_tokens, max_tokens, 40, 0.8, &mut rng)?;
        let elapsed = t0.elapsed();
        let new_count = output.len() - sample_tokens.len();
        let text = tokenizer.decode(&output).unwrap_or_else(|_| "[decode error]".into());

        let preview: String = text.chars().take(120).collect();
        println!("  Seed {seed:5} | {new_count:3} tokens | {:.1}ms", elapsed.as_secs_f64() * 1000.0);
        println!("    -> {preview}...\n");
    }

    // --- Demo 3: Temperature sweep — control diversity vs determinism ---
    println!("--- Demo 3: Temperature Sweep — Data-Dependent Branching ---");
    println!("Same prompt + seed, varying temperature changes outputs.\n");

    let temps = [0.1, 0.5, 0.8, 1.0, 1.5];
    for &temp in &temps {
        let mut rng = gpu_host::nn::models::gpt2::SimpleRng::new(42);
        let t0 = Instant::now();
        let output = model.generate_cached_sampling(&sample_tokens, 40, 40, temp, &mut rng)?;
        let elapsed = t0.elapsed();
        let new_count = output.len() - sample_tokens.len();
        let text = tokenizer.decode(&output).unwrap_or_else(|_| "[decode error]".into());

        let preview: String = text.chars().take(100).collect();
        println!("  temp={temp:.1} | {new_count:3} tokens | {:.1}ms | {preview}...",
            elapsed.as_secs_f64() * 1000.0);
    }

    // --- Demo 4: Early-Exit Inference (single forward pass) ---
    println!("\n--- Demo 4: Early-Exit Inference ---");
    println!("Probe the model's prediction at each layer. Different prompts become");
    println!("confident at different layers — the exit point is data-dependent.\n");

    let early_exit_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "In a surprising turn of events,",
        "def hello():",
        "The color of the sky is",
    ];

    for prompt in &early_exit_prompts {
        let tokens = tokenizer.encode(prompt);
        let dev = &_dev;
        let token_ids = dev.htod_sync_copy(&tokens)?;

        // Probe prediction at each layer using forward_early_exit with decreasing thresholds
        let (logits_full, _) = model.forward_early_exit(&token_ids, tokens.len(), 0.0)?;
        let logits_full_host = logits_full.to_host()?;
        let vocab = 50257usize;
        let last_pos = tokens.len() - 1;
        let full_pred = &logits_full_host[last_pos * vocab..(last_pos + 1) * vocab];
        let full_token = full_pred
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let full_word = tokenizer.decode(&[full_token]).unwrap_or_default();

        println!("  Prompt: \"{prompt}\"  (full-model predicts \"{full_word}\")");

        // Try different thresholds to find the earliest exit layer
        let thresholds = [0.3, 0.5, 0.7, 0.9];
        for &th in &thresholds {
            let (logits_ee, layers_used) =
                model.forward_early_exit(&token_ids, tokens.len(), th)?;
            let ee_host = logits_ee.to_host()?;
            let ee_pred = &ee_host[last_pos * vocab..(last_pos + 1) * vocab];
            let ee_token = ee_pred
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            let ee_word = tokenizer.decode(&[ee_token]).unwrap_or_default();
            let match_str = if ee_token == full_token {
                "MATCH"
            } else {
                "DIFF "
            };
            let bar: String = "#".repeat(layers_used);
            let pad: String = ".".repeat(12 - layers_used);
            println!(
                "    th={th:.1} | {layers_used:2}/12 layers | {bar}{pad} | \"{ee_word}\" [{match_str}]"
            );
        }
        println!();
    }

    println!("  Key insight: 'easy' tokens (predictable next words) exit early,");
    println!("  'hard' tokens (surprising continuations) need all 12 layers.");
    println!("  This per-token adaptive compute is impossible with CUDA graphs.\n");

    // --- Demo 5: Early-exit generation with full layers ---
    println!("--- Demo 5: Generation with Layer Usage Tracking ---");
    println!("Full 12-layer generation, but tracking which layer WOULD have been sufficient.\n");

    let track_prompt = "The meaning of life is";
    let track_tokens = tokenizer.encode(track_prompt);
    let t0 = Instant::now();
    let (track_output, _track_layers) =
        model.generate_cached_early_exit(&track_tokens, 30, 1.0)?; // threshold=1.0 → always use all layers
    let elapsed = t0.elapsed();
    let track_text = tokenizer
        .decode(&track_output)
        .unwrap_or_else(|_| "[decode error]".into());

    println!("  Prompt: \"{track_prompt}\"");
    let preview: String = track_text.chars().take(120).collect();
    println!("  Output: {preview}");
    println!("  Time: {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

    // Now show what would happen with early exit
    let t1 = Instant::now();
    let (ee_output, ee_layers) =
        model.generate_cached_early_exit(&track_tokens, 30, 0.9)?;
    let ee_elapsed = t1.elapsed();
    let ee_text = tokenizer
        .decode(&ee_output)
        .unwrap_or_else(|_| "[decode error]".into());
    let avg_layers: f64 =
        ee_layers.iter().map(|&l| l as f64).sum::<f64>() / ee_layers.len() as f64;
    let savings = (1.0 - avg_layers / 12.0) * 100.0;

    let ee_preview: String = ee_text.chars().take(120).collect();
    println!("  Early-exit (th=0.9): {ee_preview}");
    println!(
        "  Time: {:.1}ms | Avg layers: {avg_layers:.1}/12 | Compute saved: {savings:.0}%",
        ee_elapsed.as_secs_f64() * 1000.0
    );

    println!("\n=== Why This Matters ===");
    println!("Every generation above executed a DIFFERENT number of GPU kernel launches.");
    println!("The loop count, token choices, layer count, and stopping are all DATA-DEPENDENT.");
    println!("CUDA graphs capture a FIXED sequence — they cannot express this.");
    println!("async-gpu runs real Rust with real control flow on GPU.");

    Ok(())
}
