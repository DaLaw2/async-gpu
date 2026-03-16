//! GPU-Autonomous RAG: Retrieval-Augmented Generation on GPU.
//!
//! Demonstrates the async GPU programming model by running a full RAG pipeline:
//! 1. Embed text chunks via GPT-2 wte averaging (GPU matmul)
//! 2. Cosine similarity search (GPU matmul)
//! 3. GPT-2 text generation conditioned on retrieved context
//!
//! Usage:
//!   cargo run --release
//!   cargo run --release -- "What is Rust?"

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::models::gpt2::{Gpt2Config, Gpt2Model};
use gpu_host::nn::tensor::GpuTensor;
use gpu_host::tokenizer::Gpt2Tokenizer;

fn main() {
    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What programming language runs on GPU?".to_string());

    if let Err(e) = run(&query) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);

    println!("=== GPU-Autonomous RAG Demo ===\n");

    // --- Step 1: Load GPT-2 model ---
    let t0 = Instant::now();
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        return Err(format!(
            "GPT-2 model not found at {}. Run: bash scripts/download-gpt2.sh",
            model_path.display()
        )
        .into());
    }

    let config = Gpt2Config::small();
    let tokenizer = Gpt2Tokenizer::new()?;

    let weight_map = gpu_host::model_generic::gpt2_weight_map(config.n_layer);
    let weights =
        gpu_host::model_generic::load_safetensors_mapped(&model_path, &weight_map)?;
    let model = Gpt2Model::from_generic_weights(&weights, config.clone(), &registry)?;
    println!("GPT-2 loaded: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // --- Step 2: Build vector store from sample chunks ---
    let t1 = Instant::now();
    let chunks = sample_knowledge_base();
    println!("Knowledge base: {} chunks", chunks.len());

    // Embed each chunk via wte averaging (fast, no transformer forward pass)
    let wte_host = extract_wte(&model, &dev)?;
    let embed_dim = config.n_embd;

    let mut store_embeddings = Vec::with_capacity(chunks.len() * embed_dim);
    for chunk in &chunks {
        let tokens = tokenizer.encode(chunk);
        let emb = wte_average(&wte_host, &tokens, embed_dim);
        let norm = l2_normalize(&emb);
        store_embeddings.extend_from_slice(&norm);
    }

    // Upload store to GPU: [N, embed_dim]
    let n_chunks = chunks.len();
    let store_gpu = GpuTensor::from_host(&store_embeddings, &[n_chunks, embed_dim], &dev)?;
    // Transpose for matmul: [embed_dim, N]
    let store_t = store_gpu.transpose(0, 1)?;
    println!(
        "Vector store built: {n_chunks} chunks × {embed_dim}d, {:.1}ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // --- Step 3: Embed query + cosine similarity search ---
    let t2 = Instant::now();
    let query_tokens = tokenizer.encode(query);
    let query_emb = wte_average(&wte_host, &query_tokens, embed_dim);
    let query_norm = l2_normalize(&query_emb);
    let query_gpu = GpuTensor::from_host(&query_norm, &[1, embed_dim], &dev)?;

    // Cosine similarity: [1, embed_dim] × [embed_dim, N] → [1, N]
    let scores = gpu_host::nn::ops::matmul(&query_gpu, &store_t, &registry)?;
    let scores_host = scores.to_host()?;

    // Top-K selection (CPU, trivial for N=1000)
    let top_k = 3;
    let mut indexed: Vec<(usize, f32)> = scores_host.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nQuery: \"{query}\"");
    println!(
        "Search: {:.1}ms ({n_chunks} chunks)",
        t2.elapsed().as_secs_f64() * 1000.0
    );
    println!("\nTop-{top_k} retrieved chunks:");
    let mut context = String::new();
    for (rank, &(idx, score)) in indexed.iter().take(top_k).enumerate() {
        println!("  [{rank}] score={score:.4}: {}", &chunks[idx][..chunks[idx].len().min(80)]);
        context.push_str(chunks[idx]);
        context.push('\n');
    }

    // --- Step 4: GPT-2 generation conditioned on retrieved context ---
    let t3 = Instant::now();
    let prompt = format!("Context:\n{context}\nQuestion: {query}\nAnswer:");
    let prompt_tokens = tokenizer.encode(&prompt);
    let prompt_len = prompt_tokens.len();

    // Generate tokens
    let max_new = 40;
    let mut all_tokens = prompt_tokens.clone();
    let mut gen_tokens = Vec::new();

    println!("\nGenerating ({prompt_len} prompt tokens + {max_new} new)...");

    for _ in 0..max_new {
        let seq_len = all_tokens.len();
        let token_buf =
            dev.htod_copy(all_tokens.iter().copied().collect::<Vec<u32>>())?;
        let logits = model.forward(&token_buf, seq_len)?;
        let logits_host = logits.to_host()?;

        // Greedy: pick argmax of last position
        let vocab = config.vocab_size;
        let last_logits = &logits_host[(seq_len - 1) * vocab..seq_len * vocab];
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        if next_token == gpu_host::tokenizer::ENDOFTEXT_TOKEN_ID {
            break;
        }
        all_tokens.push(next_token);
        gen_tokens.push(next_token);
    }

    let generated_text = tokenizer.decode(&gen_tokens)?;
    let gen_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let ms_per_token = if !gen_tokens.is_empty() {
        gen_ms / gen_tokens.len() as f64
    } else {
        0.0
    };

    println!("\nGenerated ({} tokens, {gen_ms:.0}ms, {ms_per_token:.1}ms/tok):", gen_tokens.len());
    println!("  {generated_text}");

    let total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n--- Summary ---");
    println!("Total: {total:.0}ms (model load + embed + search + generate)");
    println!(
        "Pipeline: embed({:.1}ms) → search({:.1}ms) → generate({gen_ms:.0}ms)",
        t1.elapsed().as_secs_f64() * 1000.0,
        t2.elapsed().as_secs_f64() * 1000.0,
    );
    println!("PASSED (GPU-Autonomous RAG pipeline complete)");
    Ok(())
}

/// Extract wte (token embedding table) from GPT-2 model.
/// Returns [vocab_size, embed_dim] as host Vec.
fn extract_wte(
    model: &Gpt2Model,
    _dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (wte, _wpe) = model.embedding_table();
    Ok(wte.to_host()?)
}

/// Average token embeddings from wte to produce a fixed-size embedding.
fn wte_average(wte: &[f32], tokens: &[u32], embed_dim: usize) -> Vec<f32> {
    let mut avg = vec![0.0f32; embed_dim];
    if tokens.is_empty() {
        return avg;
    }
    for &tok in tokens {
        let offset = tok as usize * embed_dim;
        if offset + embed_dim <= wte.len() {
            for d in 0..embed_dim {
                avg[d] += wte[offset + d];
            }
        }
    }
    let scale = 1.0 / tokens.len() as f32;
    for v in &mut avg {
        *v *= scale;
    }
    avg
}

/// L2-normalize a vector in place.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Sample knowledge base for the RAG demo.
fn sample_knowledge_base() -> Vec<&'static str> {
    vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without garbage collection through its ownership system.",
        "CUDA is NVIDIA's parallel computing platform that enables GPU-accelerated computing. It provides APIs for C, C++, and Fortran to access GPU hardware.",
        "async_gpu is a research project that brings Rust async/await to GPU programming. It uses a patched rustc compiler to compile Rust futures to PTX assembly.",
        "GPT-2 is a transformer-based language model by OpenAI with 124 million parameters. It uses self-attention and feed-forward layers for text generation.",
        "The hostcall mechanism allows GPU kernels to request services from the host CPU. This enables file I/O, network access, and memory allocation from within GPU code.",
        "Tensor Cores are specialized hardware units in NVIDIA GPUs that accelerate matrix multiply-accumulate operations, particularly for deep learning inference.",
        "YOLOv8 is a real-time object detection model that processes images through a backbone network, feature pyramid, and detection head to identify objects.",
        "A warp in CUDA consists of 32 threads that execute instructions in lockstep. Warp-level primitives like shuffle allow efficient intra-warp communication.",
        "Batch normalization normalizes activations across the batch dimension, stabilizing training and allowing higher learning rates in deep neural networks.",
        "The im2col algorithm transforms convolution into matrix multiplication by unfolding input patches into columns, enabling efficient GPU convolution via GEMM.",
        "SafeTensors is a file format for storing neural network weights safely. It supports zero-copy deserialization and prevents arbitrary code execution during loading.",
        "ResNet-18 is a convolutional neural network with 18 layers and residual connections. Skip connections allow training very deep networks without vanishing gradients.",
        "Flash Attention computes attention with O(N) memory instead of O(N²) by tiling the computation and never materializing the full attention matrix in GPU memory.",
        "The KV cache stores previously computed key and value tensors in transformer inference, eliminating redundant computation for each new generated token.",
        "PTX (Parallel Thread Execution) is NVIDIA's intermediate representation for GPU programs. It is compiled by the GPU driver into native machine code at load time.",
        "Cosine similarity measures the angle between two vectors, commonly used in information retrieval and semantic search for comparing text embeddings.",
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. The retriever finds relevant documents, which the generator uses as context.",
        "The autograd tape records forward operations and replays them in reverse for backpropagation. Each operation stores the inputs needed for its gradient computation.",
        "Mixed precision training uses FP16 for forward/backward computation while maintaining FP32 master weights, achieving 2x speedup with minimal accuracy loss.",
        "GPU memory hierarchy includes registers (fastest), shared memory (block-local), L1/L2 cache, and global memory (slowest). Optimizing data placement is key to performance.",
        "LoRA (Low-Rank Adaptation) fine-tunes large models by training small rank-decomposed matrices A and B instead of updating all weights, reducing trainable parameters by 100x.",
        "The Treiber stack is a lock-free concurrent data structure using compare-and-swap. It is used in the hostcall protocol for packet allocation and deallocation.",
        "Formal verification with TLA+ can prove that concurrent protocols satisfy safety and liveness properties. The hostcall protocol was verified for 367M states with zero errors.",
        "Convolution backward pass computes two gradients: dInput via col2im and dWeight via im2col transposed matmul. Both use the same unfolding machinery as forward.",
        "Cross-entropy loss measures the difference between predicted probabilities and true labels. Its gradient simplifies to softmax(logits) - one_hot(target) for classification.",
        "The SPPF (Spatial Pyramid Pooling Fast) module in YOLOv8 captures multi-scale features by applying max pooling at different kernel sizes and concatenating results.",
        "Adam optimizer combines momentum with adaptive learning rates per parameter. It maintains running averages of gradients (m) and squared gradients (v) for smooth updates.",
        "Global average pooling replaces the fully connected classifier in modern CNNs. It averages each feature map to a single value, reducing parameters and overfitting.",
        "The C2f module in YOLOv8 uses cross-stage partial connections with two convolutions and N bottleneck blocks, efficiently combining features from different depths.",
        "Warp-cooperative async enables GPU threads within a warp to share a single hostcall packet. Lane 0 submits the request, and the result is broadcast via shuffle instructions.",
    ]
}
