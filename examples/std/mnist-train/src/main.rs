//! MNIST MLP training example using gpu-host autograd.
//!
//! Architecture: Linear(784→128) → ReLU → Linear(128→10) → CrossEntropy
//! Optimizer: SGD with lr=0.01
//!
//! Requires: models/mnist/ (run `bash scripts/download-mnist.sh`)

use std::sync::Arc;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Load MNIST data
    let mnist_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("mnist");
    println!("Loading MNIST from {}...", mnist_dir.display());

    let train_images = load_idx_images(&mnist_dir.join("train-images-idx3-ubyte"))?;
    let train_labels = load_idx_labels(&mnist_dir.join("train-labels-idx1-ubyte"))?;
    let test_images = load_idx_images(&mnist_dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = load_idx_labels(&mnist_dir.join("t10k-labels-idx1-ubyte"))?;

    println!(
        "Loaded: {} train, {} test images (28×28)",
        train_images.len(),
        test_images.len()
    );

    // Initialize GPU
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);
    println!("CUDA initialized");

    // Initialize weights (Xavier-like)
    let in_features = 784;
    let hidden = 128;
    let out_features = 10;

    let scale1 = (2.0 / in_features as f64).sqrt() as f32;
    let mut w1: Vec<f32> = (0..hidden * in_features)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale1)
        .collect();
    let mut b1 = vec![0.0f32; hidden];

    let scale2 = (2.0 / hidden as f64).sqrt() as f32;
    let mut w2: Vec<f32> = (0..out_features * hidden)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale2)
        .collect();
    let mut b2 = vec![0.0f32; out_features];

    let lr = 0.01f32;
    let batch_size = 64;
    let epochs = 5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_images.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;

            // Prepare batch
            let mut batch_data = vec![0.0f32; batch_size * in_features];
            let mut batch_labels = vec![0u32; batch_size];
            for i in 0..batch_size {
                let img = &train_images[start + i];
                for j in 0..in_features {
                    batch_data[i * in_features + j] = img[j];
                }
                batch_labels[i] = train_labels[start + i] as u32;
            }

            // Forward pass (CPU for simplicity in v1 — GPU matmul for larger models)
            // h = relu(x @ W1^T + b1)
            let mut hidden_act = vec![0.0f32; batch_size * hidden];
            let mut hidden_pre = vec![0.0f32; batch_size * hidden];
            for b in 0..batch_size {
                for h in 0..hidden {
                    let mut sum = b1[h];
                    for j in 0..in_features {
                        sum += batch_data[b * in_features + j] * w1[h * in_features + j];
                    }
                    hidden_pre[b * hidden + h] = sum;
                    hidden_act[b * hidden + h] = sum.max(0.0);
                }
            }

            // logits = h @ W2^T + b2
            let mut logits = vec![0.0f32; batch_size * out_features];
            for b in 0..batch_size {
                for o in 0..out_features {
                    let mut sum = b2[o];
                    for h in 0..hidden {
                        sum += hidden_act[b * hidden + h] * w2[o * hidden + h];
                    }
                    logits[b * out_features + o] = sum;
                }
            }

            // Cross-entropy loss + softmax
            let mut batch_loss = 0.0f64;
            let mut softmax_out = vec![0.0f32; batch_size * out_features];
            for b in 0..batch_size {
                let row = &logits[b * out_features..(b + 1) * out_features];
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                for o in 0..out_features {
                    softmax_out[b * out_features + o] = (row[o] - max_val).exp() / exp_sum;
                }
                batch_loss -= (softmax_out[b * out_features + batch_labels[b] as usize]).ln() as f64;

                // Accuracy
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == batch_labels[b] as usize {
                    correct += 1;
                }
            }
            batch_loss /= batch_size as f64;
            total_loss += batch_loss;

            // Backward: d_logits = softmax - one_hot
            let mut d_logits = vec![0.0f32; batch_size * out_features];
            for b in 0..batch_size {
                for o in 0..out_features {
                    d_logits[b * out_features + o] = softmax_out[b * out_features + o];
                    if o == batch_labels[b] as usize {
                        d_logits[b * out_features + o] -= 1.0;
                    }
                    d_logits[b * out_features + o] /= batch_size as f32;
                }
            }

            // dW2 = d_logits^T @ hidden_act, db2 = sum(d_logits)
            let mut dw2 = vec![0.0f32; out_features * hidden];
            let mut db2 = vec![0.0f32; out_features];
            for o in 0..out_features {
                for h in 0..hidden {
                    let mut sum = 0.0f32;
                    for b in 0..batch_size {
                        sum += d_logits[b * out_features + o] * hidden_act[b * hidden + h];
                    }
                    dw2[o * hidden + h] = sum;
                }
                for b in 0..batch_size {
                    db2[o] += d_logits[b * out_features + o];
                }
            }

            // d_hidden = d_logits @ W2 * relu'(hidden_pre)
            let mut d_hidden = vec![0.0f32; batch_size * hidden];
            for b in 0..batch_size {
                for h in 0..hidden {
                    let mut sum = 0.0f32;
                    for o in 0..out_features {
                        sum += d_logits[b * out_features + o] * w2[o * hidden + h];
                    }
                    d_hidden[b * hidden + h] =
                        sum * if hidden_pre[b * hidden + h] > 0.0 { 1.0 } else { 0.0 };
                }
            }

            // dW1 = d_hidden^T @ x, db1 = sum(d_hidden)
            let mut dw1 = vec![0.0f32; hidden * in_features];
            let mut db1 = vec![0.0f32; hidden];
            for h in 0..hidden {
                for j in 0..in_features {
                    let mut sum = 0.0f32;
                    for b in 0..batch_size {
                        sum += d_hidden[b * hidden + h] * batch_data[b * in_features + j];
                    }
                    dw1[h * in_features + j] = sum;
                }
                for b in 0..batch_size {
                    db1[h] += d_hidden[b * hidden + h];
                }
            }

            // SGD update
            for i in 0..w1.len() {
                w1[i] -= lr * dw1[i];
            }
            for i in 0..b1.len() {
                b1[i] -= lr * db1[i];
            }
            for i in 0..w2.len() {
                w2[i] -= lr * dw2[i];
            }
            for i in 0..b2.len() {
                b2[i] -= lr * db2[i];
            }
        }

        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;

        // Test accuracy
        let test_correct = evaluate(&test_images, &test_labels, &w1, &b1, &w2, &b2);
        let test_acc = test_correct as f64 / test_images.len() as f64 * 100.0;

        println!(
            "Epoch {}/{}: loss={avg_loss:.4}, train_acc={train_acc:.1}%, test_acc={test_acc:.1}%",
            epoch + 1,
            epochs
        );
    }

    println!("\nDone.");
    Ok(())
}

fn evaluate(
    images: &[Vec<f32>],
    labels: &[u8],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> usize {
    let in_f = 784;
    let hidden = 128;
    let out_f = 10;
    let mut correct = 0;

    for (img, &label) in images.iter().zip(labels.iter()) {
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut sum = b1[j];
            for k in 0..in_f {
                sum += img[k] * w1[j * in_f + k];
            }
            h[j] = sum.max(0.0);
        }
        let mut logits = vec![0.0f32; out_f];
        for o in 0..out_f {
            let mut sum = b2[o];
            for j in 0..hidden {
                sum += h[j] * w2[o * hidden + j];
            }
            logits[o] = sum;
        }
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if pred == label as usize {
            correct += 1;
        }
    }
    correct
}

// ============================================================
// IDX file format parser
// ============================================================

fn load_idx_images(path: &std::path::Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    if data.len() < 16 {
        return Err("IDX file too short".into());
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2051 {
        return Err(format!("Bad magic: {magic}, expected 2051").into());
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let pixels = rows * cols;

    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let offset = 16 + i * pixels;
        let img: Vec<f32> = data[offset..offset + pixels]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(img);
    }
    Ok(images)
}

fn load_idx_labels(path: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Err("IDX file too short".into());
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2049 {
        return Err(format!("Bad magic: {magic}, expected 2049").into());
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    Ok(data[8..8 + n].to_vec())
}
