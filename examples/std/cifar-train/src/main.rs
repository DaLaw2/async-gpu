//! CIFAR-10 tiny CNN training example.
//!
//! Architecture: Conv2d(3→8,3×3) → ReLU → AvgPool(4) → Flatten → Linear(512→10)
//! Optimizer: SGD with lr=0.005
//!
//! Uses a 1000-image subset for fast training demo.
//! Requires: models/cifar10/ (run `bash scripts/download-cifar10.sh`)

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    println!("Loading CIFAR-10 from {}...", cifar_dir.display());

    // Load subset (first 1000 train, first 500 test)
    let (train_imgs, train_labels) =
        load_cifar_batch(&cifar_dir.join("data_batch_1.bin"), 1000)?;
    let (test_imgs, test_labels) = load_cifar_batch(&cifar_dir.join("test_batch.bin"), 500)?;

    println!(
        "Loaded: {} train, {} test (3×32×32)",
        train_imgs.len(),
        test_imgs.len()
    );

    // Architecture: Conv2d(3→8, 3×3, pad=1) → ReLU → AvgPool(4×4) → Linear(8*8*8=512→10)
    let c_in = 3;
    let c_out = 8;
    let h = 32;
    let w = 32;
    let kh = 3;
    let kw = 3;
    let pool_size = 4;
    let h_pooled = h / pool_size;
    let w_pooled = w / pool_size;
    let flat_dim = c_out * h_pooled * w_pooled; // 8*8*8 = 512
    let n_classes = 10;

    // Initialize weights
    let scale_conv = (2.0 / (c_in * kh * kw) as f64).sqrt() as f32;
    let mut conv_w: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_conv)
        .collect();

    let scale_fc = (2.0 / flat_dim as f64).sqrt() as f32;
    let mut fc_w: Vec<f32> = (0..n_classes * flat_dim)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_fc)
        .collect();
    let mut fc_b = vec![0.0f32; n_classes];

    let lr = 0.005f32;
    let batch_size = 32;
    let epochs = 10;

    for epoch in 0..epochs {
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_imgs.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;

            let mut batch_loss = 0.0f64;

            for bi in 0..batch_size {
                let img = &train_imgs[start + bi];
                let label = train_labels[start + bi] as usize;

                // Forward: conv2d → relu → avg_pool → flatten → linear → softmax
                let conv_out = cpu_conv2d(img, &conv_w, c_in, h, w, c_out, kh, kw, 1, 1);
                let relu_out: Vec<f32> = conv_out.iter().map(|&v| v.max(0.0)).collect();
                let pooled = cpu_avg_pool(&relu_out, c_out, h, w, pool_size);

                // Linear: pooled [flat_dim] → logits [n_classes]
                let mut logits = vec![0.0f32; n_classes];
                for o in 0..n_classes {
                    logits[o] = fc_b[o];
                    for j in 0..flat_dim {
                        logits[o] += pooled[j] * fc_w[o * flat_dim + j];
                    }
                }

                // Softmax + cross-entropy
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = logits.iter().map(|&x| (x - max_l).exp()).sum();
                let mut probs = vec![0.0f32; n_classes];
                for o in 0..n_classes {
                    probs[o] = (logits[o] - max_l).exp() / exp_sum;
                }
                batch_loss -= probs[label].ln() as f64;

                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == label {
                    correct += 1;
                }

                // Backward: d_logits = probs - one_hot
                let mut d_logits = probs.clone();
                d_logits[label] -= 1.0;
                for o in 0..n_classes {
                    d_logits[o] /= batch_size as f32;
                }

                // dW_fc, db_fc
                for o in 0..n_classes {
                    fc_b[o] -= lr * d_logits[o];
                    for j in 0..flat_dim {
                        fc_w[o * flat_dim + j] -= lr * d_logits[o] * pooled[j];
                    }
                }

                // d_pooled → d_relu → d_conv → dW_conv (simplified: no conv weight grad for demo speed)
                // Full CNN backward would update conv weights, but the demo focuses on the pipeline.
            }

            batch_loss /= batch_size as f64;
            total_loss += batch_loss;
        }

        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;

        // Test
        let test_correct = evaluate_cnn(
            &test_imgs,
            &test_labels,
            &conv_w,
            &fc_w,
            &fc_b,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            pool_size,
            flat_dim,
            n_classes,
        );
        let test_acc = test_correct as f64 / test_imgs.len() as f64 * 100.0;

        println!(
            "Epoch {}/{}: loss={avg_loss:.4}, train_acc={train_acc:.1}%, test_acc={test_acc:.1}%",
            epoch + 1,
            epochs
        );
    }

    println!("\nDone.");
    Ok(())
}

// ============================================================
// CPU conv2d, avg pool, evaluation
// ============================================================

fn cpu_conv2d(
    input: &[f32],
    weight: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    for co in 0..c_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0f32;
                for ci in 0..c_in {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = (oh * stride + fh) as isize - padding as isize;
                            let iw = (ow * stride + fw) as isize - padding as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                sum += input[ci * h * w + ih as usize * w + iw as usize]
                                    * weight[co * (c_in * kh * kw) + ci * (kh * kw) + fh * kw + fw];
                            }
                        }
                    }
                }
                out[co * h_out * w_out + oh * w_out + ow] = sum;
            }
        }
    }
    out
}

fn cpu_avg_pool(input: &[f32], c: usize, h: usize, w: usize, pool_size: usize) -> Vec<f32> {
    let h_out = h / pool_size;
    let w_out = w / pool_size;
    let mut out = vec![0.0f32; c * h_out * w_out];
    let area = (pool_size * pool_size) as f32;
    for ch in 0..c {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0f32;
                for ph in 0..pool_size {
                    for pw in 0..pool_size {
                        sum += input[ch * h * w + (oh * pool_size + ph) * w + ow * pool_size + pw];
                    }
                }
                out[ch * h_out * w_out + oh * w_out + ow] = sum / area;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn evaluate_cnn(
    images: &[Vec<f32>],
    labels: &[u8],
    conv_w: &[f32],
    fc_w: &[f32],
    fc_b: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    pool_size: usize,
    flat_dim: usize,
    n_classes: usize,
) -> usize {
    let mut correct = 0;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let conv_out = cpu_conv2d(img, conv_w, c_in, h, w, c_out, kh, kw, 1, 1);
        let relu_out: Vec<f32> = conv_out.iter().map(|&v| v.max(0.0)).collect();
        let pooled = cpu_avg_pool(&relu_out, c_out, h, w, pool_size);
        let mut logits = vec![0.0f32; n_classes];
        for o in 0..n_classes {
            logits[o] = fc_b[o];
            for j in 0..flat_dim {
                logits[o] += pooled[j] * fc_w[o * flat_dim + j];
            }
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
// CIFAR-10 binary format parser
// ============================================================

fn load_cifar_batch(
    path: &std::path::Path,
    max_samples: usize,
) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let record_size = 3073; // 1 label + 3072 pixels
    let n = (data.len() / record_size).min(max_samples);

    let mut images = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * record_size;
        labels.push(data[offset]);
        let pixels = &data[offset + 1..offset + record_size];
        let img: Vec<f32> = pixels.iter().map(|&b| b as f32 / 255.0).collect();
        images.push(img);
    }

    Ok((images, labels))
}
