//! MNIST MLP training on GPU using autograd.
//!
//! Forward pass: GPU matmul (gemm_f32 kernel) + CPU ReLU + GPU matmul + CPU softmax
//! Backward pass: autograd tape → backward() → GPU matmul for weight gradients
//! Optimizer: SGD on host
//!
//! Requires: models/mnist/ (run `bash scripts/download-mnist.sh`)

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::autograd;
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    let use_cpu = std::env::args().any(|a| a == "--cpu");
    if let Err(e) = if use_cpu { run_cpu() } else { run_gpu() } {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// CPU-only training for benchmark comparison.
fn run_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let mnist_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("mnist");
    let train_images = load_idx_images(&mnist_dir.join("train-images-idx3-ubyte"))?;
    let train_labels = load_idx_labels(&mnist_dir.join("train-labels-idx1-ubyte"))?;
    let test_images = load_idx_images(&mnist_dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = load_idx_labels(&mnist_dir.join("t10k-labels-idx1-ubyte"))?;
    println!("MNIST CPU training ({} train, {} test)", train_images.len(), test_images.len());

    let in_f = 784;
    let hidden = 128;
    let out_f = 10;
    let scale1 = (2.0 / in_f as f64).sqrt() as f32;
    let mut w1: Vec<f32> = (0..hidden * in_f).map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale1).collect();
    let mut b1 = vec![0.0f32; hidden];
    let scale2 = (2.0 / hidden as f64).sqrt() as f32;
    let mut w2: Vec<f32> = (0..out_f * hidden).map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale2).collect();
    let mut b2 = vec![0.0f32; out_f];

    let lr = 0.01f32;
    let batch_size = 64;
    let epochs = 5;
    let total_start = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_images.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let mut batch_x = vec![0.0f32; batch_size * in_f];
            let mut batch_y = vec![0u32; batch_size];
            for i in 0..batch_size {
                batch_x[i * in_f..(i + 1) * in_f].copy_from_slice(&train_images[start + i]);
                batch_y[i] = train_labels[start + i] as u32;
            }

            // CPU forward
            let mut h_pre = vec![0.0f32; batch_size * hidden];
            let mut h_act = vec![0.0f32; batch_size * hidden];
            for b in 0..batch_size {
                for j in 0..hidden {
                    let mut s = b1[j];
                    for k in 0..in_f { s += batch_x[b * in_f + k] * w1[j * in_f + k]; }
                    h_pre[b * hidden + j] = s;
                    h_act[b * hidden + j] = s.max(0.0);
                }
            }
            let mut logits = vec![0.0f32; batch_size * out_f];
            for b in 0..batch_size {
                for o in 0..out_f {
                    let mut s = b2[o];
                    for j in 0..hidden { s += h_act[b * hidden + j] * w2[o * hidden + j]; }
                    logits[b * out_f + o] = s;
                }
            }

            // Softmax + CE
            let mut d_logits = vec![0.0f32; batch_size * out_f];
            for b in 0..batch_size {
                let row = &logits[b * out_f..(b + 1) * out_f];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let es: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                for o in 0..out_f {
                    let sm = (row[o] - mx).exp() / es;
                    d_logits[b * out_f + o] = (sm - if o == batch_y[b] as usize { 1.0 } else { 0.0 }) / batch_size as f32;
                }
                total_loss -= ((row[batch_y[b] as usize] - mx).exp() / es).ln() as f64;
                let pred = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
                if pred == batch_y[b] as usize { correct += 1; }
            }

            // CPU backward
            let mut dw2 = vec![0.0f32; out_f * hidden];
            let mut db2 = vec![0.0f32; out_f];
            for o in 0..out_f {
                for j in 0..hidden {
                    let mut s = 0.0f32;
                    for b in 0..batch_size { s += d_logits[b * out_f + o] * h_act[b * hidden + j]; }
                    dw2[o * hidden + j] = s;
                }
                for b in 0..batch_size { db2[o] += d_logits[b * out_f + o]; }
            }
            let mut dh = vec![0.0f32; batch_size * hidden];
            for b in 0..batch_size {
                for j in 0..hidden {
                    let mut s = 0.0f32;
                    for o in 0..out_f { s += d_logits[b * out_f + o] * w2[o * hidden + j]; }
                    dh[b * hidden + j] = s * if h_pre[b * hidden + j] > 0.0 { 1.0 } else { 0.0 };
                }
            }
            let mut dw1 = vec![0.0f32; hidden * in_f];
            let mut db1 = vec![0.0f32; hidden];
            for j in 0..hidden {
                for k in 0..in_f {
                    let mut s = 0.0f32;
                    for b in 0..batch_size { s += dh[b * hidden + j] * batch_x[b * in_f + k]; }
                    dw1[j * in_f + k] = s;
                }
                for b in 0..batch_size { db1[j] += dh[b * hidden + j]; }
            }

            for i in 0..w1.len() { w1[i] -= lr * dw1[i]; }
            for i in 0..b1.len() { b1[i] -= lr * db1[i]; }
            for i in 0..w2.len() { w2[i] -= lr * dw2[i]; }
            for i in 0..b2.len() { b2[i] -= lr * db2[i]; }
        }

        let avg_loss = total_loss / n_batches as f64 / batch_size as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;
        let test_correct = evaluate_cpu(&test_images, &test_labels, &w1, &b1, &w2, &b2);
        let test_acc = test_correct as f64 / test_images.len() as f64 * 100.0;
        println!("Epoch {}/{}: loss={avg_loss:.4}, train={train_acc:.1}%, test={test_acc:.1}%, time={:.1}s",
            epoch + 1, epochs, es.elapsed().as_secs_f64());
    }
    println!("\nTotal: {:.1}s (CPU)", total_start.elapsed().as_secs_f64());
    Ok(())
}

fn evaluate_cpu(images: &[Vec<f32>], labels: &[u8], w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> usize {
    let (in_f, hidden, out_f) = (784, 128, 10);
    let mut correct = 0;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut s = b1[j];
            for k in 0..in_f { s += img[k] * w1[j * in_f + k]; }
            h[j] = s.max(0.0);
        }
        let mut logits = vec![0.0f32; out_f];
        for o in 0..out_f {
            let mut s = b2[o];
            for j in 0..hidden { s += h[j] * w2[o * hidden + j]; }
            logits[o] = s;
        }
        let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        if pred == label as usize { correct += 1; }
    }
    correct
}

fn run_gpu() -> Result<(), Box<dyn std::error::Error>> {
    // Load MNIST
    let mnist_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("mnist");
    println!("Loading MNIST from {}...", mnist_dir.display());

    let train_images = load_idx_images(&mnist_dir.join("train-images-idx3-ubyte"))?;
    let train_labels = load_idx_labels(&mnist_dir.join("train-labels-idx1-ubyte"))?;
    let test_images = load_idx_images(&mnist_dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = load_idx_labels(&mnist_dir.join("t10k-labels-idx1-ubyte"))?;

    println!(
        "Loaded: {} train, {} test images",
        train_images.len(),
        test_images.len()
    );

    // Initialize GPU
    let (dev, registry) = gpu_host::nn::KernelRegistry::init_default()?;
    println!("CUDA device initialized (GPU GEMM for matmul)");

    // Model: Linear(784→128) → ReLU → Linear(128→10)
    let in_f = 784;
    let hidden = 128;
    let out_f = 10;

    // Xavier init
    let scale1 = (2.0 / in_f as f64).sqrt() as f32;
    let mut w1_host: Vec<f32> = (0..hidden * in_f)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale1)
        .collect();
    let mut b1_host = vec![0.0f32; hidden];

    let scale2 = (2.0 / hidden as f64).sqrt() as f32;
    let mut w2_host: Vec<f32> = (0..out_f * hidden)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale2)
        .collect();
    let mut b2_host = vec![0.0f32; out_f];

    let lr = 0.01f32;
    let batch_size = 64;
    let epochs = 5;

    let total_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_images.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;

            // Prepare batch data
            let mut batch_x = vec![0.0f32; batch_size * in_f];
            let mut batch_y = vec![0u32; batch_size];
            for i in 0..batch_size {
                batch_x[i * in_f..(i + 1) * in_f].copy_from_slice(&train_images[start + i]);
                batch_y[i] = train_labels[start + i] as u32;
            }

            // ========================================
            // Forward pass — GPU matmul via autograd
            // ========================================
            let tape = autograd::Tape::new();
            let mut pool = autograd::TensorPool::new();

            let (logits_host, tape) = autograd::with_tape(tape, || {
                // Upload batch to GPU (need tensor_id for matmul backward to compute dW)
                let mut x_gpu =
                    GpuTensor::from_host(&batch_x, &[batch_size, in_f], &dev).unwrap();
                let x_id = autograd::alloc_tensor_id().unwrap();
                x_gpu.set_tensor_id(x_id);
                pool.insert(x_id, x_gpu.clone_tensor().unwrap());

                // W1: [in_f, hidden] for matmul x[batch, in_f] × W1[in_f, hidden] = h[batch, hidden]
                let mut w1_gpu =
                    GpuTensor::from_host(&w1_host, &[in_f, hidden], &dev).unwrap();
                let w1_id = autograd::alloc_tensor_id().unwrap();
                w1_gpu.set_tensor_id(w1_id);
                w1_gpu.set_requires_grad(true);
                pool.insert(w1_id, w1_gpu.clone_tensor().unwrap());

                // GPU matmul: x × W1 → [batch, hidden]
                let mut h1 = gpu_host::nn::ops::matmul(&x_gpu, &w1_gpu, &registry).unwrap();
                let h1_id = h1.tensor_id().unwrap();
                pool.insert(h1_id, h1.clone_tensor().unwrap());

                // Bias add (GPU kernel)
                let b1_gpu = GpuTensor::from_host(&b1_host, &[hidden], &dev).unwrap();
                gpu_host::nn::ops::bias_add(&mut h1, &b1_gpu, &registry).unwrap();

                // ReLU on host (download, apply, upload)
                let h1_vals = h1.to_host().unwrap();
                let h1_relu: Vec<f32> = h1_vals.iter().map(|&v| v.max(0.0)).collect();

                // Upload ReLU output as new tracked tensor
                let mut h1r_gpu =
                    GpuTensor::from_host(&h1_relu, &[batch_size, hidden], &dev).unwrap();
                h1r_gpu.set_requires_grad(true);
                let h1r_id = autograd::alloc_tensor_id().unwrap();
                h1r_gpu.set_tensor_id(h1r_id);
                pool.insert(h1r_id, h1r_gpu.clone_tensor().unwrap());

                // W2: [hidden, out_f] for matmul h1r[batch, hidden] × W2[hidden, out_f]
                let mut w2_gpu =
                    GpuTensor::from_host(&w2_host, &[hidden, out_f], &dev).unwrap();
                let w2_id = autograd::alloc_tensor_id().unwrap();
                w2_gpu.set_tensor_id(w2_id);
                w2_gpu.set_requires_grad(true);
                pool.insert(w2_id, w2_gpu.clone_tensor().unwrap());

                // GPU matmul: h1r × W2 → [batch, out_f]
                let mut logits =
                    gpu_host::nn::ops::matmul(&h1r_gpu, &w2_gpu, &registry).unwrap();
                let logits_id = logits.tensor_id().unwrap();
                pool.insert(logits_id, logits.clone_tensor().unwrap());

                // Bias add
                let b2_gpu = GpuTensor::from_host(&b2_host, &[out_f], &dev).unwrap();
                gpu_host::nn::ops::bias_add(&mut logits, &b2_gpu, &registry).unwrap();
                let final_id = logits.tensor_id().unwrap();
                pool.insert(final_id, logits.clone_tensor().unwrap());

                // Download logits for loss computation
                let logits_vals = logits.to_host().unwrap();
                (logits_vals, final_id)
            });
            let (logits_vals, loss_tensor_id) = logits_host;

            // ========================================
            // Loss + accuracy (CPU softmax + CE)
            // ========================================
            let mut batch_loss = 0.0f64;
            let mut d_logits = vec![0.0f32; batch_size * out_f];

            for b in 0..batch_size {
                let row = &logits_vals[b * out_f..(b + 1) * out_f];
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

                for o in 0..out_f {
                    let softmax_o = (row[o] - max_val).exp() / exp_sum;
                    let target = if o == batch_y[b] as usize { 1.0 } else { 0.0 };
                    d_logits[b * out_f + o] = (softmax_o - target) / batch_size as f32;
                }

                batch_loss -=
                    ((row[batch_y[b] as usize] - max_val).exp() / exp_sum).ln() as f64;

                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == batch_y[b] as usize {
                    correct += 1;
                }
            }
            total_loss += batch_loss / batch_size as f64;

            // ========================================
            // Backward pass — autograd on GPU
            // ========================================
            // Upload d_logits as the gradient seed (replaces the default ones)
            let d_logits_gpu =
                GpuTensor::from_host(&d_logits, &[batch_size, out_f], &dev)?;
            let mut grads = std::collections::HashMap::new();
            grads.insert(loss_tensor_id, d_logits_gpu);

            // Backward through the tape (GPU matmul for weight gradients)
            for entry in tape.entries().iter().rev() {
                let d_out = match grads.get(&entry.output) {
                    Some(g) => g,
                    None => continue,
                };

                match entry.op {
                    autograd::OpKind::Matmul => {
                        let d_out_clone = d_out.clone_tensor()?;
                        let a_id = entry.saved[0];
                        let b_id = entry.saved[1];

                        // dA = dC × B^T
                        if a_id.0 != u32::MAX {
                            if let Some(b_tensor) = pool.get(b_id) {
                                let bt = b_tensor.transpose(0, 1)?;
                                let da =
                                    gpu_host::nn::ops::matmul(&d_out_clone, &bt, &registry)?;
                                grads
                                    .entry(a_id)
                                    .and_modify(|existing| {
                                        gpu_host::nn::ops::elementwise_add(
                                            existing, &da, &registry,
                                        )
                                        .ok();
                                    })
                                    .or_insert(da);
                            }
                        }
                        // dB = A^T × dC
                        if b_id.0 != u32::MAX {
                            if let Some(a_tensor) = pool.get(a_id) {
                                let at = a_tensor.transpose(0, 1)?;
                                let db =
                                    gpu_host::nn::ops::matmul(&at, &d_out_clone, &registry)?;
                                grads
                                    .entry(b_id)
                                    .and_modify(|existing| {
                                        gpu_host::nn::ops::elementwise_add(
                                            existing, &db, &registry,
                                        )
                                        .ok();
                                    })
                                    .or_insert(db);
                            }
                        }
                    }
                    autograd::OpKind::BiasAdd | autograd::OpKind::ElemAdd => {
                        let d_clone = d_out.clone_tensor()?;
                        grads.entry(entry.inputs[0]).or_insert(d_clone);
                    }
                    _ => {}
                }
            }

            // ========================================
            // SGD update — download grads, update on host
            // ========================================
            // Bridge the ReLU gap: propagate gradient from h1r (id=4) → h1_biased (id=3)
            // via ReLU mask: dh1_biased = dh1r * (h1_pre > 0)
            if let Some(dh1r) = grads.get(&autograd::TensorId(4)) {
                let dh1r_host = dh1r.to_host()?;
                // h1_biased was stored in pool at TensorId(3) — get it for the ReLU mask
                // Actually h1_biased is id=3 but we didn't store the biased version.
                // We stored matmul output (id=2) and bias add changes it to id=3.
                // The ReLU mask comes from the h1 values after bias add.
                // We already have h1_vals from the forward pass (before ReLU).
                // But those are local vars... We need to re-derive from pool.
                // Simplest: use the h1r values (post-ReLU) as the mask: if h1r > 0 → 1, else 0
                let h1r_data = pool.get(autograd::TensorId(4)).unwrap().to_host()?;
                let masked: Vec<f32> = dh1r_host
                    .iter()
                    .zip(h1r_data.iter())
                    .map(|(&dh, &h)| if h > 0.0 { dh } else { 0.0 })
                    .collect();
                let dh1_gpu =
                    GpuTensor::from_host(&masked, &[batch_size, hidden], &dev)?;
                grads.insert(autograd::TensorId(3), dh1_gpu);
            }

            // Continue backward through tape[1] BiasAdd and tape[0] Matmul
            for entry in tape.entries()[..2].iter().rev() {
                let d_out = match grads.get(&entry.output) {
                    Some(g) => g,
                    None => continue,
                };
                match entry.op {
                    autograd::OpKind::Matmul => {
                        let d_out_clone = d_out.clone_tensor()?;
                        let a_id = entry.saved[0];
                        let b_id = entry.saved[1];
                        if a_id.0 != u32::MAX {
                            if let Some(b_t) = pool.get(b_id) {
                                let bt = b_t.transpose(0, 1)?;
                                let da = gpu_host::nn::ops::matmul(&d_out_clone, &bt, &registry)?;
                                grads.entry(a_id).and_modify(|e| {
                                    gpu_host::nn::ops::elementwise_add(e, &da, &registry).ok();
                                }).or_insert(da);
                            }
                        }
                        if b_id.0 != u32::MAX {
                            if let Some(a_t) = pool.get(a_id) {
                                let at = a_t.transpose(0, 1)?;
                                let db = gpu_host::nn::ops::matmul(&at, &d_out_clone, &registry)?;
                                grads.entry(b_id).and_modify(|e| {
                                    gpu_host::nn::ops::elementwise_add(e, &db, &registry).ok();
                                }).or_insert(db);
                            }
                        }
                    }
                    autograd::OpKind::BiasAdd => {
                        let d_clone = d_out.clone_tensor()?;
                        grads.entry(entry.inputs[0]).or_insert(d_clone);
                    }
                    _ => {}
                }
            }

            // W1=TensorId(1), W2=TensorId(5), h1r=TensorId(4)
            if let Some(dw1) = grads.get(&autograd::TensorId(1)) {
                let dw1_host = dw1.to_host()?;
                for i in 0..w1_host.len() {
                    w1_host[i] -= lr * dw1_host[i];
                }
            }
            // W2 gradient
            if let Some(dw2) = grads.get(&autograd::TensorId(5)) {
                let dw2_host = dw2.to_host()?;
                for i in 0..w2_host.len() {
                    w2_host[i] -= lr * dw2_host[i];
                }
            }
            // Bias gradients via column sum of d_logits
            for o in 0..out_f {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    sum += d_logits[b * out_f + o];
                }
                b2_host[o] -= lr * sum;
            }
            // b1 gradient from d_hidden (simplified: sum of d_h1r * relu_mask)
            // For simplicity, b1 gradient comes from the chain through matmul backward
            // The h1r_id gradient flows through the second matmul backward → dA
            if let Some(dh1r) = grads.get(&autograd::TensorId(4)) {
                let dh1r_host = dh1r.to_host()?;
                for h in 0..hidden {
                    let mut sum = 0.0f32;
                    for b in 0..batch_size {
                        sum += dh1r_host[b * hidden + h];
                    }
                    b1_host[h] -= lr * sum;
                }
            }
        }

        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;

        // Test accuracy (CPU eval for speed)
        let test_correct =
            evaluate_gpu(&test_images, &test_labels, &w1_host, &b1_host, &w2_host, &b2_host, &dev, &registry);
        let test_acc = test_correct as f64 / test_images.len() as f64 * 100.0;

        println!(
            "Epoch {}/{}: loss={avg_loss:.4}, train_acc={train_acc:.1}%, test_acc={test_acc:.1}%, time={:.1}s",
            epoch + 1,
            epochs,
            epoch_time.as_secs_f64()
        );
    }

    let total_time = total_start.elapsed();
    println!("\nTotal training time: {:.1}s", total_time.as_secs_f64());
    println!("Done.");
    Ok(())
}

/// Evaluate on test set using GPU matmul.
fn evaluate_gpu(
    images: &[Vec<f32>],
    labels: &[u8],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    dev: &Arc<cudarc::driver::CudaDevice>,
    registry: &Arc<gpu_host::nn::KernelRegistry>,
) -> usize {
    let in_f = 784;
    let hidden = 128;
    let out_f = 10;
    let batch = images.len();

    // Batch all test images
    let mut x_flat = vec![0.0f32; batch * in_f];
    for (i, img) in images.iter().enumerate() {
        x_flat[i * in_f..(i + 1) * in_f].copy_from_slice(img);
    }

    // GPU matmul: x × W1
    let x_gpu = GpuTensor::from_host(&x_flat, &[batch, in_f], dev).unwrap();
    let w1_gpu = GpuTensor::from_host(w1, &[in_f, hidden], dev).unwrap();
    let mut h1 = gpu_host::nn::ops::matmul(&x_gpu, &w1_gpu, registry).unwrap();

    // Bias add
    let b1_gpu = GpuTensor::from_host(b1, &[hidden], dev).unwrap();
    gpu_host::nn::ops::bias_add(&mut h1, &b1_gpu, registry).unwrap();

    // ReLU on host
    let h1_vals = h1.to_host().unwrap();
    let h1_relu: Vec<f32> = h1_vals.iter().map(|&v| v.max(0.0)).collect();

    // GPU matmul: h1r × W2
    let h1r_gpu = GpuTensor::from_host(&h1_relu, &[batch, hidden], dev).unwrap();
    let w2_gpu = GpuTensor::from_host(w2, &[hidden, out_f], dev).unwrap();
    let mut logits = gpu_host::nn::ops::matmul(&h1r_gpu, &w2_gpu, registry).unwrap();

    let b2_gpu = GpuTensor::from_host(b2, &[out_f], dev).unwrap();
    gpu_host::nn::ops::bias_add(&mut logits, &b2_gpu, registry).unwrap();

    let logits_host = logits.to_host().unwrap();

    let mut correct = 0;
    for (i, &label) in labels.iter().enumerate() {
        let row = &logits_host[i * out_f..(i + 1) * out_f];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j)
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
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2051 {
        return Err(format!("Bad image magic: {magic}").into());
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let pixels = rows * cols;
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let offset = 16 + i * pixels;
        images.push(data[offset..offset + pixels].iter().map(|&b| b as f32 / 255.0).collect());
    }
    Ok(images)
}

fn load_idx_labels(path: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2049 {
        return Err(format!("Bad label magic: {magic}").into());
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    Ok(data[8..8 + n].to_vec())
}
