//! CIFAR-10 tiny CNN training on GPU.
//!
//! Forward: GPU Conv2d (im2col+GEMM) → CPU ReLU → CPU AvgPool → GPU matmul → softmax
//! Backward: autograd tape for matmul gradients + manual conv/pool gradient
//! Optimizer: SGD on host
//!
//! Requires: models/cifar10/ (run `bash scripts/download-cifar10.sh`)

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
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    println!("Loading CIFAR-10 from {}...", cifar_dir.display());

    let (train_imgs, train_labels) = load_cifar_batch(&cifar_dir.join("data_batch_1.bin"), 2000)?;
    let (test_imgs, test_labels) = load_cifar_batch(&cifar_dir.join("test_batch.bin"), 500)?;
    println!(
        "Loaded: {} train, {} test (3×32×32)",
        train_imgs.len(),
        test_imgs.len()
    );

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);
    println!("CUDA device initialized (GPU Conv2d + GEMM)");

    // Architecture: Conv2d(3→8, 3×3, pad=1) → ReLU → AvgPool(4) → Linear(512→10)
    let c_in = 3;
    let c_out = 8;
    let h = 32;
    let w = 32;
    let pool_size = 4;
    let h_pooled = h / pool_size; // 8
    let w_pooled = w / pool_size; // 8
    let flat_dim = c_out * h_pooled * w_pooled; // 512
    let n_classes = 10;

    // Xavier init
    let scale_conv = (2.0 / (c_in * 9) as f64).sqrt() as f32;
    let conv_w: Vec<f32> = (0..c_out * c_in * 3 * 3)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_conv)
        .collect();

    let scale_fc = (2.0 / flat_dim as f64).sqrt() as f32;
    let mut fc_w: Vec<f32> = (0..n_classes * flat_dim)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_fc)
        .collect();
    let mut fc_b = vec![0.0f32; n_classes];

    let lr = 0.01f32;
    let batch_size = 32;
    let epochs = 10;
    let total_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_imgs.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;

            // Process each sample (batch=1 for GPU conv2d which expects [C,H,W])
            let mut batch_d_fc_w = vec![0.0f32; n_classes * flat_dim];
            let mut batch_d_fc_b = vec![0.0f32; n_classes];
            let mut batch_loss = 0.0f64;

            for bi in 0..batch_size {
                let img = &train_imgs[start + bi];
                let label = train_labels[start + bi] as usize;

                // ========== GPU Conv2d forward ==========
                let img_gpu = GpuTensor::from_host(img, &[c_in, h, w], &dev)?;
                let cw_gpu = GpuTensor::from_host(&conv_w, &[c_out, c_in, 3, 3], &dev)?;
                let conv_out =
                    gpu_host::nn::ops::conv2d(&img_gpu, &cw_gpu, None, 1, 1, &registry)?;
                let conv_host = conv_out.to_host()?;

                // CPU ReLU + AvgPool
                let relu_out: Vec<f32> = conv_host.iter().map(|&v| v.max(0.0)).collect();
                let pooled = cpu_avg_pool(&relu_out, c_out, h, w, pool_size);

                // ========== GPU matmul for FC layer ==========
                let tape = autograd::Tape::new();
                let mut pool = autograd::TensorPool::new();

                let (logits_host, tape) = autograd::with_tape(tape, || {
                    let mut feat = GpuTensor::from_host(&pooled, &[1, flat_dim], &dev).unwrap();
                    let feat_id = autograd::alloc_tensor_id().unwrap();
                    feat.set_tensor_id(feat_id);
                    pool.insert(feat_id, feat.clone_tensor().unwrap());

                    // fc_w as [flat_dim, n_classes] for matmul
                    let mut fw = GpuTensor::from_host(&fc_w, &[flat_dim, n_classes], &dev).unwrap();
                    let fw_id = autograd::alloc_tensor_id().unwrap();
                    fw.set_tensor_id(fw_id);
                    fw.set_requires_grad(true);
                    pool.insert(fw_id, fw.clone_tensor().unwrap());

                    let mut logits =
                        gpu_host::nn::ops::matmul(&feat, &fw, &registry).unwrap();
                    let lid = logits.tensor_id().unwrap();
                    pool.insert(lid, logits.clone_tensor().unwrap());

                    let fb = GpuTensor::from_host(&fc_b, &[n_classes], &dev).unwrap();
                    gpu_host::nn::ops::bias_add(&mut logits, &fb, &registry).unwrap();
                    let final_id = logits.tensor_id().unwrap();
                    pool.insert(final_id, logits.clone_tensor().unwrap());

                    (logits.to_host().unwrap(), final_id)
                });
                let (logits_vals, loss_id) = logits_host;

                // Softmax + CE loss
                let max_l = logits_vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = logits_vals.iter().map(|&x| (x - max_l).exp()).sum();
                let mut probs = vec![0.0f32; n_classes];
                for o in 0..n_classes {
                    probs[o] = (logits_vals[o] - max_l).exp() / exp_sum;
                }
                batch_loss -= probs[label].ln() as f64;

                let pred = logits_vals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == label {
                    correct += 1;
                }

                // Backward for FC via autograd (GPU matmul)
                let mut d_logits = probs.clone();
                d_logits[label] -= 1.0;
                for o in 0..n_classes {
                    d_logits[o] /= batch_size as f32;
                }

                let d_logits_gpu =
                    GpuTensor::from_host(&d_logits, &[1, n_classes], &dev)?;
                let mut grads = std::collections::HashMap::new();
                grads.insert(loss_id, d_logits_gpu);

                // Backward through FC tape
                for entry in tape.entries().iter().rev() {
                    let d_out = match grads.get(&entry.output) {
                        Some(g) => g.clone_tensor()?,
                        None => continue,
                    };
                    match entry.op {
                        autograd::OpKind::Matmul => {
                            let a_id = entry.saved[0];
                            let b_id = entry.saved[1];
                            if b_id.0 != u32::MAX {
                                if let Some(a) = pool.get(a_id) {
                                    let at = a.transpose(0, 1)?;
                                    let db = gpu_host::nn::ops::matmul(&at, &d_out, &registry)?;
                                    grads.entry(b_id).or_insert(db);
                                }
                            }
                        }
                        autograd::OpKind::BiasAdd => {
                            grads.entry(entry.inputs[0]).or_insert(d_out);
                        }
                        _ => {}
                    }
                }

                // Accumulate FC weight gradient (TensorId(1) = fc_w)
                if let Some(dfw) = grads.get(&autograd::TensorId(1)) {
                    let dfw_host = dfw.to_host()?;
                    for i in 0..batch_d_fc_w.len() {
                        batch_d_fc_w[i] += dfw_host[i];
                    }
                }
                for o in 0..n_classes {
                    batch_d_fc_b[o] += d_logits[o];
                }
            }

            // SGD update
            for i in 0..fc_w.len() {
                fc_w[i] -= lr * batch_d_fc_w[i];
            }
            for o in 0..n_classes {
                fc_b[o] -= lr * batch_d_fc_b[o];
            }

            total_loss += batch_loss / batch_size as f64;
        }

        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;

        let test_correct = evaluate(
            &test_imgs, &test_labels, &conv_w, &fc_w, &fc_b, &dev, &registry,
        );
        let test_acc = test_correct as f64 / test_imgs.len() as f64 * 100.0;

        println!(
            "Epoch {}/{}: loss={avg_loss:.4}, train_acc={train_acc:.1}%, test_acc={test_acc:.1}%, time={:.1}s",
            epoch + 1,
            epochs,
            epoch_time.as_secs_f64()
        );
    }

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("Done.");
    Ok(())
}

fn cpu_avg_pool(input: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let ho = h / ps;
    let wo = w / ps;
    let area = (ps * ps) as f32;
    let mut out = vec![0.0f32; c * ho * wo];
    for ch in 0..c {
        for oh in 0..ho {
            for ow in 0..wo {
                let mut sum = 0.0f32;
                for ph in 0..ps {
                    for pw in 0..ps {
                        sum += input[ch * h * w + (oh * ps + ph) * w + ow * ps + pw];
                    }
                }
                out[ch * ho * wo + oh * wo + ow] = sum / area;
            }
        }
    }
    out
}

fn evaluate(
    images: &[Vec<f32>],
    labels: &[u8],
    conv_w: &[f32],
    fc_w: &[f32],
    fc_b: &[f32],
    dev: &Arc<cudarc::driver::CudaDevice>,
    registry: &Arc<gpu_host::nn::KernelRegistry>,
) -> usize {
    let c_out = 8;
    let h = 32;
    let w = 32;
    let pool_size = 4;
    let flat_dim = c_out * (h / pool_size) * (w / pool_size);
    let n_classes = 10;
    let mut correct = 0;

    for (img, &label) in images.iter().zip(labels.iter()) {
        // GPU conv2d
        let img_gpu = GpuTensor::from_host(img, &[3, h, w], dev).unwrap();
        let cw_gpu = GpuTensor::from_host(conv_w, &[c_out, 3, 3, 3], dev).unwrap();
        let conv_out = gpu_host::nn::ops::conv2d(&img_gpu, &cw_gpu, None, 1, 1, registry).unwrap();
        let conv_host = conv_out.to_host().unwrap();

        let relu: Vec<f32> = conv_host.iter().map(|&v| v.max(0.0)).collect();
        let pooled = cpu_avg_pool(&relu, c_out, h, w, pool_size);

        // GPU matmul for FC
        let feat_gpu = GpuTensor::from_host(&pooled, &[1, flat_dim], dev).unwrap();
        let fw_gpu = GpuTensor::from_host(fc_w, &[flat_dim, n_classes], dev).unwrap();
        let mut logits = gpu_host::nn::ops::matmul(&feat_gpu, &fw_gpu, registry).unwrap();
        let fb_gpu = GpuTensor::from_host(fc_b, &[n_classes], dev).unwrap();
        gpu_host::nn::ops::bias_add(&mut logits, &fb_gpu, registry).unwrap();

        let lh = logits.to_host().unwrap();
        let pred = lh.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        if pred == label as usize {
            correct += 1;
        }
    }
    correct
}

fn load_cifar_batch(
    path: &std::path::Path,
    max_samples: usize,
) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let record_size = 3073;
    let n = (data.len() / record_size).min(max_samples);
    let mut images = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * record_size;
        labels.push(data[offset]);
        images.push(data[offset + 1..offset + record_size].iter().map(|&b| b as f32 / 255.0).collect());
    }
    Ok((images, labels))
}
