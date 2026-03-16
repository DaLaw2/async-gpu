//! CIFAR-10 tiny CNN training on GPU.
//!
//! Forward: GPU Conv2d + CPU ReLU + CPU AvgPool + GPU matmul (FC) → softmax
//! Backward: FC via GPU matmul backward + conv weight gradient via CPU transposed conv
//!
//! Usage: cargo run --release [--cpu]
//! Requires: models/cifar10/ (run `bash scripts/download-cifar10.sh`)

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

fn run_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    let (train_imgs, train_labels) = load_cifar_batch(&cifar_dir.join("data_batch_1.bin"), 2000)?;
    let (test_imgs, test_labels) = load_cifar_batch(&cifar_dir.join("test_batch.bin"), 500)?;
    println!("CIFAR-10 GPU training ({} train, {} test)", train_imgs.len(), test_imgs.len());

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev), gpu_host::ptx::KERNEL,
    )?);

    let (c_in, c_out, h, w, kh, kw) = (3, 8, 32, 32, 3, 3);
    let pool_size = 4;
    let flat_dim = c_out * (h / pool_size) * (w / pool_size); // 512
    let n_classes = 10;

    let scale_c = (2.0 / (c_in * kh * kw) as f64).sqrt() as f32;
    let mut conv_w: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_c)
        .collect();
    let scale_f = (2.0 / flat_dim as f64).sqrt() as f32;
    let mut fc_w: Vec<f32> = (0..n_classes * flat_dim)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_f)
        .collect();
    let mut fc_b = vec![0.0f32; n_classes];

    let lr = 0.01f32;
    let batch_size = 32;
    let epochs = 10;
    let total_start = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_imgs.len() / batch_size;
        let mut batch_d_conv_w = vec![0.0f32; conv_w.len()];

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let mut batch_d_fc_w = vec![0.0f32; fc_w.len()];
            let mut batch_d_fc_b = vec![0.0f32; n_classes];
            batch_d_conv_w.iter_mut().for_each(|v| *v = 0.0);

            for bi in 0..batch_size {
                let img = &train_imgs[start + bi];
                let label = train_labels[start + bi] as usize;

                // === Forward: GPU Conv2d ===
                let img_gpu = GpuTensor::from_host(img, &[c_in, h, w], &dev)?;
                let cw_gpu = GpuTensor::from_host(&conv_w, &[c_out, c_in, kh, kw], &dev)?;
                let conv_out = gpu_host::nn::ops::conv2d(&img_gpu, &cw_gpu, None, 1, 1, &registry)?;
                let conv_host = conv_out.to_host()?;

                // CPU ReLU
                let relu_out: Vec<f32> = conv_host.iter().map(|&v| v.max(0.0)).collect();
                // CPU AvgPool
                let pooled = cpu_avg_pool(&relu_out, c_out, h, w, pool_size);

                // === Forward: GPU matmul (FC) ===
                let tape = autograd::Tape::new();
                let mut pool = autograd::TensorPool::new();
                let (logits_vals, tape) = autograd::with_tape(tape, || {
                    let mut feat = GpuTensor::from_host(&pooled, &[1, flat_dim], &dev).unwrap();
                    let fid = autograd::alloc_tensor_id().unwrap();
                    feat.set_tensor_id(fid);
                    pool.insert(fid, feat.clone_tensor().unwrap());

                    let mut fw = GpuTensor::from_host(&fc_w, &[flat_dim, n_classes], &dev).unwrap();
                    let fwid = autograd::alloc_tensor_id().unwrap();
                    fw.set_tensor_id(fwid);
                    fw.set_requires_grad(true);
                    pool.insert(fwid, fw.clone_tensor().unwrap());

                    let mut logits = gpu_host::nn::ops::matmul(&feat, &fw, &registry).unwrap();
                    let lid = logits.tensor_id().unwrap();
                    pool.insert(lid, logits.clone_tensor().unwrap());

                    let fb = GpuTensor::from_host(&fc_b, &[n_classes], &dev).unwrap();
                    gpu_host::nn::ops::bias_add(&mut logits, &fb, &registry).unwrap();
                    let fid2 = logits.tensor_id().unwrap();
                    pool.insert(fid2, logits.clone_tensor().unwrap());

                    (logits.to_host().unwrap(), fid2)
                });
                let (logits_vals, loss_id) = logits_vals;

                // Softmax + CE
                let mx = logits_vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let es: f32 = logits_vals.iter().map(|&x| (x - mx).exp()).sum();
                let mut probs = vec![0.0f32; n_classes];
                for o in 0..n_classes { probs[o] = (logits_vals[o] - mx).exp() / es; }
                total_loss -= probs[label].ln() as f64;
                let pred = logits_vals.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
                if pred == label { correct += 1; }

                let mut d_logits = probs.clone();
                d_logits[label] -= 1.0;
                for o in 0..n_classes { d_logits[o] /= batch_size as f32; }

                // === Backward: FC via GPU matmul ===
                let d_logits_gpu = GpuTensor::from_host(&d_logits, &[1, n_classes], &dev)?;
                let mut grads = std::collections::HashMap::new();
                grads.insert(loss_id, d_logits_gpu);

                for entry in tape.entries().iter().rev() {
                    let d_out = match grads.get(&entry.output) {
                        Some(g) => g.clone_tensor()?,
                        None => continue,
                    };
                    match entry.op {
                        autograd::OpKind::Matmul => {
                            let a_id = entry.saved[0];
                            let b_id = entry.saved[1];
                            // dA (d_feat) for conv backward chain
                            if a_id.0 != u32::MAX {
                                if let Some(b_t) = pool.get(b_id) {
                                    let bt = b_t.transpose(0, 1)?;
                                    let da = gpu_host::nn::ops::matmul(&d_out, &bt, &registry)?;
                                    grads.entry(a_id).or_insert(da);
                                }
                            }
                            // dB (dW_fc)
                            if b_id.0 != u32::MAX {
                                if let Some(a_t) = pool.get(a_id) {
                                    let at = a_t.transpose(0, 1)?;
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

                // Accumulate FC gradients
                if let Some(dfw) = grads.get(&autograd::TensorId(1)) {
                    let dfw_h = dfw.to_host()?;
                    for i in 0..batch_d_fc_w.len() { batch_d_fc_w[i] += dfw_h[i]; }
                }
                for o in 0..n_classes { batch_d_fc_b[o] += d_logits[o]; }

                // === Backward: Conv weight gradient ===
                // d_feat [1, flat_dim] → unpool → relu_mask → conv2d_backward → dW_conv
                if let Some(d_feat) = grads.get(&autograd::TensorId(0)) {
                    let d_feat_host = d_feat.to_host()?;
                    // Unpool: [c_out, h/ps, w/ps] → [c_out, h, w]
                    let d_pooled = cpu_avg_unpool(&d_feat_host, c_out, h, w, pool_size);
                    // ReLU mask
                    let d_relu: Vec<f32> = d_pooled.iter().zip(conv_host.iter())
                        .map(|(&dv, &cv)| if cv > 0.0 { dv } else { 0.0 })
                        .collect();
                    // Conv weight gradient: dW[co,ci,fh,fw] = sum over spatial of d_relu[co,oh,ow] * input[ci,ih,iw]
                    let dw = cpu_conv2d_weight_grad(img, &d_relu, c_in, h, w, c_out, kh, kw, 1, 1);
                    for i in 0..batch_d_conv_w.len() { batch_d_conv_w[i] += dw[i]; }
                }
            }

            // SGD update
            for i in 0..fc_w.len() { fc_w[i] -= lr * batch_d_fc_w[i]; }
            for o in 0..n_classes { fc_b[o] -= lr * batch_d_fc_b[o]; }
            for i in 0..conv_w.len() { conv_w[i] -= lr * batch_d_conv_w[i]; }

            total_loss /= batch_size as f64;
        }

        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;
        let test_c = evaluate_gpu(&test_imgs, &test_labels, &conv_w, &fc_w, &fc_b, &dev, &registry);
        let test_acc = test_c as f64 / test_imgs.len() as f64 * 100.0;
        println!("Epoch {}/{}: loss={avg_loss:.4}, train={train_acc:.1}%, test={test_acc:.1}%, time={:.1}s",
            epoch + 1, epochs, es.elapsed().as_secs_f64());
    }
    println!("\nTotal: {:.1}s (GPU)", total_start.elapsed().as_secs_f64());
    Ok(())
}

// ============================================================
// CPU-only mode for benchmark
// ============================================================

fn run_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    let (train_imgs, train_labels) = load_cifar_batch(&cifar_dir.join("data_batch_1.bin"), 2000)?;
    let (test_imgs, test_labels) = load_cifar_batch(&cifar_dir.join("test_batch.bin"), 500)?;
    println!("CIFAR-10 CPU training ({} train, {} test)", train_imgs.len(), test_imgs.len());

    let (c_in, c_out, h, w, kh, kw) = (3, 8, 32, 32, 3, 3);
    let pool_size = 4;
    let flat_dim = c_out * (h / pool_size) * (w / pool_size);
    let n_classes = 10;

    let scale_c = (2.0 / (c_in * kh * kw) as f64).sqrt() as f32;
    let mut conv_w: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_c).collect();
    let scale_f = (2.0 / flat_dim as f64).sqrt() as f32;
    let mut fc_w: Vec<f32> = (0..n_classes * flat_dim)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale_f).collect();
    let mut fc_b = vec![0.0f32; n_classes];

    let lr = 0.01f32;
    let batch_size = 32;
    let epochs = 10;
    let total_start = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let n_batches = train_imgs.len() / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let mut bd_fc_w = vec![0.0f32; fc_w.len()];
            let mut bd_fc_b = vec![0.0f32; n_classes];
            let mut bd_conv_w = vec![0.0f32; conv_w.len()];

            for bi in 0..batch_size {
                let img = &train_imgs[start + bi];
                let label = train_labels[start + bi] as usize;

                let conv_out = cpu_conv2d(img, &conv_w, c_in, h, w, c_out, kh, kw, 1, 1);
                let relu_out: Vec<f32> = conv_out.iter().map(|&v| v.max(0.0)).collect();
                let pooled = cpu_avg_pool(&relu_out, c_out, h, w, pool_size);

                // FC
                let mut logits = vec![0.0f32; n_classes];
                for o in 0..n_classes {
                    logits[o] = fc_b[o];
                    for j in 0..flat_dim { logits[o] += pooled[j] * fc_w[o * flat_dim + j]; }
                }

                let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let esum: f32 = logits.iter().map(|&x| (x - mx).exp()).sum();
                let mut probs = vec![0.0f32; n_classes];
                for o in 0..n_classes { probs[o] = (logits[o] - mx).exp() / esum; }
                total_loss -= probs[label].ln() as f64;
                let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
                if pred == label { correct += 1; }

                let mut dl = probs.clone();
                dl[label] -= 1.0;
                for o in 0..n_classes { dl[o] /= batch_size as f32; }

                // FC backward
                let mut d_pooled = vec![0.0f32; flat_dim];
                for o in 0..n_classes {
                    bd_fc_b[o] += dl[o];
                    for j in 0..flat_dim {
                        bd_fc_w[o * flat_dim + j] += dl[o] * pooled[j];
                        d_pooled[j] += dl[o] * fc_w[o * flat_dim + j];
                    }
                }

                // Unpool + ReLU mask + conv weight grad
                let d_up = cpu_avg_unpool(&d_pooled, c_out, h, w, pool_size);
                let d_relu: Vec<f32> = d_up.iter().zip(conv_out.iter())
                    .map(|(&dv, &cv)| if cv > 0.0 { dv } else { 0.0 }).collect();
                let dw = cpu_conv2d_weight_grad(img, &d_relu, c_in, h, w, c_out, kh, kw, 1, 1);
                for i in 0..bd_conv_w.len() { bd_conv_w[i] += dw[i]; }
            }

            for i in 0..fc_w.len() { fc_w[i] -= lr * bd_fc_w[i]; }
            for o in 0..n_classes { fc_b[o] -= lr * bd_fc_b[o]; }
            for i in 0..conv_w.len() { conv_w[i] -= lr * bd_conv_w[i]; }
            total_loss /= batch_size as f64;
        }

        let avg_loss = total_loss / n_batches as f64;
        let train_acc = correct as f64 / (n_batches * batch_size) as f64 * 100.0;
        let test_c = evaluate_cpu(&test_imgs, &test_labels, &conv_w, &fc_w, &fc_b);
        let test_acc = test_c as f64 / test_imgs.len() as f64 * 100.0;
        println!("Epoch {}/{}: loss={avg_loss:.4}, train={train_acc:.1}%, test={test_acc:.1}%, time={:.1}s",
            epoch + 1, epochs, es.elapsed().as_secs_f64());
    }
    println!("\nTotal: {:.1}s (CPU)", total_start.elapsed().as_secs_f64());
    Ok(())
}

// ============================================================
// CPU helpers
// ============================================================

fn cpu_conv2d(input: &[f32], weight: &[f32], c_in: usize, h: usize, w: usize,
    c_out: usize, kh: usize, kw: usize, stride: usize, pad: usize) -> Vec<f32> {
    let ho = (h + 2 * pad - kh) / stride + 1;
    let wo = (w + 2 * pad - kw) / stride + 1;
    let mut out = vec![0.0f32; c_out * ho * wo];
    for co in 0..c_out {
        for oh in 0..ho {
            for ow in 0..wo {
                let mut s = 0.0f32;
                for ci in 0..c_in {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = (oh * stride + fh) as isize - pad as isize;
                            let iw = (ow * stride + fw) as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                s += input[ci * h * w + ih as usize * w + iw as usize]
                                    * weight[co * (c_in * kh * kw) + ci * (kh * kw) + fh * kw + fw];
                            }
                        }
                    }
                }
                out[co * ho * wo + oh * wo + ow] = s;
            }
        }
    }
    out
}

fn cpu_conv2d_weight_grad(input: &[f32], d_output: &[f32], c_in: usize, h: usize, w: usize,
    c_out: usize, kh: usize, kw: usize, stride: usize, pad: usize) -> Vec<f32> {
    let ho = (h + 2 * pad - kh) / stride + 1;
    let wo = (w + 2 * pad - kw) / stride + 1;
    let mut dw = vec![0.0f32; c_out * c_in * kh * kw];
    for co in 0..c_out {
        for ci in 0..c_in {
            for fh in 0..kh {
                for fw in 0..kw {
                    let mut s = 0.0f32;
                    for oh in 0..ho {
                        for ow in 0..wo {
                            let ih = (oh * stride + fh) as isize - pad as isize;
                            let iw = (ow * stride + fw) as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                s += d_output[co * ho * wo + oh * wo + ow]
                                    * input[ci * h * w + ih as usize * w + iw as usize];
                            }
                        }
                    }
                    dw[co * (c_in * kh * kw) + ci * (kh * kw) + fh * kw + fw] = s;
                }
            }
        }
    }
    dw
}

fn cpu_avg_pool(input: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho, wo) = (h / ps, w / ps);
    let area = (ps * ps) as f32;
    let mut out = vec![0.0f32; c * ho * wo];
    for ch in 0..c {
        for oh in 0..ho {
            for ow in 0..wo {
                let mut s = 0.0f32;
                for ph in 0..ps { for pw in 0..ps {
                    s += input[ch * h * w + (oh * ps + ph) * w + ow * ps + pw];
                }}
                out[ch * ho * wo + oh * wo + ow] = s / area;
            }
        }
    }
    out
}

fn cpu_avg_unpool(d_pooled: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho, wo) = (h / ps, w / ps);
    let area = (ps * ps) as f32;
    let mut out = vec![0.0f32; c * h * w];
    for ch in 0..c {
        for oh in 0..ho {
            for ow in 0..wo {
                let val = d_pooled[ch * ho * wo + oh * wo + ow] / area;
                for ph in 0..ps { for pw in 0..ps {
                    out[ch * h * w + (oh * ps + ph) * w + ow * ps + pw] = val;
                }}
            }
        }
    }
    out
}

fn evaluate_gpu(images: &[Vec<f32>], labels: &[u8], conv_w: &[f32], fc_w: &[f32], fc_b: &[f32],
    dev: &Arc<cudarc::driver::CudaDevice>, registry: &Arc<gpu_host::nn::KernelRegistry>) -> usize {
    let (c_out, h, w, ps, flat_dim, nc) = (8, 32, 32, 4, 512, 10);
    let mut correct = 0;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let ig = GpuTensor::from_host(img, &[3, h, w], dev).unwrap();
        let cg = GpuTensor::from_host(conv_w, &[c_out, 3, 3, 3], dev).unwrap();
        let co = gpu_host::nn::ops::conv2d(&ig, &cg, None, 1, 1, registry).unwrap();
        let ch = co.to_host().unwrap();
        let relu: Vec<f32> = ch.iter().map(|&v| v.max(0.0)).collect();
        let pooled = cpu_avg_pool(&relu, c_out, h, w, ps);
        let fg = GpuTensor::from_host(&pooled, &[1, flat_dim], dev).unwrap();
        let wg = GpuTensor::from_host(fc_w, &[flat_dim, nc], dev).unwrap();
        let mut lg = gpu_host::nn::ops::matmul(&fg, &wg, registry).unwrap();
        let bg = GpuTensor::from_host(fc_b, &[nc], dev).unwrap();
        gpu_host::nn::ops::bias_add(&mut lg, &bg, registry).unwrap();
        let lh = lg.to_host().unwrap();
        let pred = lh.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        if pred == label as usize { correct += 1; }
    }
    correct
}

fn evaluate_cpu(images: &[Vec<f32>], labels: &[u8], conv_w: &[f32], fc_w: &[f32], fc_b: &[f32]) -> usize {
    let (c_out, h, w, ps, flat_dim, nc) = (8, 32, 32, 4, 512, 10);
    let mut correct = 0;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let co = cpu_conv2d(img, conv_w, 3, h, w, c_out, 3, 3, 1, 1);
        let relu: Vec<f32> = co.iter().map(|&v| v.max(0.0)).collect();
        let pooled = cpu_avg_pool(&relu, c_out, h, w, ps);
        let mut logits = vec![0.0f32; nc];
        for o in 0..nc {
            logits[o] = fc_b[o];
            for j in 0..flat_dim { logits[o] += pooled[j] * fc_w[o * flat_dim + j]; }
        }
        let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        if pred == label as usize { correct += 1; }
    }
    correct
}

fn load_cifar_batch(path: &std::path::Path, max: usize) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let rs = 3073;
    let n = (data.len() / rs).min(max);
    let mut imgs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * rs;
        labels.push(data[off]);
        imgs.push(data[off + 1..off + rs].iter().map(|&b| b as f32 / 255.0).collect());
    }
    Ok((imgs, labels))
}
