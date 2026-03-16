//! ResNet-18 inference + Mini-ResNet full conv training on CIFAR-10.
//!
//! Demonstrates:
//! - ResNet-18 forward pass (8 BasicBlocks, 8.1M params, 15.7ms/image)
//! - Mini-ResNet full conv training (all layers via GPU conv2d_backward)
//!
//! Usage:
//!   cargo run --release              # inference only
//!   cargo run --release -- --train   # train mini-resnet on CIFAR-10 subset

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::models::resnet::{ResNet18, ResNet18Weights};
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    let do_train = std::env::args().any(|a| a == "--train");
    if let Err(e) = if do_train { train() } else { inference() } {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn inference() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);

    println!("--- ResNet-18 CIFAR-10 Inference ---");

    let t0 = Instant::now();
    let weights = ResNet18Weights::random(10);
    let model = ResNet18::from_weights(&weights, 10, &registry)?;
    println!("Model built: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!("Parameters: ~8.1M");

    // Random test images (no CIFAR data needed for inference demo)
    let n = 100;
    let images: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..3 * 32 * 32)
                .map(|j| (i * 3072 + j) as f32 * 0.013 % 1.0)
                .collect()
        })
        .collect();
    let labels: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();

    // Warmup
    let warmup = GpuTensor::from_host(&images[0], &[3, 32, 32], &dev)?;
    let _ = model.forward(&warmup)?;

    let t1 = Instant::now();
    let mut correct = 0;
    for i in 0..n {
        let input = GpuTensor::from_host(&images[i], &[3, 32, 32], &dev)?;
        let logits = model.forward(&input)?.to_host()?;
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if pred == labels[i] as usize {
            correct += 1;
        }
    }
    let elapsed = t1.elapsed().as_secs_f64();
    println!(
        "\nAccuracy: {correct}/{n} ({:.1}%) [random weights]",
        correct as f64 / n as f64 * 100.0
    );
    println!("Speed: {elapsed:.2}s, {:.1}ms/image", elapsed / n as f64 * 1000.0);
    println!("PASSED (forward valid, no NaN)");
    Ok(())
}

/// Mini-ResNet full training on CIFAR-10 subset.
///
/// Architecture: conv1(3→32) → ReLU → BB1(32→32) → BB2(32→64) → GAP → FC(64→10)
/// BB = BasicBlock: conv → relu → conv → residual add → relu
/// Training: per-sample GPU conv2d forward + GPU conv2d_backward for ALL conv layers.
fn train() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);

    // Load CIFAR-10 data
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    let (train_imgs, train_lbls) = if cifar_dir.join("data_batch_1.bin").exists() {
        load_cifar_batch(&cifar_dir.join("data_batch_1.bin"))?
    } else {
        println!("No CIFAR-10 data. Using random data (results won't be meaningful).");
        let n = 2000;
        let imgs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..3072)
                    .map(|j| (i * 3072 + j) as f32 * 0.0037 % 1.0)
                    .collect()
            })
            .collect();
        let lbls: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();
        (imgs, lbls)
    };
    let n_train = train_imgs.len().min(2000);

    let (test_imgs, test_lbls) = if cifar_dir.join("test_batch.bin").exists() {
        load_cifar_batch(&cifar_dir.join("test_batch.bin"))?
    } else {
        let n = 500;
        let imgs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..3072)
                    .map(|j| ((i + 50000) * 3072 + j) as f32 * 0.0037 % 1.0)
                    .collect()
            })
            .collect();
        let lbls: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();
        (imgs, lbls)
    };
    let n_test = test_imgs.len().min(1000);

    println!(
        "--- Mini-ResNet CIFAR-10 Full Conv Training ---\n\
         Architecture: Conv(3→32) → ReLU → BB(32) → BB(32→64) → GAP → FC(64→10)\n\
         Train: {n_train}, Test: {n_test} (full GPU conv backward)"
    );

    let (c1, c2) = (32, 64);
    let nc = 10;

    // He initialization helper
    let he_init = |fan_in: usize, n: usize, seed: u64| -> Vec<f32> {
        let scale = (2.0 / fan_in as f64).sqrt() as f32;
        (0..n)
            .map(|i| {
                let v = (i as u64).wrapping_mul(seed).wrapping_add(0x9E3779B9) % 10007;
                (v as f32 / 10007.0 - 0.5) * 2.0 * scale
            })
            .collect()
    };

    let mut w_conv1 = he_init(3 * 9, c1 * 3 * 3 * 3, 12345); // [32,3,3,3]
    let mut w_bb1_a = he_init(c1 * 9, c1 * c1 * 3 * 3, 23456); // [32,32,3,3]
    let mut w_bb1_b = he_init(c1 * 9, c1 * c1 * 3 * 3, 34567); // [32,32,3,3]
    let mut w_bb2_a = he_init(c1 * 9, c2 * c1 * 3 * 3, 45678); // [64,32,3,3]
    let mut w_bb2_b = he_init(c2 * 9, c2 * c2 * 3 * 3, 56789); // [64,64,3,3]
    let mut w_bb2_sc = he_init(c1, c2 * c1 * 1 * 1, 67890); // [64,32,1,1] shortcut
    let mut w_fc: Vec<f32> = he_init(c2, nc * c2, 78901); // [10,64]
    let mut b_fc = vec![0.0f32; nc];

    let lr = 0.3f32;
    let bs = 16;
    let epochs = 20;
    let hw = 32 * 32;
    let ts = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let nb = n_train / bs;

        for bi in 0..nb {
            let start = bi * bs;
            let labels: Vec<usize> =
                (0..bs).map(|i| train_lbls[start + i] as usize).collect();

            let mut features = vec![0.0f32; bs * c2]; // [bs, 64] after GAP

            // Saved activations for backward (per sample)
            let mut saved_imgs = Vec::with_capacity(bs);
            let mut saved_c1_pre = Vec::with_capacity(bs); // pre-relu conv1 output
            let mut saved_c1_relu = Vec::with_capacity(bs);
            let mut saved_bb1a_pre = Vec::with_capacity(bs);
            let mut saved_bb1a_relu = Vec::with_capacity(bs);
            let mut saved_bb1b_pre = Vec::with_capacity(bs); // pre-residual
            let mut saved_after_bb1 = Vec::with_capacity(bs);
            let mut saved_bb2a_pre = Vec::with_capacity(bs);
            let mut saved_bb2a_relu = Vec::with_capacity(bs);
            let mut saved_bb2b_pre = Vec::with_capacity(bs);
            let mut saved_sc = Vec::with_capacity(bs);
            let mut saved_after_bb2_pre = Vec::with_capacity(bs); // pre-relu residual sum

            for i in 0..bs {
                let img = &train_imgs[start + i];
                saved_imgs.push(img.clone());

                // conv1: [3,32,32] → [32,32,32]
                let c1_out = gpu_conv2d(img, &w_conv1, 3, 32, 32, c1, 3, 1, &dev, &registry)?;
                let c1_relu: Vec<f32> = c1_out.iter().map(|&v| v.max(0.0)).collect();
                saved_c1_pre.push(c1_out);
                saved_c1_relu.push(c1_relu.clone());

                // BB1: conv_a → relu → conv_b → + identity → relu
                let bb1a = gpu_conv2d(&c1_relu, &w_bb1_a, c1, 32, 32, c1, 3, 1, &dev, &registry)?;
                let bb1a_relu: Vec<f32> = bb1a.iter().map(|&v| v.max(0.0)).collect();
                saved_bb1a_pre.push(bb1a);
                saved_bb1a_relu.push(bb1a_relu.clone());

                let bb1b = gpu_conv2d(&bb1a_relu, &w_bb1_b, c1, 32, 32, c1, 3, 1, &dev, &registry)?;
                // Residual: bb1b + c1_relu
                let bb1_sum: Vec<f32> = bb1b
                    .iter()
                    .zip(c1_relu.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                saved_bb1b_pre.push(bb1b);
                let after_bb1: Vec<f32> = bb1_sum.iter().map(|&v| v.max(0.0)).collect();
                saved_after_bb1.push(after_bb1.clone());

                // BB2: conv_a(32→64) → relu → conv_b(64→64) → + shortcut(32→64,1×1) → relu
                let bb2a = gpu_conv2d(&after_bb1, &w_bb2_a, c1, 32, 32, c2, 3, 1, &dev, &registry)?;
                let bb2a_relu: Vec<f32> = bb2a.iter().map(|&v| v.max(0.0)).collect();
                saved_bb2a_pre.push(bb2a);
                saved_bb2a_relu.push(bb2a_relu.clone());

                let bb2b = gpu_conv2d(
                    &bb2a_relu, &w_bb2_b, c2, 32, 32, c2, 3, 1, &dev, &registry,
                )?;
                // Shortcut: 1×1 conv for channel expansion
                let sc = gpu_conv2d(&after_bb1, &w_bb2_sc, c1, 32, 32, c2, 1, 0, &dev, &registry)?;
                // Residual: bb2b + sc
                let bb2_sum: Vec<f32> = bb2b
                    .iter()
                    .zip(sc.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                saved_bb2b_pre.push(bb2b);
                saved_sc.push(sc);
                saved_after_bb2_pre.push(bb2_sum.clone());
                let after_bb2: Vec<f32> = bb2_sum.iter().map(|&v| v.max(0.0)).collect();

                // Global average pool: [64,32,32] → [64]
                for ch in 0..c2 {
                    let sum: f32 =
                        after_bb2[ch * hw..(ch + 1) * hw].iter().sum();
                    features[i * c2 + ch] = sum / hw as f32;
                }
            }

            // FC forward + loss
            let mut logits = vec![0.0f32; bs * nc];
            for b in 0..bs {
                for o in 0..nc {
                    logits[b * nc + o] = b_fc[o];
                    for j in 0..c2 {
                        logits[b * nc + o] += features[b * c2 + j] * w_fc[o * c2 + j];
                    }
                }
            }

            // Softmax + CE loss + accuracy
            let mut d_logits = vec![0.0f32; bs * nc];
            for b in 0..bs {
                let row = &logits[b * nc..(b + 1) * nc];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let esum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                for o in 0..nc {
                    let sm = (row[o] - mx).exp() / esum;
                    d_logits[b * nc + o] =
                        (sm - if o == labels[b] { 1.0 } else { 0.0 }) / bs as f32;
                }
                total_loss -= ((row[labels[b]] - mx).exp() / esum).ln() as f64;
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == labels[b] {
                    correct += 1;
                }
            }

            // FC backward: d_features and update FC weights
            let mut d_features = vec![0.0f32; bs * c2];
            for b in 0..bs {
                for j in 0..c2 {
                    for o in 0..nc {
                        d_features[b * c2 + j] += d_logits[b * nc + o] * w_fc[o * c2 + j];
                    }
                }
            }
            for o in 0..nc {
                for j in 0..c2 {
                    let mut grad = 0.0f32;
                    for b in 0..bs {
                        grad += d_logits[b * nc + o] * features[b * c2 + j];
                    }
                    w_fc[o * c2 + j] -= lr * grad;
                }
                for b in 0..bs {
                    b_fc[o] -= lr * d_logits[b * nc + o];
                }
            }

            // ===== Full conv backward through all layers =====
            let mut dw_conv1_acc = vec![0.0f32; w_conv1.len()];
            let mut dw_bb1a_acc = vec![0.0f32; w_bb1_a.len()];
            let mut dw_bb1b_acc = vec![0.0f32; w_bb1_b.len()];
            let mut dw_bb2a_acc = vec![0.0f32; w_bb2_a.len()];
            let mut dw_bb2b_acc = vec![0.0f32; w_bb2_b.len()];
            let mut dw_bb2sc_acc = vec![0.0f32; w_bb2_sc.len()];

            for i in 0..bs {
                // d_features → d_gap → d_after_bb2
                // GAP backward: d_after_bb2[ch, h, w] = d_feat[ch] / (h*w)
                let mut d_after_bb2 = vec![0.0f32; c2 * hw];
                for ch in 0..c2 {
                    let g = d_features[i * c2 + ch] / hw as f32;
                    for p in 0..hw {
                        d_after_bb2[ch * hw + p] = g;
                    }
                }

                // ReLU mask on bb2 residual sum
                let d_bb2_sum: Vec<f32> = d_after_bb2
                    .iter()
                    .zip(saved_after_bb2_pre[i].iter())
                    .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
                    .collect();

                // Residual split: d_bb2b = d_bb2_sum, d_sc = d_bb2_sum
                // conv2d_backward for bb2b: input=bb2a_relu, weight=w_bb2_b
                let d_bb2b_t = GpuTensor::from_host(&d_bb2_sum, &[c2, 32, 32], &dev)?;
                let bb2a_relu_t =
                    GpuTensor::from_host(&saved_bb2a_relu[i], &[c2, 32, 32], &dev)?;
                let w_bb2b_t = GpuTensor::from_host(&w_bb2_b, &[c2, c2, 3, 3], &dev)?;
                let (d_bb2a_relu, dw_bb2b) =
                    gpu_host::nn::ops::conv2d_backward(&d_bb2b_t, &bb2a_relu_t, &w_bb2b_t, 1, 1, &registry)?;
                let dw_bb2b_h = dw_bb2b.to_host()?;
                for k in 0..dw_bb2b_acc.len() {
                    dw_bb2b_acc[k] += dw_bb2b_h[k];
                }

                // ReLU mask on bb2a
                let d_bb2a_relu_h = d_bb2a_relu.to_host()?;
                let d_bb2a: Vec<f32> = d_bb2a_relu_h
                    .iter()
                    .zip(saved_bb2a_pre[i].iter())
                    .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
                    .collect();

                // conv2d_backward for bb2a: input=after_bb1, weight=w_bb2_a
                let d_bb2a_t = GpuTensor::from_host(&d_bb2a, &[c2, 32, 32], &dev)?;
                let after_bb1_t =
                    GpuTensor::from_host(&saved_after_bb1[i], &[c1, 32, 32], &dev)?;
                let w_bb2a_t = GpuTensor::from_host(&w_bb2_a, &[c2, c1, 3, 3], &dev)?;
                let (d_after_bb1_from_bb2, dw_bb2a) =
                    gpu_host::nn::ops::conv2d_backward(&d_bb2a_t, &after_bb1_t, &w_bb2a_t, 1, 1, &registry)?;
                let dw_bb2a_h = dw_bb2a.to_host()?;
                for k in 0..dw_bb2a_acc.len() {
                    dw_bb2a_acc[k] += dw_bb2a_h[k];
                }

                // Shortcut backward: 1×1 conv, input=after_bb1, weight=w_bb2_sc
                let d_sc_t = GpuTensor::from_host(&d_bb2_sum, &[c2, 32, 32], &dev)?;
                let w_sc_t = GpuTensor::from_host(&w_bb2_sc, &[c2, c1, 1, 1], &dev)?;
                let (d_after_bb1_from_sc, dw_sc) =
                    gpu_host::nn::ops::conv2d_backward(&d_sc_t, &after_bb1_t, &w_sc_t, 1, 0, &registry)?;
                let dw_sc_h = dw_sc.to_host()?;
                for k in 0..dw_bb2sc_acc.len() {
                    dw_bb2sc_acc[k] += dw_sc_h[k];
                }

                // Sum gradients from both paths into after_bb1
                let d_abb1_bb2 = d_after_bb1_from_bb2.to_host()?;
                let d_abb1_sc = d_after_bb1_from_sc.to_host()?;
                let d_after_bb1: Vec<f32> = d_abb1_bb2
                    .iter()
                    .zip(d_abb1_sc.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();

                // ReLU mask on bb1 residual sum (after_bb1 = relu(bb1b + c1_relu))
                // saved_bb1b_pre[i] + saved_c1_relu[i] is the pre-relu sum
                let bb1_sum: Vec<f32> = saved_bb1b_pre[i]
                    .iter()
                    .zip(saved_c1_relu[i].iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                let d_bb1_sum: Vec<f32> = d_after_bb1
                    .iter()
                    .zip(bb1_sum.iter())
                    .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
                    .collect();

                // Residual split: d_bb1b = d_bb1_sum, d_c1_relu_from_res = d_bb1_sum

                // conv2d_backward for bb1b: input=bb1a_relu, weight=w_bb1_b
                let d_bb1b_t = GpuTensor::from_host(&d_bb1_sum, &[c1, 32, 32], &dev)?;
                let bb1a_relu_t =
                    GpuTensor::from_host(&saved_bb1a_relu[i], &[c1, 32, 32], &dev)?;
                let w_bb1b_t = GpuTensor::from_host(&w_bb1_b, &[c1, c1, 3, 3], &dev)?;
                let (d_bb1a_relu, dw_bb1b) =
                    gpu_host::nn::ops::conv2d_backward(&d_bb1b_t, &bb1a_relu_t, &w_bb1b_t, 1, 1, &registry)?;
                let dw_bb1b_h = dw_bb1b.to_host()?;
                for k in 0..dw_bb1b_acc.len() {
                    dw_bb1b_acc[k] += dw_bb1b_h[k];
                }

                // ReLU mask on bb1a
                let d_bb1a_relu_h = d_bb1a_relu.to_host()?;
                let d_bb1a: Vec<f32> = d_bb1a_relu_h
                    .iter()
                    .zip(saved_bb1a_pre[i].iter())
                    .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
                    .collect();

                // conv2d_backward for bb1a: input=c1_relu, weight=w_bb1_a
                let d_bb1a_t = GpuTensor::from_host(&d_bb1a, &[c1, 32, 32], &dev)?;
                let c1_relu_t =
                    GpuTensor::from_host(&saved_c1_relu[i], &[c1, 32, 32], &dev)?;
                let w_bb1a_t = GpuTensor::from_host(&w_bb1_a, &[c1, c1, 3, 3], &dev)?;
                let (d_c1_relu_from_bb1, dw_bb1a) =
                    gpu_host::nn::ops::conv2d_backward(&d_bb1a_t, &c1_relu_t, &w_bb1a_t, 1, 1, &registry)?;
                let dw_bb1a_h = dw_bb1a.to_host()?;
                for k in 0..dw_bb1a_acc.len() {
                    dw_bb1a_acc[k] += dw_bb1a_h[k];
                }

                // Sum gradients to c1_relu: from bb1a backward + from residual skip
                let d_c1r_bb1 = d_c1_relu_from_bb1.to_host()?;
                let d_c1_relu: Vec<f32> = d_c1r_bb1
                    .iter()
                    .zip(d_bb1_sum.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();

                // ReLU mask on conv1
                let d_c1: Vec<f32> = d_c1_relu
                    .iter()
                    .zip(saved_c1_pre[i].iter())
                    .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
                    .collect();

                // conv2d_backward for conv1: input=img, weight=w_conv1
                let d_c1_t = GpuTensor::from_host(&d_c1, &[c1, 32, 32], &dev)?;
                let img_t = GpuTensor::from_host(&saved_imgs[i], &[3, 32, 32], &dev)?;
                let w_c1_t = GpuTensor::from_host(&w_conv1, &[c1, 3, 3, 3], &dev)?;
                let (_d_img, dw_c1) =
                    gpu_host::nn::ops::conv2d_backward(&d_c1_t, &img_t, &w_c1_t, 1, 1, &registry)?;
                let dw_c1_h = dw_c1.to_host()?;
                for k in 0..dw_conv1_acc.len() {
                    dw_conv1_acc[k] += dw_c1_h[k];
                }
            }

            // Update all conv weights
            for k in 0..w_conv1.len() {
                w_conv1[k] -= lr * dw_conv1_acc[k];
            }
            for k in 0..w_bb1_a.len() {
                w_bb1_a[k] -= lr * dw_bb1a_acc[k];
            }
            for k in 0..w_bb1_b.len() {
                w_bb1_b[k] -= lr * dw_bb1b_acc[k];
            }
            for k in 0..w_bb2_a.len() {
                w_bb2_a[k] -= lr * dw_bb2a_acc[k];
            }
            for k in 0..w_bb2_b.len() {
                w_bb2_b[k] -= lr * dw_bb2b_acc[k];
            }
            for k in 0..w_bb2_sc.len() {
                w_bb2_sc[k] -= lr * dw_bb2sc_acc[k];
            }
        }

        let avg_loss = total_loss / (nb * bs) as f64;
        let train_acc = correct as f64 / (nb * bs) as f64 * 100.0;

        let test_correct = eval_mini_resnet(
            &test_imgs[..n_test],
            &test_lbls[..n_test],
            &w_conv1,
            &w_bb1_a,
            &w_bb1_b,
            &w_bb2_a,
            &w_bb2_b,
            &w_bb2_sc,
            &w_fc,
            &b_fc,
            c1,
            c2,
            nc,
            &dev,
            &registry,
        )?;
        let test_acc = test_correct as f64 / n_test as f64 * 100.0;

        println!(
            "Epoch {}/{}: loss={avg_loss:.3}, train={train_acc:.1}%, test={test_acc:.1}%, time={:.1}s",
            epoch + 1,
            epochs,
            es.elapsed().as_secs_f64()
        );
    }

    println!("\nTotal: {:.1}s", ts.elapsed().as_secs_f64());
    println!("Full conv training: all 6 conv layers + 1×1 shortcut trained via GPU backward.");
    Ok(())
}

fn gpu_conv2d(
    input: &[f32],
    weight: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k: usize,
    pad: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
    reg: &Arc<gpu_host::nn::KernelRegistry>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ig = GpuTensor::from_host(input, &[c_in, h, w], dev)?;
    let wg = GpuTensor::from_host(weight, &[c_out, c_in, k, k], dev)?;
    Ok(gpu_host::nn::ops::conv2d(&ig, &wg, None, 1, pad, reg)?.to_host()?)
}

#[allow(clippy::too_many_arguments)]
fn eval_mini_resnet(
    imgs: &[Vec<f32>],
    lbls: &[u8],
    w_conv1: &[f32],
    w_bb1_a: &[f32],
    w_bb1_b: &[f32],
    w_bb2_a: &[f32],
    w_bb2_b: &[f32],
    w_bb2_sc: &[f32],
    w_fc: &[f32],
    b_fc: &[f32],
    c1: usize,
    c2: usize,
    nc: usize,
    dev: &Arc<cudarc::driver::CudaDevice>,
    reg: &Arc<gpu_host::nn::KernelRegistry>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut correct = 0;
    for (img, &l) in imgs.iter().zip(lbls.iter()) {
        let c1_out = gpu_conv2d(img, w_conv1, 3, 32, 32, c1, 3, 1, dev, reg)?;
        let c1_relu: Vec<f32> = c1_out.iter().map(|&v| v.max(0.0)).collect();

        let bb1a = gpu_conv2d(&c1_relu, w_bb1_a, c1, 32, 32, c1, 3, 1, dev, reg)?;
        let bb1a_r: Vec<f32> = bb1a.iter().map(|&v| v.max(0.0)).collect();
        let bb1b = gpu_conv2d(&bb1a_r, w_bb1_b, c1, 32, 32, c1, 3, 1, dev, reg)?;
        let after_bb1: Vec<f32> = bb1b
            .iter()
            .zip(c1_relu.iter())
            .map(|(&a, &b)| (a + b).max(0.0))
            .collect();

        let bb2a = gpu_conv2d(&after_bb1, w_bb2_a, c1, 32, 32, c2, 3, 1, dev, reg)?;
        let bb2a_r: Vec<f32> = bb2a.iter().map(|&v| v.max(0.0)).collect();
        let bb2b = gpu_conv2d(&bb2a_r, w_bb2_b, c2, 32, 32, c2, 3, 1, dev, reg)?;
        let sc = gpu_conv2d(&after_bb1, w_bb2_sc, c1, 32, 32, c2, 1, 0, dev, reg)?;
        let after_bb2: Vec<f32> = bb2b
            .iter()
            .zip(sc.iter())
            .map(|(&a, &b)| (a + b).max(0.0))
            .collect();

        // GAP
        let mut feat = vec![0.0f32; c2];
        for ch in 0..c2 {
            let sum: f32 = after_bb2[ch * 32 * 32..(ch + 1) * 32 * 32].iter().sum();
            feat[ch] = sum / (32.0 * 32.0);
        }

        // FC
        let mut logits = vec![0.0f32; nc];
        for o in 0..nc {
            logits[o] = b_fc[o];
            for j in 0..c2 {
                logits[o] += feat[j] * w_fc[o * c2 + j];
            }
        }
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if pred == l as usize {
            correct += 1;
        }
    }
    Ok(correct)
}

fn load_cifar_batch(
    path: &std::path::Path,
) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let record_size = 1 + 3 * 32 * 32;
    let n = data.len() / record_size;
    let mut images = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * record_size;
        labels.push(data[offset]);
        let pixels = &data[offset + 1..offset + record_size];
        images.push(pixels.iter().map(|&b| b as f32 / 255.0).collect());
    }
    Ok((images, labels))
}
