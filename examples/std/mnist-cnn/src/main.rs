//! MNIST CNN training on GPU.
//!
//! Conv2d(1→16, 3×3, pad=1) → ReLU → AvgPool(2) → Conv2d(16→32, 3×3, pad=1) → ReLU → AvgPool(2) → Linear(1568→10)
//! Uses GPU conv2d for forward, CPU backward for conv weights, GPU matmul for FC.
//!
//! Usage: cargo run --release [--cpu]
//! Requires: models/mnist/ (bash scripts/download-mnist.sh)

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::autograd;
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    let use_cpu = std::env::args().any(|a| a == "--cpu");
    if let Err(e) = run(use_cpu) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(use_cpu: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mnist_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("mnist");
    let train_images = load_idx_images(&mnist_dir.join("train-images-idx3-ubyte"))?;
    let train_labels = load_idx_labels(&mnist_dir.join("train-labels-idx1-ubyte"))?;
    let test_images = load_idx_images(&mnist_dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = load_idx_labels(&mnist_dir.join("t10k-labels-idx1-ubyte"))?;

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev), gpu_host::ptx::KERNEL,
    )?);

    let mode = if use_cpu { "CPU" } else { "GPU" };
    println!("MNIST CNN {mode} ({} train, {} test)", train_images.len(), test_images.len());

    // Architecture: Conv(1→16,3,pad1) → ReLU → AvgPool(2) → Conv(16→32,3,pad1) → ReLU → AvgPool(2) → Linear(1568→10)
    let (c1_out, c2_out) = (16, 32);
    let flat = c2_out * 7 * 7; // 1568
    let nc = 10;

    // He initialization
    let sc1 = (2.0 / (1 * 9) as f64).sqrt() as f32;
    let mut cw1: Vec<f32> = (0..c1_out * 1 * 3 * 3)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * sc1).collect();
    let sc2 = (2.0 / (c1_out * 9) as f64).sqrt() as f32;
    let mut cw2: Vec<f32> = (0..c2_out * c1_out * 3 * 3)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * sc2).collect();
    let sf = (2.0 / flat as f64).sqrt() as f32;
    let mut fw: Vec<f32> = (0..nc * flat)
        .map(|i| ((i * 1234567891 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * sf).collect();
    let mut fb = vec![0.0f32; nc];

    let lr = 0.01f32;
    let bs = 32;
    let epochs = 5;
    let ts = Instant::now();

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut tl = 0.0f64;
        let mut correct = 0usize;
        let nb = train_images.len() / bs;

        for bi in 0..nb {
            let start = bi * bs;
            let labels: Vec<usize> = (0..bs).map(|i| train_labels[start + i] as usize).collect();

            // Forward: per-sample conv → relu → pool → conv → relu → pool → batch FC
            let mut all_pooled = vec![0.0f32; bs * flat];
            let mut all_conv2_pre = Vec::with_capacity(bs); // pre-relu conv2 output
            let mut all_conv1_pre = Vec::with_capacity(bs); // pre-relu conv1 output
            let mut all_p1 = Vec::with_capacity(bs); // pooled relu1 (input to conv2)

            for i in 0..bs {
                let img = &train_images[start + i];
                // Conv1: [1,28,28] → [16,28,28]
                let c1 = if use_cpu {
                    cpu_conv2d(img, &cw1, 1, 28, 28, c1_out, 3, 3, 1, 1)
                } else {
                    let ig = GpuTensor::from_host(img, &[1, 28, 28], &dev)?;
                    let wg = GpuTensor::from_host(&cw1, &[c1_out, 1, 3, 3], &dev)?;
                    gpu_host::nn::ops::conv2d(&ig, &wg, None, 1, 1, &registry)?.to_host()?
                };
                let r1: Vec<f32> = c1.iter().map(|&v| v.max(0.0)).collect();
                let p1 = cpu_avg_pool(&r1, c1_out, 28, 28, 2); // → [16, 14, 14]
                all_conv1_pre.push(c1);
                all_p1.push(p1.clone());

                // Conv2: [16,14,14] → [32,14,14]
                let c2 = if use_cpu {
                    cpu_conv2d(&p1, &cw2, c1_out, 14, 14, c2_out, 3, 3, 1, 1)
                } else {
                    let ig2 = GpuTensor::from_host(&p1, &[c1_out, 14, 14], &dev)?;
                    let wg2 = GpuTensor::from_host(&cw2, &[c2_out, c1_out, 3, 3], &dev)?;
                    gpu_host::nn::ops::conv2d(&ig2, &wg2, None, 1, 1, &registry)?.to_host()?
                };
                let r2: Vec<f32> = c2.iter().map(|&v| v.max(0.0)).collect();
                let p2 = cpu_avg_pool(&r2, c2_out, 14, 14, 2); // → [32, 7, 7]
                all_pooled[i * flat..(i + 1) * flat].copy_from_slice(&p2);
                all_conv2_pre.push(c2);
            }

            // FC forward (batched GPU matmul)
            let tape = autograd::Tape::new();
            let mut pool = autograd::TensorPool::new();
            let (logits_all, tape) = autograd::with_tape(tape, || {
                let mut feat = GpuTensor::from_host(&all_pooled, &[bs, flat], &dev).unwrap();
                feat.set_requires_grad(true);
                let fid = autograd::alloc_tensor_id().unwrap();
                feat.set_tensor_id(fid);
                pool.insert(fid, feat.clone_tensor().unwrap());

                let mut fw_t = vec![0.0f32; flat * nc];
                for o in 0..nc { for j in 0..flat { fw_t[j * nc + o] = fw[o * flat + j]; } }
                let mut wg = GpuTensor::from_host(&fw_t, &[flat, nc], &dev).unwrap();
                wg.set_requires_grad(true);
                let wid = autograd::alloc_tensor_id().unwrap();
                wg.set_tensor_id(wid);
                pool.insert(wid, wg.clone_tensor().unwrap());

                let mut logits = gpu_host::nn::ops::matmul(&feat, &wg, &registry).unwrap();
                let lid = logits.tensor_id().unwrap();
                pool.insert(lid, logits.clone_tensor().unwrap());

                let bg = GpuTensor::from_host(&fb, &[nc], &dev).unwrap();
                gpu_host::nn::ops::bias_add(&mut logits, &bg, &registry).unwrap();
                let fid2 = logits.tensor_id().unwrap();
                pool.insert(fid2, logits.clone_tensor().unwrap());

                (logits.to_host().unwrap(), fid2)
            });
            let (logits_vals, loss_id) = logits_all;

            // Loss + softmax gradient
            let mut d_logits = vec![0.0f32; bs * nc];
            for b in 0..bs {
                let row = &logits_vals[b * nc..(b + 1) * nc];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let esum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                for o in 0..nc {
                    let sm = (row[o] - mx).exp() / esum;
                    d_logits[b * nc + o] = (sm - if o == labels[b] { 1.0 } else { 0.0 }) / bs as f32;
                }
                tl -= ((row[labels[b]] - mx).exp() / esum).ln() as f64;
                let pred = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap();
                if pred == labels[b] { correct += 1; }
            }

            // FC backward via autograd tape
            let dl_gpu = GpuTensor::from_host(&d_logits, &[bs, nc], &dev)?;
            let mut grads = std::collections::HashMap::new();
            grads.insert(loss_id, dl_gpu);
            for entry in tape.entries().iter().rev() {
                let d_out = match grads.get(&entry.output) {
                    Some(g) => g.clone_tensor()?,
                    None => continue,
                };
                match entry.op {
                    autograd::OpKind::Matmul => {
                        let (a_id, b_id) = (entry.saved[0], entry.saved[1]);
                        if a_id.0 != u32::MAX { if let Some(bt) = pool.get(b_id) {
                            let btt = bt.transpose(0, 1)?;
                            let da = gpu_host::nn::ops::matmul(&d_out, &btt, &registry)?;
                            grads.entry(a_id).or_insert(da);
                        }}
                        if b_id.0 != u32::MAX { if let Some(at) = pool.get(a_id) {
                            let att = at.transpose(0, 1)?;
                            let db = gpu_host::nn::ops::matmul(&att, &d_out, &registry)?;
                            grads.entry(b_id).or_insert(db);
                        }}
                    }
                    autograd::OpKind::BiasAdd => { grads.entry(entry.inputs[0]).or_insert(d_out); }
                    _ => {}
                }
            }

            // Update FC weights
            if let Some(dw) = grads.get(&autograd::TensorId(1)) {
                let dh = dw.to_host()?;
                for o in 0..nc { for j in 0..flat { fw[o * flat + j] -= lr * dh[j * nc + o]; } }
            }
            for b in 0..bs { for o in 0..nc { fb[o] -= lr * d_logits[b * nc + o]; } }

            // Conv backward: full chain FC → pool2 → relu2 → conv2 → pool1 → relu1 → conv1
            if let Some(d_feat) = grads.get(&autograd::TensorId(0)) {
                let df = d_feat.to_host()?;
                let mut dcw2 = vec![0.0f32; cw2.len()];
                let mut dcw1 = vec![0.0f32; cw1.len()];
                for i in 0..bs {
                    let img = &train_images[start + i];
                    // d_pooled2 → unpool → relu mask → d_conv2_out
                    let dp2 = &df[i * flat..(i + 1) * flat];
                    let du2 = cpu_avg_unpool(dp2, c2_out, 14, 14, 2);
                    let dr2: Vec<f32> = du2.iter().zip(all_conv2_pre[i].iter())
                        .map(|(&d, &c)| if c > 0.0 { d } else { 0.0 }).collect();

                    if use_cpu {
                        // CPU conv backward
                        let dw2 = cpu_conv2d_wgrad(&all_p1[i], &dr2, c1_out, 14, 14, c2_out, 3, 3, 1, 1);
                        for k in 0..dcw2.len() { dcw2[k] += dw2[k]; }
                        let dp1 = cpu_conv2d_igrad(&dr2, &cw2, c1_out, 14, 14, c2_out, 3, 3, 1, 1);
                        let du1 = cpu_avg_unpool(&dp1, c1_out, 28, 28, 2);
                        let dr1: Vec<f32> = du1.iter().zip(all_conv1_pre[i].iter())
                            .map(|(&d, &c)| if c > 0.0 { d } else { 0.0 }).collect();
                        let dw1 = cpu_conv2d_wgrad(img, &dr1, 1, 28, 28, c1_out, 3, 3, 1, 1);
                        for k in 0..dcw1.len() { dcw1[k] += dw1[k]; }
                    } else {
                        // GPU conv backward
                        let dr2_t = GpuTensor::from_host(&dr2, &[c2_out, 14, 14], &dev)?;
                        let p1_t = GpuTensor::from_host(&all_p1[i], &[c1_out, 14, 14], &dev)?;
                        let w2_t = GpuTensor::from_host(&cw2, &[c2_out, c1_out, 3, 3], &dev)?;
                        let (dp1_t, dw2_t) = gpu_host::nn::ops::conv2d_backward(&dr2_t, &p1_t, &w2_t, 1, 1, &registry)?;
                        let dw2h = dw2_t.to_host()?;
                        for k in 0..dcw2.len() { dcw2[k] += dw2h[k]; }

                        let dp1 = dp1_t.to_host()?;
                        let du1 = cpu_avg_unpool(&dp1, c1_out, 28, 28, 2);
                        let dr1: Vec<f32> = du1.iter().zip(all_conv1_pre[i].iter())
                            .map(|(&d, &c)| if c > 0.0 { d } else { 0.0 }).collect();

                        let dr1_t = GpuTensor::from_host(&dr1, &[c1_out, 28, 28], &dev)?;
                        let img_t = GpuTensor::from_host(img, &[1, 28, 28], &dev)?;
                        let w1_t = GpuTensor::from_host(&cw1, &[c1_out, 1, 3, 3], &dev)?;
                        let (_di1, dw1_t) = gpu_host::nn::ops::conv2d_backward(&dr1_t, &img_t, &w1_t, 1, 1, &registry)?;
                        let dw1h = dw1_t.to_host()?;
                        for k in 0..dcw1.len() { dcw1[k] += dw1h[k]; }
                    }
                }
                for k in 0..cw2.len() { cw2[k] -= lr * dcw2[k]; }
                for k in 0..cw1.len() { cw1[k] -= lr * dcw1[k]; }
            }
        }

        let al = tl / (nb * bs) as f64;
        let ta = correct as f64 / (nb * bs) as f64 * 100.0;
        let tc = eval(use_cpu, &test_images, &test_labels, &cw1, &cw2, &fw, &fb, &dev, &registry);
        let va = tc as f64 / test_images.len() as f64 * 100.0;
        println!("Epoch {}/{}: loss={al:.3}, train={ta:.1}%, test={va:.1}%, time={:.1}s",
            epoch + 1, epochs, es.elapsed().as_secs_f64());
    }

    println!("\nTotal: {:.1}s ({mode})", ts.elapsed().as_secs_f64());
    Ok(())
}

fn eval(use_cpu: bool, imgs: &[Vec<f32>], lbls: &[u8], cw1: &[f32], cw2: &[f32], fw: &[f32], fb: &[f32],
    dev: &Arc<cudarc::driver::CudaDevice>, reg: &Arc<gpu_host::nn::KernelRegistry>) -> usize {
    let (c1, c2, flat, nc) = (16, 32, 32 * 7 * 7, 10);
    let mut correct = 0;
    for (img, &l) in imgs.iter().zip(lbls.iter()) {
        let co1 = if use_cpu { cpu_conv2d(img, cw1, 1, 28, 28, c1, 3, 3, 1, 1) } else {
            let ig = GpuTensor::from_host(img, &[1, 28, 28], dev).unwrap();
            let wg = GpuTensor::from_host(cw1, &[c1, 1, 3, 3], dev).unwrap();
            gpu_host::nn::ops::conv2d(&ig, &wg, None, 1, 1, reg).unwrap().to_host().unwrap()
        };
        let r1: Vec<f32> = co1.iter().map(|&v| v.max(0.0)).collect();
        let p1 = cpu_avg_pool(&r1, c1, 28, 28, 2);
        let co2 = if use_cpu { cpu_conv2d(&p1, cw2, c1, 14, 14, c2, 3, 3, 1, 1) } else {
            let ig2 = GpuTensor::from_host(&p1, &[c1, 14, 14], dev).unwrap();
            let wg2 = GpuTensor::from_host(cw2, &[c2, c1, 3, 3], dev).unwrap();
            gpu_host::nn::ops::conv2d(&ig2, &wg2, None, 1, 1, reg).unwrap().to_host().unwrap()
        };
        let r2: Vec<f32> = co2.iter().map(|&v| v.max(0.0)).collect();
        let p2 = cpu_avg_pool(&r2, c2, 14, 14, 2);
        let mut logits = vec![0.0f32; nc];
        for o in 0..nc { logits[o] = fb[o]; for j in 0..flat { logits[o] += p2[j] * fw[o * flat + j]; } }
        let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        if pred == l as usize { correct += 1; }
    }
    correct
}

// CPU conv2d forward: [ci,h,w] * [co,ci,kh,kw] → [co,ho,wo]
fn cpu_conv2d(i: &[f32], w: &[f32], ci: usize, h: usize, ww: usize, co: usize, kh: usize, kw: usize, s: usize, p: usize) -> Vec<f32> {
    let (ho, wo) = ((h + 2*p - kh)/s + 1, (ww + 2*p - kw)/s + 1);
    let mut o = vec![0.0f32; co * ho * wo];
    for c in 0..co { for oh in 0..ho { for ow in 0..wo { let mut sum = 0.0f32;
        for ci2 in 0..ci { for fh in 0..kh { for fw in 0..kw {
            let ih = (oh*s+fh) as isize - p as isize; let iw = (ow*s+fw) as isize - p as isize;
            if ih >= 0 && ih < h as isize && iw >= 0 && iw < ww as isize {
                sum += i[ci2*h*ww + ih as usize*ww + iw as usize] * w[c*(ci*kh*kw) + ci2*(kh*kw) + fh*kw + fw];
            }
        }}} o[c*ho*wo + oh*wo + ow] = sum; }}} o
}

// CPU conv2d weight gradient: dW[co,ci,kh,kw] = sum over (oh,ow) of d_out[co,oh,ow] * input[ci,ih,iw]
fn cpu_conv2d_wgrad(i: &[f32], d: &[f32], ci: usize, h: usize, w: usize, co: usize, kh: usize, kw: usize, s: usize, p: usize) -> Vec<f32> {
    let (ho, wo) = ((h+2*p-kh)/s+1, (w+2*p-kw)/s+1);
    let mut dw = vec![0.0f32; co*ci*kh*kw];
    for c in 0..co { for ci2 in 0..ci { for fh in 0..kh { for fw in 0..kw { let mut sum = 0.0f32;
        for oh in 0..ho { for ow in 0..wo {
            let ih = (oh*s+fh) as isize - p as isize; let iw = (ow*s+fw) as isize - p as isize;
            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                sum += d[c*ho*wo+oh*wo+ow] * i[ci2*h*w + ih as usize*w + iw as usize]; }
        }} dw[c*(ci*kh*kw)+ci2*(kh*kw)+fh*kw+fw] = sum; }}}} dw
}

// CPU conv2d input gradient: dInput[ci,ih,iw] = sum over (co,fh,fw) of d_out[co,oh,ow] * weight[co,ci,fh,fw]
// where oh = (ih + p - fh) / s, ow = (iw + p - fw) / s
fn cpu_conv2d_igrad(d: &[f32], w: &[f32], ci: usize, h: usize, ww: usize, co: usize, kh: usize, kw: usize, s: usize, p: usize) -> Vec<f32> {
    let (ho, wo) = ((h+2*p-kh)/s+1, (ww+2*p-kw)/s+1);
    let mut di = vec![0.0f32; ci*h*ww];
    for c in 0..co { for oh in 0..ho { for ow in 0..wo {
        let dv = d[c*ho*wo+oh*wo+ow];
        if dv == 0.0 { continue; }
        for ci2 in 0..ci { for fh in 0..kh { for fw in 0..kw {
            let ih = (oh*s+fh) as isize - p as isize; let iw = (ow*s+fw) as isize - p as isize;
            if ih >= 0 && ih < h as isize && iw >= 0 && iw < ww as isize {
                di[ci2*h*ww + ih as usize*ww + iw as usize] += dv * w[c*(ci*kh*kw)+ci2*(kh*kw)+fh*kw+fw];
            }
        }}}
    }}} di
}

fn cpu_avg_pool(i: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho, wo, a) = (h/ps, w/ps, (ps*ps) as f32);
    let mut o = vec![0.0f32; c*ho*wo];
    for ch in 0..c { for oh in 0..ho { for ow in 0..wo { let mut s = 0.0f32;
        for ph in 0..ps { for pw in 0..ps { s += i[ch*h*w+(oh*ps+ph)*w+ow*ps+pw]; } }
        o[ch*ho*wo+oh*wo+ow] = s/a; }}} o
}

fn cpu_avg_unpool(d: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho, wo, a) = (h/ps, w/ps, (ps*ps) as f32);
    let mut o = vec![0.0f32; c*h*w];
    for ch in 0..c { for oh in 0..ho { for ow in 0..wo { let v = d[ch*ho*wo+oh*wo+ow]/a;
        for ph in 0..ps { for pw in 0..ps { o[ch*h*w+(oh*ps+ph)*w+ow*ps+pw] = v; } } }}} o
}

fn load_idx_images(p: &std::path::Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let d = std::fs::read(p)?; let n = u32::from_be_bytes([d[0],d[1],d[2],d[3]]);
    if n != 2051 { return Err("bad magic".into()); }
    let cnt = u32::from_be_bytes([d[4],d[5],d[6],d[7]]) as usize;
    let px = 28 * 28;
    Ok((0..cnt).map(|i| d[16+i*px..16+(i+1)*px].iter().map(|&b| b as f32/255.0).collect()).collect())
}

fn load_idx_labels(p: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let d = std::fs::read(p)?;
    let n = u32::from_be_bytes([d[4],d[5],d[6],d[7]]) as usize;
    Ok(d[8..8+n].to_vec())
}
