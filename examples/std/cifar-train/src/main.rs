//! CIFAR-10 tiny CNN training on GPU.
//!
//! Forward: GPU Conv2d per-sample + CPU ReLU + CPU AvgPool + batched GPU matmul (FC)
//! Backward: batched GPU matmul backward + CPU conv weight gradient
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
    let ps = 4;
    let flat = c_out * (h / ps) * (w / ps); // 512
    let nc = 10;

    let sc = (2.0 / (c_in * kh * kw) as f64).sqrt() as f32;
    let mut conv_w: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * sc).collect();
    let sf = (2.0 / flat as f64).sqrt() as f32;
    let mut fc_w: Vec<f32> = (0..nc * flat)
        .map(|i| ((i * 340573321 % 1000) as f32 / 1000.0 - 0.5) * 2.0 * sf).collect();
    let mut fc_b = vec![0.0f32; nc];

    let lr = 0.01f32;
    let bs = 32;
    let epochs = 10;
    let ts = Instant::now();

    // Conv weights GPU tensor — re-uploaded after each batch update
    let mut cw_gpu = GpuTensor::from_host(&conv_w, &[c_out, c_in, kh, kw], &dev)?;

    for epoch in 0..epochs {
        let es = Instant::now();
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let nb = train_imgs.len() / bs;

        for bi in 0..nb {
            let start = bi * bs;
            let batch_labels: Vec<usize> = (0..bs).map(|i| train_labels[start + i] as usize).collect();

            // === Conv2d forward (per-sample GPU) ===
            let mut all_conv_host = Vec::with_capacity(bs);
            let mut all_pooled = vec![0.0f32; bs * flat];
            for i in 0..bs {
                let img_gpu = GpuTensor::from_host(&train_imgs[start + i], &[c_in, h, w], &dev)?;
                let conv_out = gpu_host::nn::ops::conv2d(&img_gpu, &cw_gpu, None, 1, 1, &registry)?;
                let conv_h = conv_out.to_host()?;
                let relu: Vec<f32> = conv_h.iter().map(|&v| v.max(0.0)).collect();
                let pooled = cpu_avg_pool(&relu, c_out, h, w, ps);
                all_pooled[i * flat..(i + 1) * flat].copy_from_slice(&pooled);
                all_conv_host.push(conv_h);
            }

            // === BATCHED FC forward: one GPU matmul for entire batch ===
            let tape = autograd::Tape::new();
            let mut pool = autograd::TensorPool::new();

            let (logits_host, tape) = autograd::with_tape(tape, || {
                let mut feat = GpuTensor::from_host(&all_pooled, &[bs, flat], &dev).unwrap();
                feat.set_requires_grad(true); // needed for FC backward to compute d_feat
                let fid = autograd::alloc_tensor_id().unwrap();
                feat.set_tensor_id(fid);
                pool.insert(fid, feat.clone_tensor().unwrap());

                let mut fw = GpuTensor::from_host(&fc_w, &[flat, nc], &dev).unwrap();
                let fwid = autograd::alloc_tensor_id().unwrap();
                fw.set_tensor_id(fwid);
                fw.set_requires_grad(true);
                pool.insert(fwid, fw.clone_tensor().unwrap());

                // ONE GPU matmul: [bs, flat] × [flat, nc] → [bs, nc]
                let mut logits = gpu_host::nn::ops::matmul(&feat, &fw, &registry).unwrap();
                let lid = logits.tensor_id().unwrap();
                pool.insert(lid, logits.clone_tensor().unwrap());

                let fb = GpuTensor::from_host(&fc_b, &[nc], &dev).unwrap();
                gpu_host::nn::ops::bias_add(&mut logits, &fb, &registry).unwrap();
                let fid2 = logits.tensor_id().unwrap();
                pool.insert(fid2, logits.clone_tensor().unwrap());

                (logits.to_host().unwrap(), fid2)
            });
            let (logits_all, loss_id) = logits_host;

            // === Softmax + CE loss (CPU, batched) ===
            let mut d_logits = vec![0.0f32; bs * nc];
            for b in 0..bs {
                let row = &logits_all[b * nc..(b + 1) * nc];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let esum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                for o in 0..nc {
                    let sm = (row[o] - mx).exp() / esum;
                    d_logits[b * nc + o] = (sm - if o == batch_labels[b] { 1.0 } else { 0.0 }) / bs as f32;
                }
                total_loss -= ((row[batch_labels[b]] - mx).exp() / esum).ln() as f64;
                let pred = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
                if pred == batch_labels[b] { correct += 1; }
            }

            // === BATCHED FC backward: ONE GPU matmul for dW ===
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
                        if a_id.0 != u32::MAX {
                            if let Some(bt) = pool.get(b_id) {
                                let btt = bt.transpose(0, 1)?;
                                let da = gpu_host::nn::ops::matmul(&d_out, &btt, &registry)?;
                                grads.entry(a_id).or_insert(da);
                            }
                        }
                        if b_id.0 != u32::MAX {
                            if let Some(at) = pool.get(a_id) {
                                let att = at.transpose(0, 1)?;
                                let db = gpu_host::nn::ops::matmul(&att, &d_out, &registry)?;
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

            // FC weight update
            if let Some(dfw) = grads.get(&autograd::TensorId(1)) {
                let dh = dfw.to_host()?;
                for i in 0..fc_w.len() { fc_w[i] -= lr * dh[i]; }
            }
            for b in 0..bs { for o in 0..nc { fc_b[o] -= lr * d_logits[b * nc + o]; } }

            // Conv weight gradient (from d_pooled back through pool/relu/conv)
            if let Some(d_feat) = grads.get(&autograd::TensorId(0)) {
                let df = d_feat.to_host()?;
                let mut d_conv_w = vec![0.0f32; conv_w.len()];
                for b in 0..bs {
                    let d_pooled = &df[b * flat..(b + 1) * flat];
                    let d_up = cpu_avg_unpool(d_pooled, c_out, h, w, ps);
                    let d_relu: Vec<f32> = d_up.iter().zip(all_conv_host[b].iter())
                        .map(|(&dv, &cv)| if cv > 0.0 { dv } else { 0.0 }).collect();
                    let dw = cpu_conv2d_wgrad(&train_imgs[start + b], &d_relu, c_in, h, w, c_out, kh, kw, 1, 1);
                    for i in 0..d_conv_w.len() { d_conv_w[i] += dw[i]; }
                }
                for i in 0..conv_w.len() { conv_w[i] -= lr * d_conv_w[i]; }
                // Re-upload updated conv weights to GPU
                cw_gpu = GpuTensor::from_host(&conv_w, &[c_out, c_in, kh, kw], &dev)?;
            }

            total_loss /= bs as f64;
        }

        let avg_loss = total_loss / nb as f64;
        let train_acc = correct as f64 / (nb * bs) as f64 * 100.0;
        let test_c = eval_gpu(&test_imgs, &test_labels, &conv_w, &fc_w, &fc_b, &dev, &registry);
        let test_acc = test_c as f64 / test_imgs.len() as f64 * 100.0;
        println!("Epoch {}/{}: loss={avg_loss:.4}, train={train_acc:.1}%, test={test_acc:.1}%, time={:.1}s",
            epoch + 1, epochs, es.elapsed().as_secs_f64());

        // Re-upload updated conv weights for next epoch
    }
    println!("\nTotal: {:.1}s (GPU)", ts.elapsed().as_secs_f64());
    Ok(())
}

fn run_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    let (ti, tl) = load_cifar_batch(&cifar_dir.join("data_batch_1.bin"), 2000)?;
    let (vi, vl) = load_cifar_batch(&cifar_dir.join("test_batch.bin"), 500)?;
    println!("CIFAR-10 CPU training ({} train, {} test)", ti.len(), vi.len());
    let (c_in, c_out, h, w, kh, kw, ps, flat, nc) = (3, 8, 32, 32, 3, 3, 4, 512, 10);
    let sc = (2.0 / (c_in * kh * kw) as f64).sqrt() as f32;
    let mut cw: Vec<f32> = (0..c_out*c_in*kh*kw).map(|i| ((i*2654435761%1000) as f32/1000.0-0.5)*2.0*sc).collect();
    let sf = (2.0 / flat as f64).sqrt() as f32;
    let mut fw: Vec<f32> = (0..nc*flat).map(|i| ((i*340573321%1000) as f32/1000.0-0.5)*2.0*sf).collect();
    let mut fb = vec![0.0f32; nc];
    let (lr, bs, epochs) = (0.01f32, 32, 10);
    let ts = Instant::now();
    for epoch in 0..epochs {
        let es = Instant::now();
        let mut tl2 = 0.0f64; let mut correct = 0usize; let nb = ti.len() / bs;
        for bi in 0..nb {
            let s = bi * bs;
            let mut dfw = vec![0.0f32; fw.len()]; let mut dfb = vec![0.0f32; nc]; let mut dcw = vec![0.0f32; cw.len()];
            for i in 0..bs {
                let label = tl[s+i] as usize;
                let co = cpu_conv2d(&ti[s+i], &cw, c_in, h, w, c_out, kh, kw, 1, 1);
                let relu: Vec<f32> = co.iter().map(|&v| v.max(0.0)).collect();
                let pooled = cpu_avg_pool(&relu, c_out, h, w, ps);
                let mut logits = vec![0.0f32; nc];
                for o in 0..nc { logits[o] = fb[o]; for j in 0..flat { logits[o] += pooled[j]*fw[o*flat+j]; } }
                let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let esum: f32 = logits.iter().map(|&x| (x-mx).exp()).sum();
                let mut dl = vec![0.0f32; nc];
                for o in 0..nc {
                    let sm = (logits[o]-mx).exp()/esum;
                    dl[o] = (sm - if o==label {1.0} else {0.0})/bs as f32;
                }
                tl2 -= ((logits[label]-mx).exp()/esum).ln() as f64;
                let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap();
                if pred==label { correct+=1; }
                let mut dp = vec![0.0f32; flat];
                for o in 0..nc { dfb[o]+=dl[o]; for j in 0..flat { dfw[o*flat+j]+=dl[o]*pooled[j]; dp[j]+=dl[o]*fw[o*flat+j]; } }
                let du = cpu_avg_unpool(&dp, c_out, h, w, ps);
                let dr: Vec<f32> = du.iter().zip(co.iter()).map(|(&d,&c)| if c>0.0{d}else{0.0}).collect();
                let dwc = cpu_conv2d_wgrad(&ti[s+i], &dr, c_in, h, w, c_out, kh, kw, 1, 1);
                for k in 0..dcw.len() { dcw[k]+=dwc[k]; }
            }
            for k in 0..fw.len() { fw[k]-=lr*dfw[k]; } for o in 0..nc { fb[o]-=lr*dfb[o]; }
            for k in 0..cw.len() { cw[k]-=lr*dcw[k]; } tl2/=bs as f64;
        }
        let al=tl2/nb as f64; let ta=correct as f64/(nb*bs) as f64*100.0;
        let tc=eval_cpu(&vi,&vl,&cw,&fw,&fb); let va=tc as f64/vi.len() as f64*100.0;
        println!("Epoch {}/{}: loss={al:.4}, train={ta:.1}%, test={va:.1}%, time={:.1}s", epoch+1, epochs, es.elapsed().as_secs_f64());
    }
    println!("\nTotal: {:.1}s (CPU)", ts.elapsed().as_secs_f64());
    Ok(())
}

fn cpu_conv2d(i: &[f32], w: &[f32], ci: usize, h: usize, ww: usize, co: usize, kh: usize, kw: usize, s: usize, p: usize) -> Vec<f32> {
    let (ho,wo)=((h+2*p-kh)/s+1,(ww+2*p-kw)/s+1); let mut o=vec![0.0f32;co*ho*wo];
    for c in 0..co { for oh in 0..ho { for ow in 0..wo { let mut sum=0.0f32;
        for ci2 in 0..ci { for fh in 0..kh { for fw in 0..kw {
            let ih=(oh*s+fh) as isize-p as isize; let iw=(ow*s+fw) as isize-p as isize;
            if ih>=0&&ih<h as isize&&iw>=0&&iw<ww as isize { sum+=i[ci2*h*ww+ih as usize*ww+iw as usize]*w[c*(ci*kh*kw)+ci2*(kh*kw)+fh*kw+fw]; }
        }}} o[c*ho*wo+oh*wo+ow]=sum; }}} o
}
fn cpu_conv2d_wgrad(i: &[f32], d: &[f32], ci: usize, h: usize, w: usize, co: usize, kh: usize, kw: usize, s: usize, p: usize) -> Vec<f32> {
    let (ho,wo)=((h+2*p-kh)/s+1,(w+2*p-kw)/s+1); let mut dw=vec![0.0f32;co*ci*kh*kw];
    for c in 0..co { for ci2 in 0..ci { for fh in 0..kh { for fw in 0..kw { let mut sum=0.0f32;
        for oh in 0..ho { for ow in 0..wo {
            let ih=(oh*s+fh) as isize-p as isize; let iw=(ow*s+fw) as isize-p as isize;
            if ih>=0&&ih<h as isize&&iw>=0&&iw<w as isize { sum+=d[c*ho*wo+oh*wo+ow]*i[ci2*h*w+ih as usize*w+iw as usize]; }
        }} dw[c*(ci*kh*kw)+ci2*(kh*kw)+fh*kw+fw]=sum; }}}} dw
}
fn cpu_avg_pool(i: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho,wo,a)=(h/ps,w/ps,(ps*ps) as f32); let mut o=vec![0.0f32;c*ho*wo];
    for ch in 0..c { for oh in 0..ho { for ow in 0..wo { let mut s=0.0f32;
        for ph in 0..ps { for pw in 0..ps { s+=i[ch*h*w+(oh*ps+ph)*w+ow*ps+pw]; } }
        o[ch*ho*wo+oh*wo+ow]=s/a; }}} o
}
fn cpu_avg_unpool(d: &[f32], c: usize, h: usize, w: usize, ps: usize) -> Vec<f32> {
    let (ho,wo,a)=(h/ps,w/ps,(ps*ps) as f32); let mut o=vec![0.0f32;c*h*w];
    for ch in 0..c { for oh in 0..ho { for ow in 0..wo { let v=d[ch*ho*wo+oh*wo+ow]/a;
        for ph in 0..ps { for pw in 0..ps { o[ch*h*w+(oh*ps+ph)*w+ow*ps+pw]=v; } } }}} o
}
fn eval_gpu(imgs: &[Vec<f32>], lbls: &[u8], cw: &[f32], fw: &[f32], fb: &[f32],
    dev: &Arc<cudarc::driver::CudaDevice>, reg: &Arc<gpu_host::nn::KernelRegistry>) -> usize {
    let (co,h,w,ps,flat,nc)=(8,32,32,4,512,10); let mut correct=0;
    // Batch conv forward
    let cwg = GpuTensor::from_host(cw, &[co, 3, 3, 3], dev).unwrap();
    let mut all_pooled = vec![0.0f32; imgs.len() * flat];
    for (i, img) in imgs.iter().enumerate() {
        let ig = GpuTensor::from_host(img, &[3, h, w], dev).unwrap();
        let co_out = gpu_host::nn::ops::conv2d(&ig, &cwg, None, 1, 1, reg).unwrap();
        let ch = co_out.to_host().unwrap();
        let relu: Vec<f32> = ch.iter().map(|&v| v.max(0.0)).collect();
        let pooled = cpu_avg_pool(&relu, co, h, w, ps);
        all_pooled[i * flat..(i + 1) * flat].copy_from_slice(&pooled);
    }
    // ONE batched matmul for all test images
    let fg = GpuTensor::from_host(&all_pooled, &[imgs.len(), flat], dev).unwrap();
    let wg = GpuTensor::from_host(fw, &[flat, nc], dev).unwrap();
    let mut lg = gpu_host::nn::ops::matmul(&fg, &wg, reg).unwrap();
    let bg = GpuTensor::from_host(fb, &[nc], dev).unwrap();
    gpu_host::nn::ops::bias_add(&mut lg, &bg, reg).unwrap();
    let lh = lg.to_host().unwrap();
    for (i, &label) in lbls.iter().enumerate() {
        let row = &lh[i*nc..(i+1)*nc];
        let pred = row.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(j,_)|j).unwrap();
        if pred==label as usize { correct+=1; }
    }
    correct
}
fn eval_cpu(imgs: &[Vec<f32>], lbls: &[u8], cw: &[f32], fw: &[f32], fb: &[f32]) -> usize {
    let (co,h,w,ps,flat,nc)=(8,32,32,4,512,10); let mut correct=0;
    for (img, &l) in imgs.iter().zip(lbls.iter()) {
        let co2=cpu_conv2d(img,cw,3,h,w,co,3,3,1,1); let r: Vec<f32>=co2.iter().map(|&v|v.max(0.0)).collect();
        let p=cpu_avg_pool(&r,co,h,w,ps); let mut lg=vec![0.0f32;nc];
        for o in 0..nc { lg[o]=fb[o]; for j in 0..flat { lg[o]+=p[j]*fw[o*flat+j]; } }
        let pred=lg.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap();
        if pred==l as usize { correct+=1; }
    } correct
}
fn load_cifar_batch(p: &std::path::Path, max: usize) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
    let d=std::fs::read(p)?; let rs=3073; let n=(d.len()/rs).min(max);
    let mut imgs=Vec::with_capacity(n); let mut lbls=Vec::with_capacity(n);
    for i in 0..n { let off=i*rs; lbls.push(d[off]); imgs.push(d[off+1..off+rs].iter().map(|&b|b as f32/255.0).collect()); }
    Ok((imgs, lbls))
}
