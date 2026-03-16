//! ResNet-18 inference on CIFAR-10 using the nn module.
//!
//! Demonstrates:
//! - ResNet-18 model definition (BasicBlock, residual connections)
//! - Forward pass through 8 BasicBlocks (18 conv layers + BN + ReLU)
//! - Global average pooling → linear classifier
//!
//! Uses random weights (no pretrained model) — verifies forward pass produces
//! valid logits and measures inference speed.
//!
//! Usage: cargo run --release

use std::sync::Arc;
use std::time::Instant;

use gpu_host::nn::models::resnet::{ResNet18, ResNet18Weights};
use gpu_host::nn::tensor::GpuTensor;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);

    println!("--- ResNet-18 CIFAR-10 Inference ---");

    // Build model with random weights
    let t0 = Instant::now();
    let weights = ResNet18Weights::random(10);
    let model = ResNet18::from_weights(&weights, 10, &registry)?;
    println!("Model built: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Count parameters
    let conv_params = 64 * 3 * 3 * 3 // conv1
        + 2 * (64 * 64 * 3 * 3 * 2)  // layer1
        + 128 * 64 * 3 * 3 + 128 * 128 * 3 * 3 * 2 + 128 * 64 // layer2 + shortcut
        + 256 * 128 * 3 * 3 + 256 * 256 * 3 * 3 * 2 + 256 * 128 // layer3 + shortcut
        + 512 * 256 * 3 * 3 + 512 * 512 * 3 * 3 * 2 + 512 * 256; // layer4 + shortcut
    let fc_params = 512 * 10 + 10;
    println!("Parameters: ~{:.1}M ({} conv + {} FC)", (conv_params + fc_params) as f64 / 1e6, conv_params, fc_params);

    // Load CIFAR-10 test data (first 100 images for speed)
    let cifar_dir = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("cifar10");
    let (images, labels) = if cifar_dir.join("test_batch.bin").exists() {
        load_cifar_batch(&cifar_dir.join("test_batch.bin"))?
    } else {
        println!("No CIFAR-10 data found at {}, using random images", cifar_dir.display());
        let n = 100;
        let imgs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..3 * 32 * 32).map(|j| (i * 3072 + j) as f32 * 0.013 % 1.0).collect())
            .collect();
        let lbls: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();
        (imgs, lbls)
    };
    let n = images.len().min(100);
    println!("Test images: {n}");

    // Warmup
    let warmup_input = GpuTensor::from_host(&images[0], &[3, 32, 32], &dev)?;
    let _ = model.forward(&warmup_input)?;

    // Inference
    let t1 = Instant::now();
    let mut correct = 0;
    for i in 0..n {
        let input = GpuTensor::from_host(&images[i], &[3, 32, 32], &dev)?;
        let logits = model.forward(&input)?;
        let logits_host = logits.to_host()?;
        let pred = logits_host
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
    let per_image = elapsed / n as f64 * 1000.0;

    println!("\nResults:");
    println!("  Accuracy: {correct}/{n} ({:.1}%) [random weights — expected ~10%]",
        correct as f64 / n as f64 * 100.0);
    println!("  Total: {elapsed:.2}s, {per_image:.1}ms/image");
    println!("  Architecture: Conv(3→64) → 2×BB(64) → 2×BB(128) → 2×BB(256) → 2×BB(512) → GAP → FC(10)");

    // Verify no NaN in final logits
    let final_input = GpuTensor::from_host(&images[0], &[3, 32, 32], &dev)?;
    let final_logits = model.forward(&final_input)?.to_host()?;
    assert!(final_logits.iter().all(|x| x.is_finite()), "NaN in logits");
    println!("  Logits sample: [{:.3}, {:.3}, {:.3}, ...]", final_logits[0], final_logits[1], final_logits[2]);
    println!("\nPASSED (forward pass valid, no NaN)");

    Ok(())
}

/// Load a CIFAR-10 binary batch file.
/// Format: 10000 × (1 byte label + 3072 bytes pixel data [R,G,B channels × 32×32])
fn load_cifar_batch(path: &std::path::Path) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn std::error::Error>> {
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
