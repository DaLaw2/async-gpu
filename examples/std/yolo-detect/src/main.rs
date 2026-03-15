//! YOLOv8-nano object detection example using the gpu-host nn API.
//!
//! Loads YOLOv8-nano from a safetensors file, runs detection on a PPM image,
//! and prints bounding boxes with class names and confidence scores.
//!
//! Usage:
//!   cargo run --release -- [image.ppm]
//!
//! Requires:
//!   - `models/yolov8n.safetensors` in the repository root
//!   - Input image as PPM (P6 binary), resized/padded to 640x640

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// COCO class names (80 classes).
const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // 1. Initialize CUDA
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // 2. Load kernels
    let registry = Arc::new(gpu_host::nn::KernelRegistry::new(
        Arc::clone(&dev),
        gpu_host::ptx::KERNEL,
    )?);
    println!("Kernel registry loaded");

    // 3. Load YOLO weights
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/yolov8n.safetensors");
    if !model_path.exists() {
        return Err(format!(
            "Model file not found: {}\nExport YOLOv8n with scripts/export_yolo.py",
            model_path.display()
        )
        .into());
    }

    let t0 = Instant::now();
    let weights = gpu_host::model_yolo::load_yolo_weights(&model_path)
        .map_err(|e| format!("Failed to load weights: {e}"))?;
    println!("Weights loaded in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // 4. Build model
    let t1 = Instant::now();
    let model = gpu_host::nn::models::yolov8::YoloV8Nano::from_weights(&weights, &registry)?;
    println!("Model built on GPU in {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // 5. Load input image
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        let default = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/bus.ppm");
        default.to_string_lossy().to_string()
    };

    let image_path = Path::new(&image_path);
    if !image_path.exists() {
        return Err(format!(
            "Image not found: {}\nProvide a 640x640 PPM image as argument",
            image_path.display()
        )
        .into());
    }

    let img = gpu_host::model_yolo::load_ppm(image_path)
        .map_err(|e| format!("Failed to read image: {e}"))?;
    println!("Loaded image: {}x{}", img.width, img.height);

    // 6. Run detection
    let t2 = Instant::now();
    let detections = model.detect(&img.data, 0.25, 0.45)?;
    let det_time = t2.elapsed();

    println!(
        "\nDetected {} objects in {:.1}ms:",
        detections.len(),
        det_time.as_secs_f64() * 1000.0
    );

    for (i, det) in detections.iter().enumerate() {
        let class_name = if det.class_id < COCO_CLASSES.len() {
            COCO_CLASSES[det.class_id]
        } else {
            "unknown"
        };
        println!(
            "  [{i}] {class_name} ({:.1}%) [{:.0}, {:.0}, {:.0}, {:.0}]",
            det.confidence * 100.0,
            det.x1,
            det.y1,
            det.x2,
            det.y2,
        );
    }

    println!("\nDone.");
    Ok(())
}
