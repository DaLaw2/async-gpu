//! YOLOv8-nano weight loader and image I/O for GPU inference.
//!
//! Loads YOLOv8-nano weights from safetensors (exported by `scripts/export_yolo.py`)
//! and provides structured access by layer name. Also includes a minimal PPM image
//! reader (no external dependency).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::model::ModelError;
use crate::model_generic::TensorData;

// ---------------------------------------------------------------------------
// YOLO Constants
// ---------------------------------------------------------------------------

/// YOLOv8-nano input image size (square).
pub const YOLO_INPUT_SIZE: usize = 640;

/// Number of COCO classes.
pub const NUM_CLASSES: usize = 80;

/// reg_max for nano variant (DFL bins per coordinate).
/// Actual model has DFL weight [1, 16, 1, 1] — 16 bins per coordinate.
pub const REG_MAX: usize = 16;

/// YOLO layer count (0..22, 23 total).
pub const NUM_LAYERS: usize = 23;

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

/// Weights for a Conv2d + BatchNorm2d + SiLU block.
#[derive(Debug)]
pub struct ConvBnSiluWeights {
    /// Conv2d weight \[C_out, C_in, kH, kW\], stored as flat f32.
    pub conv_weight: Vec<f32>,
    /// Conv2d weight shape \[C_out, C_in, kH, kW\].
    pub conv_shape: Vec<usize>,
    /// BatchNorm gamma \[C\].
    pub bn_weight: Vec<f32>,
    /// BatchNorm beta \[C\].
    pub bn_bias: Vec<f32>,
    /// BatchNorm running mean \[C\].
    pub bn_running_mean: Vec<f32>,
    /// BatchNorm running var \[C\].
    pub bn_running_var: Vec<f32>,
}

/// Weights for a bare Conv2d (no BN, no activation) — used in detect head final layers.
#[derive(Debug)]
pub struct ConvWeights {
    /// Conv2d weight \[C_out, C_in, kH, kW\].
    pub weight: Vec<f32>,
    /// Conv2d weight shape \[C_out, C_in, kH, kW\].
    pub shape: Vec<usize>,
    /// Conv2d bias \[C_out\].
    pub bias: Vec<f32>,
}

/// All weights for YOLOv8-nano, organized by Ultralytics layer index.
///
/// Access pattern: use helper methods to get Conv/BN/C2f/SPPF/Detect weights
/// by layer index, using the raw tensor map.
#[derive(Debug)]
pub struct YoloWeights {
    /// Raw tensor map: name -> (data, shape). Access via helper methods.
    pub tensors: HashMap<String, TensorData>,
}

impl YoloWeights {
    /// Get a tensor by name, returning an error if not found.
    pub fn get(&self, name: &str) -> Result<&TensorData, ModelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
    }

    /// Get f32 data for a tensor by name.
    pub fn get_data(&self, name: &str) -> Result<&[f32], ModelError> {
        Ok(&self.get(name)?.data)
    }

    /// Get shape for a tensor by name.
    pub fn get_shape(&self, name: &str) -> Result<&[usize], ModelError> {
        Ok(&self.get(name)?.shape)
    }

    /// Load Conv+BN+SiLU weights for a simple Conv layer (e.g., backbone layer 0).
    ///
    /// Expects tensor names: `model.{idx}.conv.weight`, `model.{idx}.bn.*`
    pub fn conv_bn_silu(&self, idx: usize) -> Result<ConvBnSiluWeights, ModelError> {
        let prefix = format!("model.{idx}");
        self.load_conv_bn_block(&prefix)
    }

    /// Load Conv+BN+SiLU weights for a C2f/SPPF sub-conv (cv1 or cv2).
    ///
    /// Expects tensor names: `model.{idx}.{sub}.conv.weight`, `model.{idx}.{sub}.bn.*`
    pub fn sub_conv_bn_silu(&self, idx: usize, sub: &str) -> Result<ConvBnSiluWeights, ModelError> {
        let prefix = format!("model.{idx}.{sub}");
        self.load_conv_bn_block(&prefix)
    }

    /// Load Conv+BN+SiLU weights for a C2f bottleneck conv.
    ///
    /// E.g., bottleneck j, first conv: `model.{idx}.m.{j}.cv1`
    pub fn bottleneck_conv_bn_silu(
        &self,
        idx: usize,
        bottleneck: usize,
        cv: &str,
    ) -> Result<ConvBnSiluWeights, ModelError> {
        let prefix = format!("model.{idx}.m.{bottleneck}.{cv}");
        self.load_conv_bn_block(&prefix)
    }

    /// Load bare Conv2d weights for detect head final projection.
    ///
    /// E.g., box branch scale 0 final: `model.22.cv2.0.2`
    pub fn detect_conv(
        &self,
        branch: &str,
        scale: usize,
        sub: usize,
    ) -> Result<ConvWeights, ModelError> {
        // Final layers (sub=2) use bare Conv2d: `model.22.cv2.0.2.weight` (no .conv prefix).
        // Intermediate layers (sub=0,1) use Conv+BN: `model.22.cv2.0.0.conv.weight`.
        let (w_name, b_name) = if sub == 2 {
            (
                format!("model.22.{branch}.{scale}.{sub}.weight"),
                format!("model.22.{branch}.{scale}.{sub}.bias"),
            )
        } else {
            (
                format!("model.22.{branch}.{scale}.{sub}.conv.weight"),
                format!("model.22.{branch}.{scale}.{sub}.conv.bias"),
            )
        };

        let w = self.get(&w_name)?;
        let b = self.get(&b_name)?;

        Ok(ConvWeights {
            weight: w.data.clone(),
            shape: w.shape.clone(),
            bias: b.data.clone(),
        })
    }

    /// Load detect head Conv+BN+SiLU sub-layer.
    pub fn detect_conv_bn_silu(
        &self,
        branch: &str,
        scale: usize,
        sub: usize,
    ) -> Result<ConvBnSiluWeights, ModelError> {
        let prefix = format!("model.22.{branch}.{scale}.{sub}");
        self.load_conv_bn_block(&prefix)
    }

    /// Internal helper: load Conv+BN weights from prefix.
    fn load_conv_bn_block(&self, prefix: &str) -> Result<ConvBnSiluWeights, ModelError> {
        let conv = self.get(&format!("{prefix}.conv.weight"))?;
        let bn_weight = self.get_data(&format!("{prefix}.bn.weight"))?;
        let bn_bias = self.get_data(&format!("{prefix}.bn.bias"))?;
        let bn_running_mean = self.get_data(&format!("{prefix}.bn.running_mean"))?;
        let bn_running_var = self.get_data(&format!("{prefix}.bn.running_var"))?;

        Ok(ConvBnSiluWeights {
            conv_weight: conv.data.clone(),
            conv_shape: conv.shape.clone(),
            bn_weight: bn_weight.to_vec(),
            bn_bias: bn_bias.to_vec(),
            bn_running_mean: bn_running_mean.to_vec(),
            bn_running_var: bn_running_var.to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// YOLO weight loader
// ---------------------------------------------------------------------------

/// Load YOLOv8-nano weights from a safetensors file.
///
/// Uses the generic `load_all_tensors` to load all tensors, then wraps them
/// in a `YoloWeights` struct that provides typed accessors.
///
/// The safetensors file should be exported by `scripts/export_yolo.py`.
pub fn load_yolo_weights(path: &Path) -> Result<YoloWeights, ModelError> {
    let tensors = crate::model_generic::load_safetensors_raw(path)?;
    println!("  Loaded {} tensors from {}", tensors.len(), path.display());

    // Sanity check: verify a few expected tensors exist
    let expected = [
        "model.0.conv.weight",
        "model.0.bn.weight",
        "model.22.cv2.0.0.conv.weight",
        "model.22.cv3.0.0.conv.weight",
    ];
    for name in &expected {
        if !tensors.contains_key(*name) {
            return Err(ModelError::MissingTensor(name.to_string()));
        }
    }

    Ok(YoloWeights { tensors })
}

// ---------------------------------------------------------------------------
// PPM Image I/O (minimal, no external dependency)
// ---------------------------------------------------------------------------

/// Error type for image I/O operations.
#[derive(Debug)]
pub enum ImageError {
    /// File I/O error.
    Io(std::io::Error),
    /// Invalid PPM format.
    InvalidFormat(String),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::Io(e) => write!(f, "Image I/O error: {e}"),
            ImageError::InvalidFormat(msg) => write!(f, "Invalid image format: {msg}"),
        }
    }
}

impl From<std::io::Error> for ImageError {
    fn from(e: std::io::Error) -> Self {
        ImageError::Io(e)
    }
}

/// An RGB image stored as CHW f32 tensor (values in [0, 1]).
#[derive(Debug)]
pub struct ImageCHW {
    /// Image data in CHW layout: [3, H, W], values normalized to [0, 1].
    pub data: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl ImageCHW {
    /// Create from raw RGB bytes in HWC layout, normalizing to [0, 1] CHW.
    pub fn from_rgb_hwc(bytes: &[u8], width: usize, height: usize) -> Self {
        assert_eq!(bytes.len(), width * height * 3);
        let mut data = vec![0.0f32; 3 * height * width];
        for y in 0..height {
            for x in 0..width {
                let src = (y * width + x) * 3;
                for c in 0..3 {
                    data[c * height * width + y * width + x] = bytes[src + c] as f32 / 255.0;
                }
            }
        }
        ImageCHW {
            data,
            width,
            height,
        }
    }

    /// Resize to target dimensions using nearest-neighbor interpolation.
    ///
    /// Simple and fast, no dependency. For production use bilinear.
    pub fn resize_nearest(&self, target_w: usize, target_h: usize) -> ImageCHW {
        let mut data = vec![0.0f32; 3 * target_h * target_w];
        for c in 0..3 {
            for ty in 0..target_h {
                let sy = ty * self.height / target_h;
                for tx in 0..target_w {
                    let sx = tx * self.width / target_w;
                    data[c * target_h * target_w + ty * target_w + tx] =
                        self.data[c * self.height * self.width + sy * self.width + sx];
                }
            }
        }
        ImageCHW {
            data,
            width: target_w,
            height: target_h,
        }
    }

    /// Letterbox resize: fit image into target size maintaining aspect ratio,
    /// padding with gray (0.5). Returns the resized image and the scale/offset
    /// for mapping detections back to original coordinates.
    pub fn letterbox(&self, target: usize) -> (ImageCHW, f32, usize, usize) {
        let scale = (target as f32 / self.width as f32).min(target as f32 / self.height as f32);
        let new_w = (self.width as f32 * scale) as usize;
        let new_h = (self.height as f32 * scale) as usize;
        let pad_x = (target - new_w) / 2;
        let pad_y = (target - new_h) / 2;

        // Resize to new_w x new_h
        let resized = self.resize_nearest(new_w, new_h);

        // Create padded image filled with 0.5 (gray)
        let mut data = vec![0.5f32; 3 * target * target];
        for c in 0..3 {
            for y in 0..new_h {
                for x in 0..new_w {
                    data[c * target * target + (y + pad_y) * target + (x + pad_x)] =
                        resized.data[c * new_h * new_w + y * new_w + x];
                }
            }
        }

        (
            ImageCHW {
                data,
                width: target,
                height: target,
            },
            scale,
            pad_x,
            pad_y,
        )
    }
}

/// Load a PPM (P6 binary) image file.
///
/// PPM is a trivial format: `P6\n<width> <height>\n<maxval>\n<raw RGB bytes>`.
/// No external dependency needed.
pub fn load_ppm(path: &Path) -> Result<ImageCHW, ImageError> {
    let bytes = fs::read(path)?;

    // Parse header
    // Skip to after "P6"
    if bytes.len() < 3 || &bytes[0..2] != b"P6" {
        return Err(ImageError::InvalidFormat("not a P6 PPM file".to_string()));
    }
    let mut pos = 2;

    // Parse width, height, maxval (skipping comments and whitespace)
    let mut values = Vec::new();
    while values.len() < 3 && pos < bytes.len() {
        // Skip whitespace
        while pos < bytes.len()
            && (bytes[pos] == b' '
                || bytes[pos] == b'\n'
                || bytes[pos] == b'\r'
                || bytes[pos] == b'\t')
        {
            pos += 1;
        }
        // Skip comments
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        // Parse number
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos > start {
            let s = std::str::from_utf8(&bytes[start..pos])
                .map_err(|e| ImageError::InvalidFormat(format!("invalid header: {e}")))?;
            let val: usize = s
                .parse()
                .map_err(|e| ImageError::InvalidFormat(format!("invalid number: {e}")))?;
            values.push(val);
        }
    }

    if values.len() < 3 {
        return Err(ImageError::InvalidFormat(
            "incomplete PPM header".to_string(),
        ));
    }

    let width = values[0];
    let height = values[1];
    let _maxval = values[2];

    // Skip the single whitespace after maxval
    pos += 1;

    let pixel_data = &bytes[pos..];
    let expected = width * height * 3;
    if pixel_data.len() < expected {
        return Err(ImageError::InvalidFormat(format!(
            "expected {} bytes of pixel data, got {}",
            expected,
            pixel_data.len()
        )));
    }

    Ok(ImageCHW::from_rgb_hwc(
        &pixel_data[..expected],
        width,
        height,
    ))
}

/// Load a raw f32 CHW tensor from a binary file.
///
/// Format: just raw little-endian f32 values, [3, H, W] layout.
/// Use `scripts/export_yolo.py --image` to convert images.
pub fn load_raw_f32_image(
    path: &Path,
    width: usize,
    height: usize,
) -> Result<ImageCHW, ImageError> {
    let bytes = fs::read(path)?;
    let expected = 3 * height * width * 4;
    if bytes.len() != expected {
        return Err(ImageError::InvalidFormat(format!(
            "expected {} bytes for {}x{}x3 f32, got {}",
            expected,
            width,
            height,
            bytes.len()
        )));
    }

    let mut data = Vec::with_capacity(3 * height * width);
    for i in 0..(3 * height * width) {
        let b: [u8; 4] = bytes[i * 4..(i + 1) * 4]
            .try_into()
            .expect("slice is 4 bytes");
        data.push(f32::from_le_bytes(b));
    }

    Ok(ImageCHW {
        data,
        width,
        height,
    })
}
