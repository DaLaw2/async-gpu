//! GPT-2 model weight loader from safetensors files.
//!
//! Loads a GPT-2 small (124M params) safetensors file and provides structured
//! access to all tensors organized by layer. All weights are stored as f32
//! slices; downstream code is responsible for f16x2 packing and GPU upload.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of transformer layers in GPT-2 small.
pub const NUM_LAYERS: usize = 12;
/// Hidden dimension (n_embd) of GPT-2 small.
pub const HIDDEN_DIM: usize = 768;
/// Number of attention heads.
pub const NUM_HEADS: usize = 12;
/// FFN intermediate dimension (4 * n_embd).
pub const FFN_DIM: usize = 3072;
/// Vocabulary size.
pub const VOCAB_SIZE: usize = 50257;
/// Maximum sequence length (n_positions).
pub const MAX_SEQ_LEN: usize = 1024;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when loading a safetensors model file.
#[derive(Debug)]
pub enum ModelError {
    /// Failed to read the safetensors file from disk.
    Io(std::io::Error),
    /// Failed to parse the safetensors format.
    SafeTensors(safetensors::SafeTensorError),
    /// A required tensor was not found in the file.
    MissingTensor(String),
    /// A tensor has an unexpected dtype (expected F32).
    UnexpectedDtype {
        /// Tensor name.
        name: String,
        /// Expected dtype string.
        expected: &'static str,
        /// Actual dtype string.
        got: String,
    },
    /// A tensor has an unexpected shape.
    UnexpectedShape {
        /// Tensor name.
        name: String,
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions.
        got: Vec<usize>,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error reading safetensors file: {e}"),
            Self::SafeTensors(e) => write!(f, "safetensors parse error: {e}"),
            Self::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Self::UnexpectedDtype {
                name,
                expected,
                got,
            } => write!(f, "tensor {name}: expected dtype {expected}, got {got}"),
            Self::UnexpectedShape {
                name,
                expected,
                got,
            } => write!(f, "tensor {name}: expected shape {expected:?}, got {got:?}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::SafeTensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for ModelError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(e)
    }
}

// ---------------------------------------------------------------------------
// Tensor data structures
// ---------------------------------------------------------------------------

/// LayerNorm parameters (gamma and beta).
#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    /// Gamma (scale), shape \[hidden_dim\].
    pub weight: Vec<f32>,
    /// Beta (bias), shape \[hidden_dim\].
    pub bias: Vec<f32>,
}

/// Weights for a single transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights {
    /// Pre-attention LayerNorm.
    pub ln_1: LayerNormWeights,
    /// QKV projection weight, shape \[768, 2304\] (Conv1D layout).
    pub c_attn_weight: Vec<f32>,
    /// QKV projection bias, shape \[2304\].
    pub c_attn_bias: Vec<f32>,
    /// Attention output projection weight, shape \[768, 768\].
    pub c_proj_weight: Vec<f32>,
    /// Attention output projection bias, shape \[768\].
    pub c_proj_bias: Vec<f32>,
    /// Pre-FFN LayerNorm.
    pub ln_2: LayerNormWeights,
    /// FFN up-projection weight, shape \[768, 3072\].
    pub mlp_fc_weight: Vec<f32>,
    /// FFN up-projection bias, shape \[3072\].
    pub mlp_fc_bias: Vec<f32>,
    /// FFN down-projection weight, shape \[3072, 768\].
    pub mlp_proj_weight: Vec<f32>,
    /// FFN down-projection bias, shape \[768\].
    pub mlp_proj_bias: Vec<f32>,
}

/// All weights for GPT-2 small, organized by component.
#[derive(Debug, Clone)]
pub struct Gpt2Weights {
    /// Token embedding table, shape [50257, 768].
    pub wte: Vec<f32>,
    /// Positional embedding table, shape [1024, 768].
    pub wpe: Vec<f32>,
    /// Final LayerNorm (after all transformer layers).
    pub ln_f: LayerNormWeights,
    /// Transformer layers (12 layers for GPT-2 small).
    pub layers: Vec<TransformerLayerWeights>,
}

impl Gpt2Weights {
    /// Total number of f32 parameters across all tensors.
    pub fn total_params(&self) -> usize {
        let mut total = 0;
        total += self.wte.len();
        total += self.wpe.len();
        total += self.ln_f.weight.len() + self.ln_f.bias.len();
        for layer in &self.layers {
            total += layer.ln_1.weight.len() + layer.ln_1.bias.len();
            total += layer.c_attn_weight.len() + layer.c_attn_bias.len();
            total += layer.c_proj_weight.len() + layer.c_proj_bias.len();
            total += layer.ln_2.weight.len() + layer.ln_2.bias.len();
            total += layer.mlp_fc_weight.len() + layer.mlp_fc_bias.len();
            total += layer.mlp_proj_weight.len() + layer.mlp_proj_bias.len();
        }
        total
    }

    /// Total memory footprint of loaded weights in bytes (f32).
    pub fn memory_bytes(&self) -> usize {
        self.total_params() * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Safetensors metadata inspection
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Dtype as a string (e.g., "F32").
    pub dtype: String,
    /// Number of elements.
    pub num_elements: usize,
    /// Size in bytes.
    pub size_bytes: usize,
}

/// List all tensor metadata from a safetensors file without loading the full data.
pub fn list_tensors(path: &Path) -> Result<Vec<TensorInfo>, ModelError> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    let mut result = Vec::new();
    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let dtype = format!("{:?}", view.dtype());
        let num_elements = shape.iter().product::<usize>();
        let size_bytes = view.data().len();
        result.push(TensorInfo {
            name,
            shape,
            dtype,
            num_elements,
            size_bytes,
        });
    }
    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// Copy bytes into a Vec<f32>, handling potential unaligned data.
fn copy_bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len().is_multiple_of(4),
        "f32 data length must be a multiple of 4"
    );
    let count = data.len() / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bytes: [u8; 4] = data[i * 4..(i + 1) * 4]
            .try_into()
            .expect("slice is exactly 4 bytes");
        out.push(f32::from_le_bytes(bytes));
    }
    out
}

/// Extract a named tensor as Vec<f32>, validating dtype and shape.
fn extract_f32_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, ModelError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| ModelError::MissingTensor(name.to_string()))?;

    // Validate dtype
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(ModelError::UnexpectedDtype {
            name: name.to_string(),
            expected: "F32",
            got: format!("{:?}", view.dtype()),
        });
    }

    // Validate shape
    if view.shape() != expected_shape {
        return Err(ModelError::UnexpectedShape {
            name: name.to_string(),
            expected: expected_shape.to_vec(),
            got: view.shape().to_vec(),
        });
    }

    Ok(copy_bytes_to_f32_vec(view.data()))
}

/// Extract a named tensor as Vec<f32>, validating dtype only (flexible shape).
fn extract_f32_tensor_any_shape(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), ModelError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| ModelError::MissingTensor(name.to_string()))?;

    if view.dtype() != safetensors::Dtype::F32 {
        return Err(ModelError::UnexpectedDtype {
            name: name.to_string(),
            expected: "F32",
            got: format!("{:?}", view.dtype()),
        });
    }

    let shape = view.shape().to_vec();
    Ok((copy_bytes_to_f32_vec(view.data()), shape))
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

/// Load GPT-2 small weights from a safetensors file.
///
/// Reads the entire file into memory, parses all 148 tensors, and organizes
/// them into a `Gpt2Weights` struct. All weight data is stored as f32;
/// downstream code is responsible for f16 conversion and GPU upload.
///
/// # Errors
/// Returns `ModelError` if the file cannot be read, parsed, or if expected
/// tensors are missing or have unexpected dtypes/shapes.
pub fn load_gpt2_weights(path: &Path) -> Result<Gpt2Weights, ModelError> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Global tensors
    let wte = extract_f32_tensor(&tensors, "wte.weight", &[VOCAB_SIZE, HIDDEN_DIM])?;
    let wpe = extract_f32_tensor(&tensors, "wpe.weight", &[MAX_SEQ_LEN, HIDDEN_DIM])?;
    let ln_f = LayerNormWeights {
        weight: extract_f32_tensor(&tensors, "ln_f.weight", &[HIDDEN_DIM])?,
        bias: extract_f32_tensor(&tensors, "ln_f.bias", &[HIDDEN_DIM])?,
    };

    // Per-layer tensors
    let mut layers = Vec::with_capacity(NUM_LAYERS);
    for i in 0..NUM_LAYERS {
        let layer = TransformerLayerWeights {
            ln_1: LayerNormWeights {
                weight: extract_f32_tensor(&tensors, &format!("h.{i}.ln_1.weight"), &[HIDDEN_DIM])?,
                bias: extract_f32_tensor(&tensors, &format!("h.{i}.ln_1.bias"), &[HIDDEN_DIM])?,
            },
            c_attn_weight: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.attn.c_attn.weight"),
                &[HIDDEN_DIM, HIDDEN_DIM * 3],
            )?,
            c_attn_bias: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.attn.c_attn.bias"),
                &[HIDDEN_DIM * 3],
            )?,
            c_proj_weight: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.attn.c_proj.weight"),
                &[HIDDEN_DIM, HIDDEN_DIM],
            )?,
            c_proj_bias: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.attn.c_proj.bias"),
                &[HIDDEN_DIM],
            )?,
            ln_2: LayerNormWeights {
                weight: extract_f32_tensor(&tensors, &format!("h.{i}.ln_2.weight"), &[HIDDEN_DIM])?,
                bias: extract_f32_tensor(&tensors, &format!("h.{i}.ln_2.bias"), &[HIDDEN_DIM])?,
            },
            mlp_fc_weight: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.mlp.c_fc.weight"),
                &[HIDDEN_DIM, FFN_DIM],
            )?,
            mlp_fc_bias: extract_f32_tensor(&tensors, &format!("h.{i}.mlp.c_fc.bias"), &[FFN_DIM])?,
            mlp_proj_weight: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.mlp.c_proj.weight"),
                &[FFN_DIM, HIDDEN_DIM],
            )?,
            mlp_proj_bias: extract_f32_tensor(
                &tensors,
                &format!("h.{i}.mlp.c_proj.bias"),
                &[HIDDEN_DIM],
            )?,
        };
        layers.push(layer);
    }

    Ok(Gpt2Weights {
        wte,
        wpe,
        ln_f,
        layers,
    })
}

/// Load arbitrary tensors from a safetensors file into a name -> (data, shape) map.
///
/// This is a generic loader that does not assume GPT-2 structure. Useful for
/// debugging or loading non-standard safetensors files.
/// Tensor data: (f32 values, shape dimensions).
pub type TensorData = (Vec<f32>, Vec<usize>);

/// Load all tensors from a safetensors file into a name-indexed map.
pub fn load_all_tensors(path: &Path) -> Result<HashMap<String, TensorData>, ModelError> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    let mut result = HashMap::new();
    for (name, _) in tensors.tensors() {
        let (vec, shape) = extract_f32_tensor_any_shape(&tensors, &name)?;
        result.insert(name, (vec, shape));
    }
    Ok(result)
}
