//! Generic SafeTensors model loader.
//!
//! Provides a declarative weight loading system that maps SafeTensors keys
//! to nn module weights via [`WeightMap`] entries.
//!
//! # Example
//!
//! ```no_run
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use gpu_host::model_generic::{load_safetensors, WeightMap, Transform};
//!
//! let weights = load_safetensors("model.safetensors")?;
//! // Access raw f32 data by SafeTensors key
//! let wte = weights.get("wte.weight").unwrap();
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::model::ModelError;

/// A loaded weight: f32 data + shape.
#[derive(Clone, Debug)]
pub struct TensorData {
    /// Raw f32 weight data.
    pub data: Vec<f32>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Transpose a 2D tensor: [rows, cols] → [cols, rows].
    pub fn transpose_2d(&self) -> Result<TensorData, ModelError> {
        if self.shape.len() != 2 {
            return Err(ModelError::MissingTensor(format!(
                "transpose_2d requires 2D, got {:?}",
                self.shape
            )));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Ok(TensorData {
            data: out,
            shape: vec![cols, rows],
        })
    }
}

/// Weight transformation to apply after loading.
#[derive(Clone, Debug)]
pub enum Transform {
    /// No transformation — use data as-is.
    None,
    /// Transpose 2D: [in, out] → [out, in] (for Conv1D → Linear conversion).
    Transpose2D,
}

/// Maps a SafeTensors key to a named weight with optional transform.
#[derive(Clone, Debug)]
pub struct WeightMapping {
    /// SafeTensors key (e.g., "h.0.attn.c_attn.weight").
    pub safetensors_key: String,
    /// Logical name for this weight (e.g., "block.0.attn.qkv.weight").
    pub name: String,
    /// Transform to apply after loading.
    pub transform: Transform,
}

/// A collection of weight mappings for a model.
pub type WeightMap = Vec<WeightMapping>;

/// Loaded model weights indexed by logical name.
pub struct LoadedWeights {
    weights: HashMap<String, TensorData>,
}

impl LoadedWeights {
    /// Get a weight by logical name.
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.weights.get(name)
    }

    /// Get a weight by logical name, returning an error if not found.
    pub fn require(&self, name: &str) -> Result<&TensorData, ModelError> {
        self.weights
            .get(name)
            .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
    }

    /// Number of weights loaded.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Total number of parameters across all weights.
    pub fn total_params(&self) -> usize {
        self.weights.values().map(|t| t.numel()).sum()
    }

    /// Iterate over all (name, tensor) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorData)> {
        self.weights.iter()
    }
}

/// Load a SafeTensors file and return raw tensors indexed by SafeTensors key.
///
/// This is the lowest-level loader — no weight mapping or transforms.
/// Returns `HashMap<String, TensorData>`.
pub fn load_safetensors_raw(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, TensorData>, ModelError> {
    let bytes = std::fs::read(path.as_ref()).map_err(ModelError::Io)?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(ModelError::SafeTensors)?;

    let mut result = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != safetensors::Dtype::F32 {
            return Err(ModelError::UnexpectedDtype {
                name: name.to_string(),
                expected: "F32",
                got: format!("{:?}", view.dtype()),
            });
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let bytes = view.data();
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        result.insert(name.to_string(), TensorData { data, shape });
    }
    Ok(result)
}

/// Load a SafeTensors file with a weight map, applying transforms.
///
/// Each entry in `weight_map` specifies a SafeTensors key, a logical name,
/// and an optional transform. Returns [`LoadedWeights`] indexed by logical name.
pub fn load_safetensors_mapped(
    path: impl AsRef<Path>,
    weight_map: &[WeightMapping],
) -> Result<LoadedWeights, ModelError> {
    let raw = load_safetensors_raw(path)?;
    let mut weights = HashMap::new();

    for mapping in weight_map {
        let tensor = raw
            .get(&mapping.safetensors_key)
            .ok_or_else(|| ModelError::MissingTensor(mapping.safetensors_key.clone()))?;

        let transformed = match mapping.transform {
            Transform::None => tensor.clone(),
            Transform::Transpose2D => tensor.transpose_2d()?,
        };

        weights.insert(mapping.name.clone(), transformed);
    }

    Ok(LoadedWeights { weights })
}

/// Load a SafeTensors file and return all tensors indexed by their original key.
///
/// No weight mapping — just loads everything. Useful for models that
/// can figure out weight mapping from the key names directly.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<LoadedWeights, ModelError> {
    let raw = load_safetensors_raw(path)?;
    let weights = raw.into_iter().collect();
    Ok(LoadedWeights { weights })
}

// --- Helper functions for building weight maps ---

/// Create a weight mapping: safetensors key → logical name (no transform).
pub fn map_weight(st_key: impl Into<String>, name: impl Into<String>) -> WeightMapping {
    WeightMapping {
        safetensors_key: st_key.into(),
        name: name.into(),
        transform: Transform::None,
    }
}

/// Create a weight mapping with 2D transpose.
pub fn map_weight_t(st_key: impl Into<String>, name: impl Into<String>) -> WeightMapping {
    WeightMapping {
        safetensors_key: st_key.into(),
        name: name.into(),
        transform: Transform::Transpose2D,
    }
}

/// Generate GPT-2 weight map for the generic loader.
///
/// Maps SafeTensors keys to logical names, with Conv1D→Linear transposes.
pub fn gpt2_weight_map(num_layers: usize) -> WeightMap {
    let mut map = vec![
        map_weight("wte.weight", "wte.weight"),
        map_weight("wpe.weight", "wpe.weight"),
        map_weight("ln_f.weight", "ln_f.weight"),
        map_weight("ln_f.bias", "ln_f.bias"),
    ];

    for i in 0..num_layers {
        let prefix = format!("h.{i}");
        // LayerNorm (no transpose)
        map.push(map_weight(
            format!("{prefix}.ln_1.weight"),
            format!("{prefix}.ln_1.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.ln_1.bias"),
            format!("{prefix}.ln_1.bias"),
        ));
        map.push(map_weight(
            format!("{prefix}.ln_2.weight"),
            format!("{prefix}.ln_2.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.ln_2.bias"),
            format!("{prefix}.ln_2.bias"),
        ));
        // Attention (transpose Conv1D → Linear)
        map.push(map_weight_t(
            format!("{prefix}.attn.c_attn.weight"),
            format!("{prefix}.attn.c_attn.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.attn.c_attn.bias"),
            format!("{prefix}.attn.c_attn.bias"),
        ));
        map.push(map_weight_t(
            format!("{prefix}.attn.c_proj.weight"),
            format!("{prefix}.attn.c_proj.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.attn.c_proj.bias"),
            format!("{prefix}.attn.c_proj.bias"),
        ));
        // FFN (transpose Conv1D → Linear)
        map.push(map_weight_t(
            format!("{prefix}.mlp.c_fc.weight"),
            format!("{prefix}.mlp.c_fc.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.mlp.c_fc.bias"),
            format!("{prefix}.mlp.c_fc.bias"),
        ));
        map.push(map_weight_t(
            format!("{prefix}.mlp.c_proj.weight"),
            format!("{prefix}.mlp.c_proj.weight"),
        ));
        map.push(map_weight(
            format!("{prefix}.mlp.c_proj.bias"),
            format!("{prefix}.mlp.c_proj.bias"),
        ));
    }

    map
}
