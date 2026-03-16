//! Tensor pool — maps TensorId to GpuTensor for autograd graph.

use std::collections::HashMap;

use super::TensorId;
use crate::nn::tensor::GpuTensor;

/// Pool of GPU tensors keyed by [`TensorId`].
///
/// Holds intermediate activations during the forward pass and gradients
/// during the backward pass. Tensors are added via [`insert`] and looked
/// up via [`get`].
pub struct TensorPool {
    tensors: HashMap<TensorId, GpuTensor>,
}

impl TensorPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Insert a tensor into the pool.
    pub fn insert(&mut self, id: TensorId, tensor: GpuTensor) {
        self.tensors.insert(id, tensor);
    }

    /// Get a reference to a tensor by ID.
    pub fn get(&self, id: TensorId) -> Option<&GpuTensor> {
        self.tensors.get(&id)
    }

    /// Get a mutable reference to a tensor by ID.
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut GpuTensor> {
        self.tensors.get_mut(&id)
    }

    /// Remove a tensor from the pool (e.g., to free memory after backward).
    pub fn remove(&mut self, id: TensorId) -> Option<GpuTensor> {
        self.tensors.remove(&id)
    }

    /// Check if a tensor exists in the pool.
    pub fn contains(&self, id: TensorId) -> bool {
        self.tensors.contains_key(&id)
    }

    /// Number of tensors in the pool.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Clear all tensors from the pool.
    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_insert_get() {
        // Can't test with real GpuTensor without CUDA, but test the HashMap logic
        let pool = TensorPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }
}
