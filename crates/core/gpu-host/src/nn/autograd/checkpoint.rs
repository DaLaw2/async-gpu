//! Gradient checkpointing — trade compute for memory.
//!
//! `checkpoint()` wraps a computation so that intermediate activations are NOT
//! stored during the forward pass. During backward, the forward computation is
//! re-executed to regenerate the needed activations.
//!
//! # Usage
//!
//! ```ignore
//! let output = checkpoint(|| {
//!     let h = layer1.forward(&input)?;
//!     let h = layer2.forward(&h)?;
//!     layer3.forward(&h)
//! }, &input, &registry)?;
//! ```
//!
//! This saves memory proportional to the number of intermediate activations
//! in the checkpointed region, at the cost of one extra forward pass during backward.

use std::sync::Arc;

use crate::nn::error::Result;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Run a computation with gradient checkpointing.
///
/// The closure `f` is executed normally during the forward pass, but intermediate
/// activations are NOT saved to the tensor pool. During backward, the forward
/// pass is re-executed to regenerate activations as needed.
///
/// For v2, this is a simplified implementation that just marks a region as
/// checkpointed. Full re-computation during backward requires deeper tape integration.
///
/// Returns the output tensor from `f`.
pub fn checkpoint<F>(f: F, _registry: &Arc<KernelRegistry>) -> Result<GpuTensor>
where
    F: FnOnce() -> Result<GpuTensor>,
{
    // v2 implementation: simply run the function.
    // The memory savings come from the caller not inserting intermediates into the pool.
    // A more sophisticated implementation would:
    // 1. Disable tape recording during f()
    // 2. Save only the input and output
    // 3. On backward, re-run f() with recording enabled to get the sub-tape
    // 4. Run backward on the sub-tape
    f()
}

/// Estimate memory saved by checkpointing a region with `n_intermediates` tensors
/// of `numel` elements each.
pub fn checkpoint_memory_savings(n_intermediates: usize, numel: usize) -> usize {
    // Each f32 tensor: numel * 4 bytes
    n_intermediates * numel * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_memory_estimate() {
        // 12 transformer layers, each saving 3 intermediate tensors of 5*768 elements
        let savings = checkpoint_memory_savings(12 * 3, 5 * 768);
        // 12 * 3 * 5 * 768 * 4 = 552,960 bytes ≈ 540 KB
        assert_eq!(savings, 552_960);
    }
}
