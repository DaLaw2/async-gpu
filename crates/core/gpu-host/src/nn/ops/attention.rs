//! Attention operations: scaled dot-product attention via flash_attention kernel.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Scaled dot-product attention.
///
/// Q: `[seq_len, d_head]`, K: `[seq_len, d_head]`, V: `[seq_len, d_head]`
/// → output: `[seq_len, d_head]`.
///
/// Uses the `flash_attention` kernel with causal masking.
pub fn scaled_dot_product_attention(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    causal: bool,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors [seq_len, d_head]".to_string(),
            actual: format!(
                "q.ndim={}, k.ndim={}, v.ndim={}",
                q.ndim(),
                k.ndim(),
                v.ndim()
            ),
        });
    }

    let seq_len = q.shape()[0];
    let d_head = q.shape()[1];

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[seq_len, d_head], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("flash_attention")?;
    let config = KernelRegistry::config_attention(seq_len as u32);
    let causal_mask: u32 = if causal { 1 } else { 0 };
    unsafe {
        func.launch(
            config,
            (
                q.data(),
                k.data(),
                v.data(),
                output.data_mut(),
                seq_len as u32,
                d_head as u32,
                causal_mask,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(output)
}

/// Scaled dot-product attention with separate KV cache lengths.
///
/// Q: `[q_len, d_head]`, K: `[kv_len, d_head]`, V: `[kv_len, d_head]`
/// → output: `[q_len, d_head]`.
///
/// Uses `flash_attention_kv` kernel for incremental decoding.
pub fn scaled_dot_product_attention_kv(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    causal: bool,
    q_offset: usize,
    kv_stride: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let q_len = q.shape()[0];
    let kv_len = k.shape()[0];
    let d_head = q.shape()[1];

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[q_len, d_head], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("flash_attention_kv")?;
    let config = KernelRegistry::config_attention(q_len as u32);
    let causal_mask: u32 = if causal { 1 } else { 0 };
    unsafe {
        func.launch(
            config,
            (
                q.data(),
                k.data(),
                v.data(),
                output.data_mut(),
                q_len as u32,
                kv_len as u32,
                d_head as u32,
                causal_mask,
                q_offset as u32,
                kv_stride as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(output)
}
