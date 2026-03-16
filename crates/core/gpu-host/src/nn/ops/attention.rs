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
    // flash_attention kernel: grid = (1, n_q_tiles, 1), block = (32, 1, 1)
    // One warp per query tile. Single head (MHA splits externally).
    let n_q_tiles = seq_len.div_ceil(32) as u32;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (1, n_q_tiles, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2 * 32 * d_head as u32 * 4, // K tile + V tile
    };
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

    // Record on autograd tape
    if q.requires_grad() || k.requires_grad() || v.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            let q_id = q
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            let k_id = k
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            let v_id = v
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::Attention,
                inputs: vec![q_id, k_id, v_id],
                output: out_id,
                saved: vec![q_id, k_id, v_id],
                meta: crate::nn::autograd::OpMeta::Attention {
                    seq: seq_len,
                    d: d_head,
                    causal,
                },
            });
        }
    }

    Ok(output)
}

/// Split QKV from `[seq, 3*d_model]` into Q, K, V as `[n_heads, seq, d_head]` on GPU.
///
/// Uses the `split_qkv` kernel — zero host transfers.
pub fn split_qkv(
    qkv: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
    let dev = registry.device();
    let head_total = n_heads * seq_len * d_head;

    let mut q = GpuTensor::zeros(&[n_heads * seq_len, d_head], dev)?;
    let mut k = GpuTensor::zeros(&[n_heads * seq_len, d_head], dev)?;
    let mut v = GpuTensor::zeros(&[n_heads * seq_len, d_head], dev)?;

    let func = registry.get("split_qkv")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            config,
            (
                qkv.data(),
                q.data_mut(),
                k.data_mut(),
                v.data_mut(),
                seq_len as u32,
                n_heads as u32,
                d_head as u32,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok((q, k, v))
}

/// Multi-head flash attention — all heads in one kernel launch.
///
/// Q, K, V: `[n_heads * seq_len, d_head]` (head-major layout from split_qkv).
/// Output: `[n_heads * seq_len, d_head]`.
///
/// Uses `flash_attention` with grid=(n_heads, n_q_tiles, 1).
#[allow(clippy::too_many_arguments)]
pub fn multi_head_flash_attention(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    causal: bool,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let dev = registry.device();
    let total = n_heads * seq_len * d_head;
    let mut output = GpuTensor::zeros(&[n_heads * seq_len, d_head], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32]).map_err(NnError::Cuda)?;

    let func = registry.get("flash_attention")?;
    let n_q_tiles = seq_len.div_ceil(32) as u32;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (n_heads as u32, n_q_tiles, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2 * 32 * d_head as u32 * 4,
    };
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

    // Reshape output metadata (data unchanged)
    let _ = total; // suppress unused warning
    Ok(output)
}

/// Concat attention heads from `[n_heads, seq, d_head]` → `[seq, n_heads * d_head]` on GPU.
///
/// Uses the `concat_heads` kernel — zero host transfers.
pub fn concat_heads(
    attn_out: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let dev = registry.device();
    let d_model = n_heads * d_head;
    let total = seq_len * d_model;

    let mut output = GpuTensor::zeros(&[seq_len, d_model], dev)?;

    let func = registry.get("concat_heads")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            config,
            (
                attn_out.data(),
                output.data_mut(),
                seq_len as u32,
                n_heads as u32,
                d_head as u32,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

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
    // flash_attention_kv: grid = (1, n_q_tiles, 1), block = (32, 1, 1)
    // Single head (MHA splits externally).
    let n_q_tiles = q_len.div_ceil(32) as u32;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (1, n_q_tiles, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2 * 32 * d_head as u32 * 4,
    };
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

    Ok(output)
}
