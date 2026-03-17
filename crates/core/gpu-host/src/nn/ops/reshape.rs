//! Reshape and channel operations: concat_channels, split_channels, bias_add, elementwise_add.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Concatenate two tensors along the channel dimension (dim 0).
///
/// A: `[C_a, H, W]`, B: `[C_b, H, W]` → output: `[C_a + C_b, H, W]`.
///
/// Uses the `concat_channels` GPU kernel.
pub fn concat_channels(
    a: &GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if a.ndim() != 3 || b.ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "3D tensors [C, H, W]".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let hw_a = a.shape()[1] * a.shape()[2];
    let hw_b = b.shape()[1] * b.shape()[2];
    if hw_a != hw_b {
        return Err(NnError::ShapeMismatch {
            expected: format!("same spatial dims, a has H*W={hw_a}"),
            actual: format!("b has H*W={hw_b}"),
        });
    }

    let c_a = a.shape()[0];
    let c_b = b.shape()[0];
    let h = a.shape()[1];
    let w = a.shape()[2];
    let hw = h * w;

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[c_a + c_b, h, w], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("concat_channels")?;
    let total = ((c_a + c_b) * hw) as u32;
    let config = KernelRegistry::config_1d(total);
    unsafe {
        func.launch(
            config,
            (
                a.data(),
                b.data(),
                output.data_mut(),
                c_a as u32,
                c_b as u32,
                hw as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(output)
}

/// Add bias per channel to a CHW tensor.
///
/// Input: `[C, H, W]`, bias: `[C]` → output: `[C, H, W]`.
///
/// Uses the `bias_add_chw` GPU kernel.
pub fn bias_add_chw(
    input: &GpuTensor,
    bias: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let n = input.numel();
    let hw: usize = input.shape()[1..].iter().product();

    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("bias_add_chw")?;
    let config = KernelRegistry::config_1d(n as u32);
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                bias.data(),
                n as u32,
                hw as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(output)
}

/// Add bias to a 2D tensor (per-column).
///
/// Input: `[rows, cols]`, bias: `[cols]` → modifies input in place.
///
/// Uses the `bias_add` GPU kernel.
pub fn bias_add(
    input: &mut GpuTensor,
    bias: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<()> {
    let n_cols = input.shape()[input.ndim() - 1];
    let total = input.numel();

    let status_dev = registry.device().htod_sync_copy(&[0u32])?;

    let func = registry.get("bias_add")?;
    let config = KernelRegistry::config_1d(total as u32);
    unsafe {
        func.launch(
            config,
            (
                input.data_mut(),
                bias.data(),
                n_cols as u32,
                total as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    // Record on autograd tape
    if input.requires_grad() {
        let old_id = input
            .tensor_id()
            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            input.set_tensor_id(out_id);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::BiasAdd,
                inputs: vec![old_id],
                output: out_id,
                saved: vec![],
                meta: crate::nn::autograd::OpMeta::BiasAdd { n_cols },
            });
        }
    }

    Ok(())
}

/// Element-wise addition: a += b (in-place).
///
/// Uses the `elementwise_add` GPU kernel.
pub fn elementwise_add(
    a: &mut GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<()> {
    if a.numel() != b.numel() {
        return Err(NnError::ShapeMismatch {
            expected: format!("same numel, a has {}", a.numel()),
            actual: format!("b has {}", b.numel()),
        });
    }

    let n = a.numel();

    // V3: explicit PTX float4 loads for maximum bandwidth
    let func = registry.get("elementwise_add_v3")?;
    let grid = ((n as u32 + 1023) / 1024, 1, 1);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(config, (a.data_mut(), b.data(), n as u32))
            .map_err(NnError::Cuda)?;
    }

    // Record on autograd tape
    if a.requires_grad() || b.requires_grad() {
        // Capture input IDs BEFORE overwriting with output ID
        let old_a_id = a
            .tensor_id()
            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
        let b_id = b
            .tensor_id()
            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            a.set_tensor_id(out_id);
            a.set_requires_grad(true);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::ElemAdd,
                inputs: vec![old_a_id, b_id],
                output: out_id,
                saved: vec![],
                meta: crate::nn::autograd::OpMeta::None,
            });
        }
    }

    Ok(())
}

/// Embedding lookup: wte[token_ids] + wpe[positions].
///
/// wte: `[vocab_size, d_model]`, wpe: `[max_seq, d_model]`, token_ids: device buffer of u32.
/// Output: `[seq_len, d_model]`.
///
/// Uses the `embedding_lookup` GPU kernel.
pub fn embedding_lookup(
    wte: &GpuTensor,
    wpe: &GpuTensor,
    token_ids: &cudarc::driver::CudaSlice<u32>,
    seq_len: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let d_model = wte.shape()[1];

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[seq_len, d_model], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("embedding_lookup")?;
    let total_elements = (seq_len * d_model) as u32;
    let config = KernelRegistry::config_embedding(total_elements);
    unsafe {
        func.launch(
            config,
            (
                wte.data(),
                wpe.data(),
                token_ids,
                output.data_mut(),
                seq_len as u32,
                d_model as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(output)
}
