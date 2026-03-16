//! Matrix multiplication via `gemm_f32` kernel.
//!
//! The kernel expects A=[M,K] row-major, B=[K,N] column-major, output D=[M,N] row-major.
//! M and K must be padded to multiples of 32 and 16 respectively.
//! N must be padded to a multiple of 16.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Matrix multiplication: C = A * B.
///
/// A: `[M, K]`, B: `[K, N]` → output: `[M, N]`.
///
/// Handles padding to tile boundaries internally (M→32, K→16, N→16).
/// B is transposed to column-major internally.
pub fn matmul(a: &GpuTensor, b: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            actual: format!("a.ndim={}, b.ndim={}", a.ndim(), b.ndim()),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    if k != k2 {
        return Err(NnError::ShapeMismatch {
            expected: format!("a.shape[1]={k} == b.shape[0]"),
            actual: format!("b.shape[0]={k2}"),
        });
    }

    // Pad dimensions to tile boundaries
    let m_pad = m.div_ceil(32) * 32;
    let k_pad = k.div_ceil(16) * 16;
    let n_pad = n.div_ceil(16) * 16;

    let dev = registry.device();

    // Pad A to [m_pad, k_pad] row-major
    let a_host = a.to_host()?;
    let mut a_padded = vec![0.0f32; m_pad * k_pad];
    for r in 0..m {
        for c in 0..k {
            a_padded[r * k_pad + c] = a_host[r * k + c];
        }
    }
    let a_dev = dev.htod_sync_copy(&a_padded)?;

    // B is [K, N] row-major on host. Kernel expects column-major: b_cm[col * K_pad + row]
    let b_host = b.to_host()?;
    let mut b_cm = vec![0.0f32; k_pad * n_pad];
    for r in 0..k {
        for c in 0..n {
            b_cm[c * k_pad + r] = b_host[r * n + c];
        }
    }
    let b_dev = dev.htod_sync_copy(&b_cm)?;

    // Allocate output [m_pad, n_pad]
    let mut d_dev = dev.alloc_zeros::<f32>(m_pad * n_pad)?;

    // Allocate status
    let status_dev = dev.htod_sync_copy(&[0u32])?;

    // Launch gemm_f32
    let func = registry.get("gemm_f32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: ((m_pad as u32) / 32, (n_pad as u32) / 16, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 3072,
    };
    unsafe {
        func.launch(
            config,
            (
                &a_dev,
                &b_dev,
                &mut d_dev,
                k_pad as u32,
                n_pad as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    // Extract unpadded result [M, N]
    let d_host = dev.dtoh_sync_copy(&d_dev)?;
    let mut result = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            result[r * n + c] = d_host[r * n_pad + c];
        }
    }

    let mut output = GpuTensor::from_host(&result, &[m, n], dev)?;

    // Record on autograd tape if inputs require gradients
    if a.requires_grad() || b.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::Matmul,
                inputs: vec![
                    a.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    b.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                ],
                output: out_id,
                saved: vec![
                    a.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                    b.tensor_id()
                        .unwrap_or(crate::nn::autograd::TensorId(u32::MAX)),
                ],
                meta: crate::nn::autograd::OpMeta::Matmul { m, k, n },
            });
        }
    }

    Ok(output)
}
