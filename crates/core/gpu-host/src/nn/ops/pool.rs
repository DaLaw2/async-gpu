//! Pooling operations: max_pool2d.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// 2D max pooling.
///
/// Input: `[C, H, W]` → output: `[C, H_out, W_out]`.
pub fn max_pool2d(
    input: &GpuTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    if input.ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "3D input [C, H, W]".to_string(),
            actual: format!("ndim={}", input.ndim()),
        });
    }

    let c = input.shape()[0];
    let h = input.shape()[1];
    let w = input.shape()[2];
    let h_out = (h + 2 * padding - kernel_size) / stride + 1;
    let w_out = (w + 2 * padding - kernel_size) / stride + 1;

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[c, h_out, w_out], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("maxpool2d")?;
    let total = (c * h_out * w_out) as u32;
    let config = KernelRegistry::config_1d(total);
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                c as u32,
                h as u32,
                w as u32,
                kernel_size as u32,
                stride as u32,
                padding as u32,
                h_out as u32,
                w_out as u32,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(output)
}
