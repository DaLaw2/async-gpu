//! Upsampling operations.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Nearest-neighbor 2x upsampling.
///
/// Input: `[C, H, W]` → output: `[C, 2*H, 2*W]`.
pub fn upsample_nearest_2x(input: &GpuTensor, registry: &Arc<KernelRegistry>) -> Result<GpuTensor> {
    if input.ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "3D input [C, H, W]".to_string(),
            actual: format!("ndim={}", input.ndim()),
        });
    }

    let c = input.shape()[0];
    let h = input.shape()[1];
    let w = input.shape()[2];

    let dev = registry.device();
    let mut output = GpuTensor::zeros(&[c, h * 2, w * 2], dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("upsample_nearest_2x")?;
    let total = (c * h * 2 * w * 2) as u32;
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
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }
    dev.synchronize().map_err(NnError::Cuda)?;

    Ok(output)
}
