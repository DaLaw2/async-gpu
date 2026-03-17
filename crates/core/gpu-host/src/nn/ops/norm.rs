//! Normalization operations: layer_norm, batch_norm, batch_norm_silu.

use std::sync::Arc;

use cudarc::driver::LaunchAsync;

use crate::nn::error::{NnError, Result};
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// Layer normalization over the last dimension.
///
/// Input: `[*, d_model]`, gamma/beta: `[d_model]` → output: same shape as input.
pub fn layer_norm(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let ndim = input.ndim();
    let d_model = input.shape()[ndim - 1];
    let num_rows = input.numel() / d_model;

    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    // V2 LayerNorm (256 threads, single-pass Welford, coalesced access)
    let func = registry.get("layer_norm_v2")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 2048,
    };
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                gamma.data(),
                beta.data(),
                d_model as u32,
                eps,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    // Record on autograd tape
    if input.requires_grad() {
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            let in_id = input
                .tensor_id()
                .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::LayerNorm,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![in_id], // save input for backward
                meta: crate::nn::autograd::OpMeta::LayerNorm {
                    rows: num_rows,
                    d: d_model,
                    eps,
                },
            });
        }
    }

    Ok(output)
}

/// Fused layer normalization + residual add: output = LN(input + residual).
///
/// Saves 1 kernel launch and 1 extra read of the input tensor vs separate ops.
/// Uses NVRTC-compiled CUDA C kernel for proper float4 loads.
#[cfg(feature = "cublas")]
pub fn layer_norm_residual(
    input: &GpuTensor,
    residual: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx;

    let ndim = input.ndim();
    let d_model = input.shape()[ndim - 1];
    let num_rows = input.numel() / d_model;
    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    static LN_RESIDUAL_SRC: &str = r#"
// Fused LayerNorm + residual add: output = LN(input + residual)
// Single pass for statistics, then normalize.
extern "C" __global__ void layer_norm_residual(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    unsigned int d_model,
    float eps
) {
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;

    extern __shared__ float smem[];

    const float* in_row = input + row * d_model;
    const float* res_row = residual + row * d_model;
    float* out_row = output + row * d_model;

    // Phase 1: sum and sq_sum of (input + residual)
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (unsigned int idx = tid; idx < d_model; idx += 256) {
        float x = in_row[idx] + res_row[idx];
        local_sum += x;
        local_sq_sum += x * x;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        local_sq_sum += __shfl_xor_sync(0xFFFFFFFF, local_sq_sum, offset);
    }

    // Block reduction via smem
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        smem[warp_id] = local_sum;
        smem[warp_id + 8] = local_sq_sum;
    }
    __syncthreads();

    float mean, inv_std;
    if (tid == 0) {
        float total_sum = 0.0f, total_sq = 0.0f;
        for (int w = 0; w < 8; w++) {
            total_sum += smem[w];
            total_sq += smem[w + 8];
        }
        mean = total_sum / d_model;
        float var = total_sq / d_model - mean * mean;
        inv_std = rsqrtf(var + eps);
        smem[16] = mean;
        smem[17] = inv_std;
    }
    __syncthreads();
    mean = smem[16];
    inv_std = smem[17];

    // Phase 2: normalize and write
    for (unsigned int idx = tid; idx < d_model; idx += 256) {
        float x = in_row[idx] + res_row[idx];  // re-compute add (cheaper than extra read)
        float g = gamma[idx];
        float b = beta[idx];
        out_row[idx] = g * (x - mean) * inv_std + b;
    }
}
"#;

    // Cache compiled kernel
    use std::sync::OnceLock;
    static COMPILED: OnceLock<bool> = OnceLock::new();
    COMPILED.get_or_init(|| {
        let ptx = compile_ptx(LN_RESIDUAL_SRC).expect("NVRTC LN+residual compile failed");
        dev.load_ptx(ptx, "ln_res", &["layer_norm_residual"])
            .expect("LN+residual PTX load failed");
        true
    });

    let func = dev
        .get_func("ln_res", "layer_norm_residual")
        .ok_or(NnError::KernelNotFound {
            name: "layer_norm_residual",
        })?;

    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 2048,
    };

    unsafe {
        func.launch(
            config,
            (
                input.data(),
                residual.data(),
                output.data_mut(),
                gamma.data(),
                beta.data(),
                d_model as u32,
                eps,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(output)
}

/// Batch normalization for CHW tensors.
///
/// Input: `[C, H, W]`, gamma/beta/mean/var: `[C]` → output: same shape.
pub fn batch_norm(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    running_mean: &GpuTensor,
    running_var: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    // Use batchnorm_silu kernel but we need a separate batchnorm-only kernel.
    // For now, fall back to host-side implementation.
    // TODO: add a dedicated batchnorm (no SiLU) kernel, or use batchnorm_silu
    // and apply inverse SiLU. For v1, compute on host.
    let inp = input.to_host()?;
    let g = gamma.to_host()?;
    let b = beta.to_host()?;
    let mean = running_mean.to_host()?;
    let var = running_var.to_host()?;

    let c = input.shape()[0];
    let hw: usize = input.shape()[1..].iter().product();
    let mut out = vec![0.0f32; inp.len()];
    let mut inv_stds = vec![0.0f32; c];
    let mut x_norm = vec![0.0f32; inp.len()];

    for ch in 0..c {
        let is = 1.0 / (var[ch] + eps).sqrt();
        inv_stds[ch] = is;
        for i in 0..hw {
            let idx = ch * hw + i;
            let xn = (inp[idx] - mean[ch]) * is;
            x_norm[idx] = xn;
            out[idx] = g[ch] * xn + b[ch];
        }
    }

    let dev = registry.device();
    let mut output = GpuTensor::from_host(&out, input.shape(), dev)?;

    // Record on autograd tape
    if input.requires_grad() {
        let in_id = input
            .tensor_id()
            .unwrap_or(crate::nn::autograd::TensorId(u32::MAX));
        if let Some(out_id) = crate::nn::autograd::alloc_tensor_id() {
            output.set_tensor_id(out_id);
            output.set_requires_grad(true);
            crate::nn::autograd::record_op(crate::nn::autograd::TapeEntry {
                op: crate::nn::autograd::OpKind::BatchNorm,
                inputs: vec![in_id],
                output: out_id,
                saved: vec![],
                meta: crate::nn::autograd::OpMeta::BatchNorm {
                    channels: c,
                    hw,
                    gamma: g.clone(),
                    inv_std: inv_stds,
                    x_norm,
                },
            });
        }
    }

    Ok(output)
}

/// Fused BatchNorm + SiLU for CHW tensors.
///
/// Input: `[C, H, W]`, gamma/beta/mean/var: `[C]` → output: same shape.
/// Uses the fused `batchnorm_silu` GPU kernel.
pub fn batch_norm_silu(
    input: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    running_mean: &GpuTensor,
    running_var: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    let _c = input.shape()[0];
    let hw: usize = input.shape()[1..].iter().product();
    let n = input.numel();

    let dev = registry.device();
    let mut output = GpuTensor::zeros(input.shape(), dev)?;

    let status_dev = dev.htod_sync_copy(&[0u32])?;

    let func = registry.get("batchnorm_silu")?;
    let config = KernelRegistry::config_1d(n as u32);
    unsafe {
        func.launch(
            config,
            (
                input.data(),
                output.data_mut(),
                gamma.data(),
                beta.data(),
                running_mean.data(),
                running_var.data(),
                n as u32,
                hw as u32,
                eps,
                &status_dev,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok(output)
}
