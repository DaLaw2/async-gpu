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

    // Use v3 (float4 vectorized) when d_model is divisible by 4, else fall back to v2
    let kernel_name = if d_model % 4 == 0 {
        "layer_norm_v3"
    } else {
        "layer_norm_v2"
    };
    let func = registry.get(kernel_name)?;
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
// Uses float4 vectorized loads when d_model % 4 == 0.
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

    const float4* in_row  = (const float4*)(input    + row * d_model);
    const float4* res_row = (const float4*)(residual + row * d_model);
    float4*       out_row = (float4*)(output + row * d_model);
    const float4* g4      = (const float4*)gamma;
    const float4* b4      = (const float4*)beta;

    unsigned int d_model_v4 = d_model / 4;

    // Phase 1: sum and sq_sum of (input + residual) with float4 loads
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (unsigned int v = tid; v < d_model_v4; v += 256) {
        float4 iv = in_row[v];
        float4 rv = res_row[v];
        float4 sv;
        sv.x = iv.x + rv.x;
        sv.y = iv.y + rv.y;
        sv.z = iv.z + rv.z;
        sv.w = iv.w + rv.w;
        local_sum += sv.x + sv.y + sv.z + sv.w;
        local_sq_sum += sv.x*sv.x + sv.y*sv.y + sv.z*sv.z + sv.w*sv.w;
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

    // Phase 2: normalize and write with float4 loads/stores
    for (unsigned int v = tid; v < d_model_v4; v += 256) {
        float4 iv = in_row[v];
        float4 rv = res_row[v];
        float4 gv = g4[v];
        float4 bv = b4[v];
        float4 result;
        float sx = iv.x + rv.x;
        float sy = iv.y + rv.y;
        float sz = iv.z + rv.z;
        float sw = iv.w + rv.w;
        result.x = gv.x * (sx - mean) * inv_std + bv.x;
        result.y = gv.y * (sy - mean) * inv_std + bv.y;
        result.z = gv.z * (sz - mean) * inv_std + bv.z;
        result.w = gv.w * (sw - mean) * inv_std + bv.w;
        out_row[v] = result;
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

/// Fused layer normalization + residual add with dual output:
/// `norm_out = LN(input + residual)` and `sum_out = input + residual`.
///
/// Returns `(norm_out, sum_out)`. Saves 1 kernel launch vs separate
/// `elementwise_add` + `layer_norm` while preserving the un-normalized sum
/// for downstream residual connections (e.g., GPT-2 transformer blocks).
/// Uses NVRTC-compiled CUDA C kernel with float4 vectorized loads.
/// Requires `d_model % 4 == 0`.
#[cfg(feature = "cublas")]
pub fn layer_norm_residual_dual(
    input: &GpuTensor,
    residual: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    eps: f32,
    registry: &Arc<KernelRegistry>,
) -> Result<(GpuTensor, GpuTensor)> {
    use cudarc::nvrtc::compile_ptx;

    let ndim = input.ndim();
    let d_model = input.shape()[ndim - 1];
    let num_rows = input.numel() / d_model;
    let dev = registry.device();
    let mut norm_out = GpuTensor::zeros(input.shape(), dev)?;
    let mut sum_out = GpuTensor::zeros(input.shape(), dev)?;

    assert!(
        d_model % 4 == 0,
        "layer_norm_residual_dual requires d_model divisible by 4, got {d_model}"
    );

    static LN_RESIDUAL_DUAL_SRC: &str = r#"
// Fused LayerNorm + residual add with dual output and float4 vectorized loads.
// norm_out = LN(input + residual), sum_out = input + residual.
// d_model must be divisible by 4.
extern "C" __global__ void layer_norm_residual_dual(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ norm_out,
    float* __restrict__ sum_out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    unsigned int d_model,
    float eps
) {
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;

    extern __shared__ float smem[];

    const float4* in_row  = (const float4*)(input    + row * d_model);
    const float4* res_row = (const float4*)(residual + row * d_model);
    float4*       nrm_row = (float4*)(norm_out + row * d_model);
    float4*       sum_row = (float4*)(sum_out  + row * d_model);
    const float4* g4      = (const float4*)gamma;
    const float4* b4      = (const float4*)beta;

    unsigned int d_model_v4 = d_model / 4;

    // Phase 1: compute sums, write sum_out, accumulate statistics
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (unsigned int v = tid; v < d_model_v4; v += 256) {
        float4 iv = in_row[v];
        float4 rv = res_row[v];
        float4 sv;
        sv.x = iv.x + rv.x;
        sv.y = iv.y + rv.y;
        sv.z = iv.z + rv.z;
        sv.w = iv.w + rv.w;
        sum_row[v] = sv;  // write un-normalized sum
        local_sum += sv.x + sv.y + sv.z + sv.w;
        local_sq_sum += sv.x*sv.x + sv.y*sv.y + sv.z*sv.z + sv.w*sv.w;
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

    // Phase 2: normalize and write norm_out (read sum_out back from global)
    for (unsigned int v = tid; v < d_model_v4; v += 256) {
        float4 sv = sum_row[v];  // read back from sum_out
        float4 gv = g4[v];
        float4 bv = b4[v];
        float4 result;
        result.x = gv.x * (sv.x - mean) * inv_std + bv.x;
        result.y = gv.y * (sv.y - mean) * inv_std + bv.y;
        result.z = gv.z * (sv.z - mean) * inv_std + bv.z;
        result.w = gv.w * (sv.w - mean) * inv_std + bv.w;
        nrm_row[v] = result;
    }
}
"#;

    use std::sync::OnceLock;
    static COMPILED_DUAL: OnceLock<bool> = OnceLock::new();
    COMPILED_DUAL.get_or_init(|| {
        let ptx = compile_ptx(LN_RESIDUAL_DUAL_SRC).expect("NVRTC LN+residual dual compile failed");
        dev.load_ptx(ptx, "ln_res_dual", &["layer_norm_residual_dual"])
            .expect("LN+residual dual PTX load failed");
        true
    });

    let func = dev
        .get_func("ln_res_dual", "layer_norm_residual_dual")
        .ok_or(NnError::KernelNotFound {
            name: "layer_norm_residual_dual",
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
                norm_out.data_mut(),
                sum_out.data_mut(),
                gamma.data(),
                beta.data(),
                d_model as u32,
                eps,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

    Ok((norm_out, sum_out))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// LayerNorm v3 correctness check — verify float4 vectorized kernel matches CPU reference.
    #[test]
    fn test_layer_norm_v3_correctness() {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry =
            Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"));

        let seq_len = 4;
        let d_model = 768;
        let n = seq_len * d_model;
        let input_data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let gamma_data: Vec<f32> = (0..d_model).map(|i| 0.9 + (i % 17) as f32 * 0.01).collect();
        let beta_data: Vec<f32> = (0..d_model).map(|i| (i % 13) as f32 * 0.001).collect();
        let eps = 1e-5f32;

        let input_t = GpuTensor::from_host(&input_data, &[seq_len, d_model], &dev).expect("input");
        let gamma_t = GpuTensor::from_host(&gamma_data, &[d_model], &dev).expect("gamma");
        let beta_t = GpuTensor::from_host(&beta_data, &[d_model], &dev).expect("beta");

        eprintln!("  Running layer_norm (should use v3 for d_model=768)...");
        let out_t = layer_norm(&input_t, &gamma_t, &beta_t, eps, &registry).unwrap();
        dev.synchronize().unwrap();
        let out_host = out_t.to_host().unwrap();

        // CPU reference
        for row in 0..seq_len {
            let start = row * d_model;
            let row_data = &input_data[start..start + d_model];
            let mean: f32 = row_data.iter().sum::<f32>() / d_model as f32;
            let var: f32 = row_data
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>()
                / d_model as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for j in 0..d_model {
                let expected =
                    gamma_data[j] * (input_data[start + j] - mean) * inv_std + beta_data[j];
                let got = out_host[start + j];
                let diff = (expected - got).abs();
                assert!(
                    diff < 1e-3,
                    "row={row} j={j}: expected={expected} got={got} diff={diff}"
                );
            }
        }
        eprintln!("  Correctness: PASS (v3 float4 matches CPU reference)");
    }

    /// LayerNorm bandwidth benchmark — measures GB/s for standalone and fused variants.
    ///
    /// Target: >= 180 GB/s (60% of GTX 1660's 336 GB/s peak).
    ///
    /// Bandwidth formula for standalone LayerNorm:
    ///   reads: input (2 passes) + gamma + beta = (2*N + 2*d) * 4 bytes
    ///   writes: output = N * 4 bytes
    ///   total = (3*N + 2*d) * 4 bytes
    #[test]
    fn bench_layer_norm_bandwidth() {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry =
            Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"));

        let num_warmup = 5;
        let num_runs = 50;

        eprintln!("\n=== LayerNorm Bandwidth Benchmark ===");
        eprintln!("  Warmup: {num_warmup}, Runs: {num_runs}");
        eprintln!();

        // GPT-2 Small: 128 tokens, 768 hidden
        let seq_len = 128;
        let d_model = 768;
        let n = seq_len * d_model;

        let input_data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let gamma_data: Vec<f32> = (0..d_model).map(|i| 0.9 + (i % 17) as f32 * 0.01).collect();
        let beta_data: Vec<f32> = (0..d_model).map(|i| (i % 13) as f32 * 0.001).collect();
        let eps = 1e-5f32;

        let shape = &[seq_len, d_model];
        let g_shape = &[d_model];

        let input_t = GpuTensor::from_host(&input_data, shape, &dev).expect("input");
        let gamma_t = GpuTensor::from_host(&gamma_data, g_shape, &dev).expect("gamma");
        let beta_t = GpuTensor::from_host(&beta_data, g_shape, &dev).expect("beta");

        // Warmup
        for _ in 0..num_warmup {
            let _ = layer_norm(&input_t, &gamma_t, &beta_t, eps, &registry).unwrap();
            dev.synchronize().unwrap();
        }

        // Benchmark
        dev.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..num_runs {
            let _ = layer_norm(&input_t, &gamma_t, &beta_t, eps, &registry).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed_s = t0.elapsed().as_secs_f64() / num_runs as f64;

        // Bandwidth: input read (2x for 2 passes) + gamma + beta reads + output write
        let bytes_moved = ((3 * n + 2 * d_model) * 4) as f64;
        let gbps = bytes_moved / elapsed_s / 1e9;
        let us = elapsed_s * 1e6;

        eprintln!("  LN v3 (128x768):  {us:7.2} us | {gbps:6.1} GB/s");
        assert!(
            gbps > 50.0,
            "LayerNorm bandwidth {gbps:.1} GB/s is too low (expected > 50 GB/s)"
        );
    }

    /// Micro-benchmark: fused vs unfused LayerNorm + residual add with bandwidth.
    #[test]
    fn bench_fused_ln_residual_vs_unfused() {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA");
        let registry =
            Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX"));

        // GPT-2 Small dimensions: 128 tokens, 768 hidden
        let seq_len = 128;
        let d_model = 768;
        let n = seq_len * d_model;

        // Create test tensors
        let input_data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let residual_data: Vec<f32> = (0..n).map(|i| ((i % 83) as f32 - 41.0) * 0.01).collect();
        let gamma_data: Vec<f32> = (0..d_model).map(|i| 0.9 + (i % 17) as f32 * 0.01).collect();
        let beta_data: Vec<f32> = (0..d_model).map(|i| (i % 13) as f32 * 0.001).collect();
        let eps = 1e-5f32;

        let shape = &[seq_len, d_model];
        let g_shape = &[d_model];

        let input_t = GpuTensor::from_host(&input_data, shape, &dev).expect("input");
        let residual_t = GpuTensor::from_host(&residual_data, shape, &dev).expect("residual");
        let gamma_t = GpuTensor::from_host(&gamma_data, g_shape, &dev).expect("gamma");
        let beta_t = GpuTensor::from_host(&beta_data, g_shape, &dev).expect("beta");

        let num_warmup = 10;
        let num_runs = 200;

        // ---------- Unfused: elementwise_add + layer_norm ----------
        for _ in 0..num_warmup {
            let mut tmp = input_t.clone_tensor().unwrap();
            crate::nn::ops::elementwise_add(&mut tmp, &residual_t, &registry).unwrap();
            let _ = layer_norm(&tmp, &gamma_t, &beta_t, eps, &registry).unwrap();
            dev.synchronize().unwrap();
        }

        dev.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..num_runs {
            let mut tmp = input_t.clone_tensor().unwrap();
            crate::nn::ops::elementwise_add(&mut tmp, &residual_t, &registry).unwrap();
            let _ = layer_norm(&tmp, &gamma_t, &beta_t, eps, &registry).unwrap();
        }
        dev.synchronize().unwrap();
        let unfused_s = t0.elapsed().as_secs_f64() / num_runs as f64;
        let unfused_ms = unfused_s * 1000.0;

        // Bandwidth for unfused: add reads 2*N, writes N; LN reads 2*N+2*d, writes N
        let unfused_bytes = ((5 * n + 2 * d_model) * 4) as f64;
        let unfused_gbps = unfused_bytes / unfused_s / 1e9;

        eprintln!("\n=== LN+Residual Micro-Benchmark (seq={seq_len}, d={d_model}) ===");
        eprintln!("  Unfused (add + LN):      {unfused_ms:.4} ms | {unfused_gbps:.1} GB/s");

        // ---------- Fused: layer_norm_residual_dual ----------
        #[cfg(feature = "cublas")]
        {
            for _ in 0..num_warmup {
                let _ = layer_norm_residual_dual(
                    &input_t,
                    &residual_t,
                    &gamma_t,
                    &beta_t,
                    eps,
                    &registry,
                )
                .unwrap();
                dev.synchronize().unwrap();
            }

            dev.synchronize().unwrap();
            let t1 = std::time::Instant::now();
            for _ in 0..num_runs {
                let _ = layer_norm_residual_dual(
                    &input_t,
                    &residual_t,
                    &gamma_t,
                    &beta_t,
                    eps,
                    &registry,
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            let fused_s = t1.elapsed().as_secs_f64() / num_runs as f64;
            let fused_ms = fused_s * 1000.0;

            // Fused dual: reads input+residual+gamma+beta = (2*N+2*d)*4, writes norm+sum = 2*N*4
            let fused_bytes = ((4 * n + 2 * d_model) * 4) as f64;
            let fused_gbps = fused_bytes / fused_s / 1e9;

            let speedup = unfused_ms / fused_ms;
            eprintln!("  Fused (LN+res dual):     {fused_ms:.4} ms | {fused_gbps:.1} GB/s");
            eprintln!("  Speedup:                 {speedup:.2}x");

            // --- Fused single: layer_norm_residual ---
            for _ in 0..num_warmup {
                let _ =
                    layer_norm_residual(&input_t, &residual_t, &gamma_t, &beta_t, eps, &registry)
                        .unwrap();
                dev.synchronize().unwrap();
            }

            dev.synchronize().unwrap();
            let t2 = std::time::Instant::now();
            for _ in 0..num_runs {
                let _ =
                    layer_norm_residual(&input_t, &residual_t, &gamma_t, &beta_t, eps, &registry)
                        .unwrap();
            }
            dev.synchronize().unwrap();
            let fused_single_s = t2.elapsed().as_secs_f64() / num_runs as f64;
            let fused_single_ms = fused_single_s * 1000.0;

            // Fused single: reads input+residual (2x for 2 passes)+gamma+beta, writes output
            let fused_single_bytes = ((4 * n + 2 * d_model) * 4) as f64;
            let fused_single_gbps = fused_single_bytes / fused_single_s / 1e9;

            eprintln!(
                "  Fused (LN+res single):   {fused_single_ms:.4} ms | {fused_single_gbps:.1} GB/s"
            );
        }

        #[cfg(not(feature = "cublas"))]
        {
            eprintln!("  Fused: SKIPPED (requires --features cublas)");
        }
    }
}
