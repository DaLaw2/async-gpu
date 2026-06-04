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

/// Out-of-place element-wise addition: c = a + b.
///
/// Unlike `elementwise_add` (in-place a += b), this creates a new output tensor.
/// Avoids read-write conflicts on the same buffer, achieving higher bandwidth.
/// Uses NVRTC-compiled kernel with float4 vectorized loads.
#[cfg(feature = "cublas")]
pub fn elementwise_add_out(
    a: &GpuTensor,
    b: &GpuTensor,
    registry: &Arc<KernelRegistry>,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx;

    if a.numel() != b.numel() {
        return Err(NnError::ShapeMismatch {
            expected: format!("same numel, a has {}", a.numel()),
            actual: format!("b has {}", b.numel()),
        });
    }

    let n = a.numel();
    let dev = registry.device();
    let mut output = GpuTensor::zeros(a.shape(), dev)?;

    static ELEM_ADD_OOP_SRC: &str = r#"
extern "C" __global__ void elementwise_add_oop(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 av = *reinterpret_cast<const float4*>(&a[idx]);
        float4 bv = *reinterpret_cast<const float4*>(&b[idx]);
        float4 cv;
        cv.x = av.x + bv.x;
        cv.y = av.y + bv.y;
        cv.z = av.z + bv.z;
        cv.w = av.w + bv.w;
        *reinterpret_cast<float4*>(&c[idx]) = cv;
    } else {
        for (unsigned int i = idx; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}
"#;

    use std::sync::OnceLock;
    static COMPILED: OnceLock<bool> = OnceLock::new();
    COMPILED.get_or_init(|| {
        let ptx = compile_ptx(ELEM_ADD_OOP_SRC).expect("NVRTC elementwise_add_oop failed");
        dev.load_ptx(ptx, "elem_oop", &["elementwise_add_oop"])
            .expect("load elementwise_add_oop");
        true
    });

    let func = dev
        .get_func("elem_oop", "elementwise_add_oop")
        .ok_or(NnError::KernelNotFound {
            name: "elementwise_add_oop",
        })?;

    let threads = 256u32;
    let total_threads = (n as u32 + 3) / 4; // each thread handles 4 elements
    let grid = ((total_threads + threads - 1) / threads, 1, 1);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(config, (a.data(), b.data(), output.data_mut(), n as u32))
            .map_err(NnError::Cuda)?;
    }

    Ok(output)
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
