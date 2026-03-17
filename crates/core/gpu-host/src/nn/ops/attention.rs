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

    // Use flash_attention_v2 kernel (Rust PTX, 4-way unrolled)
    let func = registry.get("flash_attention_v2")?;
    {
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

        let _ = total;
        Ok(output)
    } // end #[cfg(not(feature = "cublas"))]
}

/// NVRTC-compiled flash attention with tiled GEMM for score and P·V computation.
///
/// Uses cudarc NVRTC to compile a CUDA C kernel at runtime that performs:
/// - Q·K^T via tiled register-blocked GEMM (not scalar dot products)
/// - Online softmax with shared memory reduction
/// - P·V via tiled GEMM
///
/// Q, K, V: [n_heads * seq_len, d_head], output: same shape.
#[cfg(feature = "cublas")]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_flash_attention_nvrtc(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    causal: bool,
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<GpuTensor> {
    use cudarc::driver::LaunchAsync;
    use cudarc::nvrtc::compile_ptx;

    let mut output = GpuTensor::zeros(&[n_heads * seq_len, d_head], dev)?;

    // Compile CUDA C flash attention kernel via NVRTC
    static FLASH_ATTN_SRC: &str = r#"
// Flash Attention with cooperative dot products
// 128 threads (4 warps), 4 threads per Q row for parallel dot product reduction
// Block: 128 threads (4 warps), processes BQ=32 query rows per Q-tile
// For each KV tile of BKV=32:
//   1. Load K_tile[32,64] and V_tile[32,64] into shared memory
//   2. Compute scores S[32,32] = Q_tile[32,64] × K_tile^T[64,32] cooperatively
//   3. Online softmax on S
//   4. Accumulate O += P × V_tile

extern "C" __global__ void flash_attn_tiled(
    const float* __restrict__ Q,  // [n_heads * seq_len, d_head]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    unsigned int seq_len,
    unsigned int d_head,
    unsigned int causal
) {
    // Block assignment: grid = (n_heads, ceil(seq_len/BQ))
    const unsigned int BQ = 32;  // Q rows per block
    const unsigned int BKV = 32; // KV rows per tile

    unsigned int head = blockIdx.x;
    unsigned int q_tile = blockIdx.y;
    unsigned int tid = threadIdx.x;

    unsigned int head_off = head * seq_len * d_head;
    const float* q_head = Q + head_off;
    const float* k_head = K + head_off;
    const float* v_head = V + head_off;
    float* out_head = Out + head_off;

    float scale = rsqrtf((float)d_head);

    // Shared memory: K_tile[32][64] + V_tile[32][64] + scores[32][32]
    extern __shared__ float smem[];
    float* k_tile = smem;                    // 32*64 = 2048
    float* v_tile = smem + 2048;             // 32*64 = 2048
    float* s_tile = smem + 4096;             // 32*32 = 1024

    unsigned int my_q_row = q_tile * BQ + tid;  // Each of 32 threads owns one Q row

    // Load Q row into registers (persistent across all KV tiles)
    float q_reg[64];
    if (my_q_row < seq_len) {
        for (unsigned int d = 0; d < d_head; d++) {
            q_reg[d] = q_head[my_q_row * d_head + d];
        }
    }

    // Online softmax state
    float m_val = -1e38f;  // running max
    float l_val = 0.0f;    // running sum
    float o_acc[64];        // output accumulator
    for (unsigned int d = 0; d < d_head; d++) o_acc[d] = 0.0f;

    unsigned int n_kv_tiles = (seq_len + BKV - 1) / BKV;

    for (unsigned int t = 0; t < n_kv_tiles; t++) {
        unsigned int kv_start = t * BKV;

        // Causal: skip tiles entirely above diagonal
        if (causal && kv_start > q_tile * BQ + BQ - 1) break;

        unsigned int tile_sz = (kv_start + BKV <= seq_len) ? BKV : (seq_len - kv_start);

        // === Load K and V tiles (32 threads load 32 rows) ===
        unsigned int kv_row = kv_start + tid;
        for (unsigned int d = 0; d < d_head; d++) {
            float kv = (kv_row < seq_len) ? k_head[kv_row * d_head + d] : 0.0f;
            k_tile[tid * d_head + d] = kv;
            float vv = (kv_row < seq_len) ? v_head[kv_row * d_head + d] : 0.0f;
            v_tile[tid * d_head + d] = vv;
        }
        __syncthreads();

        if (my_q_row < seq_len) {
            // === Compute scores: S[my_row, 0..tile_sz] ===
            float tile_max = -1e38f;
            float scores[32];

            for (unsigned int c = 0; c < tile_sz; c++) {
                unsigned int kv_col = kv_start + c;
                if (causal && kv_col > my_q_row) {
                    scores[c] = -1e38f;
                } else {
                    // Dot product with 4-way unrolling
                    float dot = 0.0f;
                    unsigned int d = 0;
                    for (; d + 3 < d_head; d += 4) {
                        dot += q_reg[d]   * k_tile[c * d_head + d];
                        dot += q_reg[d+1] * k_tile[c * d_head + d+1];
                        dot += q_reg[d+2] * k_tile[c * d_head + d+2];
                        dot += q_reg[d+3] * k_tile[c * d_head + d+3];
                    }
                    for (; d < d_head; d++) {
                        dot += q_reg[d] * k_tile[c * d_head + d];
                    }
                    float s = dot * scale;
                    scores[c] = s;
                    if (s > tile_max) tile_max = s;
                }
            }

            // === Online softmax update ===
            float m_new = (tile_max > m_val) ? tile_max : m_val;
            float row_sum = 0.0f;
            float exp_scores[32];

            for (unsigned int c = 0; c < tile_sz; c++) {
                float e = expf(scores[c] - m_new);
                exp_scores[c] = e;
                row_sum += e;
            }

            float correction = expf(m_val - m_new);

            // Rescale old output
            for (unsigned int d = 0; d < d_head; d++) {
                o_acc[d] *= correction;
            }

            // Accumulate P × V with 4-way unrolling
            for (unsigned int c = 0; c < tile_sz; c++) {
                float p = exp_scores[c];
                if (p > 1e-30f) {
                    unsigned int v_off = c * d_head;
                    unsigned int d = 0;
                    for (; d + 3 < d_head; d += 4) {
                        o_acc[d]   += p * v_tile[v_off + d];
                        o_acc[d+1] += p * v_tile[v_off + d+1];
                        o_acc[d+2] += p * v_tile[v_off + d+2];
                        o_acc[d+3] += p * v_tile[v_off + d+3];
                    }
                    for (; d < d_head; d++) {
                        o_acc[d] += p * v_tile[v_off + d];
                    }
                }
            }

            l_val = l_val * correction + row_sum;
            m_val = m_new;
        }

        __syncthreads();
    }

    // Write output
    if (my_q_row < seq_len && l_val > 0.0f) {
        float inv_l = 1.0f / l_val;
        for (unsigned int d = 0; d < d_head; d++) {
            out_head[my_q_row * d_head + d] = o_acc[d] * inv_l;
        }
    }
}
"#;

    // Cache the compiled PTX across calls
    use std::sync::OnceLock;
    static COMPILED: OnceLock<bool> = OnceLock::new();
    COMPILED.get_or_init(|| {
        let ptx = compile_ptx(FLASH_ATTN_SRC).expect("NVRTC compile failed");
        dev.load_ptx(ptx, "flash_attn", &["flash_attn_tiled"])
            .expect("PTX load failed");
        true
    });

    let func = dev
        .get_func("flash_attn", "flash_attn_tiled")
        .ok_or(NnError::KernelNotFound {
            name: "flash_attn_tiled",
        })?;

    let n_q_tiles = seq_len.div_ceil(32) as u32;
    let smem_bytes = (2048 + 2048 + 1024) * 4; // K_tile + V_tile + S_tile
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (n_heads as u32, n_q_tiles, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    let causal_flag: u32 = if causal { 1 } else { 0 };

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
                causal_flag,
            ),
        )
        .map_err(NnError::Cuda)?;
    }

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

/// Multi-head attention via matmul: S = Q·K^T, P = softmax(S), O = P·V.
///
/// Uses cuBLAS-backed matmul for the heavy GEMM operations.
/// Requires softmax + causal mask as separate kernels.
#[cfg(feature = "cublas")]
#[allow(clippy::too_many_arguments)]
fn multi_head_matmul_attention(
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
    let scale = 1.0 / (d_head as f32).sqrt();

    // Q, K, V are [n_heads * seq_len, d_head] in head-major layout
    let q_host = q.to_host()?;
    let k_host = k.to_host()?;
    let v_host = v.to_host()?;
    let mut all_output = vec![0.0f32; n_heads * seq_len * d_head];

    for h in 0..n_heads {
        let head_off = h * seq_len * d_head;

        // Extract per-head slices
        let q_slice = &q_host[head_off..head_off + seq_len * d_head];
        let k_slice = &k_host[head_off..head_off + seq_len * d_head];
        let v_slice = &v_host[head_off..head_off + seq_len * d_head];

        // Transpose K: [seq_len, d_head] → [d_head, seq_len]
        let mut k_t = vec![0.0f32; d_head * seq_len];
        for r in 0..seq_len {
            for c in 0..d_head {
                k_t[c * seq_len + r] = k_slice[r * d_head + c];
            }
        }

        // Upload Q_h and K^T to GPU
        let q_dev = GpuTensor::from_host(q_slice, &[seq_len, d_head], dev)?;
        let kt_dev = GpuTensor::from_host(&k_t, &[d_head, seq_len], dev)?;

        // S = Q × K^T: [seq_len, seq_len]
        let s = super::matmul_v2(&q_dev, &kt_dev, registry)?;

        // Scale + causal mask + softmax on host (simple for correctness)
        let mut s_host = s.to_host()?;
        for i in 0..seq_len {
            // Apply scale and causal mask
            let mut max_val: f32 = f32::NEG_INFINITY;
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if causal && j > i {
                    s_host[idx] = f32::NEG_INFINITY;
                } else {
                    s_host[idx] *= scale;
                }
                if s_host[idx] > max_val {
                    max_val = s_host[idx];
                }
            }
            // Softmax: exp(x - max) / sum(exp(x - max))
            let mut sum: f32 = 0.0;
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                let e = (s_host[idx] - max_val).exp();
                s_host[idx] = e;
                sum += e;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..seq_len {
                s_host[i * seq_len + j] *= inv_sum;
            }
        }

        // Upload P and V_h to GPU
        let p = GpuTensor::from_host(&s_host, &[seq_len, seq_len], dev)?;
        let v_dev = GpuTensor::from_host(v_slice, &[seq_len, d_head], dev)?;

        // O = P × V: [seq_len, d_head]
        let o_h = super::matmul_v2(&p, &v_dev, registry)?;
        let o_host = o_h.to_host()?;
        all_output[head_off..head_off + seq_len * d_head].copy_from_slice(&o_host);
    }

    GpuTensor::from_host(&all_output, &[n_heads * seq_len, d_head], dev)
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
