//! Kernel registry — maps operation names to loaded CUDA functions with auto launch config.
//!
//! [`KernelRegistry`] abstracts PTX function names, grid/block sizes, and unsafe
//! kernel launches behind a clean API. Users never need to write `LaunchConfig`,
//! `get_func()`, or PTX function names directly.
//!
//! # Usage
//!
//! ```no_run
//! # use std::sync::Arc;
//! # fn example() -> gpu_host::nn::error::Result<()> {
//! use gpu_host::nn::KernelRegistry;
//! let dev = cudarc::driver::CudaDevice::new(0)?;
//! let registry = KernelRegistry::new(Arc::clone(&dev), gpu_host::ptx::KERNEL)?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};

use super::error::{NnError, Result};

/// Registry of loaded GPU kernel functions with auto-config launch helpers.
///
/// Loads all ML-relevant kernels from a PTX source at construction time and
/// provides typed access by name with pre-computed launch configurations.
pub struct KernelRegistry {
    device: Arc<CudaDevice>,
}

/// All ML kernel function names loaded from the main kernel PTX.
const ML_KERNELS: &[&str] = &[
    // CNN ops (compute_cnn.rs)
    "batchnorm_silu",
    "silu_forward",
    "im2col",
    "maxpool2d",
    "upsample_nearest_2x",
    "concat_channels",
    "sigmoid_forward",
    "bias_add_chw",
    "elementwise_mul",
    "elementwise_sub",
    "elementwise_neg",
    "scalar_mul",
    "channel_scale_chw",
    // Persistent kernel (compute_persistent.rs)
    "persistent_worker",
    // Fused GEMM + activation ops (compute_fused.rs)
    "gemm_bias_gelu",
    "gemm_bias_relu",
    // INT8 GEMM ops (compute_gemm.rs)
    "int8_gemm_dp4a",
    "int8_dequantize",
    "int4_gemm_w4a16",
    // Physics simulation ops (compute_physics.rs)
    "spring_forces",
    "gravity_forces",
    "euler_step",
    // Transformer ops (compute_transformer.rs)
    "layer_norm",
    "gelu_forward",
    "attention_head",
    "flash_attention",
    "flash_attention_kv",
    "embedding_lookup",
    "bias_add",
    "elementwise_add",
    "split_qkv",
    "concat_heads",
    "f32_to_f16x2_pack",
    "zero_pad",
    "kv_cache_append",
    // GEMM (compute_gemm.rs)
    "gemm_f32",
    "gemm_f32_v2",
    "gemm_f32_v3",
    "layer_norm_v2",
    "elementwise_add_v2",
    "gelu_forward_v2",
    "flash_attention_v2",
    "full_gemm_splitk",
    "sgd_step",
    "im2col_offset",
    // Matrix utilities
    "matrix_transpose",
    "matrix_pad",
    "matrix_unpad",
    // Conv backward
    "col2im",
    // Backward kernels (autograd)
    "gelu_backward",
    "silu_backward",
    "sigmoid_backward",
    "relu_backward",
    "bias_add_backward",
];

impl KernelRegistry {
    /// Load all ML kernels from PTX source.
    ///
    /// Loads the PTX module and registers all known ML kernel functions.
    /// Returns an error if the PTX cannot be loaded.
    pub fn new(device: Arc<CudaDevice>, ptx_src: &str) -> Result<Self> {
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx_src),
                "nn_kernels",
                ML_KERNELS,
            )
            .map_err(NnError::Cuda)?;

        Ok(Self { device })
    }

    /// Create a registry on GPU 0 with the default embedded kernel PTX.
    ///
    /// Convenience wrapper that initializes `CudaDevice::new(0)` and loads
    /// the default `ptx::KERNEL`. Returns `(Arc<CudaDevice>, Arc<KernelRegistry>)`.
    ///
    /// # Example
    /// ```no_run
    /// let (dev, registry) = gpu_host::nn::KernelRegistry::init_default()?;
    /// ```
    pub fn init_default() -> Result<(Arc<CudaDevice>, Arc<Self>)> {
        let dev = CudaDevice::new(0).map_err(NnError::Cuda)?;
        let registry = Arc::new(Self::new(Arc::clone(&dev), crate::ptx::KERNEL)?);
        Ok((dev, registry))
    }

    /// Get a kernel function by name.
    ///
    /// Returns [`NnError::KernelNotFound`] if the function is not in the registry.
    pub fn get(&self, name: &'static str) -> Result<CudaFunction> {
        self.device
            .get_func("nn_kernels", name)
            .ok_or(NnError::KernelNotFound { name })
    }

    /// Reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Standard 1D launch config for element-wise ops.
    ///
    /// Block size = 256, grid = ceil(n / 256).
    pub fn config_1d(n: u32) -> LaunchConfig {
        let block = 256;
        let grid = n.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// GEMM launch config for `gemm_f32` kernel.
    ///
    /// Each block computes a 32x16 output tile using 4 warps.
    /// Grid = (ceil(M/32), ceil(N/16), 1), Block = (128, 1, 1).
    pub fn config_gemm(m: u32, n: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (m.div_ceil(32), n.div_ceil(16), 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// LayerNorm launch config.
    ///
    /// One warp (32 threads) per row. Grid = (num_rows, 1, 1).
    pub fn config_layernorm(num_rows: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (num_rows, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Flash attention launch config.
    ///
    /// One warp per query position. Grid = (seq_len, 1, 1), Block = (32, 1, 1).
    pub fn config_attention(seq_len: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (seq_len, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Embedding lookup launch config.
    ///
    /// 1D grid over total output elements (seq_len * d_model).
    /// Block = (256, 1, 1), grid = (ceil(total/256), 1, 1).
    pub fn config_embedding(total_elements: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (total_elements.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// im2col launch config.
    ///
    /// 1D grid over total output elements.
    pub fn config_im2col(total_elements: u32) -> LaunchConfig {
        Self::config_1d(total_elements)
    }

    /// BatchNorm launch config.
    ///
    /// 1D grid over total elements (C * H * W).
    pub fn config_batchnorm(total_elements: u32) -> LaunchConfig {
        Self::config_1d(total_elements)
    }
}
