//! Stateless GPU operations — functional API wrapping GPU kernels.
//!
//! Each function takes [`GpuTensor`] inputs and returns a new [`GpuTensor`].
//! Launch configuration is handled automatically via [`KernelRegistry`].
//!
//! [`GpuTensor`]: super::GpuTensor
//! [`KernelRegistry`]: super::KernelRegistry

pub mod activation;
pub mod attention;
pub mod conv;
pub mod gemm;
pub mod norm;
pub mod pool;
pub mod reshape;
pub mod upsample;

pub use activation::{gelu, relu, sigmoid, silu};
pub use attention::{
    concat_heads, multi_head_flash_attention, scaled_dot_product_attention,
    scaled_dot_product_attention_kv, split_qkv,
};
pub use conv::{conv2d, conv2d_backward};
pub use gemm::{
    int4_matmul, int8_matmul, matmul, matmul_fused, matmul_prepadded_b, FusedActivation,
};
pub use norm::{batch_norm, batch_norm_silu, layer_norm};
pub use pool::max_pool2d;
pub use reshape::{bias_add, bias_add_chw, concat_channels, elementwise_add, embedding_lookup};
pub use upsample::upsample_nearest_2x;
