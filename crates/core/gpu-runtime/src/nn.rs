use crate::math;

/// GELU activation: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// Approximate GELU used in GPT-2, BERT, etc.
#[inline(always)]
pub fn gelu_f32(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978846_f32 * (x + 0.044715_f32 * x3); // sqrt(2/π) ≈ 0.7978846
    x * 0.5 * (1.0 + math::tanh_f32(inner))
}

/// ReLU activation: `max(0, x)`.
#[inline(always)]
pub fn relu_f32(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Leaky ReLU: `x if x > 0, else alpha * x`.
#[inline(always)]
pub fn leaky_relu_f32(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// SiLU (Swish) activation: `x * sigmoid(x)`.
#[inline(always)]
pub fn silu_f32(x: f32) -> f32 {
    x * math::sigmoid_f32(x)
}

/// Warp-cooperative softmax: compute softmax of `val` across all 32 lanes.
///
/// Each lane provides one element. Returns the softmax of that element
/// relative to all 32 values. Uses butterfly reduction for max and sum.
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
pub unsafe fn warp_softmax_f32(val: f32) -> f32 {
    use crate::warp;
    let max_val = warp::reduce_max_f32(val);
    let exp_val = math::exp_f32(val - max_val);
    let sum = warp::reduce_sum_f32(exp_val);
    exp_val / sum
}

/// Warp-cooperative layer normalization across 32 lanes.
///
/// Each lane holds one element. Computes: `gamma * (x - mean) / sqrt(var + eps) + beta`
/// where mean and variance are computed across all 32 lanes.
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
pub unsafe fn warp_layer_norm_f32(val: f32, gamma: f32, beta: f32) -> f32 {
    use crate::warp;
    let mean = warp::reduce_sum_f32(val) / 32.0;
    let diff = val - mean;
    let var = warp::reduce_sum_f32(diff * diff) / 32.0;
    let inv_std = math::rsqrt_f32(var + 1e-5);
    gamma * diff * inv_std + beta
}
