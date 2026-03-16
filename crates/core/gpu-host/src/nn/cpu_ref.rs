//! CPU f64 reference implementations for numerical verification.
//!
//! All functions compute in f64 and return f32 — used to verify GPU kernel
//! outputs via finite differences and direct comparison.

/// CPU f64 matrix multiplication: C = A * B.
///
/// A: `[m, k]`, B: `[k, n]` → C: `[m, n]`, all row-major.
pub fn cpu_matmul_f64(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
    c
}

/// CPU f64 layer normalization.
///
/// Input: `[rows, d]`, gamma/beta: `[d]`.
/// Normalizes each row independently.
pub fn cpu_layer_norm_f64(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    for i in 0..rows {
        let row = &input[i * d..(i + 1) * d];
        let mean: f64 = row.iter().map(|&x| x as f64).sum::<f64>() / d as f64;
        let var: f64 = row
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / d as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for j in 0..d {
            let norm = (row[j] as f64 - mean) * inv_std;
            out[i * d + j] = (norm * gamma[j] as f64 + beta[j] as f64) as f32;
        }
    }
    out
}

/// CPU f64 2D convolution (direct, no im2col).
///
/// Input: `[c_in, h, w]`, weight: `[c_out, c_in, kh, kw]`, bias: `[c_out]` (optional).
/// Output: `[c_out, h_out, w_out]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_conv2d_f64(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let mut out = vec![0.0f32; c_out * h_out * w_out];

    for co in 0..c_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0f64;
                for ci in 0..c_in {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = (oh * stride + fh) as isize - padding as isize;
                            let iw = (ow * stride + fw) as isize - padding as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_val =
                                    input[ci * h * w + ih as usize * w + iw as usize] as f64;
                                let w_val = weight
                                    [co * (c_in * kh * kw) + ci * (kh * kw) + fh * kw + fw]
                                    as f64;
                                sum += in_val * w_val;
                            }
                        }
                    }
                }
                if let Some(b) = bias {
                    sum += b[co] as f64;
                }
                out[co * h_out * w_out + oh * w_out + ow] = sum as f32;
            }
        }
    }
    out
}

/// CPU f64 softmax (row-wise).
///
/// Input: `[rows, cols]` → output: `[rows, cols]`.
pub fn cpu_softmax_f64(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let row = &input[i * cols..(i + 1) * cols];
        let max_val: f64 = row
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&x| ((x as f64) - max_val).exp()).sum();
        for j in 0..cols {
            out[i * cols + j] = (((row[j] as f64) - max_val).exp() / exp_sum) as f32;
        }
    }
    out
}

/// CPU f64 scaled dot-product attention (single head, causal).
///
/// Q: `[seq, d]`, K: `[seq, d]`, V: `[seq, d]` → output: `[seq, d]`.
pub fn cpu_attention_f64(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f64).sqrt();

    // Compute attention scores: Q * K^T * scale
    let mut scores = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += q[i * d + p] as f64 * k[j * d + p] as f64;
            }
            // Causal mask: set future positions to -inf
            if j > i {
                scores[i * seq + j] = f32::NEG_INFINITY;
            } else {
                scores[i * seq + j] = (dot * scale) as f32;
            }
        }
    }

    // Softmax over scores
    let probs = cpu_softmax_f64(&scores, seq, seq);

    // Output = probs * V
    let mut out = vec![0.0f32; seq * d];
    for i in 0..seq {
        for j in 0..d {
            let mut sum = 0.0f64;
            for p in 0..seq {
                sum += probs[i * seq + p] as f64 * v[p * d + j] as f64;
            }
            out[i * d + j] = sum as f32;
        }
    }
    out
}
