//! Numerical comparison utilities for GPU vs CPU testing.
//!
//! Provides [`assert_close`] for tensor comparison with configurable tolerance,
//! plus convenience functions like [`max_abs_error`] and [`mse`].
//!
//! # Tolerance Presets
//!
//! - [`Tolerance::f32_loose`]: for f32 GEMM outputs (rtol=1e-3, atol=1e-3)
//! - [`Tolerance::f32_strict`]: for elementwise ops (rtol=1e-5, atol=1e-5)
//! - [`Tolerance::gradient`]: for finite-difference gradient checks (rtol=1e-2, atol=1e-4)

/// Tolerance configuration for numerical comparison.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Relative tolerance: |a-b| <= rtol * max(|a|, |b|)
    pub rtol: f64,
    /// Absolute tolerance: |a-b| <= atol
    pub atol: f64,
}

impl Tolerance {
    /// Loose tolerance for f32 GEMM and multi-op pipelines.
    pub fn f32_loose() -> Self {
        Self {
            rtol: 1e-3,
            atol: 1e-3,
        }
    }

    /// Strict tolerance for simple elementwise operations.
    pub fn f32_strict() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-5,
        }
    }

    /// Tolerance for finite-difference gradient checks.
    pub fn gradient() -> Self {
        Self {
            rtol: 1e-2,
            atol: 1e-4,
        }
    }
}

/// Maximum absolute error between two slices.
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slices must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Mean squared error between two slices.
pub fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "slices must have same length");
    let n = a.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / n
}

/// Assert two f32 slices are close within the given tolerance.
///
/// Uses the formula: |a[i] - b[i]| <= atol + rtol * max(|a[i]|, |b[i]|)
///
/// On failure, prints the first few mismatches with indices and values.
///
/// # Panics
///
/// Panics if any element exceeds the tolerance, or if slices differ in length.
pub fn assert_close(actual: &[f32], expected: &[f32], tol: Tolerance, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );

    let mut mismatches: Vec<(usize, f32, f32, f64)> = Vec::new();
    let mut max_err: f64 = 0.0;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a as f64 - e as f64).abs();
        let threshold = tol.atol + tol.rtol * (a as f64).abs().max((e as f64).abs());

        if diff > max_err {
            max_err = diff;
        }

        if diff > threshold && mismatches.len() < 5 {
            mismatches.push((i, a, e, diff));
        }
    }

    if !mismatches.is_empty() {
        let mut msg = format!(
            "{label}: {}/{} elements exceed tolerance (rtol={}, atol={})\n",
            mismatches.len().min(5),
            actual.len(),
            tol.rtol,
            tol.atol,
        );
        msg.push_str(&format!("  max abs error: {max_err:.6e}\n"));
        msg.push_str("  first mismatches:\n");
        for (i, a, e, diff) in &mismatches {
            msg.push_str(&format!(
                "    [{i}] actual={a:.6}, expected={e:.6}, diff={diff:.6e}\n"
            ));
        }
        panic!("{msg}");
    }
}

// ============================================================
// CPU f64 reference implementations
// ============================================================

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

// ============================================================
// Golden file infrastructure
// ============================================================

/// A golden file entry: named f32 array with metadata.
///
/// Golden files are stored as simple text files with the format:
/// ```text
/// # Golden: <label>
/// # Shape: <dim0>,<dim1>,...
/// # Tolerance: <rtol>,<atol>
/// <f32 value per line>
/// ```
pub struct GoldenEntry {
    /// Human-readable label (e.g., "gpt2_logits_prompt0_pos4_top5").
    pub label: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Expected values.
    pub data: Vec<f32>,
    /// Tolerance for comparison.
    pub tolerance: Tolerance,
}

impl GoldenEntry {
    /// Save golden data to a text file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "# Golden: {}", self.label)?;
        let shape_str: Vec<String> = self.shape.iter().map(|s| s.to_string()).collect();
        writeln!(f, "# Shape: {}", shape_str.join(","))?;
        writeln!(
            f,
            "# Tolerance: {:.e},{:.e}",
            self.tolerance.rtol, self.tolerance.atol
        )?;
        for v in &self.data {
            writeln!(f, "{v:.8e}")?;
        }
        Ok(())
    }

    /// Load golden data from a text file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut label = String::new();
        let mut shape = Vec::new();
        let mut tolerance = Tolerance::f32_loose();
        let mut data = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("# Golden: ") {
                label = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("# Shape: ") {
                shape = rest
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            } else if let Some(rest) = line.strip_prefix("# Tolerance: ") {
                let parts: Vec<&str> = rest.split(',').collect();
                if parts.len() == 2 {
                    tolerance = Tolerance {
                        rtol: parts[0].trim().parse().unwrap_or(1e-3),
                        atol: parts[1].trim().parse().unwrap_or(1e-3),
                    };
                }
            } else if !line.is_empty() && !line.starts_with('#') {
                if let Ok(v) = line.parse::<f32>() {
                    data.push(v);
                }
            }
        }

        Ok(Self {
            label,
            shape,
            data,
            tolerance,
        })
    }

    /// Compare actual data against this golden entry.
    pub fn assert_matches(&self, actual: &[f32]) {
        assert_close(actual, &self.data, self.tolerance, &self.label);
    }
}

/// Returns the path to the golden files directory (repo_root/.research/golden/).
pub fn golden_dir() -> std::path::PathBuf {
    let root = crate::model_dir(Some(env!("CARGO_MANIFEST_DIR")));
    // model_dir returns repo_root/models, we want repo_root/.research/golden
    root.parent()
        .unwrap_or(std::path::Path::new("."))
        .join(".research")
        .join("golden")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_abs_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 2.9];
        let err = max_abs_error(&a, &b);
        assert!((err - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_mse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(mse(&a, &b) < 1e-10);
    }

    #[test]
    fn test_assert_close_passes() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert_close(&a, &b, Tolerance::f32_loose(), "test");
    }

    #[test]
    #[should_panic(expected = "exceed tolerance")]
    fn test_assert_close_fails() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0]; // diff = 1.0
        assert_close(&a, &b, Tolerance::f32_strict(), "test");
    }

    #[test]
    fn test_cpu_matmul_identity() {
        // 2x2 identity × [1,2; 3,4] = [1,2; 3,4]
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = cpu_matmul_f64(&eye, &b, 2, 2, 2);
        assert_close(&c, &b, Tolerance::f32_strict(), "matmul identity");
    }

    #[test]
    fn test_cpu_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = cpu_softmax_f64(&input, 2, 3);
        let sum0: f32 = out[0..3].iter().sum();
        let sum1: f32 = out[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5, "row 0 sum={sum0}");
        assert!((sum1 - 1.0).abs() < 1e-5, "row 1 sum={sum1}");
    }

    #[test]
    fn test_cpu_attention_causal() {
        // With causal mask, position 0 should only attend to itself
        let seq = 3;
        let d = 4;
        let q: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq * d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let out = cpu_attention_f64(&q, &k, &v, seq, d);
        // Position 0 attends only to itself → output should be v[0..d]
        assert_close(&out[0..d], &v[0..d], Tolerance::f32_strict(), "pos0 = v[0]");
    }

    // ============================================================
    // GPU vs CPU validation tests (require CUDA device)
    // ============================================================

    use crate::nn::registry::KernelRegistry;
    use crate::nn::tensor::GpuTensor;
    use std::sync::Arc;

    fn gpu_registry() -> Arc<KernelRegistry> {
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        Arc::new(KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).expect("PTX load"))
    }

    #[test]
    fn test_gpu_vs_cpu_matmul() {
        let registry = gpu_registry();
        let dev = registry.device();

        let m = 8;
        let k = 32;
        let n = 16;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 73) as f32 - 36.0) * 0.01)
            .collect();

        let expected = cpu_matmul_f64(&a, &b, m, k, n);

        let a_gpu = GpuTensor::from_host(&a, &[m, k], dev).unwrap();
        let b_gpu = GpuTensor::from_host(&b, &[k, n], dev).unwrap();
        let c_gpu = crate::nn::ops::matmul(&a_gpu, &b_gpu, &registry).unwrap();
        let actual = c_gpu.to_host().unwrap();

        assert_close(&actual, &expected, Tolerance::f32_loose(), "GPU matmul");
    }

    #[test]
    fn test_gpu_vs_cpu_layer_norm() {
        let registry = gpu_registry();
        let dev = registry.device();

        let rows = 4;
        let d = 64;
        let eps = 1e-5;
        let input: Vec<f32> = (0..rows * d)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.01)
            .collect();
        let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let beta: Vec<f32> = (0..d).map(|i| (i as f32) * 0.001).collect();

        let expected = cpu_layer_norm_f64(&input, &gamma, &beta, rows, d, eps);

        let input_gpu = GpuTensor::from_host(&input, &[rows, d], dev).unwrap();
        let gamma_gpu = GpuTensor::from_host(&gamma, &[d], dev).unwrap();
        let beta_gpu = GpuTensor::from_host(&beta, &[d], dev).unwrap();
        let out_gpu =
            crate::nn::ops::layer_norm(&input_gpu, &gamma_gpu, &beta_gpu, eps, &registry).unwrap();
        let actual = out_gpu.to_host().unwrap();

        assert_close(&actual, &expected, Tolerance::f32_loose(), "GPU layer_norm");
    }

    #[test]
    fn test_gpu_vs_cpu_gelu() {
        let registry = gpu_registry();
        let dev = registry.device();

        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) - 128.0) * 0.05).collect();

        // CPU GELU reference: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| {
                let x64 = x as f64;
                let inner =
                    (2.0_f64 / std::f64::consts::PI).sqrt() * (x64 + 0.044715 * x64 * x64 * x64);
                (0.5 * x64 * (1.0 + inner.tanh())) as f32
            })
            .collect();

        let input_gpu = GpuTensor::from_host(&input, &[n], dev).unwrap();
        let out_gpu = crate::nn::ops::gelu(&input_gpu, &registry).unwrap();
        let actual = out_gpu.to_host().unwrap();

        assert_close(&actual, &expected, Tolerance::f32_strict(), "GPU gelu");
    }
}
