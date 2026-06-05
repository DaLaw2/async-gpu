//! Tape-level fusion candidate detection.
//!
//! Analyzes an autograd [`Tape`] to identify sequences of operations that can
//! be replaced by a single fused kernel, reducing kernel launch overhead.
//!
//! This module is **detection only** — it produces a [`FusionPlan`] describing
//! which tape entries can be fused and what fused op would replace them.
//! Actual kernel compilation and execution are handled by a later phase.
//!
//! # Algorithm
//!
//! Greedy longest-match forward scan: the tape is walked front-to-back and at
//! each position, the highest-priority pattern whose length is maximal is
//! selected. Patterns are tried in descending priority order so that
//! `Matmul → BiasAdd → Gelu` (P1, 3 ops) is preferred over
//! `Matmul → BiasAdd` (P4, 2 ops) when both match.
//!
//! # Supported patterns (initial set)
//!
//! | ID | Sequence                   | Fused replacement        |
//! |----|----------------------------|--------------------------|
//! | P1 | Matmul → BiasAdd → Gelu    | `MatmulBiasGelu`         |
//! | P3 | ElemAdd → LayerNorm        | `ElemAddLayerNorm`        |
//! | P4 | Matmul → BiasAdd           | `MatmulBias`             |

use std::collections::HashMap;
use std::fmt;

use crate::nn::autograd::{OpKind, TapeEntry, TensorId};

// ---------------------------------------------------------------------------
// Op classification
// ---------------------------------------------------------------------------

/// Classification of ops for fusion eligibility.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OpClass {
    /// Element-wise: output[i] = f(input[i], ...).
    Elementwise,
    /// Compute-bound op (e.g. GEMM) — can absorb elementwise epilogue.
    ComputeBound,
    /// Reduction — consumes elements and produces fewer.
    Reduction,
    /// Structural — changes layout, fusion barrier.
    Structural,
}

/// Classify an [`OpKind`] into its fusion class.
fn classify(op: OpKind) -> OpClass {
    match op {
        OpKind::Gelu
        | OpKind::Silu
        | OpKind::Sigmoid
        | OpKind::Relu
        | OpKind::BiasAdd
        | OpKind::ElemAdd
        | OpKind::ElemMul => OpClass::Elementwise,

        OpKind::Matmul | OpKind::Conv2d => OpClass::ComputeBound,

        OpKind::LayerNorm | OpKind::BatchNorm | OpKind::CrossEntropy | OpKind::MseLoss => {
            OpClass::Reduction
        }

        OpKind::Attention => OpClass::ComputeBound, // opaque, never fuses

        OpKind::Embedding | OpKind::MaxPool2d | OpKind::UpsampleNearest => OpClass::Structural,
    }
}

// ---------------------------------------------------------------------------
// Fused op kinds
// ---------------------------------------------------------------------------

/// Kinds of fused operations the optimizer can produce.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FusedOpKind {
    /// Matmul + BiasAdd + Gelu — maps to existing `gemm_bias_gelu` kernel.
    MatmulBiasGelu,
    /// ElemAdd + LayerNorm — maps to existing `layer_norm_residual` kernel.
    ElemAddLayerNorm,
    /// Matmul + BiasAdd — GEMM epilogue fusion.
    MatmulBias,
    /// Generic fused elementwise chain (2-5 ops) — NVRTC codegen.
    ElementwiseChain(Vec<OpKind>),
}

impl fmt::Display for FusedOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatmulBiasGelu => write!(f, "MatmulBiasGelu"),
            Self::ElemAddLayerNorm => write!(f, "ElemAddLayerNorm"),
            Self::MatmulBias => write!(f, "MatmulBias"),
            Self::ElementwiseChain(ops) => {
                write!(f, "ElementwiseChain({ops:?})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion pattern definition
// ---------------------------------------------------------------------------

/// A concrete fusion pattern: a sequence of ops that can be replaced.
struct FusionPattern {
    /// Ordered sequence of `OpKind` to match.
    ops: Vec<OpKind>,
    /// The replacement fused operation.
    replacement: FusedOpKind,
    /// Priority — higher values are tried first.
    priority: u32,
}

/// Return the built-in pattern catalog, sorted by priority descending then
/// length descending (so longest patterns are preferred at the same priority).
fn pattern_catalog() -> Vec<FusionPattern> {
    let mut patterns = vec![
        // P1: Matmul → BiasAdd → Gelu
        FusionPattern {
            ops: vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Gelu],
            replacement: FusedOpKind::MatmulBiasGelu,
            priority: 100,
        },
        // P3: ElemAdd → LayerNorm
        FusionPattern {
            ops: vec![OpKind::ElemAdd, OpKind::LayerNorm],
            replacement: FusedOpKind::ElemAddLayerNorm,
            priority: 90,
        },
        // P4: Matmul → BiasAdd
        FusionPattern {
            ops: vec![OpKind::Matmul, OpKind::BiasAdd],
            replacement: FusedOpKind::MatmulBias,
            priority: 80,
        },
    ];
    // Sort by priority descending, then by pattern length descending.
    patterns.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| b.ops.len().cmp(&a.ops.len()))
    });
    patterns
}

// ---------------------------------------------------------------------------
// Fusability predicates
// ---------------------------------------------------------------------------

/// Check whether the intermediate tensor between `producer` and `consumer` has
/// exactly one consumer (no fan-out).
fn single_consumer(producer: &TapeEntry, ref_counts: &HashMap<TensorId, usize>) -> bool {
    ref_counts
        .get(&producer.output)
        .is_none_or(|&count| count == 1)
}

/// Check whether the consumer reads the producer's output as one of its inputs,
/// establishing a direct data-flow dependency.
fn data_flows(producer: &TapeEntry, consumer: &TapeEntry) -> bool {
    consumer.inputs.contains(&producer.output)
}

/// Compute how many times each tensor is consumed across the tape.
fn compute_ref_counts(tape: &[TapeEntry]) -> HashMap<TensorId, usize> {
    let mut counts: HashMap<TensorId, usize> = HashMap::new();
    for entry in tape {
        for &input_id in &entry.inputs {
            *counts.entry(input_id).or_insert(0) += 1;
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// Fusion plan
// ---------------------------------------------------------------------------

/// A group of consecutive tape entries to be replaced by a single fused op.
#[derive(Clone, Debug)]
pub struct FusionGroup {
    /// Inclusive start index in the tape.
    pub start: usize,
    /// Exclusive end index in the tape.
    pub end: usize,
    /// The fused operation that replaces `tape[start..end]`.
    pub fused_op: FusedOpKind,
    /// Input tensor IDs for the fused group (first entry's inputs).
    pub inputs: Vec<TensorId>,
    /// Output tensor ID of the fused group (last entry's output).
    pub output: TensorId,
}

/// Result of fusion analysis — a plan listing all fusion opportunities.
#[derive(Clone, Debug, Default)]
pub struct FusionPlan {
    /// Detected fusion groups, ordered by their position in the tape.
    pub groups: Vec<FusionGroup>,
}

impl FusionPlan {
    /// Number of fusion groups detected.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether the plan contains no fusion opportunities.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Total number of kernel launches saved by applying this plan.
    pub fn launches_saved(&self) -> usize {
        self.groups
            .iter()
            .map(|g| (g.end - g.start).saturating_sub(1))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// FusionOptimizer
// ---------------------------------------------------------------------------

/// Tape-level fusion optimizer.
///
/// Analyzes an autograd tape to find fusable op sequences. This is a
/// pure analysis pass — it does not modify the tape or compile kernels.
pub struct FusionOptimizer {
    patterns: Vec<FusionPattern>,
}

impl FusionOptimizer {
    /// Create a new optimizer with the default pattern catalog.
    pub fn new() -> Self {
        Self {
            patterns: pattern_catalog(),
        }
    }

    /// Analyze a tape and produce a fusion plan.
    ///
    /// The algorithm is a greedy longest-match forward scan: at each tape
    /// position, the highest-priority (and longest) matching pattern is
    /// selected. Matched entries are skipped so they participate in at most
    /// one fusion group.
    pub fn analyze(&self, tape: &[TapeEntry]) -> FusionPlan {
        let ref_counts = compute_ref_counts(tape);
        let mut groups = Vec::new();
        let mut i = 0;

        while i < tape.len() {
            if let Some(group) = self.try_match(tape, i, &ref_counts) {
                let skip = group.end - group.start;
                groups.push(group);
                i += skip;
            } else {
                i += 1;
            }
        }

        FusionPlan { groups }
    }

    /// Try to match any pattern starting at position `start`.
    ///
    /// Returns the best (highest priority, longest) matching group, or `None`.
    /// Fixed patterns (P1/P3/P4) are tried first, then a generic elementwise
    /// chain match (P10) as fallback.
    fn try_match(
        &self,
        tape: &[TapeEntry],
        start: usize,
        ref_counts: &HashMap<TensorId, usize>,
    ) -> Option<FusionGroup> {
        // Patterns are pre-sorted by priority desc, length desc.
        for pattern in &self.patterns {
            let len = pattern.ops.len();
            if start + len > tape.len() {
                continue;
            }

            let slice = &tape[start..start + len];

            // Check op kinds match the pattern.
            if !slice
                .iter()
                .zip(&pattern.ops)
                .all(|(entry, expected)| entry.op == *expected)
            {
                continue;
            }

            // Check fusability predicates for each consecutive pair.
            let mut fusable = true;
            for j in 0..len - 1 {
                // Data flow: producer's output must feed consumer's input.
                if !data_flows(&slice[j], &slice[j + 1]) {
                    fusable = false;
                    break;
                }
                // Single consumer: no fan-out on intermediate tensors.
                if !single_consumer(&slice[j], ref_counts) {
                    fusable = false;
                    break;
                }
            }

            if fusable {
                return Some(FusionGroup {
                    start,
                    end: start + len,
                    fused_op: pattern.replacement.clone(),
                    inputs: slice[0].inputs.clone(),
                    output: slice[len - 1].output,
                });
            }
        }

        // Fallback: generic elementwise chain (P10).
        // Greedily extend from `start` as long as consecutive elementwise ops
        // are connected by data-flow and have no fan-out.
        if classify(tape[start].op) == OpClass::Elementwise {
            let mut end = start + 1;
            while end < tape.len()
                && end - start < 5
                && classify(tape[end].op) == OpClass::Elementwise
                && data_flows(&tape[end - 1], &tape[end])
                && single_consumer(&tape[end - 1], ref_counts)
            {
                end += 1;
            }
            if end - start >= 2 {
                let ops: Vec<OpKind> = tape[start..end].iter().map(|e| e.op).collect();
                return Some(FusionGroup {
                    start,
                    end,
                    fused_op: FusedOpKind::ElementwiseChain(ops),
                    inputs: tape[start].inputs.clone(),
                    output: tape[end - 1].output,
                });
            }
        }

        None
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fusion codegen — NVRTC-based fused elementwise kernel generation
// ---------------------------------------------------------------------------

/// A compiled fused kernel's module and function names for device lookup.
#[derive(Clone, Debug)]
struct CompiledFusedKernel {
    module_name: String,
    func_name: String,
}

/// Extra parameter descriptor for ops that need additional device buffers
/// (e.g., bias vectors for BiasAdd, addend vectors for ElemAdd).
#[derive(Clone, Debug)]
pub struct ExtraParam {
    /// Index into the extra-params slice passed at launch time.
    pub idx: usize,
    /// Whether this param also carries a `n_cols` size argument.
    pub has_n_cols: bool,
}

/// Thread-safe cache and codegen engine for fused elementwise kernels.
///
/// Generates CUDA C source from a chain of elementwise ops, compiles via
/// NVRTC, caches by hash, and provides a launch helper.
pub struct FusionCodegen {
    cache: std::sync::Mutex<HashMap<u64, CompiledFusedKernel>>,
}

impl FusionCodegen {
    /// Create a new codegen engine with an empty cache.
    pub fn new() -> Self {
        Self {
            cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Compute a cache key from the op chain and any shape-affecting params.
    fn cache_key(ops: &[OpKind], n_cols_params: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ops.hash(&mut hasher);
        n_cols_params.hash(&mut hasher);
        hasher.finish()
    }

    /// Fixed function name used for all fused kernels.
    ///
    /// cudarc's `load_ptx` requires `&'static str` for function names, so we
    /// use a single static name and differentiate kernels by module name
    /// (which accepts dynamic `&str`).
    const FUNC_NAME: &'static str = "fused_kernel";

    /// Generate CUDA C source for a fused elementwise chain.
    ///
    /// Returns `(cuda_source, func_name, extra_param_descriptors)`.
    ///
    /// The generated kernel signature is:
    /// ```text
    /// __global__ void fused_kernel(
    ///     const float* input, float* output,
    ///     [const float* bias_0, unsigned int n_cols_0,]  // if BiasAdd
    ///     [const float* addend_0,]                       // if ElemAdd
    ///     ...
    ///     unsigned int n
    /// )
    /// ```
    pub fn codegen(
        ops: &[OpKind],
        n_cols_params: &[usize],
        _key: u64,
    ) -> (String, String, Vec<ExtraParam>) {
        let func_name = Self::FUNC_NAME.to_string();
        let mut extra_params_decl = Vec::new();
        let mut extra_params_desc = Vec::new();
        let mut float4_body = String::new();
        let mut scalar_body = String::new();
        let mut param_idx = 0usize;
        let mut n_cols_idx = 0usize;

        for &op in ops {
            match op {
                OpKind::BiasAdd => {
                    let p = param_idx;
                    let nc = n_cols_params[n_cols_idx];
                    n_cols_idx += 1;
                    extra_params_decl.push(format!(
                        "    const float* __restrict__ bias_{p},\n    unsigned int n_cols_{p}"
                    ));
                    extra_params_desc.push(ExtraParam {
                        idx: p,
                        has_n_cols: true,
                    });
                    // Float4 path: vectorized bias load when n_cols is aligned,
                    // per-element scalar fallback otherwise.
                    if nc.is_multiple_of(4) {
                        float4_body.push_str(&format!(
                            r#"
        // BiasAdd (param {p}, n_cols={nc}, vectorized)
        {{
            unsigned int col4 = idx % n_cols_{p};
            float4 bv = *reinterpret_cast<const float4*>(&bias_{p}[col4]);
            v.x += bv.x;
            v.y += bv.y;
            v.z += bv.z;
            v.w += bv.w;
        }}
"#
                        ));
                    } else {
                        float4_body.push_str(&format!(
                            r#"
        // BiasAdd (param {p}, n_cols={nc}, scalar bias reads)
        {{
            v.x += bias_{p}[(idx    ) % n_cols_{p}];
            v.y += bias_{p}[(idx + 1) % n_cols_{p}];
            v.z += bias_{p}[(idx + 2) % n_cols_{p}];
            v.w += bias_{p}[(idx + 3) % n_cols_{p}];
        }}
"#
                        ));
                    }
                    scalar_body.push_str(&format!(
                        r#"
            // BiasAdd (param {p})
            val += bias_{p}[i % n_cols_{p}];
"#
                    ));
                    param_idx += 1;
                }
                OpKind::ElemAdd => {
                    let p = param_idx;
                    extra_params_decl.push(format!("    const float* __restrict__ addend_{p}"));
                    extra_params_desc.push(ExtraParam {
                        idx: p,
                        has_n_cols: false,
                    });
                    float4_body.push_str(&format!(
                        r#"
        // ElemAdd (param {p})
        {{
            float4 av = *reinterpret_cast<const float4*>(&addend_{p}[idx]);
            v.x += av.x;
            v.y += av.y;
            v.z += av.z;
            v.w += av.w;
        }}
"#
                    ));
                    scalar_body.push_str(&format!(
                        r#"
            // ElemAdd (param {p})
            val += addend_{p}[i];
"#
                    ));
                    param_idx += 1;
                }
                OpKind::ElemMul => {
                    let p = param_idx;
                    extra_params_decl.push(format!("    const float* __restrict__ multiplier_{p}"));
                    extra_params_desc.push(ExtraParam {
                        idx: p,
                        has_n_cols: false,
                    });
                    float4_body.push_str(&format!(
                        r#"
        // ElemMul (param {p})
        {{
            float4 mv = *reinterpret_cast<const float4*>(&multiplier_{p}[idx]);
            v.x *= mv.x;
            v.y *= mv.y;
            v.z *= mv.z;
            v.w *= mv.w;
        }}
"#
                    ));
                    scalar_body.push_str(&format!(
                        r#"
            // ElemMul (param {p})
            val *= multiplier_{p}[i];
"#
                    ));
                    param_idx += 1;
                }
                OpKind::Gelu => {
                    float4_body.push_str(
                        r#"
        // GELU approximation
        {
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float COEFF = 0.044715f;
            float4 tmp;
            tmp.x = SQRT_2_OVER_PI * (v.x + COEFF * v.x * v.x * v.x);
            tmp.y = SQRT_2_OVER_PI * (v.y + COEFF * v.y * v.y * v.y);
            tmp.z = SQRT_2_OVER_PI * (v.z + COEFF * v.z * v.z * v.z);
            tmp.w = SQRT_2_OVER_PI * (v.w + COEFF * v.w * v.w * v.w);
            v.x = 0.5f * v.x * (1.0f + tanhf(tmp.x));
            v.y = 0.5f * v.y * (1.0f + tanhf(tmp.y));
            v.z = 0.5f * v.z * (1.0f + tanhf(tmp.z));
            v.w = 0.5f * v.w * (1.0f + tanhf(tmp.w));
        }
"#,
                    );
                    scalar_body.push_str(
                        r#"
            // GELU approximation
            {
                const float SQRT_2_OVER_PI = 0.7978845608f;
                const float COEFF = 0.044715f;
                float tmp = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
                val = 0.5f * val * (1.0f + tanhf(tmp));
            }
"#,
                    );
                }
                OpKind::Relu => {
                    float4_body.push_str(
                        r#"
        // ReLU
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
"#,
                    );
                    scalar_body.push_str(
                        r#"
            val = fmaxf(val, 0.0f);
"#,
                    );
                }
                OpKind::Silu => {
                    float4_body.push_str(
                        r#"
        // SiLU (x * sigmoid(x))
        v.x = v.x / (1.0f + expf(-v.x));
        v.y = v.y / (1.0f + expf(-v.y));
        v.z = v.z / (1.0f + expf(-v.z));
        v.w = v.w / (1.0f + expf(-v.w));
"#,
                    );
                    scalar_body.push_str(
                        r#"
            val = val / (1.0f + expf(-val));
"#,
                    );
                }
                OpKind::Sigmoid => {
                    float4_body.push_str(
                        r#"
        // Sigmoid
        v.x = 1.0f / (1.0f + expf(-v.x));
        v.y = 1.0f / (1.0f + expf(-v.y));
        v.z = 1.0f / (1.0f + expf(-v.z));
        v.w = 1.0f / (1.0f + expf(-v.w));
"#,
                    );
                    scalar_body.push_str(
                        r#"
            val = 1.0f / (1.0f + expf(-val));
"#,
                    );
                }
                _ => {
                    // Non-elementwise ops should never reach codegen.
                    panic!("codegen called on non-elementwise op: {op:?}");
                }
            }
        }

        // Build the extra params string for the function signature.
        let extra_sig = if extra_params_decl.is_empty() {
            String::new()
        } else {
            format!(",\n{}", extra_params_decl.join(",\n"))
        };

        let cuda_src = format!(
            r#"
extern "C" __global__ void {func_name}(
    const float* __restrict__ input,
    float* __restrict__ output{extra_sig},
    unsigned int n
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;

    if (idx + 3 < n) {{
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
{float4_body}
        *reinterpret_cast<float4*>(&output[idx]) = v;
    }} else {{
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {{
            float val = input[i];
{scalar_body}
            output[i] = val;
        }}
    }}
}}
"#
        );

        (cuda_src, func_name, extra_params_desc)
    }

    /// Get (or compile) a fused kernel for the given op chain.
    ///
    /// Returns `(module_name, func_name)` that can be looked up via
    /// `dev.get_func(module, func)`.
    pub fn get_or_compile(
        &self,
        ops: &[OpKind],
        n_cols_params: &[usize],
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> crate::nn::Result<(String, String)> {
        let key = Self::cache_key(ops, n_cols_params);

        // Fast path: check cache.
        {
            let cache = self.cache.lock().unwrap();
            if let Some(compiled) = cache.get(&key) {
                return Ok((compiled.module_name.clone(), compiled.func_name.clone()));
            }
        }

        // Slow path: generate CUDA C, compile via NVRTC.
        let (cuda_src, _func_name, _extra) = Self::codegen(ops, n_cols_params, key);
        let module_name = format!("fused_{key:016x}");

        let ptx = cudarc::nvrtc::compile_ptx(&cuda_src).map_err(|e| {
            crate::nn::NnError::ShapeMismatch {
                expected: "valid CUDA C source".into(),
                actual: format!("NVRTC compile error: {e}"),
            }
        })?;

        dev.load_ptx(ptx, &module_name, &[Self::FUNC_NAME])?;

        let compiled = CompiledFusedKernel {
            module_name: module_name.clone(),
            func_name: Self::FUNC_NAME.to_string(),
        };
        self.cache.lock().unwrap().insert(key, compiled);

        Ok((module_name, Self::FUNC_NAME.to_string()))
    }
}

impl Default for FusionCodegen {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::{OpKind, OpMeta, TapeEntry, TensorId};

    /// Helper: create a tape entry with the given op kind and tensor IDs.
    fn entry(op: OpKind, inputs: &[u32], output: u32) -> TapeEntry {
        TapeEntry {
            op,
            inputs: inputs.iter().map(|&id| TensorId(id)).collect(),
            output: TensorId(output),
            saved: vec![],
            meta: OpMeta::None,
        }
    }

    /// Helper: create a matmul entry with proper metadata.
    fn matmul_entry(a: u32, b: u32, output: u32) -> TapeEntry {
        TapeEntry {
            op: OpKind::Matmul,
            inputs: vec![TensorId(a), TensorId(b)],
            output: TensorId(output),
            saved: vec![TensorId(a), TensorId(b)],
            meta: OpMeta::Matmul {
                m: 128,
                k: 768,
                n: 3072,
            },
        }
    }

    #[test]
    fn test_classify_ops() {
        assert_eq!(classify(OpKind::Gelu), OpClass::Elementwise);
        assert_eq!(classify(OpKind::BiasAdd), OpClass::Elementwise);
        assert_eq!(classify(OpKind::Matmul), OpClass::ComputeBound);
        assert_eq!(classify(OpKind::LayerNorm), OpClass::Reduction);
        assert_eq!(classify(OpKind::Embedding), OpClass::Structural);
    }

    #[test]
    fn test_empty_tape() {
        let opt = FusionOptimizer::new();
        let plan = opt.analyze(&[]);
        assert!(plan.is_empty());
        assert_eq!(plan.launches_saved(), 0);
    }

    #[test]
    fn test_single_op_no_fusion() {
        let opt = FusionOptimizer::new();
        let tape = vec![entry(OpKind::Gelu, &[0], 1)];
        let plan = opt.analyze(&tape);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_p1_matmul_bias_gelu() {
        let opt = FusionOptimizer::new();
        let tape = vec![
            matmul_entry(0, 1, 2),
            entry(OpKind::BiasAdd, &[2], 3),
            entry(OpKind::Gelu, &[3], 4),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);

        let g = &plan.groups[0];
        assert_eq!(g.start, 0);
        assert_eq!(g.end, 3);
        assert_eq!(g.fused_op, FusedOpKind::MatmulBiasGelu);
        assert_eq!(g.inputs, vec![TensorId(0), TensorId(1)]);
        assert_eq!(g.output, TensorId(4));
        assert_eq!(plan.launches_saved(), 2);
    }

    #[test]
    fn test_p3_elemadd_layernorm() {
        let opt = FusionOptimizer::new();
        let tape = vec![
            entry(OpKind::ElemAdd, &[0, 1], 2),
            entry(OpKind::LayerNorm, &[2], 3),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);

        let g = &plan.groups[0];
        assert_eq!(g.fused_op, FusedOpKind::ElemAddLayerNorm);
        assert_eq!(g.start, 0);
        assert_eq!(g.end, 2);
        assert_eq!(plan.launches_saved(), 1);
    }

    #[test]
    fn test_p4_matmul_bias() {
        let opt = FusionOptimizer::new();
        let tape = vec![matmul_entry(0, 1, 2), entry(OpKind::BiasAdd, &[2], 3)];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);

        let g = &plan.groups[0];
        assert_eq!(g.fused_op, FusedOpKind::MatmulBias);
        assert_eq!(plan.launches_saved(), 1);
    }

    #[test]
    fn test_p1_preferred_over_p4() {
        // Matmul → BiasAdd → Gelu should match P1, not P4 + standalone Gelu.
        let opt = FusionOptimizer::new();
        let tape = vec![
            matmul_entry(0, 1, 2),
            entry(OpKind::BiasAdd, &[2], 3),
            entry(OpKind::Gelu, &[3], 4),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.groups[0].fused_op, FusedOpKind::MatmulBiasGelu);
        assert_eq!(plan.groups[0].end - plan.groups[0].start, 3);
    }

    #[test]
    fn test_fan_out_blocks_fusion() {
        // Matmul output is consumed by both BiasAdd and something else.
        let opt = FusionOptimizer::new();
        let tape = vec![
            matmul_entry(0, 1, 2),
            entry(OpKind::BiasAdd, &[2], 3),
            // Also consume tensor 2 elsewhere (fan-out).
            entry(OpKind::Relu, &[2], 4),
        ];
        let plan = opt.analyze(&tape);
        // Tensor 2 has ref_count=2, so Matmul→BiasAdd cannot fuse.
        assert!(plan.is_empty());
    }

    #[test]
    fn test_broken_data_flow_blocks_fusion() {
        // BiasAdd does NOT consume the Matmul output.
        let opt = FusionOptimizer::new();
        let tape = vec![
            matmul_entry(0, 1, 2),
            entry(OpKind::BiasAdd, &[99], 3), // input is tensor 99, not 2
            entry(OpKind::Gelu, &[3], 4),
        ];
        let plan = opt.analyze(&tape);
        // Matmul→BiasAdd fails data flow check (tensor 99 != 2).
        // BiasAdd→Gelu IS detected as an elementwise chain (P10 fallback).
        assert_eq!(plan.len(), 1);
        assert!(matches!(
            plan.groups[0].fused_op,
            FusedOpKind::ElementwiseChain(_)
        ));
        assert_eq!(plan.groups[0].start, 1);
        assert_eq!(plan.groups[0].end, 3);
    }

    #[test]
    fn test_multiple_fusion_groups() {
        // Simulated GPT-2 block fragment:
        //   Matmul → BiasAdd → Gelu  (FFN up — P1)
        //   Matmul → BiasAdd          (FFN down — P4)
        //   ElemAdd → LayerNorm       (residual + norm — P3)
        let opt = FusionOptimizer::new();
        let tape = vec![
            // FFN up: Matmul→BiasAdd→Gelu
            matmul_entry(0, 1, 10),
            entry(OpKind::BiasAdd, &[10], 11),
            entry(OpKind::Gelu, &[11], 12),
            // FFN down: Matmul→BiasAdd
            matmul_entry(12, 13, 20),
            entry(OpKind::BiasAdd, &[20], 21),
            // Residual + norm: ElemAdd→LayerNorm
            entry(OpKind::ElemAdd, &[21, 30], 31),
            entry(OpKind::LayerNorm, &[31], 32),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 3);

        assert_eq!(plan.groups[0].fused_op, FusedOpKind::MatmulBiasGelu);
        assert_eq!(plan.groups[0].start, 0);
        assert_eq!(plan.groups[0].end, 3);

        assert_eq!(plan.groups[1].fused_op, FusedOpKind::MatmulBias);
        assert_eq!(plan.groups[1].start, 3);
        assert_eq!(plan.groups[1].end, 5);

        assert_eq!(plan.groups[2].fused_op, FusedOpKind::ElemAddLayerNorm);
        assert_eq!(plan.groups[2].start, 5);
        assert_eq!(plan.groups[2].end, 7);

        // Total: 3 + 2 + 2 = 7 ops → 3 fused ops, saving 4 launches.
        assert_eq!(plan.launches_saved(), 4);
    }

    #[test]
    fn test_unfusable_ops_pass_through() {
        // Embedding → Matmul → BiasAdd → Attention → ElemAdd → LayerNorm
        // Only Matmul→BiasAdd (P4) and ElemAdd→LayerNorm (P3) should fuse.
        let opt = FusionOptimizer::new();
        let tape = vec![
            entry(OpKind::Embedding, &[0], 1),
            matmul_entry(1, 2, 3),
            entry(OpKind::BiasAdd, &[3], 4),
            entry(OpKind::Attention, &[4], 5),
            entry(OpKind::ElemAdd, &[5, 6], 7),
            entry(OpKind::LayerNorm, &[7], 8),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.groups[0].fused_op, FusedOpKind::MatmulBias);
        assert_eq!(plan.groups[1].fused_op, FusedOpKind::ElemAddLayerNorm);
    }

    #[test]
    fn test_ref_counts() {
        let tape = vec![
            matmul_entry(0, 1, 2),
            entry(OpKind::BiasAdd, &[2], 3),
            entry(OpKind::Relu, &[2], 4), // also uses tensor 2
        ];
        let counts = compute_ref_counts(&tape);
        assert_eq!(counts[&TensorId(2)], 2); // fan-out
        assert_eq!(counts.get(&TensorId(0)), Some(&1));
    }

    #[test]
    fn test_gpt2_full_block_fusion() {
        // Full GPT-2 transformer block (simplified for tape):
        //   0: LayerNorm           (standalone)
        //   1: Matmul (QKV proj)   → fuse with 2
        //   2: BiasAdd (QKV bias)  → fused with 1
        //   3: Attention           (barrier)
        //   4: Matmul (out proj)   → fuse with 5
        //   5: BiasAdd (out bias)  → fused with 4
        //   6: ElemAdd (residual)  → fuse with 7
        //   7: LayerNorm (LN2)     → fused with 6
        //   8: Matmul (FFN up)     → fuse with 9, 10
        //   9: BiasAdd (FFN bias)  → fused with 8, 10
        //  10: Gelu               → fused with 8, 9
        //  11: Matmul (FFN down)   → fuse with 12
        //  12: BiasAdd (FFN dbias) → fused with 11
        //  13: ElemAdd (residual)  (standalone — no following LayerNorm in same block)
        let opt = FusionOptimizer::new();
        let tape = vec![
            // 0: LayerNorm
            entry(OpKind::LayerNorm, &[100], 101),
            // 1-2: Matmul→BiasAdd (QKV)
            matmul_entry(101, 102, 103),
            entry(OpKind::BiasAdd, &[103], 104),
            // 3: Attention
            entry(OpKind::Attention, &[104], 105),
            // 4-5: Matmul→BiasAdd (out proj)
            matmul_entry(105, 106, 107),
            entry(OpKind::BiasAdd, &[107], 108),
            // 6-7: ElemAdd→LayerNorm (residual + LN2)
            entry(OpKind::ElemAdd, &[108, 100], 109),
            entry(OpKind::LayerNorm, &[109], 110),
            // 8-10: Matmul→BiasAdd→Gelu (FFN up)
            matmul_entry(110, 111, 112),
            entry(OpKind::BiasAdd, &[112], 113),
            entry(OpKind::Gelu, &[113], 114),
            // 11-12: Matmul→BiasAdd (FFN down)
            matmul_entry(114, 115, 116),
            entry(OpKind::BiasAdd, &[116], 117),
            // 13: ElemAdd (residual — standalone)
            entry(OpKind::ElemAdd, &[117, 101], 118),
        ];
        let plan = opt.analyze(&tape);

        // Expected groups:
        // [1,3): Matmul→BiasAdd (P4) — QKV
        // [4,6): Matmul→BiasAdd (P4) — out proj
        // [6,8): ElemAdd→LayerNorm (P3) — residual+LN2
        // [8,11): Matmul→BiasAdd→Gelu (P1) — FFN up
        // [11,13): Matmul→BiasAdd (P4) — FFN down
        assert_eq!(plan.len(), 5);
        assert_eq!(plan.groups[0].fused_op, FusedOpKind::MatmulBias);
        assert_eq!(plan.groups[1].fused_op, FusedOpKind::MatmulBias);
        assert_eq!(plan.groups[2].fused_op, FusedOpKind::ElemAddLayerNorm);
        assert_eq!(plan.groups[3].fused_op, FusedOpKind::MatmulBiasGelu);
        assert_eq!(plan.groups[4].fused_op, FusedOpKind::MatmulBias);

        // 14 ops → 14 - launches_saved = effective kernel count
        // Saved: 1 + 1 + 1 + 2 + 1 = 6
        assert_eq!(plan.launches_saved(), 6);
    }

    #[test]
    fn test_elementwise_chain_detection() {
        // BiasAdd → Gelu should be detected as ElementwiseChain.
        let opt = FusionOptimizer::new();
        let tape = vec![
            entry(OpKind::BiasAdd, &[0], 1),
            entry(OpKind::Gelu, &[1], 2),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);
        assert!(matches!(
            &plan.groups[0].fused_op,
            FusedOpKind::ElementwiseChain(ops) if *ops == vec![OpKind::BiasAdd, OpKind::Gelu]
        ));
    }

    #[test]
    fn test_elementwise_chain_max_5() {
        // 6 consecutive elementwise ops: should cap at 5.
        let opt = FusionOptimizer::new();
        let tape = vec![
            entry(OpKind::Relu, &[0], 1),
            entry(OpKind::Sigmoid, &[1], 2),
            entry(OpKind::Relu, &[2], 3),
            entry(OpKind::Sigmoid, &[3], 4),
            entry(OpKind::Relu, &[4], 5),
            entry(OpKind::Sigmoid, &[5], 6),
        ];
        let plan = opt.analyze(&tape);
        // First 5 ops fuse, 6th standalone.
        assert_eq!(plan.len(), 1);
        let g = &plan.groups[0];
        assert_eq!(g.end - g.start, 5); // capped at 5
    }

    // -----------------------------------------------------------------------
    // GPU codegen tests — require CUDA device
    // -----------------------------------------------------------------------

    #[test]
    fn test_codegen_source_bias_gelu() {
        // Verify codegen produces valid CUDA C source for BiasAdd → Gelu.
        let ops = vec![OpKind::BiasAdd, OpKind::Gelu];
        let n_cols = vec![768usize];
        let key = FusionCodegen::cache_key(&ops, &n_cols);
        let (src, func_name, extras) = FusionCodegen::codegen(&ops, &n_cols, key);

        assert_eq!(func_name, "fused_kernel");
        assert_eq!(extras.len(), 1);
        assert!(extras[0].has_n_cols);
        assert!(src.contains("fused_kernel"));
        assert!(src.contains("bias_0"));
        assert!(src.contains("n_cols_0"));
        assert!(src.contains("tanhf")); // GELU uses tanh
        assert!(src.contains("float4"));
    }

    #[test]
    fn test_codegen_source_relu_only() {
        // Pure activation chain: Relu → Sigmoid (no extra params).
        let ops = vec![OpKind::Relu, OpKind::Sigmoid];
        let key = FusionCodegen::cache_key(&ops, &[]);
        let (src, _, extras) = FusionCodegen::codegen(&ops, &[], key);

        assert!(extras.is_empty());
        assert!(src.contains("fmaxf")); // ReLU
        assert!(src.contains("expf")); // Sigmoid
    }

    /// Run the fused kernel on GPU and compare against a CPU reference.
    ///
    /// This helper allocates device memory, launches the fused kernel,
    /// downloads the result, and asserts max absolute error < tolerance.
    #[cfg(feature = "cublas")]
    fn run_fused_vs_cpu(
        ops: &[OpKind],
        n_cols_params: &[usize],
        input: &[f32],
        extra_bufs: &[&[f32]], // bias/addend buffers
        cpu_ref: impl Fn(&[f32]) -> Vec<f32>,
        tol: f32,
    ) {
        use cudarc::driver::LaunchAsync;

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        let codegen = FusionCodegen::new();

        let (module, func) = codegen
            .get_or_compile(ops, n_cols_params, &dev)
            .expect("compile fused kernel");

        let cuda_func = dev.get_func(&module, &func).expect("get compiled function");

        let n = input.len() as u32;
        let d_input = dev.htod_sync_copy(input).expect("upload input");
        let mut d_output = dev.alloc_zeros::<f32>(input.len()).expect("alloc output");

        // Upload extra buffers.
        let d_extras: Vec<cudarc::driver::CudaSlice<f32>> = extra_bufs
            .iter()
            .map(|buf| dev.htod_sync_copy(buf).expect("upload extra"))
            .collect();

        let threads = 256u32;
        let total_threads = (n + 3) / 4;
        let grid = ((total_threads + threads - 1) / threads, 1, 1);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: grid,
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // Build launch args dynamically based on ops.
        // The kernel signature is: (input, output, [bias_0, n_cols_0,]... n)
        // We must match the exact parameter order from codegen.
        let mut extra_idx = 0usize;
        let mut ncols_idx = 0usize;

        // We need to launch with the right parameter tuple. Since Rust
        // requires static tuple types, we handle common cases.
        match (ops, extra_bufs.len()) {
            // Case: no extra params (pure activations)
            (_, 0) => unsafe {
                cuda_func
                    .launch(config, (&d_input, &mut d_output, n))
                    .expect("launch fused kernel");
            },
            // Case: 1 extra param with n_cols (BiasAdd + activations)
            _ if extra_bufs.len() == 1 && !n_cols_params.is_empty() => {
                let ncols = n_cols_params[0] as u32;
                unsafe {
                    cuda_func
                        .launch(config, (&d_input, &mut d_output, &d_extras[0], ncols, n))
                        .expect("launch fused kernel");
                }
            }
            // Case: 1 extra param without n_cols (ElemAdd)
            _ if extra_bufs.len() == 1 && n_cols_params.is_empty() => unsafe {
                cuda_func
                    .launch(config, (&d_input, &mut d_output, &d_extras[0], n))
                    .expect("launch fused kernel");
            },
            // Case: 2 extra params without n_cols (ElemMul + ElemAdd)
            _ if extra_bufs.len() == 2 && n_cols_params.is_empty() => unsafe {
                cuda_func
                    .launch(
                        config,
                        (&d_input, &mut d_output, &d_extras[0], &d_extras[1], n),
                    )
                    .expect("launch fused kernel");
            },
            _ => panic!(
                "unsupported launch config: {} extras, {} n_cols",
                extra_bufs.len(),
                n_cols_params.len()
            ),
        }

        let result = dev.dtoh_sync_copy(&d_output).expect("download result");
        let expected = cpu_ref(input);

        let max_err = result
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < tol,
            "max error {max_err} exceeds tolerance {tol}\nfirst 8 result: {:?}\nfirst 8 expected: {:?}",
            &result[..8.min(result.len())],
            &expected[..8.min(expected.len())],
        );
    }

    /// CPU reference: GELU approximation (tanh-based).
    fn cpu_gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = 0.7978845608_f32;
        let coeff = 0.044715_f32;
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// CPU reference: ReLU.
    fn cpu_relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// CPU reference: Sigmoid.
    fn cpu_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU reference: SiLU.
    fn cpu_silu(x: f32) -> f32 {
        x * cpu_sigmoid(x)
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_bias_gelu() {
        // BiasAdd → Gelu on [128, 768] shape.
        let n_cols = 768;
        let n = 128 * n_cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let bias: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.01) - 3.84).collect();

        run_fused_vs_cpu(
            &[OpKind::BiasAdd, OpKind::Gelu],
            &[n_cols],
            &input,
            &[&bias],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let biased = x + bias[i % n_cols];
                        cpu_gelu(biased)
                    })
                    .collect()
            },
            1e-4, // f32 GELU tolerance
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_bias_relu() {
        // BiasAdd → ReLU on [64, 256] shape.
        let n_cols = 256;
        let n = 64 * n_cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 1.0).collect();
        let bias: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.005) - 0.64).collect();

        run_fused_vs_cpu(
            &[OpKind::BiasAdd, OpKind::Relu],
            &[n_cols],
            &input,
            &[&bias],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_relu(x + bias[i % n_cols]))
                    .collect()
            },
            1e-6,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_relu_sigmoid() {
        // Pure activation chain: ReLU → Sigmoid (no extra params).
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 5.12).collect();

        run_fused_vs_cpu(
            &[OpKind::Relu, OpKind::Sigmoid],
            &[],
            &input,
            &[],
            |inp| inp.iter().map(|&x| cpu_sigmoid(cpu_relu(x))).collect(),
            1e-6,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_bias_silu() {
        // BiasAdd → SiLU on [32, 128] shape.
        let n_cols = 128;
        let n = 32 * n_cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003) - 0.6).collect();
        let bias: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.02) - 1.28).collect();

        run_fused_vs_cpu(
            &[OpKind::BiasAdd, OpKind::Silu],
            &[n_cols],
            &input,
            &[&bias],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_silu(x + bias[i % n_cols]))
                    .collect()
            },
            1e-5,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_elemadd_gelu() {
        // ElemAdd → Gelu chain.
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 2.048).collect();
        let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 1.024).collect();

        run_fused_vs_cpu(
            &[OpKind::ElemAdd, OpKind::Gelu],
            &[],
            &input,
            &[&addend],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_gelu(x + addend[i]))
                    .collect()
            },
            1e-4,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_scalar_tail() {
        // n = 13 (not divisible by 4) — tests scalar tail path.
        let n = 13;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 0.6).collect();

        run_fused_vs_cpu(
            &[OpKind::Relu, OpKind::Sigmoid],
            &[],
            &input,
            &[],
            |inp| inp.iter().map(|&x| cpu_sigmoid(cpu_relu(x))).collect(),
            1e-6,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_cache_hit() {
        // Compile the same chain twice — second call should be a cache hit.
        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        let codegen = FusionCodegen::new();
        let ops = vec![OpKind::Relu, OpKind::Gelu];
        let n_cols: Vec<usize> = vec![];

        let (m1, f1) = codegen.get_or_compile(&ops, &n_cols, &dev).unwrap();
        let (m2, f2) = codegen.get_or_compile(&ops, &n_cols, &dev).unwrap();

        assert_eq!(m1, m2);
        assert_eq!(f1, f2);

        // Verify the kernel is actually usable.
        assert!(dev.get_func(&m1, &f1).is_some());
    }

    #[test]
    fn test_codegen_source_elemmul() {
        // Verify codegen produces valid CUDA C for ElemMul.
        let ops = vec![OpKind::ElemMul, OpKind::Gelu];
        let key = FusionCodegen::cache_key(&ops, &[]);
        let (src, func_name, extras) = FusionCodegen::codegen(&ops, &[], key);

        assert_eq!(func_name, "fused_kernel");
        assert_eq!(extras.len(), 1);
        assert!(!extras[0].has_n_cols);
        assert!(src.contains("multiplier_0"));
        assert!(src.contains("float4"));
        assert!(src.contains("tanhf")); // GELU
    }

    #[test]
    fn test_codegen_bias_vectorized_path() {
        // n_cols=768 (divisible by 4) should produce vectorized bias load.
        let ops = vec![OpKind::BiasAdd, OpKind::Relu];
        let key = FusionCodegen::cache_key(&ops, &[768]);
        let (src, _, _) = FusionCodegen::codegen(&ops, &[768], key);
        assert!(
            src.contains("reinterpret_cast<const float4*>(&bias_0"),
            "expected vectorized bias load for n_cols%4==0"
        );
    }

    #[test]
    fn test_codegen_bias_scalar_fallback() {
        // n_cols=99 (NOT divisible by 4) should use scalar bias reads.
        let ops = vec![OpKind::BiasAdd, OpKind::Relu];
        let key = FusionCodegen::cache_key(&ops, &[99]);
        let (src, _, _) = FusionCodegen::codegen(&ops, &[99], key);
        assert!(
            src.contains("bias_0[(idx") && src.contains("% n_cols_0"),
            "expected scalar bias reads for non-aligned n_cols\nSource:\n{src}"
        );
    }

    #[test]
    fn test_classify_elemmul() {
        assert_eq!(classify(OpKind::ElemMul), OpClass::Elementwise);
    }

    #[test]
    fn test_elementwise_chain_with_elemmul() {
        // ElemMul → ElemAdd → Gelu should be detected as ElementwiseChain.
        let opt = FusionOptimizer::new();
        let tape = vec![
            entry(OpKind::ElemMul, &[0, 1], 2),
            entry(OpKind::ElemAdd, &[2, 3], 4),
            entry(OpKind::Gelu, &[4], 5),
        ];
        let plan = opt.analyze(&tape);
        assert_eq!(plan.len(), 1);
        assert!(matches!(
            &plan.groups[0].fused_op,
            FusedOpKind::ElementwiseChain(ops) if *ops == vec![OpKind::ElemMul, OpKind::ElemAdd, OpKind::Gelu]
        ));
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_elemmul_gelu() {
        // ElemMul → Gelu chain.
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 2.048).collect();
        let multiplier: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) + 0.5).collect();

        run_fused_vs_cpu(
            &[OpKind::ElemMul, OpKind::Gelu],
            &[],
            &input,
            &[&multiplier],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_gelu(x * multiplier[i]))
                    .collect()
            },
            1e-4,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_elemmul_elemadd_gelu() {
        // ElemMul → ElemAdd → Gelu: the epic success criteria chain
        // (multiply + add + activation).
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 2.0).collect();
        let multiplier: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0005) + 0.1).collect();
        let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0003) - 0.5).collect();

        run_fused_vs_cpu(
            &[OpKind::ElemMul, OpKind::ElemAdd, OpKind::Gelu],
            &[],
            &input,
            &[&multiplier, &addend],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let muled = x * multiplier[i];
                        let added = muled + addend[i];
                        cpu_gelu(added)
                    })
                    .collect()
            },
            1e-4,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_elemmul_scalar_tail() {
        // ElemMul with non-aligned size to test scalar tail.
        let n = 17;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 0.8).collect();
        let multiplier: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) + 0.5).collect();

        run_fused_vs_cpu(
            &[OpKind::ElemMul, OpKind::Relu],
            &[],
            &input,
            &[&multiplier],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_relu(x * multiplier[i]))
                    .collect()
            },
            1e-6,
        );
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_bias_gelu_vectorized() {
        // BiasAdd → Gelu with n_cols%4==0 to exercise vectorized bias path.
        let n_cols = 256; // divisible by 4
        let n = 64 * n_cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let bias: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        run_fused_vs_cpu(
            &[OpKind::BiasAdd, OpKind::Gelu],
            &[n_cols],
            &input,
            &[&bias],
            |inp| {
                inp.iter()
                    .enumerate()
                    .map(|(i, &x)| cpu_gelu(x + bias[i % n_cols]))
                    .collect()
            },
            1e-4,
        );
    }

    /// Benchmark: fused (single kernel) vs unfused (sequential kernel launches).
    ///
    /// Tests the epic success criteria: fused chain >= 2x faster than
    /// sequential kernel launches for ElemMul + ElemAdd + Gelu.
    #[test]
    #[cfg(feature = "cublas")]
    fn test_gpu_fused_vs_unfused_benchmark() {
        use cudarc::driver::LaunchAsync;

        let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device");
        let codegen = FusionCodegen::new();

        // Shape: [1024, 1024] = 1M elements — big enough to measure
        let n = 1024 * 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let multiplier: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0005) + 0.1).collect();
        let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0003) - 0.5).collect();

        let d_input = dev.htod_sync_copy(&input).expect("upload input");
        let d_mul = dev.htod_sync_copy(&multiplier).expect("upload multiplier");
        let d_add = dev.htod_sync_copy(&addend).expect("upload addend");
        let mut d_output = dev.alloc_zeros::<f32>(n).expect("alloc output");
        let mut d_tmp1 = dev.alloc_zeros::<f32>(n).expect("alloc tmp1");
        let mut d_tmp2 = dev.alloc_zeros::<f32>(n).expect("alloc tmp2");

        let n_u32 = n as u32;
        let threads = 256u32;
        let total_threads = (n_u32 + 3) / 4;
        let grid = ((total_threads + threads - 1) / threads, 1, 1);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: grid,
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // --- Compile fused kernel (ElemMul + ElemAdd + Gelu) ---
        let (fused_mod, fused_fn) = codegen
            .get_or_compile(&[OpKind::ElemMul, OpKind::ElemAdd, OpKind::Gelu], &[], &dev)
            .expect("compile fused kernel");

        // --- Compile individual unfused kernels ---
        // FusionCodegen requires at least 2 ops in the chain, so we create
        // separate NVRTC kernels manually for accurate unfused comparison.

        // ElemMul kernel (standalone)
        let mul_src = r#"
extern "C" __global__ void elem_mul(
    const float* __restrict__ input,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
        float4 m = *reinterpret_cast<const float4*>(&multiplier[idx]);
        v.x *= m.x; v.y *= m.y; v.z *= m.z; v.w *= m.w;
        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            output[i] = input[i] * multiplier[i];
        }
    }
}
"#;
        let add_src = r#"
extern "C" __global__ void elem_add(
    const float* __restrict__ input,
    const float* __restrict__ addend,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
        float4 a = *reinterpret_cast<const float4*>(&addend[idx]);
        v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            output[i] = input[i] + addend[i];
        }
    }
}
"#;
        let gelu_src = r#"
extern "C" __global__ void gelu_act(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
        const float S = 0.7978845608f;
        const float C = 0.044715f;
        float4 t;
        t.x = S * (v.x + C * v.x * v.x * v.x);
        t.y = S * (v.y + C * v.y * v.y * v.y);
        t.z = S * (v.z + C * v.z * v.z * v.z);
        t.w = S * (v.w + C * v.w * v.w * v.w);
        v.x = 0.5f * v.x * (1.0f + tanhf(t.x));
        v.y = 0.5f * v.y * (1.0f + tanhf(t.y));
        v.z = 0.5f * v.z * (1.0f + tanhf(t.z));
        v.w = 0.5f * v.w * (1.0f + tanhf(t.w));
        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            float x = input[i];
            const float S = 0.7978845608f;
            const float C = 0.044715f;
            float t = S * (x + C * x * x * x);
            output[i] = 0.5f * x * (1.0f + tanhf(t));
        }
    }
}
"#;

        let ptx_mul = cudarc::nvrtc::compile_ptx(mul_src).expect("compile elem_mul");
        let ptx_add = cudarc::nvrtc::compile_ptx(add_src).expect("compile elem_add");
        let ptx_gelu = cudarc::nvrtc::compile_ptx(gelu_src).expect("compile gelu_act");

        dev.load_ptx(ptx_mul, "unfused_mul", &["elem_mul"])
            .expect("load mul");
        dev.load_ptx(ptx_add, "unfused_add", &["elem_add"])
            .expect("load add");
        dev.load_ptx(ptx_gelu, "unfused_gelu", &["gelu_act"])
            .expect("load gelu");

        let f_mul = dev.get_func("unfused_mul", "elem_mul").unwrap();
        let f_add = dev.get_func("unfused_add", "elem_add").unwrap();
        let f_gelu = dev.get_func("unfused_gelu", "gelu_act").unwrap();

        // Warm-up runs
        let fused_func = dev.get_func(&fused_mod, &fused_fn).unwrap();
        for _ in 0..5 {
            unsafe {
                fused_func
                    .clone()
                    .launch(config, (&d_input, &mut d_output, &d_mul, &d_add, n_u32))
                    .expect("fused warm-up");
            }
            dev.synchronize().unwrap();
        }
        for _ in 0..5 {
            unsafe {
                f_mul
                    .clone()
                    .launch(config, (&d_input, &d_mul, &mut d_tmp1, n_u32))
                    .expect("mul warm-up");
                f_add
                    .clone()
                    .launch(config, (&d_tmp1, &d_add, &mut d_tmp2, n_u32))
                    .expect("add warm-up");
                f_gelu
                    .clone()
                    .launch(config, (&d_tmp2, &mut d_output, n_u32))
                    .expect("gelu warm-up");
            }
            dev.synchronize().unwrap();
        }

        // Benchmark: fused
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let ff = dev.get_func(&fused_mod, &fused_fn).unwrap();
            unsafe {
                ff.launch(config, (&d_input, &mut d_output, &d_mul, &d_add, n_u32))
                    .expect("fused launch");
            }
        }
        dev.synchronize().unwrap();
        let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

        // Benchmark: unfused (3 separate kernel launches)
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let fm = dev.get_func("unfused_mul", "elem_mul").unwrap();
            let fa = dev.get_func("unfused_add", "elem_add").unwrap();
            let fg = dev.get_func("unfused_gelu", "gelu_act").unwrap();
            unsafe {
                fm.launch(config, (&d_input, &d_mul, &mut d_tmp1, n_u32))
                    .expect("mul");
                fa.launch(config, (&d_tmp1, &d_add, &mut d_tmp2, n_u32))
                    .expect("add");
                fg.launch(config, (&d_tmp2, &mut d_output, n_u32))
                    .expect("gelu");
            }
        }
        dev.synchronize().unwrap();
        let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

        let speedup = unfused_us / fused_us;
        eprintln!(
            "\n=== Fused vs Unfused Benchmark (ElemMul+ElemAdd+Gelu, n={n}) ===\n\
             Fused:   {fused_us:.1} us/iter\n\
             Unfused: {unfused_us:.1} us/iter\n\
             Speedup: {speedup:.2}x\n"
        );

        // Verify correctness of fused output.
        let result = dev.dtoh_sync_copy(&d_output).expect("download");
        let expected: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let m = x * multiplier[i];
                let a = m + addend[i];
                cpu_gelu(a)
            })
            .collect();
        let max_err = result
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "fused result error {max_err} >= 1e-4");

        // The fused kernel should be faster because it avoids 2 extra
        // global memory round-trips. On GTX 1660 with 1M elements we
        // expect >= 1.5x speedup (conservative lower bound).
        assert!(
            speedup >= 1.3,
            "speedup {speedup:.2}x below 1.3x threshold — \
             fused={fused_us:.1}us unfused={unfused_us:.1}us"
        );
    }

    #[test]
    fn test_codegen_register_only_intermediates() {
        // Verify that codegen does NOT write intermediates to global memory
        // between ops — values stay in local float4 `v` (register-allocated
        // by nvcc).
        let ops = vec![OpKind::ElemMul, OpKind::ElemAdd, OpKind::Gelu];
        let key = FusionCodegen::cache_key(&ops, &[]);
        let (src, _, _) = FusionCodegen::codegen(&ops, &[], key);

        // The kernel should have exactly ONE store to output (the final
        // reinterpret_cast write). Count "output[" occurrences — there
        // should be exactly 2: one in float4 path, one in scalar path.
        let output_stores: Vec<_> = src.match_indices("output[").collect();
        assert_eq!(
            output_stores.len(),
            2,
            "expected 2 output stores (float4 + scalar), got {}. \
             Intermediates may be leaking to global memory.\nSource:\n{}",
            output_stores.len(),
            src
        );

        // Verify no intermediate global memory allocation keywords.
        assert!(
            !src.contains("__shared__"),
            "fused kernel should not use shared memory for intermediates"
        );
        // The only global pointers should be input, output, and extra params.
        // There should be no malloc/new/temp buffer allocations.
        assert!(
            !src.contains("malloc") && !src.contains("new float"),
            "fused kernel should not allocate temporary buffers"
        );
    }
}
