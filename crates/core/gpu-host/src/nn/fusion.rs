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
        | OpKind::ElemAdd => OpClass::Elementwise,

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
}

impl fmt::Display for FusedOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatmulBiasGelu => write!(f, "MatmulBiasGelu"),
            Self::ElemAddLayerNorm => write!(f, "ElemAddLayerNorm"),
            Self::MatmulBias => write!(f, "MatmulBias"),
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
        .map_or(true, |&count| count == 1)
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

        None
    }
}

impl Default for FusionOptimizer {
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
        // Matmul→BiasAdd fails data flow check.
        // BiasAdd→Gelu is elementwise chain but not in our pattern catalog.
        assert!(plan.is_empty());
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
}
