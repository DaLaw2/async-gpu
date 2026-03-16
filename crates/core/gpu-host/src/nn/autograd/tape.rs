//! Autograd tape — records forward operations for backward traversal.

/// Unique identifier for a tensor in the autograd computation graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

/// Kind of operation recorded on the tape.
#[derive(Copy, Clone, Debug)]
pub enum OpKind {
    /// Matrix multiplication: C = A * B
    Matmul,
    /// Bias addition: Y = X + bias (broadcast along last dim)
    BiasAdd,
    /// Layer normalization
    LayerNorm,
    /// GELU activation
    Gelu,
    /// SiLU activation
    Silu,
    /// Sigmoid activation
    Sigmoid,
    /// ReLU activation
    Relu,
    /// Element-wise addition: C = A + B
    ElemAdd,
    /// Embedding lookup
    Embedding,
    /// Cross-entropy loss (log_softmax + NLL)
    CrossEntropy,
    /// Mean squared error loss
    MseLoss,
}

/// Operation-specific metadata for backward computation.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum OpMeta {
    /// No extra metadata needed.
    None,
    /// Matmul dimensions: A=[m,k], B=[k,n], C=[m,n].
    Matmul { m: usize, k: usize, n: usize },
    /// LayerNorm parameters.
    LayerNorm { rows: usize, d: usize, eps: f32 },
    /// BiasAdd: number of columns (bias dimension).
    BiasAdd { n_cols: usize },
    /// Embedding: vocabulary size.
    Embedding { vocab_size: usize },
    /// Loss reduction mode.
    Loss { reduction: Reduction },
}

/// Loss reduction mode.
#[derive(Copy, Clone, Debug)]
pub enum Reduction {
    /// Average over elements.
    Mean,
    /// Sum over elements.
    Sum,
}

/// A single recorded operation on the tape.
#[derive(Clone, Debug)]
pub struct TapeEntry {
    /// The operation that was performed.
    pub op: OpKind,
    /// Input tensor IDs (e.g., [A, B] for matmul).
    pub inputs: Vec<TensorId>,
    /// Output tensor ID.
    pub output: TensorId,
    /// Tensor IDs saved for backward (e.g., pre-activation for GELU').
    pub saved: Vec<TensorId>,
    /// Operation-specific metadata.
    pub meta: OpMeta,
}

/// Autograd tape: append-only record of forward operations.
///
/// During the forward pass with `recording = true`, each op appends a [`TapeEntry`].
/// The backward pass reads entries in reverse order (natural reverse topological sort
/// since the tape is written in forward execution order).
pub struct Tape {
    entries: Vec<TapeEntry>,
    recording: bool,
    next_id: u32,
}

impl Tape {
    /// Create a new empty tape with recording enabled.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            recording: true,
            next_id: 0,
        }
    }

    /// Allocate a new tensor ID.
    pub fn alloc_id(&mut self) -> TensorId {
        let id = TensorId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Whether the tape is currently recording operations.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Enable or disable recording.
    pub fn set_recording(&mut self, on: bool) {
        self.recording = on;
    }

    /// Record an operation on the tape (only if recording is enabled).
    pub fn record(&mut self, entry: TapeEntry) {
        if self.recording {
            self.entries.push(entry);
        }
    }

    /// Get all recorded entries (for backward traversal).
    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tape has no recorded operations.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries and reset (e.g., between training batches).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 0;
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_record_and_iterate() {
        let mut tape = Tape::new();
        let a = tape.alloc_id();
        let b = tape.alloc_id();
        let c = tape.alloc_id();

        tape.record(TapeEntry {
            op: OpKind::Matmul,
            inputs: vec![a, b],
            output: c,
            saved: vec![a, b],
            meta: OpMeta::Matmul { m: 4, k: 8, n: 6 },
        });

        assert_eq!(tape.len(), 1);
        assert_eq!(tape.entries()[0].output, c);
    }

    #[test]
    fn test_tape_recording_toggle() {
        let mut tape = Tape::new();
        let a = tape.alloc_id();
        let b = tape.alloc_id();

        tape.set_recording(false);
        tape.record(TapeEntry {
            op: OpKind::Relu,
            inputs: vec![a],
            output: b,
            saved: vec![],
            meta: OpMeta::None,
        });

        assert_eq!(tape.len(), 0); // Not recorded
    }

    #[test]
    fn test_tape_clear() {
        let mut tape = Tape::new();
        let _ = tape.alloc_id();
        tape.record(TapeEntry {
            op: OpKind::Gelu,
            inputs: vec![TensorId(0)],
            output: TensorId(1),
            saved: vec![TensorId(0)],
            meta: OpMeta::None,
        });
        assert_eq!(tape.len(), 1);

        tape.clear();
        assert_eq!(tape.len(), 0);
        assert_eq!(tape.alloc_id(), TensorId(0)); // IDs reset
    }
}
