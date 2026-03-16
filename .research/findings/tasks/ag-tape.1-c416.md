# ag-tape.1: Design TapeEntry, Tape, TensorPool, OpKind, OpMeta structs
**Cycle**: 416 | **Theme**: ag-tape | **Kind**: design | **Status**: done

## Summary
Design for GPU autograd tape with TensorId-based tracking, operation recording,
and reverse-topological backward traversal.

## Architecture

### Core Types

```rust
/// Unique identifier for a tensor in the autograd graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(u32);

/// Kind of operation recorded on the tape.
#[derive(Copy, Clone, Debug)]
pub enum OpKind {
    Matmul,       // dA = dC * B^T, dB = A^T * dC
    BiasAdd,      // dBias = sum(dOut, dim=0)
    LayerNorm,    // fused backward kernel
    Gelu,         // dX = dOut * gelu'(X)
    Silu,         // dX = dOut * silu'(X)
    Sigmoid,      // dX = dOut * sig(X) * (1 - sig(X))
    Relu,         // dX = dOut * (X > 0)
    ElemAdd,      // dA = dOut, dB = dOut (passthrough)
    Embedding,    // scatter_add to weight grad
    CrossEntropy, // combined log_softmax + nll backward
    MseLoss,      // dX = 2/n * (X - target)
}

/// Metadata for a recorded operation (saved tensors for backward).
pub struct TapeEntry {
    pub op: OpKind,
    pub inputs: Vec<TensorId>,      // input tensor IDs
    pub output: TensorId,           // output tensor ID
    pub saved: Vec<TensorId>,       // tensors saved for backward (e.g., input for GELU')
    pub meta: OpMeta,               // operation-specific metadata
}

/// Operation-specific metadata (shapes, hyperparams).
pub enum OpMeta {
    None,
    Matmul { m: usize, k: usize, n: usize },
    LayerNorm { rows: usize, d: usize, eps: f32 },
    BiasAdd { n_cols: usize },
    Embedding { vocab_size: usize },
    Loss { reduction: Reduction },
}

pub enum Reduction { Mean, Sum }

/// The autograd tape: records forward ops for backward traversal.
pub struct Tape {
    entries: Vec<TapeEntry>,
    recording: bool,         // toggle recording on/off
    next_tensor_id: u32,     // monotonic counter
}

/// Pool of GPU tensors keyed by TensorId.
/// Holds intermediate activations and gradients.
pub struct TensorPool {
    tensors: HashMap<TensorId, GpuTensor>,
}
```

### Key Design Decisions

1. **TensorId, not references**: GPU tensors live in a HashMap pool, addressed by ID.
   This avoids borrow checker issues with the computation graph. IDs are cheap to copy.

2. **Tape is append-only during forward**: Each op appends a TapeEntry. No mutation
   of existing entries. Backward reads in reverse.

3. **Saved tensors**: Operations that need inputs during backward (e.g., GELU needs
   the pre-activation value) explicitly save tensor IDs. The pool keeps these alive.

4. **No graph pruning in v1**: Keep all entries. Memory management via explicit
   `tape.clear()` between batches.

5. **OpKind as enum**: Finite set of ops. The backward dispatch is a match on OpKind.
   No dynamic dispatch needed — all backward kernels are known at compile time.

6. **Recording toggle**: `tape.recording = false` during inference (no tape overhead).
   During training, `tape.recording = true`.

### Backward Algorithm

```
fn backward(tape: &Tape, pool: &mut TensorPool, loss_id: TensorId):
    // Initialize gradient of loss as 1.0
    grads[loss_id] = ones_like(pool[loss_id])

    // Reverse topological order = reverse tape order (tape is topologically sorted)
    for entry in tape.entries.iter().rev():
        if grads[entry.output] is None:
            continue  // not in the backward path

        let d_out = grads[entry.output]
        match entry.op:
            Matmul =>
                d_a = matmul(d_out, pool[B].T)
                d_b = matmul(pool[A].T, d_out)
                accumulate(grads, entry.inputs[0], d_a)
                accumulate(grads, entry.inputs[1], d_b)
            Gelu =>
                d_x = gelu_backward(d_out, pool[saved_x])
                accumulate(grads, entry.inputs[0], d_x)
            ...
```

### File Structure

```
nn/
├── autograd/
│   ├── mod.rs        // pub use, Tape, backward()
│   ├── tape.rs       // TapeEntry, OpKind, OpMeta, Tape impl
│   ├── pool.rs       // TensorPool
│   └── backward.rs   // backward dispatch per OpKind
```

## Impact on Downstream Tasks
- ag-tape.2: Implement TensorPool (straightforward HashMap wrapper)
- ag-tape.3: Implement Tape with recording toggle
- ag-tape.4: Add requires_grad to GpuTensor
- ag-tape.5: Wire forward ops to record on tape
- ag-tape.6: Implement backward() traversal
