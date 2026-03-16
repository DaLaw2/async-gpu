# gs-spike.2: Decision — GpuRef vs Arena vs Current RAII

## Question

Should we invest in a gpu-safety epic (GpuRef<'a, T>, arena allocator,
lifetime-tracked GPU references)? Or is the current design already sufficient?

---

## Option A: Current RAII (status quo)

**How it works:** Each `GpuTensor` owns a `CudaSlice<f32>`. When the tensor
is dropped, the device memory is freed. Every kernel launch is followed by
`dev.synchronize()`. No shared views, no aliasing.

**What it prevents:**
- Use-after-free: impossible (RAII + synchronous execution)
- Double-free: impossible (CudaSlice has no Clone)
- Dangling pointers: impossible (no raw pointers exposed to users)
- Buffer aliasing: impossible (every op copies)

**What it doesn't prevent:**
- Nothing safety-related. The audit (gs-spike.1) found zero safety bugs.

**Cost:** Extra copies on reshape/transpose/split/concat. Every operation
allocates new GPU memory and copies data.

---

## Option B: GpuRef<'a, T> (borrow from arena)

**Concept:** An arena owns all GPU memory. Tensors are `GpuRef<'a, T>`
references that borrow from the arena. The arena is freed all at once
(e.g., at the end of a forward pass).

**What it would add:**
- Compile-time guarantee that tensors don't outlive the arena
- Batch deallocation (potentially faster than individual CUDA frees)

**What it would NOT add:**
- Any safety beyond what we already have. The current RAII design is already
  safe. GpuRef would add a lifetime parameter to every function signature
  for zero additional safety benefit.

**Cost:**
- Lifetime annotations propagate virally through the entire API
- Every function gains `<'a>` parameters
- Arena lifetime must be threaded through layers, attention, model code
- Significantly worse ergonomics for users
- Complex interaction with autograd tape (tape entries would need lifetime)

**Real bugs it would prevent:** None found. The gs-spike.1 audit identified
zero lifetime safety gaps.

---

## Option C: Tensor Views (zero-copy reshape/slice)

**Concept:** Allow `GpuTensor` to reference a sub-region of another tensor's
memory without copying. A view shares the underlying `CudaSlice` (via
`Arc<CudaSlice<f32>>` or similar).

**What it would add:**
- Zero-copy `reshape`: O(1) instead of O(n) memcpy
- Zero-copy `slice`: extract sub-tensors without allocation
- Zero-copy `transpose`: change strides without copying data (for
  contiguous-only kernels, materialize on demand)

**Current waste:** Every reshape in the conv2d pipeline copies data:
- `weight.reshape(&[c_out, col_h])` — copies entire weight tensor
- `gemm_out.reshape(&[c_out, h_out, w_out])` — copies entire output
- `transpose()` round-trips through host memory (!)

**Performance impact estimate:**
- GPT-2 forward pass: ~6 reshapes per layer x 12 layers = 72 unnecessary
  copies
- YOLO inference: reshape + transpose in every conv layer
- For a 768-dim model with seq_len=1024: each reshape copies 3MB

**Real value:** This solves an actual performance problem that users would
care about. "My model is slow because reshape copies" is a real complaint.
"My model crashed because of a lifetime bug" has never happened.

**Cost:**
- Need `Arc<CudaSlice<f32>>` or equivalent shared ownership
- Views need offset + length fields
- Must track whether a view is contiguous (already have `is_contiguous()`)
- Kernels that require contiguous data need a `materialize()` path
- Slightly more complex Drop semantics (but Arc handles this)

---

## Option D: Skip entirely

Do nothing. The current design is safe and functional. Focus engineering
effort on features that directly improve model inference quality and speed
(better kernels, more model support, etc.).

---

## Decision Matrix

| Criterion | RAII (A) | GpuRef (B) | Views (C) | Skip (D) |
|-----------|----------|------------|-----------|----------|
| Safety improvement | baseline | +0 | +0 | baseline |
| Performance improvement | baseline | ~0 | significant | baseline |
| API ergonomics | good | much worse | slightly better | good |
| Implementation effort | 0 | high | medium | 0 |
| User-visible benefit | none | none | faster inference | none |
| Risk | none | lifetime virality | Arc complexity | none |

---

## Recommendation: **Reframe as tensor-views (Option C), skip GpuRef (Option B)**

### Rationale

1. **GpuRef solves a non-problem.** The audit found zero safety bugs. Adding
   lifetime annotations everywhere would hurt ergonomics for zero safety
   gain. This is classic over-engineering.

2. **Tensor views solve a real problem.** The current design copies on every
   reshape, transpose, and split. In a 12-layer transformer, that's dozens
   of unnecessary GPU memcpy operations per forward pass. Views would
   provide measurable speedup.

3. **The safety epic framing is wrong.** The codebase is already safe. The
   useful work is performance optimization (views), not safety hardening
   (GpuRef/arena). Reframing avoids wasting effort on a non-issue.

### Suggested scope for a tensor-views theme

- **Task 1:** Add `Arc<CudaSlice<f32>>` shared ownership + offset/length to
  `GpuTensor`. Reshape becomes O(1) metadata change.
- **Task 2:** Zero-copy `slice()` and `narrow()` operations.
- **Task 3:** Lazy contiguous materialization — `materialize_contiguous()`
  called automatically before kernel launch if strides are non-standard.
- **Task 4:** Benchmark: measure memcpy elimination in GPT-2 forward pass.

### What NOT to do

- Do not add lifetime parameters to the tensor API
- Do not build an arena allocator
- Do not add `GpuRef<'a, T>` or any borrow-from-pool pattern
- Do not create a "gpu-safety" epic — there is no safety problem to solve

### When to revisit

Re-evaluate if:
- Async stream-based execution is introduced (concurrent kernel + host access)
- Multi-GPU tensor migration is added (cross-device aliasing)
- Users report actual safety bugs (none so far)

Until then, the combination of cudarc RAII + synchronous execution +
Rust ownership is sufficient.
