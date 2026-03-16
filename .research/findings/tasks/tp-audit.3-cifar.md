# tp-audit.3 — CIFAR-10 CNN Training Audit

## Summary

The CIFAR-10 GPU training example produces learning curves comparable to CPU
but is **~2x slower** (12.7s GPU vs 6.5s CPU for 10 epochs). The gradient flow
is complete and correct in structure, but there is an **FC weight layout bug**
that happens to be masked by the symmetric optimization landscape, and a
**loss averaging bug** present in both modes. The primary performance problem
is an astronomical number of redundant H2D/D2H memory transfers per batch.

## A. Forward Pass — Data Flow Verification

**Path:** input image → GPU im2col + GEMM (conv2d) → D2H → CPU ReLU → CPU avg_pool → H2D → GPU matmul (FC) → GPU bias_add → D2H → CPU softmax

The data flow is **structurally correct** but suffers from excessive transfers:

1. **conv2d per sample** (lines 68-76): Each of the 32 samples does a full
   GPU conv2d. Inside `conv2d()` (conv.rs), the im2col output is **copied back
   to CPU** (line 93: `dtoh_sync_copy`), transposed on CPU, then **re-uploaded**
   (line 104: `from_host`). Then `matmul()` copies both inputs back to CPU for
   padding, re-uploads them, runs the kernel, copies the result back, and
   re-uploads the unpadded version. The final conv output is also copied to host
   (line 71). Total: **~13 transfers + 2 kernel launches per sample**.

2. **No data corruption** in the conv→relu→pool→FC pipeline. Values flow
   correctly through the CPU intermediaries. The conv output layout [C_out, H, W]
   matches what `cpu_avg_pool` expects.

3. **FC forward** (lines 82-106): The batched matmul [32, 512] × [512, 10]
   correctly produces [32, 10] logits. bias_add broadcasts correctly along the
   last dimension.

## B. Backward Pass — Gradient Chain

**Path:** d_logits → GPU (BiasAdd passthrough) → GPU matmul backward (dW_fc, d_feat) → D2H → CPU avg_unpool → CPU relu_mask → CPU conv_weight_grad

The gradient chain is **complete**:

1. **BiasAdd backward** (line 153): `entry.op == BiasAdd` passes `d_out`
   through to `entry.inputs[0]` (the pre-bias logits tensor). Correct — bias
   gradient is handled separately on CPU (line 164).

2. **Matmul backward** (lines 136-150): Computes both:
   - `dW_fc = feat^T × d_logits` → stored at TensorId(1)
   - `d_feat = d_logits × W_fc^T` → stored at TensorId(0)
   Both use pool lookups for saved activations. Correct.

3. **d_feat reaches conv weight update** (line 167): `grads.get(&TensorId(0))`
   retrieves the feature gradient. It is unpooled, masked by ReLU, and used
   to compute `cpu_conv2d_wgrad` per sample. The gradient accumulates across
   the batch, then updates `conv_w`. **The chain is complete.**

4. **Conv weight re-upload** (line 180): `cw_gpu = GpuTensor::from_host(...)`.
   Correct — updated weights are uploaded for the next batch.

## C. Weight Update Verification

| Weight   | Updated? | Re-uploaded to GPU? | Correct? |
|----------|----------|---------------------|----------|
| `conv_w` | Yes (line 178) | Yes (line 180) | Yes |
| `fc_w`   | Yes (line 162) | Yes (re-created each batch, line 89) | **BUG** |
| `fc_b`   | Yes (line 164) | Yes (re-created each batch, line 100) | Yes |

### FC Weight Layout Bug

`fc_w` is initialized with layout `[nc=10, flat=512]` row-major (line 44:
index `o*flat+j`), matching the CPU code (line 224: `fw[o*flat+j]`).

However, the GPU path uploads it as `[flat=512, nc=10]` (line 89):
```rust
let mut fw = GpuTensor::from_host(&fc_w, &[flat, nc], &dev).unwrap();
```

This reinterprets the memory: `GpuTensor[row=j, col=o] = fc_w[j*10 + o]`,
but the intended weight `W[j,o]` should be `fc_w[o*512 + j]`. These differ.

The matmul backward computes `dW` in `[flat, nc]` layout, then the update
(line 162) applies it element-wise to `fc_w` which is in `[nc, flat]` layout.
**Gradients are applied to wrong weight positions.**

This bug is partially masked because:
- Both modes use the same random seed → same init
- SGD with per-element updates finds a local minimum regardless of which
  logical weight each memory position represents
- The loss curves are similar but NOT identical (GPU epoch 10: train=28.6%
  vs CPU epoch 10: train=27.2%)

### Loss Averaging Bug (both modes)

Line 183: `total_loss /= bs as f64;` is inside the batch loop but `total_loss`
accumulates across all batches. After batch 0, it divides by bs. After batch 1,
it divides the residual + new by bs again. This compounds the division, making
displayed loss values smaller than the true average cross-entropy.

The CPU code has the identical bug (line 243: `tl2/=bs as f64;`), so the
loss curves appear to match between modes.

## D. Actual Training Results

### GPU Mode
```
CIFAR-10 GPU training (2000 train, 500 test)
Epoch  1/10: loss=0.0382, train=10.8%, test=12.4%, time=1.3s
Epoch  5/10: loss=0.0375, train=12.2%, test=14.8%, time=1.3s
Epoch 10/10: loss=0.0346, train=28.6%, test=22.6%, time=1.3s
Total: 12.7s
```

### CPU Mode
```
CIFAR-10 CPU training (2000 train, 500 test)
Epoch  1/10: loss=0.0381, train=11.7%, test=13.6%, time=0.6s
Epoch  5/10: loss=0.0373, train=13.7%, test=16.0%, time=0.6s
Epoch 10/10: loss=0.0347, train=27.2%, test=21.0%, time=0.6s
Total: 6.5s
```

**Key observations:**
- GPU is **~2x slower** (12.7s vs 6.5s)
- Loss curves are nearly identical (both start ~0.038, end ~0.035)
- Final accuracy is similar (GPU 28.6%/22.6% vs CPU 27.2%/21.0% train/test)
- Small divergence is consistent with the FC weight layout bug — GPU and CPU
  are effectively training different (but structurally similar) networks

## E. Performance Bottleneck Analysis

### Transfers and Kernel Launches Per Batch (bs=32)

| Phase | H2D | D2H | Kernels | Notes |
|-------|-----|-----|---------|-------|
| Conv forward (×32 samples) | ~224 | ~192 | 64 | im2col+GEMM per sample, each matmul does to_host→pad→htod |
| FC forward (1 batched) | ~8 | ~5 | 2 | matmul + bias_add |
| FC backward (2 matmuls) | ~12 | ~10 | 2 | dW + d_feat |
| Conv backward (CPU) | 1 | 1 | 0 | Single d_feat download |
| **Total per batch** | **~245** | **~208** | **~68** | |

With 62 batches per epoch: **~15,190 H2D + ~12,896 D2H + ~4,216 kernel launches per epoch**.

### Root Cause: matmul copies everything to CPU for padding

The `matmul()` function in `gemm.rs` (lines 47-54, 57-63) calls `to_host()`
on BOTH input tensors, pads them on CPU, then re-uploads. This means **every
GPU matmul does 2 D2H + 2 H2D + 1 kernel + 1 D2H (result)** — 5 transfers
for one multiplication. The actual GEMM kernel execution is likely <1% of
wall-clock time.

### Estimated Time Breakdown

- Each PCIe transfer ~10-50µs for small tensors (3072 floats = 12KB)
- ~453 transfers per batch × ~30µs avg = ~13.6ms transfer overhead per batch
- 62 batches × 13.6ms = ~843ms per epoch just in transfers
- Actual kernel time: 68 kernels × ~5µs = ~0.34ms per batch
- Measured: 1.3s per epoch → **~65% is transfer overhead, ~30% CUDA synchronization,
  <5% actual computation**

### conv2d is the worst offender

The `conv2d()` function in `conv.rs` downloads the im2col result to CPU
(line 93), transposes it on CPU (lines 96-101), re-uploads it (line 104),
and its internal `matmul()` call does the same download-pad-upload dance.
A single conv2d sample triggers **~7 H2D + ~6 D2H transfers**.

## F. conv2d_batched (conv.rs) — Correctness Review

The batched path (lines 160-266) is **not used** by the CIFAR training example
(it passes 3D `[C, H, W]` tensors, so the single-sample path is taken). But
reviewing it for correctness:

### Column Rearrangement Logic (lines 252-263)

```rust
// GEMM output is [c_out, batch*col_w]
// Rearrange to [batch, c_out, h_out, w_out]
for b in 0..batch {
    for ch in 0..c_out {
        for s in 0..col_w {
            output_data[b * c_out * col_w + ch * col_w + s] =
                result[ch * big_col_w + b * col_w + s];
        }
    }
}
```

The big matmul concatenates im2col columns as:
`BigCol = [col_sample0 | col_sample1 | ... | col_sample_{N-1}]`

So `result[ch, b*col_w + s] = output for channel ch, sample b, spatial pos s`.

The rearrangement reads `result[ch * big_col_w + b * col_w + s]` and writes to
`output_data[b * c_out * col_w + ch * col_w + s]`. This correctly converts from
`[C_out, N*spatial]` to `[N, C_out, spatial]`. **No bug here.**

However, the bias addition (lines 240-249) indexes:
```rust
result[b * c_out * col_w + ch * col_w + i] += b_val;
```
This uses `[N, C_out, spatial]` indexing on `result`, but `result` is still in
`[C_out, N*spatial]` layout at this point (rearrangement happens after). **This
is a bug** — bias is added to wrong positions in the batched path. Since the
CIFAR example doesn't use this path (no bias in conv), it has no effect here.

## Recommendations

1. **Fix FC weight layout**: Either initialize `fc_w` as `[flat, nc]` or
   transpose before upload. The GPU matmul expects `[flat, nc]` row-major.

2. **Fix loss averaging**: Replace `total_loss /= bs as f64` inside the loop
   with accumulation, then divide once: `let avg_loss = total_loss / (nb * bs) as f64`.

3. **Eliminate redundant transfers in matmul**: Pad on GPU (a simple kernel) or
   allocate pre-padded tensors. The current approach defeats the purpose of GPU
   acceleration for small matrices.

4. **Batch the conv forward**: Instead of 32 individual conv2d calls, use
   conv2d_batched (after fixing its bias bug) or batch the im2col+GEMM.

5. **Fix conv2d_batched bias order**: Move the bias addition after the
   rearrangement, or index with `[C_out, N*spatial]` layout.
