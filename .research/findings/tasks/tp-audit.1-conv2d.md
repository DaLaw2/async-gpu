# tp-audit.1 — Conv2d Implementation Audit

## Summary

Audited the conv2d implementation across three files: the im2col GPU kernel,
the host-side conv ops, and the Conv2d layer + tests. Found and fixed a bug
in `conv2d_batched` (batch > 1 path). Single-sample path was correct.

## A. im2col Kernel Output Layout

**Layout: `[spatial, K]`** — i.e. `[H_out*W_out, C_in*kH*kW]`, row-major.

The kernel decomposes `global_id` as:
- `out_row = global_id / col_width` (spatial position index)
- `out_col = global_id % col_width` (K = C_in*kH*kW index)
- Write: `output[out_row * col_width + out_col]`

Each row is one output spatial position; each column within that row is one
element of the unfolded filter patch. This is correct.

## B. Transpose in conv.rs (lines ~91-101)

**Correct.** The host code transposes from `[spatial, K]` to `[K, spatial]`
via `col_t[k * col_w + s] = col_raw[s * col_h + k]`. This is a standard
matrix transpose and produces the correct `[K, spatial]` layout needed for
the GEMM.

## C. Matmul: W[c_out, K] x Col[K, spatial] -> [c_out, spatial]

**Correct.** Weight is reshaped from `[C_out, C_in, kH, kW]` to `[C_out, K]`
where `K = C_in * kH * kW`. The matmul produces `[C_out, H_out*W_out]` which
is already CHW-order output. Bias addition iterates correctly per channel.

## D. Batched conv2d — BUG FOUND AND FIXED

**Bug:** The original code stored per-batch transposed columns in a flat array
with layout `[batch, K, col_w]` (each batch's `[K, col_w]` block was
contiguous), but then uploaded the array as shape `[K, batch*col_w]`. These
two layouts are NOT equivalent when batch > 1.

For batch 0, indices match because `offset=0`. For batch b > 0:
- Actual storage: `all_cols_t[b * K * col_w + k * col_w + s]`
- Expected for `[K, batch*col_w]`: `all_cols_t[k * batch*col_w + b * col_w + s]`

These differ, causing incorrect matmul results. Confirmed by test:
max_err = 2.4 before fix, 3.6e-7 after fix.

**Additional bug in bias addition:** The bias code indexed the GEMM result
as `result[b * c_out * col_w + ch * col_w + i]`, treating it as
`[batch, c_out, spatial]`, but the actual GEMM output is `[c_out, big_col_w]`
where `big_col_w = batch * col_w`. Fixed to iterate `result[ch * big_col_w + j]`.

The rearrangement loop (result -> output_data) was already correct:
`result[ch * big_col_w + b * col_w + s]` -> `output_data[b * c_out * col_w + ch * col_w + s]`.

### Fix Applied

File: `crates/core/gpu-host/src/nn/ops/conv.rs`

1. Changed column storage to write directly into `[K, batch*col_w]` layout:
   `all_cols_t[k * big_col_w + b * col_w + s] = col_raw[s * col_h + k]`
2. Fixed bias addition to index the GEMM result correctly.

## E. Test Results

All 5 tests pass after fix:

| Test | Status | Max Error |
|------|--------|-----------|
| test_conv2d_1x1_identity | PASS | < 1e-3 |
| test_conv2d_multichannel | PASS | < 0.1 |
| test_conv2d_3x3_matches_cpu | PASS | < 1e-2 |
| test_conv2d_cifar10_dims (NEW) | PASS | 4.77e-7 |
| test_conv2d_batched_matches_cpu (NEW) | PASS | 3.58e-7 |

## F. New Tests Added

File: `crates/core/gpu-host/src/nn/layers/conv.rs`

1. **test_conv2d_cifar10_dims** — Input `[3, 32, 32]`, weight `[8, 3, 3, 3]`,
   stride=1, pad=1, with bias. Verifies output shape `[8, 32, 32]` and
   GPU vs CPU reference with max_err < 0.1.

2. **test_conv2d_batched_matches_cpu** — Input `[2, 3, 8, 8]`, weight `[4, 3, 3, 3]`,
   stride=1, pad=1. Verifies each batch sample independently against CPU
   reference. This test caught the batched layout bug (max_err=2.4 before fix).

## Performance Note

Both single and batched paths do a device->host round-trip for the im2col
output to perform the transpose on CPU. This is functional but suboptimal.
A GPU transpose kernel or changing the im2col kernel to output `[K, spatial]`
directly would eliminate these transfers.
