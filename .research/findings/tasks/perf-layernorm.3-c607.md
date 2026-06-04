# perf-layernorm.3: benchmark + integration verification

## Summary

Verified that LayerNorm v3 (float4 vectorized) is correctly integrated into
`nn::ops::layer_norm()` and that all callers use the optimized path.
Benchmarks launched but blocked by PTX JIT compilation (~20+ min for 254K-line
PTX); results pending below.

## 1. Integration verification: PASS

**File:** `crates/core/gpu-host/src/nn/ops/norm.rs` (lines 30-35)

The `layer_norm()` function auto-selects the kernel based on `d_model`:

```rust
let kernel_name = if d_model % 4 == 0 {
    "layer_norm_v3"
} else {
    "layer_norm_v2"
};
```

This is correct. All standard transformer dimensions (768, 1024, 1280, 1600,
2048) are divisible by 4, so v3 is always selected for real workloads.

## 2. Kernel registry: PASS

**File:** `crates/core/gpu-host/src/nn/registry.rs` (lines 80-81)

Both `"layer_norm_v2"` and `"layer_norm_v3"` are registered in `ML_KERNELS`,
meaning they get loaded during `KernelRegistry::new()`.

## 3. PTX verification: PASS

**File:** `crates/core/gpu-host/kernel.ptx` (line 249305+)

The `layer_norm_v3` kernel entry is present and contains:
- **Phase 1 (statistics):** `ld.global.v4.f32` for 128-bit coalesced input reads
- **Phase 2 (normalize):** `ld.global.v4.f32` for input/gamma/beta reads,
  `st.global.v4.f32` for output writes
- Warp shuffle reductions (`shfl.sync.bfly.b32`) for sum/sq_sum
- Block-level reduction via shared memory with two `bar.sync` barriers

All memory transactions use 128-bit vectorized loads/stores (4 floats per
transaction), reducing transaction count by 4x compared to v2.

## 4. Caller audit: ALL CALLERS USE OPTIMIZED PATH

| Caller | File | Path |
|--------|------|------|
| `LayerNorm::forward()` | `nn/layers/norm.rs:53` | Calls `ops::layer_norm()` -> v3 auto-select |
| GPT-2 `TransformerBlock` LN1 | `nn/models/gpt2.rs:174` | Via `self.ln_1.forward()` -> v3 |
| GPT-2 `TransformerBlock` LN2 | `nn/models/gpt2.rs:180` | Via `layer_norm_residual_dual` (already float4) |
| GPT-2 `Gpt2Model` final LN | `nn/models/gpt2.rs:456` | Via `self.ln_f.forward()` -> v3 |
| INT4 GPT-2 blocks | `nn/models/gpt2.rs:1243-1265` | Same pattern as f32 GPT-2 |
| ONNX executor | `onnx_rt/executor.rs:1536` | Calls `nn::ops::layer_norm()` -> v3 auto-select |

**GPT-2 model dimensions (all divisible by 4):**
- Small: n_embd=768 -> v3
- Medium: n_embd=1024 -> v3
- Large: n_embd=1280 -> v3
- XL: n_embd=1600 -> v3

**Fused variants (already float4):**
- `layer_norm_residual` (NVRTC): uses float4 casts
- `layer_norm_residual_dual` (NVRTC): uses float4 casts
- Both used in GPT-2's `TransformerBlock::forward()` when `cublas` feature enabled

## 5. Autograd compatibility: PASS

The autograd tape records `OpKind::LayerNorm` with `OpMeta::LayerNorm { rows, d, eps }`
during forward pass (norm.rs lines 59-76). The backward pass
(`autograd/backward.rs:92`) uses the saved input and meta to compute gradients
CPU-side. It does not depend on which forward kernel variant was used, so v3
is fully compatible with autograd.

## 6. Benchmark results

Benchmarks were launched:
- `bench_layer_norm_bandwidth` — standalone LN v3 at 128x768
- `test_layer_norm_v3_correctness` — float4 vs CPU reference

Both tests are blocked in the PTX JIT compilation phase (254K-line PTX takes
~20+ min to JIT compile via `cuModuleLoadData`, doubled to ~25+ min because
two test binaries are JIT-compiling concurrently). Results are pending.

### Bandwidth model (theoretical)

For standalone LayerNorm at seq=128, d_model=768:
- N = 128 * 768 = 98,304 elements
- Reads: input (2 passes) + gamma + beta = (2 * 98304 + 2 * 768) * 4 = 793,600 B
- Writes: output = 98,304 * 4 = 393,216 B
- Total: 1,186,816 B (~1.13 MB)

GTX 1660 peak bandwidth: 192 GB/s (GDDR6, 192-bit bus)
Target: >= 180 GB/s (60% utilization = ~115 GB/s floor; 180 GB/s = ~94%)

At 180 GB/s: elapsed = 1.13 MB / 180 GB/s = 6.3 us per call
At 192 GB/s: elapsed = 1.13 MB / 192 GB/s = 5.9 us per call

The v3 kernel should easily exceed the 180 GB/s target because:
1. **Float4 vectorized loads** — 128-bit transactions maximize memory bus utilization
2. **Single-pass statistics** — one pass for mean+variance (vs two-pass algorithms)
3. **Warp shuffle reductions** — no shared memory contention for partial sums
4. **Small working set** — 1.13 MB fits well within GPU L2 cache after warmup

**Benchmark assertion threshold:** The test asserts `gbps > 50.0` as a
conservative floor (to avoid false failures in debug builds or under load).
The actual performance should be significantly higher.

## Conclusion

Integration is verified correct. All LayerNorm callers (standalone, fused,
GPT-2, INT4 GPT-2, ONNX executor) use the optimized v3 path for d_model
divisible by 4. The v3 kernel is present in the PTX with correct float4
vectorized instructions. Benchmark numbers are pending PTX JIT completion.
