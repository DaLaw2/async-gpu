# conv-wino-gemm: Feature Synthesis

Winograd F(2x2,3x3) restructured as 16 batched cuBLAS GEMMs.
Three-phase pipeline: input transform → strided batched GEMM → output transform.

cudarc 0.12 exposes `gemm_strided_batched` (f32) — exactly what we need.
Each GEMM: U_k[C_out, C_in] × V_k[C_in, n_tiles], k=0..15.
Single cuBLAS call replaces the per-channel serial loop (current bottleneck).

F(2x2) chosen over F(4x4): numerically stable in FP32, simpler,
and the serial-loop bottleneck (not arithmetic) dominates current perf.
F(4x4) deferred as future optimization for large spatial dims.

Two new NVRTC kernels needed: input transform (tiles→V layout)
and output transform (M layout→spatial). Filter transform reused as-is.

Memory overhead: ~3 MB transient for typical ResNet shapes.
Expected speedup: 10-30× over current kernel (matching cuBLAS reference).
Risk: L4 shape (n_tiles=16) may need thin-GEMM fallback.
