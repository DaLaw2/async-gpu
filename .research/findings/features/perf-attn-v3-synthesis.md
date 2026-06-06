# perf-attn-v3: Attention V3 — Warp-cooperative Tiled GEMM
**Epic**: kernel-perf | **Status**: active | **Updated**: 2026-06-04

## Progress
- V3 kernel designed and implemented: 128 threads, 4 per Q row, cooperative shuffles for softmax
- Optimized: smem padding (stride 65), float4 global loads, cooperative Q load, launch_bounds
- Integrated into `multi_head_flash_attention_v3()` with NVRTC compilation

## Verified Conclusions
- 4-thread-per-row cooperative design works correctly with online softmax
- Smem bank conflict elimination (stride 65) is verified mathematically and in code
- The `p_val > 1e-30f` branch helps causal attention (saves V reads for masked positions)
- V3 at seq=512: 559 GFLOPS causal, 600 GFLOPS bidirectional on GTX 1660

## Rejected Approaches
- Removing P·V branch: slower due to lost V-read savings for masked positions
- Q in shared memory (persistent): increases smem from 16KB to 25KB, reducing occupancy
- BQ=64 with 256 threads: would require architectural redesign, uncertain benefit

## Open Questions
- How much would BKV=16 (halved tiles, better occupancy) improve things?
- Is 70% cuDNN achievable on sm_75 without tensor cores? (theoretical max ~1200 GFLOPS)

## Key Metrics
- seq=512 causal: 0.721 ms, 559 GFLOPS
- seq=512 bidir: 1.342 ms, 600 GFLOPS
- 12% of peak FP32 utilization (limited by occupancy + memory latency)

## Next Steps
- perf-attn-v3.3: Try alternative P·V scheduling (less shuffles, register-file tiling)
- perf-attn-v3.4: Final benchmark + integration, determine if 70% is achievable on this HW
