# conv-direct-opt: Feature Synthesis

Multi-output-channel direct conv kernel: CO_PER_BLOCK=4 output
channels share one shared-memory input tile per iteration.
CI_CHUNK_WR=8 channels loaded per chunk for higher reuse.

3x3 stride=2 (YOLO backbone): 100-111 -> 214-278 GFLOPS (2.1-2.6x).
5x5 stride=2: 194-220 -> 317-340 GFLOPS (1.5-1.7x).
7x7 stride=2 (ResNet stem): 270 -> 394 GFLOPS (1.46x).
5x5/7x7 stride=1: 296-395 -> 312-410 GFLOPS (1.04-1.26x).

Still at 5-8% of peak. Warp-level C_in reduction (distributing
C_in across threads) was tried first but was 15-34% slower due
to 1024 thread/block register pressure. The dominant bottleneck
is one-thread-per-output serial C_in loop structure.

Kernel routing: warp_reduce for C_out>=4, tiled fallback otherwise.
Correctness verified against CPU f64 reference (<1e-5 rel error).
