#!/usr/bin/env python3
"""Benchmark cuBLAS/cuDNN baselines for comparison with async-gpu custom kernels.

Measures throughput for the same shapes used in examples/std/benchmark/.
Run with: uv run --with torch scripts/bench_cudnn_baseline.py
"""

import time
import torch
import torch.nn.functional as F


def sync():
    torch.cuda.synchronize()


def bench(fn, warmup=5, iters=20):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def sgemm_gflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms / 1000) / 1e9


def conv2d_gflops(C_out, C_in, H, W, K, ms):
    # FLOPs = 2 * C_out * H_out * W_out * C_in * K * K
    H_out = H  # assume pad = K//2, stride=1
    W_out = W
    return 2.0 * C_out * H_out * W_out * C_in * K * K / (ms / 1000) / 1e9


def main():
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}")
    print()

    # =========================================================================
    # SGEMM (cuBLAS)
    # =========================================================================
    print("=" * 70)
    print("SGEMM — cuBLAS (via torch.mm)")
    print("=" * 70)
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # GPT-2 shapes
        (128, 768, 768),      # attention projection
        (128, 768, 3072),     # FFN up
        (128, 3072, 768),     # FFN down
        (128, 768, 50257),    # LM head
    ]
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'ms':>8} {'GFLOPS':>10}")
    print("-" * 50)
    for M, N, K in sizes:
        A = torch.randn(M, K, device=dev)
        B = torch.randn(K, N, device=dev)
        ms = bench(lambda: torch.mm(A, B))
        gf = sgemm_gflops(M, N, K, ms)
        print(f"{M:>6} {N:>6} {K:>6} | {ms:>8.3f} {gf:>10.1f}")
    print()

    # =========================================================================
    # Conv2D (cuDNN)
    # =========================================================================
    print("=" * 70)
    print("Conv2D — cuDNN (via F.conv2d)")
    print("=" * 70)
    conv_configs = [
        # (C_in, C_out, H, W, K, stride, pad, name)
        (3, 64, 224, 224, 7, 2, 3, "ResNet conv1 (224x224)"),
        (3, 64, 32, 32, 3, 1, 1, "ResNet CIFAR conv1"),
        (64, 64, 32, 32, 3, 1, 1, "ResNet layer1"),
        (64, 128, 32, 32, 3, 2, 1, "ResNet layer2 (stride=2)"),
        (128, 256, 16, 16, 3, 2, 1, "ResNet layer3 (stride=2)"),
        (256, 512, 8, 8, 3, 2, 1, "ResNet layer4 (stride=2)"),
        # YOLO shapes
        (3, 16, 640, 640, 3, 2, 1, "YOLO P1 (640x640)"),
        (16, 32, 320, 320, 3, 2, 1, "YOLO P2"),
        (32, 64, 160, 160, 3, 2, 1, "YOLO P3"),
    ]
    print(f"{'Config':<30} | {'ms':>8} {'GFLOPS':>10}")
    print("-" * 55)
    for C_in, C_out, H, W, K, stride, pad, name in conv_configs:
        x = torch.randn(1, C_in, H, W, device=dev)
        w = torch.randn(C_out, C_in, K, K, device=dev)
        ms = bench(lambda: F.conv2d(x, w, stride=stride, padding=pad))
        H_out = (H + 2 * pad - K) // stride + 1
        W_out = (W + 2 * pad - K) // stride + 1
        flops = 2.0 * C_out * H_out * W_out * C_in * K * K
        gf = flops / (ms / 1000) / 1e9
        print(f"{name:<30} | {ms:>8.3f} {gf:>10.1f}")
    print()

    # =========================================================================
    # Attention (cuDNN / FlashAttention via SDPA)
    # =========================================================================
    print("=" * 70)
    print("Scaled Dot-Product Attention — cuDNN/FlashAttention (via torch SDPA)")
    print("=" * 70)
    attn_configs = [
        # (batch, n_heads, seq_len, d_head, causal)
        (1, 12, 64, 64, True),
        (1, 12, 128, 64, True),
        (1, 12, 256, 64, True),
        (1, 12, 512, 64, True),
        (1, 12, 1024, 64, True),
    ]
    print(f"{'seq_len':>8} {'n_heads':>8} {'d_head':>7} | {'ms':>8} {'GFLOPS':>10}")
    print("-" * 55)
    for batch, n_heads, seq_len, d_head, causal in attn_configs:
        q = torch.randn(batch, n_heads, seq_len, d_head, device=dev)
        k = torch.randn(batch, n_heads, seq_len, d_head, device=dev)
        v = torch.randn(batch, n_heads, seq_len, d_head, device=dev)
        ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        # Attention FLOPs: 2*n_heads*(seq^2*d_head + seq^2*d_head) = 4*n_heads*seq^2*d_head
        flops = 4.0 * batch * n_heads * seq_len * seq_len * d_head
        gf = flops / (ms / 1000) / 1e9
        print(f"{seq_len:>8} {n_heads:>8} {d_head:>7} | {ms:>8.3f} {gf:>10.1f}")
    print()

    # =========================================================================
    # Memory-Bound Operations
    # =========================================================================
    print("=" * 70)
    print("Memory-Bound Ops — cuDNN/PyTorch")
    print("=" * 70)
    N = 128 * 768  # GPT-2 shaped
    x = torch.randn(N, device=dev)

    # Elementwise add
    y = torch.randn(N, device=dev)
    ms = bench(lambda: torch.add(x, y))
    bytes_moved = 3 * N * 4  # read x, read y, write out
    gbps = bytes_moved / (ms / 1000) / 1e9
    print(f"elementwise_add ({N} floats):  {ms:.3f} ms, {gbps:.1f} GB/s")

    # GELU
    ms = bench(lambda: F.gelu(x))
    bytes_moved = 2 * N * 4  # read + write
    gbps = bytes_moved / (ms / 1000) / 1e9
    print(f"gelu ({N} floats):             {ms:.3f} ms, {gbps:.1f} GB/s")

    # LayerNorm
    x_ln = torch.randn(128, 768, device=dev)
    ln = torch.nn.LayerNorm(768).cuda()
    ms = bench(lambda: ln(x_ln))
    bytes_moved = 2 * 128 * 768 * 4
    gbps = bytes_moved / (ms / 1000) / 1e9
    print(f"layer_norm (128×768):          {ms:.3f} ms, {gbps:.1f} GB/s")
    print()

    # =========================================================================
    # GPT-2 End-to-End (PyTorch, all cuBLAS/cuDNN)
    # =========================================================================
    print("=" * 70)
    print("GPT-2 Small Forward Pass — PyTorch (cuBLAS + cuDNN)")
    print("=" * 70)
    try:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
        input_ids = torch.randint(0, 50257, (1, 128), device=dev)
        with torch.no_grad():
            ms = bench(lambda: model(input_ids), warmup=3, iters=10)
        print(f"Forward pass (seq_len=128):    {ms:.1f} ms")
        print(f"Per-token:                     {ms/128:.2f} ms/token")
    except ImportError:
        print("(transformers not installed — skipping GPT-2 e2e)")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
