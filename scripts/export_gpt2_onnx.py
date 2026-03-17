#!/usr/bin/env python3
"""Export GPT-2 Small (124M) to ONNX format and list all required operators.

Builds a clean GPT-2 model from scratch, loads HuggingFace weights,
and exports to ONNX -- bypassing all transformers tracing issues.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, max_pos=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_pos, n_embd)
        self.drop = nn.Dropout(0.0)  # eval mode, no dropout
        self.h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def load_hf_weights(model, hf_model):
    """Copy weights from HuggingFace GPT2LMHeadModel into our model."""
    sd = model.state_dict()
    hf_sd = hf_model.state_dict()

    # Direct mappings
    direct = {
        "wte.weight": "transformer.wte.weight",
        "wpe.weight": "transformer.wpe.weight",
        "ln_f.weight": "transformer.ln_f.weight",
        "ln_f.bias": "transformer.ln_f.bias",
    }
    for ours, theirs in direct.items():
        sd[ours].copy_(hf_sd[theirs])

    # lm_head shares weights with wte
    sd["lm_head.weight"].copy_(hf_sd["transformer.wte.weight"])

    # Per-layer weights
    for i in range(12):
        prefix_ours = f"h.{i}."
        prefix_hf = f"transformer.h.{i}."

        # LayerNorm
        for ln in ["ln_1", "ln_2"]:
            sd[f"{prefix_ours}{ln}.weight"].copy_(hf_sd[f"{prefix_hf}{ln}.weight"])
            sd[f"{prefix_ours}{ln}.bias"].copy_(hf_sd[f"{prefix_hf}{ln}.bias"])

        # Attention: HF uses Conv1D (transposed), we use nn.Linear
        # Conv1D stores weight as (in_features, out_features), Linear as (out, in)
        sd[f"{prefix_ours}attn.c_attn.weight"].copy_(
            hf_sd[f"{prefix_hf}attn.c_attn.weight"].t()
        )
        sd[f"{prefix_ours}attn.c_attn.bias"].copy_(
            hf_sd[f"{prefix_hf}attn.c_attn.bias"]
        )
        sd[f"{prefix_ours}attn.c_proj.weight"].copy_(
            hf_sd[f"{prefix_hf}attn.c_proj.weight"].t()
        )
        sd[f"{prefix_ours}attn.c_proj.bias"].copy_(
            hf_sd[f"{prefix_hf}attn.c_proj.bias"]
        )

        # MLP: same Conv1D -> Linear transpose
        sd[f"{prefix_ours}mlp.c_fc.weight"].copy_(
            hf_sd[f"{prefix_hf}mlp.c_fc.weight"].t()
        )
        sd[f"{prefix_ours}mlp.c_fc.bias"].copy_(
            hf_sd[f"{prefix_hf}mlp.c_fc.bias"]
        )
        sd[f"{prefix_ours}mlp.c_proj.weight"].copy_(
            hf_sd[f"{prefix_hf}mlp.c_proj.weight"].t()
        )
        sd[f"{prefix_ours}mlp.c_proj.bias"].copy_(
            hf_sd[f"{prefix_hf}mlp.c_proj.bias"]
        )

    model.load_state_dict(sd)


def main():
    from transformers import GPT2LMHeadModel as HF_GPT2

    print("Loading HuggingFace GPT-2 Small weights...")
    hf_model = HF_GPT2.from_pretrained("gpt2")
    hf_model.eval()

    print("Building clean GPT-2 model and loading weights...")
    model = GPT2Model()
    load_hf_weights(model, hf_model)
    model.eval()

    # Verify correctness
    dummy = torch.randint(0, 50257, (1, 16), dtype=torch.long)
    with torch.no_grad():
        hf_out = hf_model(dummy).logits
        our_out = model(dummy)
        max_err = (hf_out - our_out).abs().max().item()
        print(f"Max error vs HuggingFace: {max_err:.6e}")
        assert max_err < 1e-4, f"Error too large: {max_err}"

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy,),
        "models/gpt2.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Exported to models/gpt2.onnx")

    # Inspect the ONNX file
    import onnx
    model_onnx = onnx.load("models/gpt2.onnx")
    onnx.checker.check_model(model_onnx)
    print("ONNX model validated successfully.")

    # Collect all unique op_types
    op_types = set()
    for node in model_onnx.graph.node:
        op_types.add(node.op_type)

    print(f"\nTotal unique ONNX operators: {len(op_types)}")
    print("Operators:")
    for op in sorted(op_types):
        print(f"  - {op}")

    # File size
    import os
    size_mb = os.path.getsize("models/gpt2.onnx") / (1024 * 1024)
    print(f"\nModel file size: {size_mb:.1f} MB")

    # Validate with ONNX Runtime
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession("models/gpt2.onnx")
    np_input = dummy.numpy()
    ort_out = sess.run(None, {"input_ids": np_input})[0]
    ort_err = np.abs(hf_out.numpy() - ort_out).max()
    print(f"ONNX Runtime vs PyTorch max error: {ort_err:.6e}")


if __name__ == "__main__":
    main()
