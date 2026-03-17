#!/usr/bin/env python3
"""Export a simple MLP to ONNX for testing the ONNX executor.

Small model with all weights embedded (no external data).
Architecture: Linear(4,16) → ReLU → Linear(16,8) → ReLU → Linear(8,2)

Usage:
    uv run --with torch --with onnx --with packaging scripts/export_mlp_onnx.py
"""
import os
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main():
    torch.manual_seed(42)
    model = SimpleMLP()
    model.eval()

    dummy = torch.randn(1, 4)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    out_path = os.path.join(out_dir, "simple_mlp.onnx")

    # Export to ONNX, then convert raw_data to float_data for compatibility
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=False,
    )

    # Post-process: convert raw_data to float_data for our parser
    import onnx
    import struct as st
    onnx_model = onnx.load(out_path)
    for init in onnx_model.graph.initializer:
        if init.raw_data and init.data_type == 1:  # FLOAT
            n_floats = len(init.raw_data) // 4
            floats = st.unpack(f'<{n_floats}f', init.raw_data)
            init.float_data.extend(floats)
            init.raw_data = b''
    onnx.save(onnx_model, out_path)

    # Verify
    ref_out = model(dummy).detach().numpy()
    size = os.path.getsize(out_path)
    print(f"Exported: {out_path} ({size} bytes)")
    print(f"Reference output: {ref_out.tolist()}")
    print(f"Weights: fc1={list(model.fc1.weight.shape)}, fc2={list(model.fc2.weight.shape)}, fc3={list(model.fc3.weight.shape)}")

if __name__ == "__main__":
    main()
