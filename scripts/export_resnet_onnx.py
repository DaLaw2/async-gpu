#!/usr/bin/env python3
"""Export ResNet-18 CIFAR-10 variant to ONNX with all weights inline.

Usage:
    uv run --with torch --with torchvision --with onnx --with packaging --with onnxscript scripts/export_resnet_onnx.py
"""
import os, tempfile
import torch
import torch.nn as nn
import onnx

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)

def main():
    torch.manual_seed(42)
    model = ResNet18CIFAR(num_classes=10)
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    out_path = os.path.join(out_dir, "resnet18_cifar10.onnx")

    # Export to temp first (may create external data)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    torch.onnx.export(
        model, dummy, tmp.name,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )

    # Load and save with all data inline
    onnx_model = onnx.load(tmp.name, load_external_data=True)
    onnx.save(onnx_model, out_path)

    # Clean up temp
    os.unlink(tmp.name)
    data_file = tmp.name + ".data"
    if os.path.exists(data_file):
        os.unlink(data_file)

    # Verify initializers
    total_params = 0
    for init in onnx_model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total_params += n

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported: {out_path}")
    print(f"Size: {size_mb:.1f} MB, {total_params:,} params, {len(onnx_model.graph.initializer)} initializers")
    print(f"Nodes: {len(onnx_model.graph.node)}")

    # Print unique op types
    ops = set(n.op_type for n in onnx_model.graph.node)
    print(f"Ops: {sorted(ops)}")

    # Reference output
    ref = model(dummy).detach()
    print(f"Ref output (first 5): {ref[0, :5].tolist()}")

if __name__ == "__main__":
    main()
