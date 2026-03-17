#!/usr/bin/env python3
"""Export ResNet-18 CIFAR-10 variant to ONNX format.

Usage:
    uv run --with torch --with torchvision --with onnx scripts/export_resnet_onnx.py
"""
import os
import torch
import torch.nn as nn

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
    model = ResNet18CIFAR(num_classes=10)
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resnet18_cifar10.onnx")

    # Use opset 13, embed all weights (no external data)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported: {out_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
