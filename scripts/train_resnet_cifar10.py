#!/usr/bin/env python3
"""Train ResNet-18 (CIFAR variant) on CIFAR-10, export weights to SafeTensors.

The CIFAR variant uses 3x3 conv1 with stride=1, no maxpool, matching the
Rust ResNet18 model definition.

Usage:
    uv run --with torch --with torchvision --with safetensors scripts/train_resnet_cifar10.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from safetensors.torch import save_file


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
    """ResNet-18 for CIFAR-10 (3x3 conv1, no maxpool)."""

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


def export_weights(model, path):
    """Export model weights with naming matching Rust ResNet18Weights."""
    tensors = {}
    sd = model.state_dict()

    # conv1 + bn1
    tensors["conv1.weight"] = sd["conv1.weight"].float().contiguous()
    tensors["bn1.weight"] = sd["bn1.weight"].float().contiguous()
    tensors["bn1.bias"] = sd["bn1.bias"].float().contiguous()
    tensors["bn1.running_mean"] = sd["bn1.running_mean"].float().contiguous()
    tensors["bn1.running_var"] = sd["bn1.running_var"].float().contiguous()

    # Layers 1-4, each with 2 BasicBlocks
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        for block_idx in range(2):
            prefix = f"{layer_name}.{block_idx}"
            out_prefix = f"{layer_name}.{block_idx}"

            tensors[f"{out_prefix}.conv1.weight"] = sd[f"{prefix}.conv1.weight"].float().contiguous()
            tensors[f"{out_prefix}.bn1.weight"] = sd[f"{prefix}.bn1.weight"].float().contiguous()
            tensors[f"{out_prefix}.bn1.bias"] = sd[f"{prefix}.bn1.bias"].float().contiguous()
            tensors[f"{out_prefix}.bn1.running_mean"] = sd[f"{prefix}.bn1.running_mean"].float().contiguous()
            tensors[f"{out_prefix}.bn1.running_var"] = sd[f"{prefix}.bn1.running_var"].float().contiguous()

            tensors[f"{out_prefix}.conv2.weight"] = sd[f"{prefix}.conv2.weight"].float().contiguous()
            tensors[f"{out_prefix}.bn2.weight"] = sd[f"{prefix}.bn2.weight"].float().contiguous()
            tensors[f"{out_prefix}.bn2.bias"] = sd[f"{prefix}.bn2.bias"].float().contiguous()
            tensors[f"{out_prefix}.bn2.running_mean"] = sd[f"{prefix}.bn2.running_mean"].float().contiguous()
            tensors[f"{out_prefix}.bn2.running_var"] = sd[f"{prefix}.bn2.running_var"].float().contiguous()

            # Shortcut (if present)
            sc_conv_key = f"{prefix}.shortcut.0.weight"
            if sc_conv_key in sd:
                tensors[f"{out_prefix}.shortcut.conv.weight"] = sd[sc_conv_key].float().contiguous()
                tensors[f"{out_prefix}.shortcut.bn.weight"] = sd[f"{prefix}.shortcut.1.weight"].float().contiguous()
                tensors[f"{out_prefix}.shortcut.bn.bias"] = sd[f"{prefix}.shortcut.1.bias"].float().contiguous()
                tensors[f"{out_prefix}.shortcut.bn.running_mean"] = sd[f"{prefix}.shortcut.1.running_mean"].float().contiguous()
                tensors[f"{out_prefix}.shortcut.bn.running_var"] = sd[f"{prefix}.shortcut.1.running_var"].float().contiguous()

    # FC
    tensors["fc.weight"] = sd["fc.weight"].float().contiguous()
    tensors["fc.bias"] = sd["fc.bias"].float().contiguous()

    save_file(tensors, path)
    total = sum(t.numel() for t in tensors.values())
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved: {path} ({total:,} params, {size_mb:.1f} MB, {len(tensors)} keys)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cifar10_torch")
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Model
    model = ResNet18CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print(f"Training ResNet-18 CIFAR-10 variant for 50 epochs...")

    best_acc = 0.0
    for epoch in range(50):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        scheduler.step()

        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0)
                test_correct += pred.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        test_acc = 100.0 * test_correct / test_total
        avg_loss = total_loss / total
        print(f"Epoch {epoch+1:2d}/50: loss={avg_loss:.3f}, train={train_acc:.1f}%, test={test_acc:.1f}%, lr={scheduler.get_last_lr()[0]:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"\nBest test accuracy: {best_acc:.1f}%")

    # Export
    model.eval()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resnet18_cifar10.safetensors")
    export_weights(model, out_path)


if __name__ == "__main__":
    main()
