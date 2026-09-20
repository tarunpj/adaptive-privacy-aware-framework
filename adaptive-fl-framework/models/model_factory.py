"""
models/model_factory.py

Model Factory for the Adaptive FL Framework.
Provides modular, lightweight neural network architectures for:
- CIFAR-10 (3-channel 32x32 RGB)
- MNIST (1-channel 28x28 Grayscale)
- Fashion-MNIST (1-channel 28x28 Grayscale)
- Lightweight ResNet for vision benchmarking
"""

from typing import Dict, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    Standard 5-layer CNN for CIFAR-10 classification (3x32x32 -> 10 classes).
    Parameters: ~62,006
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MNISTCNN(nn.Module):
    """
    Compact 4-layer CNN for MNIST & Fashion-MNIST (1x28x28 -> 10 classes).
    Parameters: ~28,938
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResidualBlock(nn.Module):
    """Basic Residual Block for small-scale vision models."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, channels)  # GroupNorm is DP-friendly (no running batch stats)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SmallResNet(nn.Module):
    """
    Lightweight ResNet with GroupNorm (Opacus DP compliant).
    Works on 3x32x32 images.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.res1 = ResidualBlock(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_conv(x))
        x = self.res1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ModelFactory:
    """Factory creating PyTorch neural networks for specified datasets and tasks."""

    _REGISTRY: Dict[str, Type[nn.Module]] = {
        "cifar10": CIFAR10CNN,
        "cifar10_cnn": CIFAR10CNN,
        "cifar10_resnet": SmallResNet,
        "mnist": MNISTCNN,
        "mnist_cnn": MNISTCNN,
        "fashion_mnist": MNISTCNN,
        "fashion_mnist_cnn": MNISTCNN,
    }

    @classmethod
    def create_model(cls, model_name: str, dataset_name: Optional[str] = None) -> nn.Module:
        """
        Create a model instance based on model name or dataset name.
        """
        key = (model_name or dataset_name or "cifar10").lower().replace("-", "_")
        
        if key in cls._REGISTRY:
            return cls._REGISTRY[key]()
        
        # Fallback heuristic
        if "mnist" in key:
            return MNISTCNN()
        return CIFAR10CNN()

    @classmethod
    def get_model_info(cls, model: nn.Module) -> Dict:
        """Return parameter count and byte footprint of a model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_bytes": param_bytes,
            "size_kb": round(param_bytes / 1024, 2),
        }
