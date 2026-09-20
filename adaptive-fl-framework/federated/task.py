"""
federated/task.py

Defines the CNN model, data loading, training, and evaluation functions.
This is the core ML logic, kept separate from Flower-specific code.

Model: Simple CNN (same architecture as quickstart-pytorch baseline)
Dataset: CIFAR-10 via flwr-datasets (IID partitioning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import load_dataset

# Module-level cache so FederatedDataset is only initialised once per process
_fds = None

# Proper CIFAR-10 per-channel mean/std (computed from training set)
_transforms = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class Net(nn.Module):
    """
    Stronger CNN for CIFAR-10 with GroupNorm (Opacus DP-compatible) and dropout.
    Architecture: 3x[Conv-GN-ReLU-Pool] -> Dropout -> fc1 -> fc2
    GroupNorm is used instead of BatchNorm because Opacus does not support
    BatchNorm (it requires per-sample gradient computation which BN breaks).
    Input:  3 x 32 x 32 (RGB image)
    Output: 10 class logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, 32)   # 8 groups of 4 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, 64)   # 8 groups of 8 channels
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3   = nn.GroupNorm(8, 128)  # 8 groups of 16 channels
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.gn3(self.conv3(x))))  # 8x8   -> 4x4
        x = x.view(-1, 128 * 4 * 4)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def _apply_transforms(batch: dict) -> dict:
    batch["img"] = [_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    Load a single client's partition of CIFAR-10.

    Uses IID partitioning: each client gets an equal, randomly shuffled slice.
    Splits each partition 80% train / 20% validation.

    Returns:
        trainloader, valloader
    """
    global _fds
    if _fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        _fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = _fds.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=42)
    split = split.with_transform(_apply_transforms)

    trainloader = DataLoader(split["train"], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(split["test"], batch_size=batch_size)
    return trainloader, valloader


def load_centralized_testset(batch_size: int = 128) -> DataLoader:
    """
    Load the full CIFAR-10 test split for centralised server-side evaluation.
    """
    dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = dataset.with_format("torch").with_transform(_apply_transforms)
    return DataLoader(dataset, batch_size=batch_size)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """
    Train `net` for `epochs` passes over `trainloader`.

    Returns:
        avg_loss: average cross-entropy loss across all batches and epochs
    """
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    running_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1

    return running_loss / total_batches if total_batches > 0 else 0.0


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate `net` on `testloader`.

    Returns:
        (avg_loss, accuracy)  — accuracy is in [0, 1]
    """
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy
