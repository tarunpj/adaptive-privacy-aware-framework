"""
data_utils/dataset_manager.py

Dataset Manager for the Adaptive FL Framework.
Supports:
- CIFAR-10, MNIST, Fashion-MNIST
- IID Partitioning
- Non-IID Dirichlet Partitioning (configurable alpha parameter)
- Local client DataLoader generation (train/validation splits)
- Centralized server evaluation DataLoaders
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    """Configuration for loading and partitioning a dataset."""
    name: str = "cifar10"                 # "cifar10" | "mnist" | "fashion_mnist"
    num_clients: int = 10
    partitioning: str = "iid"             # "iid" | "dirichlet"
    dirichlet_alpha: float = 0.5          # Lower alpha = higher non-IID skew (0.1 to 1.0)
    batch_size: int = 32
    val_ratio: float = 0.2
    seed: int = 42
    data_dir: str = "./data"


class DatasetManager:
    """Manages downloading, preprocessing, and partitioning vision datasets for FL."""

    NORM_PARAMS = {
        "cifar10": {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "in_channels": 3,
            "img_size": (32, 32),
            "num_classes": 10,
        },
        "mnist": {
            "mean": (0.1307,),
            "std": (0.3081,),
            "in_channels": 1,
            "img_size": (28, 28),
            "num_classes": 10,
        },
        "fashion_mnist": {
            "mean": (0.2860,),
            "std": (0.3530,),
            "in_channels": 1,
            "img_size": (28, 28),
            "num_classes": 10,
        },
    }

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.dataset_name = config.name.lower().replace("-", "_")
        if self.dataset_name not in self.NORM_PARAMS:
            self.dataset_name = "cifar10"

        self.meta = self.NORM_PARAMS[self.dataset_name]
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._client_indices: Dict[int, List[int]] = {}

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Return PyTorch transform pipelines for train and test splits."""
        mean = self.meta["mean"]
        std = self.meta["std"]

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_transform, test_transform

    def load_base_datasets(self) -> Tuple[Dataset, Dataset]:
        """Download or load train and test datasets from torchvision."""
        train_tf, test_tf = self._get_transforms()
        d_dir = self.config.data_dir

        if self.dataset_name == "mnist":
            train_data = datasets.MNIST(root=d_dir, train=True, download=True, transform=train_tf)
            test_data = datasets.MNIST(root=d_dir, train=False, download=True, transform=test_tf)
        elif self.dataset_name == "fashion_mnist":
            train_data = datasets.FashionMNIST(root=d_dir, train=True, download=True, transform=train_tf)
            test_data = datasets.FashionMNIST(root=d_dir, train=False, download=True, transform=test_tf)
        else:
            train_data = datasets.CIFAR10(root=d_dir, train=True, download=True, transform=train_tf)
            test_data = datasets.CIFAR10(root=d_dir, train=False, download=True, transform=test_tf)

        self._train_dataset = train_data
        self._test_dataset = test_data
        return train_data, test_data

    def create_partitions(self) -> Dict[int, List[int]]:
        """
        Partition dataset indices across clients using IID or Dirichlet Non-IID.
        """
        if self._train_dataset is None:
            self.load_base_datasets()

        np.random.seed(self.config.seed)
        num_clients = self.config.num_clients
        labels = np.array(self._train_dataset.targets if hasattr(self._train_dataset, "targets") else self._train_dataset.labels)
        num_samples = len(labels)
        num_classes = self.meta["num_classes"]

        if self.config.partitioning.lower() == "dirichlet":
            # Non-IID Dirichlet distribution over class proportions
            client_indices = {i: [] for i in range(num_clients)}
            for k in range(num_classes):
                k_indices = np.where(labels == k)[0]
                np.random.shuffle(k_indices)
                
                # Sample proportions from Dirichlet(alpha)
                proportions = np.random.dirichlet(np.repeat(self.config.dirichlet_alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, client_indices.values())])
                if proportions.sum() == 0:
                    proportions = np.ones(num_clients) / num_clients
                else:
                    proportions = proportions / proportions.sum()

                # Split class indices according to proportions
                splits = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
                client_k_splits = np.split(k_indices, splits)
                for i in range(num_clients):
                    client_indices[i].extend(client_k_splits[i].tolist())

            for i in range(num_clients):
                np.random.shuffle(client_indices[i])
        else:
            # Uniform IID partitioning
            all_indices = np.random.permutation(num_samples)
            client_splits = np.array_split(all_indices, num_clients)
            client_indices = {i: client_splits[i].tolist() for i in range(num_clients)}

        self._client_indices = client_indices
        return client_indices

    def get_client_loaders(self, client_id: int) -> Tuple[DataLoader, DataLoader]:
        """
        Return (trainloader, valloader) for a specific client partition.
        """
        if not self._client_indices:
            self.create_partitions()

        indices = self._client_indices[client_id]
        val_size = int(len(indices) * self.config.val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_subset = Subset(self._train_dataset, train_indices)
        val_subset = Subset(self._train_dataset, val_indices)

        trainloader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)
        return trainloader, valloader

    def get_centralized_test_loader(self, batch_size: int = 128) -> DataLoader:
        """Return full centralized test set DataLoader for server evaluation."""
        if self._test_dataset is None:
            self.load_base_datasets()
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False)

    def get_metadata(self) -> Dict:
        """Return dataset metadata and class balance summary."""
        return {
            "dataset": self.dataset_name,
            "num_clients": self.config.num_clients,
            "partitioning": self.config.partitioning,
            "dirichlet_alpha": self.config.dirichlet_alpha if self.config.partitioning == "dirichlet" else None,
            "in_channels": self.meta["in_channels"],
            "img_size": self.meta["img_size"],
            "num_classes": self.meta["num_classes"],
            "total_train_samples": len(self._train_dataset) if self._train_dataset else 0,
            "total_test_samples": len(self._test_dataset) if self._test_dataset else 0,
        }
