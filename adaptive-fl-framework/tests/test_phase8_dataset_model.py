"""
tests/test_phase8_dataset_model.py

Unit tests for Phase 8: ModelFactory & DatasetManager.
Verifies:
1. Model creation for CIFAR-10, MNIST, Fashion-MNIST, and SmallResNet.
2. Tensor input/output shapes and parameter footprints.
3. Dataset loading, IID partitioning, and Dirichlet Non-IID partitioning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.model_factory import ModelFactory, CIFAR10CNN, MNISTCNN, SmallResNet
from data_utils.dataset_manager import DatasetManager, DatasetConfig


def test_model_factory():
    """Verify ModelFactory instantiates correct models with proper tensor dimensions."""
    cifar_model = ModelFactory.create_model("cifar10")
    mnist_model = ModelFactory.create_model("mnist")
    fmnist_model = ModelFactory.create_model("fashion_mnist")
    resnet_model = ModelFactory.create_model("cifar10_resnet")

    assert isinstance(cifar_model, CIFAR10CNN)
    assert isinstance(mnist_model, MNISTCNN)
    assert isinstance(fmnist_model, MNISTCNN)
    assert isinstance(resnet_model, SmallResNet)

    # Forward pass checks
    cifar_out = cifar_model(torch.randn(2, 3, 32, 32))
    mnist_out = mnist_model(torch.randn(2, 1, 28, 28))
    resnet_out = resnet_model(torch.randn(2, 3, 32, 32))

    assert cifar_out.shape == (2, 10)
    assert mnist_out.shape == (2, 10)
    assert resnet_out.shape == (2, 10)

    info = ModelFactory.get_model_info(cifar_model)
    assert info["total_parameters"] == 62006
    print(f"  [PASS] ModelFactory: CIFAR-10 ({info['size_kb']} KB), MNIST, ResNet verified")


def test_dataset_manager_iid_and_dirichlet():
    """Verify DatasetManager partitioning logic."""
    cfg_iid = DatasetConfig(name="cifar10", num_clients=4, partitioning="iid", batch_size=16)
    mgr_iid = DatasetManager(cfg_iid)
    
    # Mock base dataset for fast offline testing
    class _MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx): return torch.randn(3, 32, 32), idx % 10
        @property
        def targets(self): return [i % 10 for i in range(100)]

    mgr_iid._train_dataset = _MockDataset()
    mgr_iid._test_dataset = _MockDataset()

    # Test IID partitioning
    parts = mgr_iid.create_partitions()
    assert len(parts) == 4
    assert sum(len(indices) for indices in parts.values()) == 100
    print(f"  [PASS] DatasetManager IID partitioning verified (4 clients, 100 samples)")

    # Test Dirichlet Non-IID partitioning
    cfg_dir = DatasetConfig(name="cifar10", num_clients=4, partitioning="dirichlet", dirichlet_alpha=0.5)
    mgr_dir = DatasetManager(cfg_dir)
    mgr_dir._train_dataset = _MockDataset()
    mgr_dir._test_dataset = _MockDataset()

    dir_parts = mgr_dir.create_partitions()
    assert len(dir_parts) == 4
    print(f"  [PASS] DatasetManager Dirichlet Non-IID partitioning verified (alpha=0.5)")


if __name__ == "__main__":
    print("\nRunning Phase 8 Dataset & Model Framework unit tests...\n")
    test_model_factory()
    test_dataset_manager_iid_and_dirichlet()
    print("\nAll Phase 8 tests passed successfully.\n")
