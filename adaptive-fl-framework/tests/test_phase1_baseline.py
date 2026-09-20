"""
tests/test_phase1_baseline.py

Unit tests for Phase 1: Baseline Federated Learning components.
Tests model architecture, parameter count, forward pass, and training/evaluation loops.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from federated.task import Net, train, test


class _SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for quick offline unit testing."""
    def __init__(self, size=64):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"img": self.data[idx], "label": self.labels[idx]}


def test_model_architecture():
    """Verify model output shape and parameter size calculation."""
    model = Net()
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    
    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    assert total_params == 62006, f"Expected 62,006 parameters, got {total_params}"
    assert model_bytes == 248024, f"Expected 248,024 bytes, got {model_bytes}"
    print(f"  [PASS] Model architecture verified: {total_params:,} parameters ({model_bytes / 1024:.1f} KB)")


def test_training_and_eval_step():
    """Verify local training loop and evaluation function."""
    device = torch.device("cpu")
    model = Net()
    dataset = _SyntheticDataset(size=64)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    initial_loss, initial_acc = test(model, loader, device)
    assert 0.0 <= initial_acc <= 1.0, "Accuracy must be within [0, 1]"
    
    train_loss = train(model, loader, epochs=1, lr=0.01, device=device)
    assert train_loss > 0, "Train loss should be positive"
    
    post_loss, post_acc = test(model, loader, device)
    assert 0.0 <= post_acc <= 1.0, "Post accuracy must be within [0, 1]"
    print(f"  [PASS] Training & evaluation loops verified: train_loss={train_loss:.4f}, acc={post_acc:.4f}")


if __name__ == "__main__":
    print("\nRunning Phase 1 Baseline unit tests...\n")
    test_model_architecture()
    test_training_and_eval_step()
    print("\nAll Phase 1 tests passed successfully.\n")
