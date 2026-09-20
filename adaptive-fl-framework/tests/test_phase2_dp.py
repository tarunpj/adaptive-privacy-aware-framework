"""Quick unit test for DP training — run directly with python."""
import sys
import warnings
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
warnings.filterwarnings("ignore")

from privacy.differential_privacy import DPConfig, train_with_dp, unwrap_model
from privacy.privacy_accountant import PrivacyAccountant
from federated.task import Net


class _FakeCIFARDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset that mimics CIFAR-10 batch format."""
    def __len__(self): return 64
    def __getitem__(self, idx):
        return {"img": torch.randn(3, 32, 32), "label": torch.randint(0, 10, ()).item()}


def _collate(batch):
    return {
        "img":   torch.stack([b["img"]   for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


def test_dp_train():
    dataset = _FakeCIFARDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=_collate)
    net = Net()
    dp_config = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_delta=1e-5)
    result = train_with_dp(
        net, loader, epochs=1, lr=0.01,
        device=torch.device("cpu"), dp_config=dp_config,
    )
    assert result.epsilon > 0, "Epsilon should be positive after training"
    assert result.train_loss > 0, "Loss should be positive"
    clean = unwrap_model(net)
    assert hasattr(clean, "fc3"), "Unwrapped model should have fc3 layer"
    print(f"  [PASS] DP train: loss={result.train_loss:.4f}  epsilon={result.epsilon:.4f}")


def test_privacy_accountant():
    acc = PrivacyAccountant(target_delta=1e-5)
    acc.record_round(1, [0.5, 0.6, 0.55], noise_multiplier=1.0, max_grad_norm=1.0)
    acc.record_round(2, [0.8, 0.9, 0.85], noise_multiplier=1.0, max_grad_norm=1.0)
    assert abs(acc.cumulative_epsilon - (0.6 + 0.9)) < 1e-9, "Cumulative should be sum of per-round max epsilons"
    assert acc.latest_epsilon == 0.9
    print(f"  [PASS] PrivacyAccountant: cumulative_eps={acc.cumulative_epsilon:.2f}")


if __name__ == "__main__":
    print("\nRunning Phase 2 unit tests...\n")
    test_dp_train()
    test_privacy_accountant()
    print("\nAll tests passed.\n")
