"""
tests/test_phase7_evaluation.py

Unit tests for Phase 7: Comprehensive Evaluation & Membership Inference Attack (MIA).
Verifies:
1. Multi-metric evaluation (loss, accuracy, precision, recall, macro-F1).
2. Theoretical communication payload calculation.
3. Membership Inference Attack (MIA) execution, AUC, and risk score assignment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_classification_metrics, compute_communication_cost
from evaluation.attack_evaluation import evaluate_membership_inference
from federated.task import Net


def _create_synthetic_loader(num_samples: int = 100, num_classes: int = 10, batch_size: int = 32):
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_metrics_computation():
    """Verify accuracy, precision, recall, and macro-F1 metric calculation."""
    device = torch.device("cpu")
    model = Net()
    loader = _create_synthetic_loader(num_samples=100)

    metrics = compute_classification_metrics(model, loader, device)
    
    assert 0.0 <= metrics.accuracy <= 1.0, "Accuracy must be between 0 and 1"
    assert 0.0 <= metrics.precision_macro <= 1.0, "Precision must be between 0 and 1"
    assert 0.0 <= metrics.recall_macro <= 1.0, "Recall must be between 0 and 1"
    assert 0.0 <= metrics.f1_macro <= 1.0, "F1 must be between 0 and 1"
    assert metrics.loss > 0, "Loss must be positive"
    assert metrics.total_samples == 100
    print(f"  [PASS] Metrics computed: acc={metrics.accuracy:.4f}, f1={metrics.f1_macro:.4f}, loss={metrics.loss:.4f}")


def test_communication_cost():
    """Verify communication payload size calculations."""
    model = Net()
    cost = compute_communication_cost(model, num_clients=10, num_rounds=5, secagg_overhead_multiplier=1.2)
    
    assert cost["model_parameters"] == 62006
    assert cost["single_model_bytes"] == 248024
    assert cost["total_round_mb"] > 0
    assert cost["total_experiment_mb"] > cost["total_round_mb"]
    print(f"  [PASS] Comm cost verified: {cost['model_parameters']} params, total experiment={cost['total_experiment_mb']} MB")


def test_membership_inference_attack():
    """Verify MIA evaluation execution, AUC bounds, and risk grading."""
    device = torch.device("cpu")
    model = Net()

    member_loader = _create_synthetic_loader(num_samples=120)
    non_member_loader = _create_synthetic_loader(num_samples=120)

    mia_result = evaluate_membership_inference(model, member_loader, non_member_loader, device)

    assert 0.0 <= mia_result.attack_accuracy <= 1.0
    assert 0.0 <= mia_result.roc_auc <= 1.0
    assert -1.0 <= mia_result.advantage <= 1.0
    assert mia_result.risk_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert mia_result.num_members == 120
    assert mia_result.num_non_members == 120
    print(f"  [PASS] MIA attack audited: ASR={mia_result.attack_accuracy:.4f}, AUC={mia_result.roc_auc:.4f}, Risk={mia_result.risk_level}")


if __name__ == "__main__":
    print("\nRunning Phase 7 Evaluation & Privacy Attack unit tests...\n")
    test_metrics_computation()
    test_communication_cost()
    test_membership_inference_attack()
    print("\nAll Phase 7 tests passed successfully.\n")
