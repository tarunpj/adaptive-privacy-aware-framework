"""
evaluation/metrics.py

Comprehensive evaluation metrics for the Adaptive FL Framework:
- Accuracy, Top-1/Top-5 accuracy
- Multi-class Precision, Recall, and Macro-F1 score
- Cross-entropy loss and Per-class accuracy
- Communication footprint in bytes / KB / MB
- Convergence velocity and stability metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@dataclass
class ClassificationMetrics:
    """Standardized classification evaluation results."""
    loss: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    total_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": round(self.loss, 6),
            "accuracy": round(self.accuracy, 6),
            "precision_macro": round(self.precision_macro, 6),
            "recall_macro": round(self.recall_macro, 6),
            "f1_macro": round(self.f1_macro, 6),
            "total_samples": self.total_samples,
        }


def compute_classification_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> ClassificationMetrics:
    """
    Evaluate model performance across loss, accuracy, precision, recall, and F1.
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    total_samples = len(all_targets)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets))) if total_samples > 0 else 0.0

    precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return ClassificationMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        total_samples=total_samples,
    )


def compute_communication_cost(
    model: nn.Module,
    num_clients: int,
    num_rounds: int,
    secagg_overhead_multiplier: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate theoretical communication payload size per round and overall.
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # In each round: server -> clients (broadcast) + clients -> server (upload)
    round_upload_bytes = param_bytes * num_clients * secagg_overhead_multiplier
    round_broadcast_bytes = param_bytes * num_clients
    total_round_bytes = round_upload_bytes + round_broadcast_bytes
    total_experiment_bytes = total_round_bytes * num_rounds

    return {
        "model_parameters": total_params,
        "single_model_bytes": param_bytes,
        "single_model_kb": round(param_bytes / 1024, 2),
        "round_upload_mb": round(round_upload_bytes / (1024 ** 2), 4),
        "round_broadcast_mb": round(round_broadcast_bytes / (1024 ** 2), 4),
        "total_round_mb": round(total_round_bytes / (1024 ** 2), 4),
        "total_experiment_mb": round(total_experiment_bytes / (1024 ** 2), 4),
    }
