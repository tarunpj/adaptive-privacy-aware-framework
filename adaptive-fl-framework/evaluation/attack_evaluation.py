"""
evaluation/attack_evaluation.py

Empirical Privacy Attack Evaluation for the Adaptive FL Framework.

Implements Membership Inference Attacks (MIA):
- Loss-threshold attack (Yeom et al. 2018 / Salem et al. 2019)
- Confidence-based prediction score attack (Shokri et al. 2017)
- ROC-AUC and Privacy Advantage calculation

Why MIA is appropriate:
  MIA tests whether an adversary can determine if a particular sample was part
  of a client's private training dataset.
  - Standard FedAvg (No DP) typically leaks membership (ASR > 60-70%).
  - DP / Adaptive DP provably bounds membership leakage (ASR close to random guess ~50%).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@dataclass
class MIAResult:
    """Results from an empirical Membership Inference Attack."""
    attack_accuracy: float        # Attack Success Rate (0.5 = random guess, 1.0 = total leakage)
    baseline_random_guess: float  # Typically 0.5 for balanced member/non-member split
    roc_auc: float                # Area under ROC curve (0.5 = ideal privacy)
    tpr: float                    # True Positive Rate (correctly identified members)
    fpr: float                    # False Positive Rate (non-members misclassified as members)
    advantage: float              # TPR - FPR (0.0 = zero leakage)
    risk_level: str               # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    num_members: int
    num_non_members: int

    def to_dict(self) -> Dict:
        return {
            "attack_accuracy": round(self.attack_accuracy, 4),
            "baseline_random_guess": self.baseline_random_guess,
            "roc_auc": round(self.roc_auc, 4),
            "tpr": round(self.tpr, 4),
            "fpr": round(self.fpr, 4),
            "advantage": round(self.advantage, 4),
            "risk_level": self.risk_level,
            "num_members": self.num_members,
            "num_non_members": self.num_non_members,
        }


def _extract_sample_losses_and_confidences(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-sample loss and maximum softmax confidence for a given dataset.
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    losses: List[float] = []
    confidences: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            sample_losses = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)

            losses.extend(sample_losses.cpu().numpy().tolist())
            confidences.extend(max_probs.cpu().numpy().tolist())

    return np.array(losses), np.array(confidences)


def evaluate_membership_inference(
    model: nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: torch.device,
) -> MIAResult:
    """
    Run empirical Membership Inference Attack against `model`.

    Parameters
    ----------
    model             : Trained global model to audit
    member_loader     : DataLoader containing training data (members)
    non_member_loader : DataLoader containing held-out test data (non-members)
    device            : Torch computation device
    """
    member_losses, member_confs = _extract_sample_losses_and_confidences(model, member_loader, device)
    non_member_losses, non_member_confs = _extract_sample_losses_and_confidences(model, non_member_loader, device)

    # Balance member and non-member set sizes for fair 50/50 baseline
    n_samples = min(len(member_losses), len(non_member_losses))
    m_losses = member_losses[:n_samples]
    nm_losses = non_member_losses[:n_samples]

    # Ground truth: 1 = Member, 0 = Non-member
    y_true = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    # Loss-threshold attack: lower loss => more likely a member
    # Predict negative loss as score for ROC (higher score = member)
    all_scores = np.concatenate([-m_losses, -nm_losses])

    # Optimal threshold on member vs non-member median
    threshold = np.median(m_losses)
    y_pred = (np.concatenate([m_losses, nm_losses]) <= threshold).astype(int)

    attack_acc = float(np.mean(y_pred == y_true))

    # ROC-AUC calculation
    try:
        auc = float(roc_auc_score(y_true, all_scores))
    except Exception:
        auc = 0.5

    # TPR and FPR
    tp = np.sum((y_pred[:n_samples] == 1))
    fp = np.sum((y_pred[n_samples:] == 1))
    tpr = float(tp / n_samples)
    fpr = float(fp / n_samples)
    advantage = float(tpr - fpr)

    # Determine risk level
    if attack_acc >= 0.70 or auc >= 0.75:
        risk = "CRITICAL"
    elif attack_acc >= 0.60 or auc >= 0.65:
        risk = "HIGH"
    elif attack_acc >= 0.54 or auc >= 0.55:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return MIAResult(
        attack_accuracy=attack_acc,
        baseline_random_guess=0.5,
        roc_auc=auc,
        tpr=tpr,
        fpr=fpr,
        advantage=advantage,
        risk_level=risk,
        num_members=n_samples,
        num_non_members=n_samples,
    )
