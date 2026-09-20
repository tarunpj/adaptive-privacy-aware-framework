"""
privacy/differential_privacy.py

Differential Privacy training using Opacus.

What Opacus does under the hood:
  1. Gradient Clipping  — per-sample gradients are clipped to max_grad_norm
  2. Noise Addition     — Gaussian noise scaled by noise_multiplier is added
  3. Privacy Accounting — tracks cumulative epsilon via RDP accountant

The DP guarantee is (epsilon, delta)-DP:
  - epsilon: privacy loss budget. Lower = stronger privacy, more noise.
  - delta:   probability of catastrophic privacy failure. Typically 1e-5.
  - noise_multiplier: ratio of noise std to clipping norm. Higher = more noise.
  - max_grad_norm: per-sample gradient clipping bound (L2 norm).

Mathematical relationship (approximate, via RDP):
  epsilon grows with: more training steps, fewer samples, lower noise_multiplier
  epsilon shrinks with: higher noise_multiplier, larger dataset

Reference: Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
"""

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DPConfig:
    """
    All parameters needed to configure a DP training run.

    Attributes
    ----------
    noise_multiplier : float
        Ratio of Gaussian noise std to the clipping norm.
        Higher = more noise = stronger privacy = lower accuracy.
    max_grad_norm : float
        Per-sample gradient clipping bound (L2 norm).
        Clips each sample's gradient before noise is added.
    target_delta : float
        The delta in (epsilon, delta)-DP. Typically 1e-5.
        Should be much smaller than 1/dataset_size.
    """
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5


@dataclass
class DPTrainResult:
    """Results returned after one DP training run."""
    train_loss: float
    epsilon: float          # privacy budget consumed so far this session
    noise_multiplier: float
    max_grad_norm: float
    target_delta: float


def train_with_dp(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    dp_config: DPConfig,
) -> DPTrainResult:
    """
    Train `net` with Differential Privacy using Opacus.

    Pipeline per batch:
        forward pass
        → per-sample gradient computation
        → per-sample gradient clipping (max_grad_norm)
        → Gaussian noise addition (noise_multiplier * max_grad_norm)
        → parameter update

    Returns DPTrainResult with the actual epsilon consumed.
    """
    net.to(device)
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(secure_mode=False)

    # make_private wraps the model, optimizer, and dataloader
    # so that every optimizer.step() automatically clips + adds noise
    private_net, private_optimizer, private_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )

    running_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in private_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            private_optimizer.zero_grad()
            loss = criterion(private_net(images), labels)
            loss.backward()
            private_optimizer.step()
            running_loss += loss.item()
            total_batches += 1

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    epsilon = privacy_engine.get_epsilon(delta=dp_config.target_delta)

    return DPTrainResult(
        train_loss=avg_loss,
        epsilon=float(epsilon),
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
        target_delta=dp_config.target_delta,
    )


def unwrap_model(net: nn.Module) -> nn.Module:
    """
    Return the underlying nn.Module even if Opacus has wrapped it
    in a GradSampleModule. Needed to extract a clean state_dict.
    """
    return net._module if hasattr(net, "_module") else net
