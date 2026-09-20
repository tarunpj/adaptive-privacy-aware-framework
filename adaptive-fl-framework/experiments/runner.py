"""
experiments/runner.py

Experiment Manager and Orchestrator for the Adaptive FL Framework.

Responsibilities:
1. Load, validate, and serialize declarative JSON experiment configurations.
2. Translate configurations into execution arguments for Flower runtime.
3. Track and persist experiment manifests, run metadata, and reproducibility seeds.
4. Automatically generate diagnostic and publication-quality convergence plots.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class DPParameters:
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5


@dataclass
class SecAggParameters:
    num_shares: int = 7
    reconstruction_threshold: int = 4


@dataclass
class ControllerParameters:
    alpha: float = 0.15
    beta: float = 0.08
    threshold_high: float = 0.05
    threshold_low: float = 0.01
    noise_min: float = 0.3
    noise_max: float = 3.0
    epsilon_budget: float = 10.0
    warmup_rounds: int = 2


@dataclass
class ExperimentConfig:
    experiment_id: str
    description: str = ""
    dataset: str = "cifar10"
    model: str = "simple_cnn"
    num_clients: int = 10
    num_rounds: int = 5
    local_epochs: int = 1
    learning_rate: float = 0.01
    batch_size: int = 32
    fraction_evaluate: float = 1.0
    seed: int = 42
    save_model: bool = False
    
    # Mode flags
    differential_privacy: bool = False
    dp_params: Optional[DPParameters] = None
    secure_aggregation: bool = False
    secagg_params: Optional[SecAggParameters] = None
    adaptive_privacy: bool = False
    controller_params: Optional[ControllerParameters] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(data)
        if d.get("dp_params") and isinstance(d["dp_params"], dict):
            d["dp_params"] = DPParameters(**d["dp_params"])
        if d.get("secagg_params") and isinstance(d["secagg_params"], dict):
            d["secagg_params"] = SecAggParameters(**d["secagg_params"])
        if d.get("controller_params") and isinstance(d["controller_params"], dict):
            d["controller_params"] = ControllerParameters(**d["controller_params"])
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        assert self.experiment_id, "experiment_id cannot be empty"
        assert self.num_clients >= 2, "num_clients must be at least 2"
        assert self.num_rounds >= 1, "num_rounds must be at least 1"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size >= 1, "batch_size must be >= 1"


class ExperimentRunner:
    """Orchestrates FL experiments, logging, and evaluation plot generation."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(__file__).parent.parent
        self.results_dir = self.project_root / "results"
        self.configs_dir = self.project_root / "experiments" / "configs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: Path) -> ExperimentConfig:
        """Load and validate an experiment configuration from JSON."""
        with open(config_path, "r") as f:
            data = json.load(f)
        config = ExperimentConfig.from_dict(data)
        config.validate()
        return config

    def save_config(self, config: ExperimentConfig, destination: Optional[Path] = None) -> Path:
        """Save configuration to JSON file."""
        dst = destination or (self.configs_dir / f"{config.experiment_id}.json")
        with open(dst, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        return dst

    def build_cli_command(self, config: ExperimentConfig) -> List[str]:
        """Convert ExperimentConfig into CLI launcher arguments."""
        python_exe = sys.executable
        run_script = str(self.project_root / "run.py")

        cmd = [
            python_exe, run_script,
            "--experiment-id", config.experiment_id,
            "--rounds", str(config.num_rounds),
            "--clients", str(config.num_clients),
            "--lr", str(config.learning_rate),
            "--local-epochs", str(config.local_epochs),
            "--batch-size", str(config.batch_size),
        ]

        if config.adaptive_privacy:
            cmd.append("--adaptive")
            if config.controller_params:
                cp = config.controller_params
                cmd.extend(["--alpha", str(cp.alpha), "--beta", str(cp.beta), "--warmup", str(cp.warmup_rounds)])
            if config.dp_params:
                dp = config.dp_params
                cmd.extend(["--noise", str(dp.noise_multiplier), "--clip", str(dp.max_grad_norm), "--delta", str(dp.target_delta)])

        elif config.secure_aggregation:
            cmd.append("--secagg")
            if config.secagg_params:
                sp = config.secagg_params
                cmd.extend(["--shares", str(sp.num_shares), "--threshold", str(sp.reconstruction_threshold)])
            if config.dp_params:
                dp = config.dp_params
                cmd.extend(["--noise", str(dp.noise_multiplier), "--clip", str(dp.max_grad_norm), "--delta", str(dp.target_delta)])

        elif config.differential_privacy:
            cmd.append("--dp")
            if config.dp_params:
                dp = config.dp_params
                cmd.extend(["--noise", str(dp.noise_multiplier), "--clip", str(dp.max_grad_norm), "--delta", str(dp.target_delta)])

        return cmd

    def generate_plots(self, experiment_id: str) -> Optional[Path]:
        """Generate accuracy, loss, and privacy trajectory plots from results CSV."""
        csv_path = self.results_dir / f"{experiment_id}.csv"
        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Experiment Diagnostic Report: {experiment_id}", fontsize=14, fontweight="bold")

        # 1. Accuracy vs Round
        ax1 = axes[0, 0]
        if "test_acc_centralised" in df.columns:
            ax1.plot(df["round"], df["test_acc_centralised"] * 100, marker="o", color="#1f77b4", label="Test Accuracy (Central)")
        if "val_acc" in df.columns and not df["val_acc"].isna().all():
            ax1.plot(df["round"], df["val_acc"] * 100, marker="s", linestyle="--", color="#aec7e8", label="Val Accuracy (Clients)")
        ax1.set_title("Model Accuracy (%) vs Round")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # 2. Loss vs Round
        ax2 = axes[0, 1]
        if "train_loss" in df.columns:
            ax2.plot(df["round"], df["train_loss"], marker="o", color="#d62728", label="Train Loss")
        if "test_loss_centralised" in df.columns:
            ax2.plot(df["round"], df["test_loss_centralised"], marker="^", linestyle="--", color="#ff9896", label="Test Loss")
        ax2.set_title("Loss vs Round")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cross-Entropy Loss")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        # 3. Privacy Budget (Epsilon) vs Round
        ax3 = axes[1, 0]
        if "epsilon_mean" in df.columns and df["epsilon_mean"].sum() > 0:
            ax3.plot(df["round"], df["epsilon_mean"], marker="d", color="#2ca02c", label="Per-Round Epsilon")
            ax3.plot(df["round"], df["epsilon_mean"].cumsum(), marker="o", linestyle="--", color="#98df8a", label="Cumulative Epsilon")
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No DP (Epsilon = 0.0)", ha="center", va="center", transform=ax3.transAxes, color="gray", fontsize=11)
        ax3.set_title("Privacy Budget (Epsilon) Trajectory")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Epsilon (ε)")
        ax3.grid(True, linestyle=":", alpha=0.6)

        # 4. Noise Multiplier Schedule
        ax4 = axes[1, 1]
        if "noise_multiplier" in df.columns and df["noise_multiplier"].sum() > 0:
            ax4.step(df["round"], df["noise_multiplier"], where="mid", marker="x", color="#9467bd", label="Noise Multiplier (σ)")
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Standard Non-Private Training", ha="center", va="center", transform=ax4.transAxes, color="gray", fontsize=11)
        ax4.set_title("Noise Multiplier Schedule")
        ax4.set_xlabel("Round")
        ax4.set_ylabel("Noise Multiplier (σ)")
        ax4.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot_path = self.results_dir / f"{experiment_id}_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return plot_path
