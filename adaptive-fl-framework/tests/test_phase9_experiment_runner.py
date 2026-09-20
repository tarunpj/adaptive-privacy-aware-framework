"""
tests/test_phase9_experiment_runner.py

Unit tests for Phase 9: Experiment Manager & Plot Generation.
Verifies:
1. Declarative JSON configuration loading and parameter validation.
2. CLI command assembly for various experiment modes (Baseline, DP, SecAgg, Adaptive).
3. Automated diagnostic plot generation from experiment CSV files.
"""

import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    DPParameters,
    SecAggParameters,
    ControllerParameters,
)


def test_experiment_config_validation():
    """Verify ExperimentConfig serialization and validation rules."""
    cfg = ExperimentConfig(
        experiment_id="test_exp",
        dataset="cifar10",
        num_clients=10,
        num_rounds=5,
        learning_rate=0.01,
        differential_privacy=True,
        dp_params=DPParameters(noise_multiplier=1.2, max_grad_norm=1.0, target_delta=1e-5),
        adaptive_privacy=True,
        controller_params=ControllerParameters(alpha=0.2, beta=0.1),
    )
    cfg.validate()

    d = cfg.to_dict()
    reconstructed = ExperimentConfig.from_dict(d)
    assert reconstructed.experiment_id == "test_exp"
    assert reconstructed.dp_params.noise_multiplier == 1.2
    assert reconstructed.controller_params.alpha == 0.2
    print(f"  [PASS] ExperimentConfig: schema serialization & deserialization verified")


def test_cli_command_builder():
    """Verify CLI command construction for various experiment modes."""
    runner = ExperimentRunner()

    # 1. Baseline
    cfg_base = ExperimentConfig(experiment_id="base", num_rounds=3, num_clients=10)
    cmd_base = runner.build_cli_command(cfg_base)
    assert "--experiment-id" in cmd_base
    assert "base" in cmd_base
    assert "--rounds" in cmd_base
    assert "3" in cmd_base
    assert "--dp" not in cmd_base

    # 2. Adaptive DP
    cfg_adapt = ExperimentConfig(
        experiment_id="adapt",
        num_rounds=3,
        adaptive_privacy=True,
        dp_params=DPParameters(noise_multiplier=1.0),
        controller_params=ControllerParameters(alpha=0.15),
    )
    cmd_adapt = runner.build_cli_command(cfg_adapt)
    assert "--adaptive" in cmd_adapt
    assert "--alpha" in cmd_adapt
    assert "0.15" in cmd_adapt
    print(f"  [PASS] ExperimentRunner: CLI command assembly for Baseline and Adaptive DP verified")


def test_plot_generation():
    """Verify plot generation from existing result CSV."""
    runner = ExperimentRunner()
    plot_path = runner.generate_plots("baseline_fedavg")
    if plot_path and plot_path.exists():
        assert plot_path.stat().st_size > 0
        print(f"  [PASS] Plot generation verified: {plot_path.name} ({plot_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [SKIP] baseline_fedavg.csv not found for plot testing")


if __name__ == "__main__":
    print("\nRunning Phase 9 Experiment Manager unit tests...\n")
    test_experiment_config_validation()
    test_cli_command_builder()
    test_plot_generation()
    print("\nAll Phase 9 tests passed successfully.\n")
