"""
federated/server_adaptive.py

Flower ServerApp for Phase 4: Adaptive Privacy Controller.

The challenge: FedAvg.start() runs all rounds internally, so we cannot
intercept between rounds to update DP parameters.

Solution: We run one round at a time by calling strategy.start(num_rounds=1)
in a loop. After each round we:
  1. Feed metrics to the AdaptivePrivacyController
  2. Get new noise_multiplier and max_grad_norm
  3. Pass them to clients via ConfigRecord in the next round

This gives us full per-round control while still using Flower's FedAvg.
"""

import csv
import json
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated.task import Net, load_centralized_testset, test
from privacy.adaptive_controller import (
    AdaptivePrivacyController,
    AdaptiveControllerConfig,
    ControllerDecision,
)
from privacy.privacy_accountant import PrivacyAccountant

app = ServerApp()

RESULTS_DIR = Path("results")

CSV_FIELDS = [
    "experiment_id", "round",
    "train_loss", "val_loss", "val_acc",
    "test_loss_centralised", "test_acc_centralised",
    "epsilon_mean",
    "noise_multiplier", "max_grad_norm", "target_delta",
    "controller_action", "convergence_rate", "budget_pressure",
    "dp_enabled", "secure_agg_enabled", "adaptive_privacy",
]


def _fmt(v) -> float:
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return float("nan")


def _centralised_eval(net_weights_arrays: ArrayRecord, testloader, device) -> tuple[float, float]:
    net = Net()
    net.load_state_dict(net_weights_arrays.to_torch_state_dict())
    return test(net, testloader, device)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Adaptive server: runs one FL round at a time, updating DP params each round."""

    # ── Read configuration ────────────────────────────────────────────────
    num_rounds:        int   = int(context.run_config["num-server-rounds"])
    lr:                float = float(context.run_config["learning-rate"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    experiment_id:     str   = str(context.run_config.get("experiment-id", "adaptive_dp"))
    target_delta:      float = float(context.run_config.get("dp-target-delta", 1e-5))

    # Adaptive controller config
    ctrl_cfg = AdaptiveControllerConfig(
        initial_noise_multiplier = float(context.run_config.get("dp-noise-multiplier", 0.5)),
        initial_max_grad_norm    = float(context.run_config.get("dp-max-grad-norm", 1.2)),
        target_delta             = target_delta,
        alpha                    = float(context.run_config.get("ctrl-alpha", 0.10)),
        beta                     = float(context.run_config.get("ctrl-beta", 0.08)),
        threshold_high           = float(context.run_config.get("ctrl-threshold-high", 0.05)),
        threshold_low            = float(context.run_config.get("ctrl-threshold-low", 0.01)),
        noise_min                = float(context.run_config.get("ctrl-noise-min", 0.3)),
        noise_max                = float(context.run_config.get("ctrl-noise-max", 2.0)),
        epsilon_budget           = float(context.run_config.get("ctrl-epsilon-budget", 50.0)),
        warmup_rounds            = int(context.run_config.get("ctrl-warmup-rounds", 5)),
    )

    controller  = AdaptivePrivacyController(ctrl_cfg)
    accountant  = PrivacyAccountant(target_delta=target_delta)
    controller.initialize()

    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = load_centralized_testset()

    global_model   = Net()
    current_arrays = ArrayRecord(global_model.state_dict())
    strategy       = FedAvg(fraction_evaluate=fraction_evaluate)

    print(f"\n{'='*60}")
    print(f"  Adaptive FL Framework  |  Phase 4: Adaptive DP")
    print(f"  Rounds: {num_rounds}  |  LR: {lr}  |  ID: {experiment_id}")
    print(f"  Initial noise={ctrl_cfg.initial_noise_multiplier}  "
          f"clip={ctrl_cfg.initial_max_grad_norm}  delta={target_delta}")
    print(f"  Controller: alpha={ctrl_cfg.alpha}  beta={ctrl_cfg.beta}  "
          f"warmup={ctrl_cfg.warmup_rounds}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = RESULTS_DIR / f"{experiment_id}.csv"
    json_path = RESULTS_DIR / f"{experiment_id}_summary.json"
    csv_path.unlink(missing_ok=True)

    t_start = time.time()
    cumulative_epsilon = 0.0

    # ── Per-round loop ────────────────────────────────────────────────────
    for rnd in range(1, num_rounds + 1):

        noise = controller.current_noise
        clip  = controller.current_clip

        # Pass current DP params to clients via train_config
        train_cfg = ConfigRecord({
            "lr":                  lr,
            "dp-noise-multiplier": noise,
            "dp-max-grad-norm":    clip,
        })

        # Run exactly ONE round of FedAvg
        result = strategy.start(
            grid=grid,
            initial_arrays=current_arrays,
            num_rounds=1,
            train_config=train_cfg,
        )

        current_arrays = result.arrays   # updated global weights

        # ── Collect metrics from this round ───────────────────────────────
        train_m = result.train_metrics_clientapp.get(1, MetricRecord({}))
        eval_m  = result.evaluate_metrics_clientapp.get(1, MetricRecord({}))

        train_loss = _fmt(train_m.get("train_loss", float("nan")))
        val_loss   = _fmt(eval_m.get("val_loss",    float("nan")))
        val_acc    = _fmt(eval_m.get("val_acc",     float("nan")))
        eps_round  = _fmt(train_m.get("epsilon",    0.0))

        # Accumulate epsilon (basic composition)
        cumulative_epsilon += eps_round
        accountant.record_round(rnd, [eps_round], noise, clip)

        # ── Centralised evaluation ────────────────────────────────────────
        test_loss, test_acc = _centralised_eval(current_arrays, testloader, device)

        # ── Feed controller ───────────────────────────────────────────────
        decision: ControllerDecision = controller.observe_and_update(
            round_num=rnd,
            current_acc=test_acc,
            current_loss=train_loss,
            cumulative_epsilon=cumulative_epsilon,
        )

        print(
            f"  Round {rnd:>2}/{num_rounds} | "
            f"test_acc={test_acc:.4f} | train_loss={train_loss:.4f} | "
            f"eps={eps_round:.4f} (cum={cumulative_epsilon:.4f}) | "
            f"noise: {noise:.4f}->{decision.new_noise:.4f} | "
            f"action={decision.action}"
        )

        # ── Write CSV row ─────────────────────────────────────────────────
        write_header = (rnd == 1)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "experiment_id":         experiment_id,
                "round":                 rnd,
                "train_loss":            train_loss,
                "val_loss":              val_loss,
                "val_acc":               val_acc,
                "test_loss_centralised": _fmt(test_loss),
                "test_acc_centralised":  _fmt(test_acc),
                "epsilon_mean":          eps_round,
                "noise_multiplier":      round(noise, 6),
                "max_grad_norm":         round(clip, 6),
                "target_delta":          target_delta,
                "controller_action":     decision.action,
                "convergence_rate":      decision.convergence_rate,
                "budget_pressure":       decision.budget_pressure,
                "dp_enabled":            True,
                "secure_agg_enabled":    False,
                "adaptive_privacy":      True,
            })

    total_time = time.time() - t_start
    ctrl_summary = controller.summary()

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Controller summary: {ctrl_summary['actions']}")
    print(f"  Final noise: {ctrl_summary['final_noise']}  "
          f"Final clip: {ctrl_summary['final_clip']}")
    print(f"  Results saved -> {csv_path}")

    summary = {
        "experiment_id":      experiment_id,
        "num_rounds":         num_rounds,
        "total_time_s":       round(total_time, 2),
        "final_test_acc":     _fmt(test_acc),
        "final_test_loss":    _fmt(test_loss),
        "cumulative_epsilon": round(cumulative_epsilon, 6),
        "dp_enabled":         True,
        "adaptive_privacy":   True,
        "secure_agg":         False,
        "controller_config":  {
            "alpha":           ctrl_cfg.alpha,
            "beta":            ctrl_cfg.beta,
            "threshold_high":  ctrl_cfg.threshold_high,
            "threshold_low":   ctrl_cfg.threshold_low,
            "noise_min":       ctrl_cfg.noise_min,
            "noise_max":       ctrl_cfg.noise_max,
            "warmup_rounds":   ctrl_cfg.warmup_rounds,
            "epsilon_budget":  ctrl_cfg.epsilon_budget,
        },
        "controller_summary": ctrl_summary,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved  -> {json_path}\n")
