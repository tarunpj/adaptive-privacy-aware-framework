"""
federated/server_secagg.py

Flower ServerApp for Phase 3: FedAvg + DP + Secure Aggregation.

Uses the Flower Legacy API because SecAggPlusWorkflow requires LegacyContext.
The modern FedAvg.start() API does not support SecAgg+ workflow injection.

Architecture:
  FedAvg strategy
    └── wrapped by DifferentialPrivacyClientSideFixedClipping
          └── executed inside SecAggPlusWorkflow
                └── inside DefaultWorkflow

SecAgg+ guarantees the server only sees the aggregate, not individual updates.
DP adds calibrated noise to the aggregate for additional privacy protection.
"""

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
from flwr.app import ArrayRecord
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from federated.task import Net, load_centralized_testset, test

app = ServerApp()

RESULTS_DIR = Path("results")

CSV_FIELDS = [
    "experiment_id", "round",
    "train_loss", "val_loss", "val_acc",
    "test_loss_centralised", "test_acc_centralised",
    "epsilon_mean", "noise_multiplier", "max_grad_norm", "target_delta",
    "dp_enabled", "secure_agg_enabled", "adaptive_privacy",
    "num_shares", "reconstruction_threshold",
]


def _weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of client metrics by number of examples."""
    total = sum(n for n, _ in metrics)
    return {
        key: sum(n * m[key] for n, m in metrics if key in m) / total
        for key in metrics[0][1].keys()
        if all(key in m for _, m in metrics)
    }


def _run_centralised_eval(arrays_list, device, testloader):
    """Evaluate the global model on the centralised test set."""
    import numpy as np
    from collections import OrderedDict
    net = Net()
    # arrays_list is a list of numpy arrays (Legacy API format)
    params_dict = zip(net.state_dict().keys(), arrays_list)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    loss, accuracy = test(net, testloader, device)
    return loss, accuracy


def _save_results(
    experiment_id, history, num_rounds, total_time,
    noise_multiplier, max_grad_norm, target_delta,
    num_shares, reconstruction_threshold,
    device, testloader,
):
    """Save per-round metrics from Flower History object to CSV + JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = RESULTS_DIR / f"{experiment_id}.csv"
    json_path = RESULTS_DIR / f"{experiment_id}_summary.json"

    # Extract metrics from Flower History
    # history.losses_distributed: list of (round, loss) tuples
    # history.metrics_distributed: dict of metric_name -> list of (round, value)
    losses_dist = dict(history.losses_distributed)
    metrics_dist = history.metrics_distributed_fit or {}
    losses_central = dict(history.losses_centralized or [])
    metrics_central = dict(history.metrics_centralized or [])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for rnd in range(1, num_rounds + 1):
            train_loss = metrics_dist.get("train_loss", {})
            val_loss   = metrics_dist.get("val_loss", {})
            val_acc    = metrics_dist.get("val_accuracy", {})

            # Legacy API metrics are stored as list of (round, value)
            tl = dict(train_loss).get(rnd, float("nan")) if isinstance(train_loss, list) else float("nan")
            vl = dict(val_loss).get(rnd, float("nan"))   if isinstance(val_loss, list)   else float("nan")
            va = dict(val_acc).get(rnd, float("nan"))    if isinstance(val_acc, list)     else float("nan")

            cl = losses_central.get(rnd, float("nan"))
            ca = dict(metrics_central.get("accuracy", [])).get(rnd, float("nan"))

            writer.writerow({
                "experiment_id":          experiment_id,
                "round":                  rnd,
                "train_loss":             round(float(tl), 6),
                "val_loss":               round(float(vl), 6),
                "val_acc":                round(float(va), 6),
                "test_loss_centralised":  round(float(cl), 6),
                "test_acc_centralised":   round(float(ca), 6),
                "epsilon_mean":           0.0,
                "noise_multiplier":       noise_multiplier,
                "max_grad_norm":          max_grad_norm,
                "target_delta":           target_delta,
                "dp_enabled":             True,
                "secure_agg_enabled":     True,
                "adaptive_privacy":       False,
                "num_shares":             num_shares,
                "reconstruction_threshold": reconstruction_threshold,
            })

    print(f"\n  Results saved  -> {csv_path}")

    final_ca = dict(metrics_central.get("accuracy", [])).get(num_rounds, float("nan"))
    final_cl = losses_central.get(num_rounds, float("nan"))

    summary = {
        "experiment_id":          experiment_id,
        "num_rounds":             num_rounds,
        "total_time_s":           round(total_time, 2),
        "final_test_acc":         round(float(final_ca), 6),
        "final_test_loss":        round(float(final_cl), 6),
        "dp_enabled":             True,
        "noise_multiplier":       noise_multiplier,
        "max_grad_norm":          max_grad_norm,
        "target_delta":           target_delta,
        "secure_agg":             True,
        "num_shares":             num_shares,
        "reconstruction_threshold": reconstruction_threshold,
        "adaptive_privacy":       False,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved  -> {json_path}\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Server entry point for Phase 3: DP + SecAgg+."""

    num_rounds:               int   = int(context.run_config["num-server-rounds"])
    noise_multiplier:         float = float(context.run_config.get("dp-noise-multiplier", 1.0))
    max_grad_norm:            float = float(context.run_config.get("dp-max-grad-norm", 1.0))
    target_delta:             float = float(context.run_config.get("dp-target-delta", 1e-5))
    num_shares:               int   = int(context.run_config.get("secagg-num-shares", 3))
    reconstruction_threshold: int   = int(context.run_config.get("secagg-threshold", 2))
    num_sampled_clients:      int   = int(context.run_config.get("secagg-num-sampled-clients", 10))
    experiment_id:            str   = str(context.run_config.get("experiment-id", "fedavg_dp_sa"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = load_centralized_testset()

    # Initialise global model in Legacy API format (list of numpy arrays)
    import numpy as np
    net = Net()
    model_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    parameters = ndarrays_to_parameters(model_weights)

    print(f"\n{'='*60}")
    print(f"  Adaptive FL Framework  |  Phase 3: FedAvg + DP + SecAgg+")
    print(f"  Rounds: {num_rounds}  |  ID: {experiment_id}")
    print(f"  DP: noise={noise_multiplier}  clip={max_grad_norm}  delta={target_delta}")
    print(f"  SecAgg+: shares={num_shares}  threshold={reconstruction_threshold}")
    print(f"{'='*60}\n")

    # Centralised evaluation function for Legacy API
    def evaluate_fn(server_round, parameters, config):
        import numpy as np
        from collections import OrderedDict
        eval_net = Net()
        params_dict = zip(eval_net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(np.array(v)) for k, v in params_dict})
        eval_net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(eval_net, testloader, device)
        print(f"  [Server eval] Round {server_round} | test_acc={accuracy:.4f} | test_loss={loss:.4f}")
        return loss, {"accuracy": accuracy}

    # Base FedAvg strategy
    fraction_fit = min(1.0, num_sampled_clients / max(num_sampled_clients, 1))
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,          # evaluation handled centrally
        min_fit_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=_weighted_average,
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,
    )

    # Wrap with server-side DP clipping + noise
    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=max_grad_norm,
        num_sampled_clients=num_sampled_clients,
    )

    # Build LegacyContext required by SecAggPlusWorkflow
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # SecAgg+ workflow — wraps the default fit workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=num_shares,
            reconstruction_threshold=reconstruction_threshold,
        )
    )

    t_start = time.time()
    workflow(grid, legacy_context)
    total_time = time.time() - t_start

    print(f"\n  Total time: {total_time:.1f}s")

    _save_results(
        experiment_id,
        legacy_context.history,
        num_rounds,
        total_time,
        noise_multiplier,
        max_grad_norm,
        target_delta,
        num_shares,
        reconstruction_threshold,
        device,
        testloader,
    )
