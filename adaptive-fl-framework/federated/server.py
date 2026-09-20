"""
federated/server.py

Flower ServerApp for the Adaptive FL Framework.

Works for all experiment modes:
  - Phase 1: standard FedAvg (dp-enabled=false)
  - Phase 2: FedAvg + Differential Privacy (dp-enabled=true)

Per-round metrics saved to results/<experiment-id>.csv
Summary saved to results/<experiment-id>_summary.json
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

app = ServerApp()

RESULTS_DIR = Path("results")

CSV_FIELDS = [
    "experiment_id", "round", "participating_clients",
    "train_loss", "val_loss", "val_acc",
    "test_loss_centralised", "test_acc_centralised",
    "model_size_bytes",
    "round_time_s", "total_time_s",
    "epsilon_mean", "epsilon_max",
    "noise_multiplier", "max_grad_norm", "target_delta",
    "dp_enabled", "secure_agg_enabled", "adaptive_privacy",
]


def _centralised_eval_fn(testloader, device, round_times: dict):
    """Closure for server-side centralised evaluation after each round."""
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        r_start = time.time()
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy = test(net, testloader, device)
        r_duration = time.time() - r_start
        round_times[server_round] = round(r_duration, 3)
        print(
            f"  [Server eval] Round {server_round} | "
            f"test_acc={accuracy:.4f} | test_loss={loss:.4f}"
        )
        return MetricRecord({"test_loss": float(loss), "test_acc": float(accuracy)})
    return evaluate


def _save_results(
    experiment_id: str,
    result,
    num_rounds: int,
    num_clients: int,
    total_time: float,
    model_size_bytes: int,
    round_times: dict,
    dp_enabled: bool,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
) -> None:
    """Write per-round metrics to CSV and summary to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = RESULTS_DIR / f"{experiment_id}.csv"
    json_path = RESULTS_DIR / f"{experiment_id}_summary.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for rnd in range(1, num_rounds + 1):
            train_m = result.train_metrics_clientapp.get(rnd, MetricRecord({}))
            eval_m  = result.evaluate_metrics_clientapp.get(rnd, MetricRecord({}))
            srv_m   = result.evaluate_metrics_serverapp.get(rnd, MetricRecord({}))

            writer.writerow({
                "experiment_id":         experiment_id,
                "round":                 rnd,
                "participating_clients": num_clients,
                "train_loss":            _fmt(train_m.get("train_loss",  float("nan"))),
                "val_loss":              _fmt(eval_m.get("val_loss",     float("nan"))),
                "val_acc":               _fmt(eval_m.get("val_acc",      float("nan"))),
                "test_loss_centralised": _fmt(srv_m.get("test_loss",     float("nan"))),
                "test_acc_centralised":  _fmt(srv_m.get("test_acc",      float("nan"))),
                "model_size_bytes":      model_size_bytes,
                "round_time_s":          round_times.get(rnd, 0.0),
                "total_time_s":          round(total_time, 2),
                # epsilon is reported per-client and averaged by FedAvg
                "epsilon_mean":          _fmt(train_m.get("epsilon",     0.0)),
                "epsilon_max":           _fmt(train_m.get("epsilon",     0.0)),
                "noise_multiplier":      noise_multiplier if dp_enabled else 0.0,
                "max_grad_norm":         max_grad_norm    if dp_enabled else 0.0,
                "target_delta":          target_delta     if dp_enabled else 0.0,
                "dp_enabled":            dp_enabled,
                "secure_agg_enabled":    False,
                "adaptive_privacy":      False,
            })

    print(f"\n  Results saved  -> {csv_path}")

    final_srv = result.evaluate_metrics_serverapp.get(num_rounds, MetricRecord({}))
    final_train = result.train_metrics_clientapp.get(num_rounds, MetricRecord({}))

    summary = {
        "experiment_id":         experiment_id,
        "num_rounds":            num_rounds,
        "participating_clients": num_clients,
        "total_time_s":          round(total_time, 2),
        "model_size_bytes":      model_size_bytes,
        "model_size_kb":         round(model_size_bytes / 1024, 2),
        "final_test_acc":        _fmt(final_srv.get("test_acc",   float("nan"))),
        "final_test_loss":       _fmt(final_srv.get("test_loss",  float("nan"))),
        "final_epsilon":         _fmt(final_train.get("epsilon",  0.0)),
        "dp_enabled":            dp_enabled,
        "noise_multiplier":      noise_multiplier if dp_enabled else 0.0,
        "max_grad_norm":         max_grad_norm    if dp_enabled else 0.0,
        "target_delta":          target_delta     if dp_enabled else 0.0,
        "secure_agg":            False,
        "adaptive_privacy":      False,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved  -> {json_path}\n")


def _fmt(v) -> float:
    return round(float(v), 6)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Server entry point: initialise model, run FedAvg, log results."""

    num_rounds:        int   = int(context.run_config["num-server-rounds"])
    lr:                float = float(context.run_config["learning-rate"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    experiment_id:     str   = str(context.run_config.get("experiment-id", "baseline_fedavg"))
    num_clients:       int   = int(context.run_config.get("num-clients", 10))

    # DP config (read here for logging; actual DP happens on clients)
    dp_enabled:       bool  = bool(context.run_config.get("dp-enabled", False))
    noise_multiplier: float = float(context.run_config.get("dp-noise-multiplier", 1.0))
    max_grad_norm:    float = float(context.run_config.get("dp-max-grad-norm", 1.0))
    target_delta:     float = float(context.run_config.get("dp-target-delta", 1e-5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = load_centralized_testset()

    global_model = Net()
    model_size_bytes = sum(p.numel() * p.element_size() for p in global_model.parameters())
    initial_arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    mode = "FedAvg + DP (Opacus)" if dp_enabled else "FedAvg (Baseline)"
    print(f"\n{'='*60}")
    print(f"  Adaptive FL Framework  |  {mode}")
    print(f"  Rounds: {num_rounds}  |  Clients: {num_clients}  |  LR: {lr}  |  ID: {experiment_id}")
    print(f"  Model parameter footprint: {model_size_bytes:,} bytes (~{model_size_bytes/1024:.1f} KB)")
    if dp_enabled:
        print(f"  DP: noise={noise_multiplier}  clip={max_grad_norm}  delta={target_delta}")
    print(f"{'='*60}\n")

    round_times = {}
    t_start = time.time()

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_fn=_centralised_eval_fn(testloader, device, round_times),
    )

    total_time = time.time() - t_start
    print(f"\n  Total simulation time: {total_time:.1f}s")

    _save_results(
        experiment_id, result, num_rounds, num_clients, total_time,
        model_size_bytes, round_times,
        dp_enabled, noise_multiplier, max_grad_norm, target_delta,
    )
