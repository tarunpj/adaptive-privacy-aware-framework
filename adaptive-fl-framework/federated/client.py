"""
federated/client.py

Flower ClientApp for the Adaptive FL Framework.

Supports two training modes, selected by `dp-enabled` in run config:

  dp-enabled = false  →  standard SGD training (Phase 1 baseline)
  dp-enabled = true   →  Opacus DP training with gradient clipping + noise (Phase 2)

The client always returns:
  - updated model weights (ArrayRecord)
  - training metrics including epsilon/noise if DP is active (MetricRecord)
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated.task import Net, load_data, test, train
from privacy.differential_privacy import DPConfig, DPTrainResult, train_with_dp, unwrap_model

app = ClientApp()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_dp_config(context: Context, msg: Message) -> DPConfig:
    """
    Load DP config — prefer per-round values from train_config (msg),
    fall back to run_config defaults. This allows the adaptive server
    to update noise/clip each round via ConfigRecord.
    """
    cfg = msg.content["config"]
    # Per-round override from adaptive server takes priority
    noise = float(cfg.get("dp-noise-multiplier",
                  context.run_config.get("dp-noise-multiplier", 1.0)))
    clip  = float(cfg.get("dp-max-grad-norm",
                  context.run_config.get("dp-max-grad-norm", 1.0)))
    delta = float(context.run_config.get("dp-target-delta", 1e-5))
    return DPConfig(noise_multiplier=noise, max_grad_norm=clip, target_delta=delta)


@app.train()
def train_round(msg: Message, context: Context) -> Message:
    """Local training — standard or DP depending on run config."""

    lr: float          = float(msg.content["config"]["lr"])
    local_epochs: int  = int(context.run_config["local-epochs"])
    batch_size: int    = int(context.run_config["batch-size"])
    partition_id: int  = context.node_config["partition-id"]
    num_partitions: int = context.node_config["num-partitions"]
    dp_enabled: bool   = bool(context.run_config.get("dp-enabled", False))

    device = _device()

    net = Net()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    net.to(device)

    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)

    if dp_enabled:
        # ── DP Training path ──────────────────────────────────────────────
        dp_config = _load_dp_config(context, msg)
        dp_result: DPTrainResult = train_with_dp(
            net, trainloader, local_epochs, lr, device, dp_config
        )
        train_loss = dp_result.train_loss
        epsilon    = dp_result.epsilon
        noise_mult = dp_result.noise_multiplier
        grad_norm  = dp_result.max_grad_norm

        # Unwrap Opacus GradSampleModule to get a clean state_dict
        clean_net = unwrap_model(net)
        val_loss, val_acc = test(clean_net, valloader, device)

        arrays  = ArrayRecord(clean_net.state_dict())
        metrics = MetricRecord({
            "train_loss":      float(train_loss),
            "val_loss":        float(val_loss),
            "val_acc":         float(val_acc),
            "epsilon":         float(epsilon),
            "noise_multiplier": float(noise_mult),
            "max_grad_norm":   float(grad_norm),
            "dp_enabled":      1.0,
            "num-examples":    float(len(trainloader.dataset)),
        })

    else:
        # ── Standard Training path ────────────────────────────────────────
        train_loss = train(net, trainloader, local_epochs, lr, device)
        val_loss, val_acc = test(net, valloader, device)

        arrays  = ArrayRecord(net.state_dict())
        metrics = MetricRecord({
            "train_loss":  float(train_loss),
            "val_loss":    float(val_loss),
            "val_acc":     float(val_acc),
            "epsilon":     0.0,
            "dp_enabled":  0.0,
            "num-examples": float(len(trainloader.dataset)),
        })

    return Message(
        content=RecordDict({"arrays": arrays, "metrics": metrics}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_round(msg: Message, context: Context) -> Message:
    """Local evaluation — same for both DP and non-DP modes."""

    batch_size: int    = int(context.run_config["batch-size"])
    partition_id: int  = context.node_config["partition-id"]
    num_partitions: int = context.node_config["num-partitions"]

    device = _device()

    net = Net()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    net.to(device)

    _, valloader = load_data(partition_id, num_partitions, batch_size)
    val_loss, val_acc = test(net, valloader, device)

    metrics = MetricRecord({
        "val_loss":    float(val_loss),
        "val_acc":     float(val_acc),
        "num-examples": float(len(valloader.dataset)),
    })
    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=msg,
    )
