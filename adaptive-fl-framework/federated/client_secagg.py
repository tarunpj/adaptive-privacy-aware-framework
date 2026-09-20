"""
federated/client_secagg.py

Flower ClientApp for Phase 3: FedAvg + DP + Secure Aggregation.

Uses the Legacy NumPyClient API because SecAgg+ mods (secaggplus_mod,
fixedclipping_mod) are designed to work with the NumPyClient interface.

The two mods act as middleware:
  fixedclipping_mod:  clips the model update (delta weights) to max_grad_norm
                      before it leaves the client — client-side DP clipping
  secaggplus_mod:     encrypts and secret-shares the clipped update so the
                      server only sees the aggregate, not individual updates

Training itself is standard SGD (no Opacus here — clipping is done by
fixedclipping_mod on the model update, not per-sample gradients).
"""

import torch
from collections import OrderedDict

from flwr.client import NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from flwr.clientapp import ClientApp
from flwr.common import Context

from federated.task import Net, load_data, test, train


class SecAggClient(NumPyClient):
    """
    Standard federated client with SecAgg+ and fixed clipping middleware.

    The mods (secaggplus_mod, fixedclipping_mod) are applied automatically
    by Flower after fit() returns — the client code itself is unchanged.
    """

    def __init__(self, partition_id: int, num_partitions: int,
                 batch_size: int, local_epochs: int, lr: float) -> None:
        self.net = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader, self.valloader = load_data(
            partition_id, num_partitions, batch_size
        )
        self.local_epochs = local_epochs
        self.lr = lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss = train(
            self.net, self.trainloader,
            self.local_epochs, self.lr, self.device,
        )
        val_loss, val_acc = test(self.net, self.valloader, self.device)
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "val_loss":   float(val_loss),
                "val_accuracy": float(val_acc),
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size     = int(context.run_config.get("batch-size", 32))
    local_epochs   = int(context.run_config.get("local-epochs", 1))
    lr             = float(context.run_config.get("learning-rate", 0.01))

    return SecAggClient(
        partition_id, num_partitions, batch_size, local_epochs, lr
    ).to_client()


# SecAgg+ mods applied as middleware — order matters:
#   fixedclipping_mod clips the update first, then secaggplus_mod encrypts it
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
        fixedclipping_mod,
    ],
)
