"""
security/secure_aggregation.py

Secure Aggregation using Flower's SecAgg+ protocol.

═══════════════════════════════════════════════════════════════
THREAT MODEL AND TRUST ASSUMPTIONS
═══════════════════════════════════════════════════════════════

What SecAgg+ protects against:
  - A semi-honest (honest-but-curious) server that follows the protocol
    but tries to learn individual client model updates.
  - The server receives ONLY the aggregate of client updates, not any
    individual client's update.

What the server CAN see:
  - The final aggregated model update (sum of all client updates)
  - Which clients participated (node IDs)
  - Aggregated metrics (loss, accuracy averages)
  - The number of participating clients

What the server CANNOT see:
  - Any individual client's model update / gradient
  - Any individual client's local data distribution (inferred from updates)

Protocol: SecAgg+ (Bonawitz et al., 2017 / Bell et al., 2020)
  - Each client secret-shares its model update among other clients
  - Shares are masked with pairwise random seeds
  - The server can only reconstruct the SUM, not individual updates
  - Tolerates client dropout up to (num_shares - reconstruction_threshold)

Parameters:
  num_shares:                 Number of secret shares per client update.
                              Higher = more robust to dropout, more overhead.
  reconstruction_threshold:   Minimum shares needed to reconstruct the sum.
                              Must be <= num_shares.
                              Dropout tolerance = num_shares - reconstruction_threshold

Limitations:
  - Does NOT protect against a malicious server that deviates from protocol
  - Does NOT protect against colluding clients (> num_shares - threshold)
  - Communication overhead scales with num_shares
  - Requires a minimum number of clients to complete the protocol

Combined with DP:
  SecAgg+ + DP provides defence-in-depth:
  - SecAgg+ hides individual updates from the server
  - DP adds noise to prevent membership inference even from the aggregate

═══════════════════════════════════════════════════════════════
IMPLEMENTATION NOTE
═══════════════════════════════════════════════════════════════

Flower's SecAgg+ is implemented as:
  - Client side: secaggplus_mod + fixedclipping_mod (middleware mods)
  - Server side: SecAggPlusWorkflow wrapping the DefaultWorkflow

These are applied via the Legacy API on the server and as ClientApp mods.
The SecAgg+ cryptographic protocol (Diffie-Hellman key agreement,
secret sharing, masking) is handled entirely by Flower internals.
"""

# Configuration dataclass used by server.py and run.py
from dataclasses import dataclass


@dataclass
class SecAggConfig:
    """
    Parameters for the SecAgg+ protocol.

    num_shares:
        How many secret shares each client's update is split into.
        Recommended: roughly sqrt(num_clients) to num_clients - 1.

    reconstruction_threshold:
        Minimum shares needed to reconstruct the aggregate.
        Must satisfy: 2 <= reconstruction_threshold <= num_shares.
        Dropout tolerance = num_shares - reconstruction_threshold.

    Example for 10 clients:
        num_shares=7, reconstruction_threshold=4
        → tolerates up to 3 client dropouts
    """
    num_shares: int = 7
    reconstruction_threshold: int = 4

    def validate(self) -> None:
        assert self.num_shares >= 2, "num_shares must be >= 2"
        assert 2 <= self.reconstruction_threshold <= self.num_shares, (
            "reconstruction_threshold must be between 2 and num_shares"
        )
