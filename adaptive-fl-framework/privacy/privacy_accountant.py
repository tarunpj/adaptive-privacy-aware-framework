"""
privacy/privacy_accountant.py

Tracks privacy budget consumption across federated learning rounds.

Why this is needed:
  Opacus reports epsilon for a single training session (one client, one round).
  In federated learning, the same client may participate in multiple rounds.
  The total privacy cost for a client is the sum of per-round costs (composition).

  Under basic composition:
      total_epsilon = sum of per-round epsilons

  Under advanced composition (tighter bound, used here):
      total_epsilon grows sub-linearly with rounds (via RDP composition)

  For simplicity in Phase 2, we track per-round epsilon reported by Opacus
  and record the maximum across clients per round (worst-case client).

  In Phase 4 (Adaptive Privacy), the controller will use these values
  to decide whether to increase or decrease noise.
"""

from dataclasses import dataclass, field


@dataclass
class RoundPrivacyRecord:
    """Privacy metrics for one FL round."""
    round_num: int
    epsilon_max: float      # worst-case epsilon across all clients this round
    epsilon_mean: float     # average epsilon across clients
    noise_multiplier: float
    max_grad_norm: float
    target_delta: float


class PrivacyAccountant:
    """
    Accumulates per-round privacy metrics reported by clients.

    Usage:
        accountant = PrivacyAccountant(target_delta=1e-5)
        accountant.record_round(round_num=1, client_epsilons=[0.8, 0.9, 0.85], ...)
        print(accountant.cumulative_epsilon)   # worst-case total so far
    """

    def __init__(self, target_delta: float = 1e-5) -> None:
        self.target_delta = target_delta
        self.rounds: list[RoundPrivacyRecord] = []

    def record_round(
        self,
        round_num: int,
        client_epsilons: list[float],
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> RoundPrivacyRecord:
        """Record privacy metrics for one completed round."""
        if not client_epsilons:
            eps_max = eps_mean = float("nan")
        else:
            eps_max  = max(client_epsilons)
            eps_mean = sum(client_epsilons) / len(client_epsilons)

        record = RoundPrivacyRecord(
            round_num=round_num,
            epsilon_max=eps_max,
            epsilon_mean=eps_mean,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_delta=self.target_delta,
        )
        self.rounds.append(record)
        return record

    @property
    def cumulative_epsilon(self) -> float:
        """
        Worst-case cumulative epsilon under basic composition.
        (Sum of per-round max epsilons — conservative upper bound.)
        """
        return sum(r.epsilon_max for r in self.rounds if r.epsilon_max == r.epsilon_max)

    @property
    def latest_epsilon(self) -> float:
        """Epsilon from the most recent round."""
        return self.rounds[-1].epsilon_max if self.rounds else float("nan")

    def summary(self) -> dict:
        return {
            "rounds_recorded": len(self.rounds),
            "cumulative_epsilon": round(self.cumulative_epsilon, 6),
            "latest_epsilon": round(self.latest_epsilon, 6),
            "target_delta": self.target_delta,
            "per_round": [
                {
                    "round": r.round_num,
                    "epsilon_max": round(r.epsilon_max, 6),
                    "epsilon_mean": round(r.epsilon_mean, 6),
                    "noise_multiplier": r.noise_multiplier,
                    "max_grad_norm": r.max_grad_norm,
                }
                for r in self.rounds
            ],
        }
