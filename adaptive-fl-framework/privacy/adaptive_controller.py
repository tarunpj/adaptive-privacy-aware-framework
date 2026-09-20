"""
privacy/adaptive_controller.py

Convergence-Aware Adaptive Differential Privacy Controller.

═══════════════════════════════════════════════════════════════
RESEARCH CONTRIBUTION
═══════════════════════════════════════════════════════════════

Most existing DP-FL systems use a fixed noise multiplier throughout training.
Andrew et al. (2021) adapts only the clipping norm using gradient quantiles.
No existing work uses accuracy + convergence rate to jointly adapt both
noise multiplier and clipping norm with a documented, reproducible rule.

This controller observes multiple runtime signals each round and adjusts
DP parameters to balance privacy protection with model utility.

═══════════════════════════════════════════════════════════════
ALGORITHM
═══════════════════════════════════════════════════════════════

At round t, given:
    acc_t       = current test accuracy
    acc_{t-1}   = previous test accuracy
    epsilon_t   = cumulative privacy budget consumed
    loss_t      = current training loss

Step 1 — Compute convergence rate:
    convergence_rate = (acc_t - acc_{t-1}) / (acc_{t-1} + 1e-8)

    Interpretation:
        > threshold_high  → model learning fast, can afford more noise
        < threshold_low   → model learning slowly, reduce noise to help
        in between        → stable, maintain current noise

Step 2 — Budget pressure:
    budget_pressure = epsilon_t / epsilon_budget
    (how much of the total allowed budget has been consumed)

    When budget_pressure > 0.8: force noise increase regardless of convergence
    (prevents runaway privacy spending)

Step 3 — Compute noise adjustment:
    if budget_pressure > 0.8:
        new_noise = noise * (1 + alpha)          # forced increase
    elif convergence_rate > threshold_high:
        new_noise = noise * (1 + alpha)          # fast convergence → more privacy
    elif convergence_rate < threshold_low:
        new_noise = noise * (1 - beta)           # slow convergence → less noise
    else:
        new_noise = noise                        # stable → no change

Step 4 — Apply hard bounds:
    new_noise = clip(new_noise, noise_min, noise_max)

Step 5 — Adapt clipping norm (proportional to noise change):
    new_clip = clip_norm * (new_noise / noise)   # scale together
    new_clip = clip(new_clip, clip_min, clip_max)

═══════════════════════════════════════════════════════════════
PARAMETERS
═══════════════════════════════════════════════════════════════

alpha           : noise increase rate (default 0.15 = 15%)
beta            : noise decrease rate (default 0.08 = 8%)
                  Asymmetric: we increase privacy faster than we decrease it
threshold_high  : convergence rate above which we increase noise (default 0.05)
threshold_low   : convergence rate below which we decrease noise (default 0.01)
noise_min       : minimum allowed noise multiplier (default 0.3)
noise_max       : maximum allowed noise multiplier (default 3.0)
clip_min        : minimum clipping norm (default 0.5)
clip_max        : maximum clipping norm (default 5.0)
epsilon_budget  : total allowed epsilon before forcing max noise (default 10.0)
warmup_rounds   : rounds before adaptation starts (default 2)
                  Allows model to stabilise before we start adapting
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class AdaptiveControllerConfig:
    """All tunable parameters for the adaptive controller."""

    # Initial DP parameters
    initial_noise_multiplier: float = 0.5
    initial_max_grad_norm:    float = 1.2
    target_delta:             float = 1e-5

    # Adaptation rates (asymmetric — increase privacy faster than decrease)
    alpha: float = 0.10   # noise increase rate per round
    beta:  float = 0.08   # noise decrease rate per round

    # Convergence thresholds
    threshold_high: float = 0.05   # acc improvement > 5% → increase noise
    threshold_low:  float = 0.01   # acc improvement < 1% → decrease noise

    # Hard bounds on noise multiplier
    noise_min: float = 0.3
    noise_max: float = 2.0

    # Hard bounds on clipping norm
    clip_min: float = 0.5
    clip_max: float = 5.0

    # Budget guard: if cumulative epsilon > epsilon_budget, force noise increase
    epsilon_budget: float = 50.0

    # Warmup: don't adapt for the first N rounds (allow model to stabilise)
    warmup_rounds: int = 5


@dataclass
class ControllerDecision:
    """The output of one controller step — what changed and why."""
    round_num:          int
    prev_noise:         float
    new_noise:          float
    prev_clip:          float
    new_clip:           float
    convergence_rate:   float
    budget_pressure:    float
    reason:             str       # human-readable explanation
    action:             str       # "INCREASE_PRIVACY" | "DECREASE_NOISE" | "MAINTAIN" | "WARMUP" | "BUDGET_GUARD"
    current_acc:        float
    prev_acc:           float
    cumulative_epsilon: float


class AdaptivePrivacyController:
    """
    Convergence-Aware Adaptive DP Controller.

    Usage:
        controller = AdaptivePrivacyController(config)
        controller.initialize()

        # After each FL round:
        decision = controller.observe_and_update(
            round_num=1,
            current_acc=0.21,
            current_loss=2.1,
            cumulative_epsilon=0.62,
        )
        new_dp_config = controller.get_current_dp_config()
    """

    def __init__(self, config: AdaptiveControllerConfig) -> None:
        self.config = config
        self._noise:   float = config.initial_noise_multiplier
        self._clip:    float = config.initial_max_grad_norm
        self._prev_acc: Optional[float] = None
        self._history: list[ControllerDecision] = []

    def initialize(self) -> None:
        """Reset controller state. Call before starting a new experiment."""
        self._noise    = self.config.initial_noise_multiplier
        self._clip     = self.config.initial_max_grad_norm
        self._prev_acc = None
        self._history  = []

    def observe_and_update(
        self,
        round_num: int,
        current_acc: float,
        current_loss: float,
        cumulative_epsilon: float,
    ) -> ControllerDecision:
        """
        Observe round metrics and compute new DP parameters.

        Parameters
        ----------
        round_num          : current FL round number (1-indexed)
        current_acc        : test accuracy this round (0.0 – 1.0)
        current_loss       : training loss this round
        cumulative_epsilon : total epsilon consumed so far

        Returns
        -------
        ControllerDecision with new noise_multiplier and max_grad_norm
        """
        cfg = self.config
        prev_noise = self._noise
        prev_clip  = self._clip
        prev_acc   = self._prev_acc if self._prev_acc is not None else current_acc

        # ── Step 1: Convergence rate ──────────────────────────────────────
        convergence_rate = (current_acc - prev_acc) / (prev_acc + 1e-8)

        # ── Step 2: Budget pressure ───────────────────────────────────────
        budget_pressure = cumulative_epsilon / max(cfg.epsilon_budget, 1e-8)

        # ── Step 3: Warmup guard ──────────────────────────────────────────
        if round_num <= cfg.warmup_rounds:
            new_noise = self._noise
            new_clip  = self._clip
            action    = "WARMUP"
            reason    = (
                f"Round {round_num} <= warmup_rounds ({cfg.warmup_rounds}). "
                f"No adaptation yet."
            )

        # ── Step 4: Budget guard ──────────────────────────────────────────
        elif budget_pressure > 0.8:
            new_noise = min(self._noise * (1.0 + cfg.alpha), cfg.noise_max)
            new_clip  = self._clip  # don't change clip under budget pressure
            action    = "BUDGET_GUARD"
            reason    = (
                f"Budget pressure {budget_pressure:.2f} > 0.8. "
                f"Forcing noise increase to protect remaining budget."
            )

        # ── Step 5: Fast convergence → increase privacy ───────────────────
        elif convergence_rate > cfg.threshold_high:
            new_noise = min(self._noise * (1.0 + cfg.alpha), cfg.noise_max)
            new_clip  = max(self._clip  * (1.0 + cfg.alpha), cfg.clip_min)
            new_clip  = min(new_clip, cfg.clip_max)
            action    = "INCREASE_PRIVACY"
            reason    = (
                f"Convergence rate {convergence_rate:.4f} > threshold_high "
                f"({cfg.threshold_high}). Model learning fast — increasing noise."
            )

        # ── Step 6: Slow convergence → reduce noise ───────────────────────
        elif convergence_rate < cfg.threshold_low:
            new_noise = max(self._noise * (1.0 - cfg.beta), cfg.noise_min)
            new_clip  = max(self._clip  * (1.0 - cfg.beta), cfg.clip_min)
            action    = "DECREASE_NOISE"
            reason    = (
                f"Convergence rate {convergence_rate:.4f} < threshold_low "
                f"({cfg.threshold_low}). Model converging slowly — reducing noise."
            )

        # ── Step 7: Stable convergence → maintain ────────────────────────
        else:
            new_noise = self._noise
            new_clip  = self._clip
            action    = "MAINTAIN"
            reason    = (
                f"Convergence rate {convergence_rate:.4f} in stable range "
                f"[{cfg.threshold_low}, {cfg.threshold_high}]. Maintaining noise."
            )

        # ── Apply hard bounds ─────────────────────────────────────────────
        new_noise = float(max(cfg.noise_min, min(cfg.noise_max, new_noise)))
        new_clip  = float(max(cfg.clip_min,  min(cfg.clip_max,  new_clip)))

        # ── Update state ──────────────────────────────────────────────────
        self._noise    = new_noise
        self._clip     = new_clip
        self._prev_acc = current_acc

        decision = ControllerDecision(
            round_num=round_num,
            prev_noise=round(prev_noise, 6),
            new_noise=round(new_noise, 6),
            prev_clip=round(prev_clip, 6),
            new_clip=round(new_clip, 6),
            convergence_rate=round(convergence_rate, 6),
            budget_pressure=round(budget_pressure, 6),
            reason=reason,
            action=action,
            current_acc=round(current_acc, 6),
            prev_acc=round(prev_acc, 6),
            cumulative_epsilon=round(cumulative_epsilon, 6),
        )
        self._history.append(decision)
        return decision

    def get_current_dp_config(self) -> dict:
        """Return current DP parameters as a dict for passing to clients."""
        return {
            "dp-noise-multiplier": self._noise,
            "dp-max-grad-norm":    self._clip,
            "dp-target-delta":     self.config.target_delta,
        }

    @property
    def current_noise(self) -> float:
        return self._noise

    @property
    def current_clip(self) -> float:
        return self._clip

    @property
    def history(self) -> list[ControllerDecision]:
        return list(self._history)

    def summary(self) -> dict:
        """Return a summary of all controller decisions."""
        return {
            "total_rounds": len(self._history),
            "final_noise":  round(self._noise, 6),
            "final_clip":   round(self._clip, 6),
            "actions": {
                "WARMUP":           sum(1 for d in self._history if d.action == "WARMUP"),
                "INCREASE_PRIVACY": sum(1 for d in self._history if d.action == "INCREASE_PRIVACY"),
                "DECREASE_NOISE":   sum(1 for d in self._history if d.action == "DECREASE_NOISE"),
                "MAINTAIN":         sum(1 for d in self._history if d.action == "MAINTAIN"),
                "BUDGET_GUARD":     sum(1 for d in self._history if d.action == "BUDGET_GUARD"),
            },
            "decisions": [
                {
                    "round":             d.round_num,
                    "action":            d.action,
                    "prev_noise":        d.prev_noise,
                    "new_noise":         d.new_noise,
                    "prev_clip":         d.prev_clip,
                    "new_clip":          d.new_clip,
                    "convergence_rate":  d.convergence_rate,
                    "budget_pressure":   d.budget_pressure,
                    "current_acc":       d.current_acc,
                    "reason":            d.reason,
                }
                for d in self._history
            ],
        }
