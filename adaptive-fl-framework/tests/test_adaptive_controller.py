"""
tests/test_adaptive_controller.py

Deterministic unit tests for the AdaptivePrivacyController.

Given the same inputs, the controller must always produce the same outputs.
These tests verify every decision branch of the algorithm.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy.adaptive_controller import (
    AdaptivePrivacyController,
    AdaptiveControllerConfig,
    ControllerDecision,
)


def _make_controller(
    initial_noise=1.0,
    initial_clip=1.0,
    alpha=0.15,
    beta=0.08,
    threshold_high=0.05,
    threshold_low=0.01,
    noise_min=0.3,
    noise_max=3.0,
    epsilon_budget=10.0,
    warmup_rounds=2,
) -> AdaptivePrivacyController:
    cfg = AdaptiveControllerConfig(
        initial_noise_multiplier=initial_noise,
        initial_max_grad_norm=initial_clip,
        alpha=alpha,
        beta=beta,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        noise_min=noise_min,
        noise_max=noise_max,
        epsilon_budget=epsilon_budget,
        warmup_rounds=warmup_rounds,
    )
    ctrl = AdaptivePrivacyController(cfg)
    ctrl.initialize()
    return ctrl


def test_warmup_no_change():
    """During warmup rounds, noise must not change."""
    ctrl = _make_controller(initial_noise=1.0, warmup_rounds=2)
    d1 = ctrl.observe_and_update(round_num=1, current_acc=0.20, current_loss=2.3, cumulative_epsilon=0.5)
    d2 = ctrl.observe_and_update(round_num=2, current_acc=0.30, current_loss=2.1, cumulative_epsilon=1.0)
    assert d1.action == "WARMUP", f"Expected WARMUP, got {d1.action}"
    assert d2.action == "WARMUP", f"Expected WARMUP, got {d2.action}"
    assert d1.new_noise == 1.0
    assert d2.new_noise == 1.0
    print(f"  [PASS] warmup: noise stays at 1.0 for rounds 1-2")


def test_fast_convergence_increases_noise():
    """When accuracy improves > threshold_high, noise must increase."""
    ctrl = _make_controller(initial_noise=1.0, warmup_rounds=0, threshold_high=0.05, alpha=0.15)
    # Round 1 sets prev_acc = 0.20 (convergence_rate = 0 -> DECREASE_NOISE since 0 < threshold_low)
    ctrl.observe_and_update(round_num=1, current_acc=0.20, current_loss=2.3, cumulative_epsilon=0.3)
    noise_after_r1 = ctrl.current_noise
    # Round 2: acc goes from 0.20 to 0.30 -> rate = 0.50 >> threshold_high=0.05
    d = ctrl.observe_and_update(round_num=2, current_acc=0.30, current_loss=2.0, cumulative_epsilon=0.6)
    assert d.action == "INCREASE_PRIVACY", f"Expected INCREASE_PRIVACY, got {d.action}"
    assert d.new_noise > d.prev_noise, "Noise should increase"
    expected = round(noise_after_r1 * 1.15, 6)
    assert abs(d.new_noise - expected) < 1e-5, f"Expected {expected}, got {d.new_noise}"
    print(f"  [PASS] fast convergence: noise {d.prev_noise} -> {d.new_noise} (action={d.action})")


def test_slow_convergence_decreases_noise():
    """When accuracy barely improves < threshold_low, noise must decrease."""
    ctrl = _make_controller(initial_noise=1.0, warmup_rounds=0, threshold_low=0.01, beta=0.08)
    # Round 1 sets prev_acc = 0.30
    ctrl.observe_and_update(round_num=1, current_acc=0.30, current_loss=2.0, cumulative_epsilon=0.5)
    # Round 2: acc goes from 0.30 to 0.302 → rate = 0.002/0.30 = 0.0067 < 0.01
    d = ctrl.observe_and_update(round_num=2, current_acc=0.302, current_loss=1.99, cumulative_epsilon=1.0)
    assert d.action == "DECREASE_NOISE", f"Expected DECREASE_NOISE, got {d.action}"
    assert d.new_noise < d.prev_noise, "Noise should decrease"
    expected = round(d.prev_noise * (1.0 - 0.08), 6)
    assert abs(d.new_noise - expected) < 1e-5, f"Expected {expected}, got {d.new_noise}"
    print(f"  [PASS] slow convergence: noise {d.prev_noise} -> {d.new_noise} (action={d.action})")


def test_stable_convergence_maintains_noise():
    """When convergence rate is in the stable band, noise must not change."""
    ctrl = _make_controller(
        initial_noise=1.0, warmup_rounds=0,
        threshold_low=0.01, threshold_high=0.05
    )
    # Round 1 sets prev_acc = 0.30
    ctrl.observe_and_update(round_num=1, current_acc=0.30, current_loss=2.0, cumulative_epsilon=0.5)
    # Round 2: acc goes from 0.30 to 0.309 → rate = 0.009/0.30 = 0.03 (in [0.01, 0.05])
    d = ctrl.observe_and_update(round_num=2, current_acc=0.309, current_loss=1.95, cumulative_epsilon=1.0)
    assert d.action == "MAINTAIN", f"Expected MAINTAIN, got {d.action}"
    assert d.new_noise == d.prev_noise, "Noise should not change"
    print(f"  [PASS] stable convergence: noise stays at {d.new_noise} (action={d.action})")


def test_budget_guard_forces_noise_increase():
    """When budget_pressure > 0.8, noise must increase regardless of convergence."""
    ctrl = _make_controller(initial_noise=1.0, warmup_rounds=0, epsilon_budget=10.0, alpha=0.15)
    # Round 1 sets prev_acc
    ctrl.observe_and_update(round_num=1, current_acc=0.20, current_loss=2.3, cumulative_epsilon=0.5)
    # Round 2: cumulative_epsilon = 9.0 → budget_pressure = 0.9 > 0.8
    d = ctrl.observe_and_update(round_num=2, current_acc=0.201, current_loss=2.29, cumulative_epsilon=9.0)
    assert d.action == "BUDGET_GUARD", f"Expected BUDGET_GUARD, got {d.action}"
    assert d.new_noise > d.prev_noise
    print(f"  [PASS] budget guard: noise {d.prev_noise} -> {d.new_noise} (action={d.action})")


def test_noise_hard_bounds_respected():
    """Noise must never exceed noise_max or go below noise_min."""
    ctrl = _make_controller(
        initial_noise=2.9, warmup_rounds=0,
        noise_max=3.0, alpha=0.15, threshold_high=0.0
    )
    # Fast convergence would push noise above 3.0 → must be clamped
    d = ctrl.observe_and_update(round_num=1, current_acc=0.50, current_loss=1.5, cumulative_epsilon=0.5)
    assert d.new_noise <= 3.0, f"Noise {d.new_noise} exceeds noise_max=3.0"
    print(f"  [PASS] noise upper bound: noise clamped to {d.new_noise} (max=3.0)")

    ctrl2 = _make_controller(
        initial_noise=0.35, warmup_rounds=0,
        noise_min=0.3, beta=0.08, threshold_low=1.0  # always triggers decrease
    )
    ctrl2.observe_and_update(round_num=1, current_acc=0.20, current_loss=2.3, cumulative_epsilon=0.1)
    d2 = ctrl2.observe_and_update(round_num=2, current_acc=0.201, current_loss=2.29, cumulative_epsilon=0.2)
    assert d2.new_noise >= 0.3, f"Noise {d2.new_noise} below noise_min=0.3"
    print(f"  [PASS] noise lower bound: noise clamped to {d2.new_noise} (min=0.3)")


def test_determinism():
    """Same inputs must always produce the same outputs."""
    def run_sequence():
        ctrl = _make_controller(initial_noise=1.0, warmup_rounds=1)
        decisions = []
        accs = [0.20, 0.30, 0.31, 0.50, 0.51]
        eps  = [0.5,  1.0,  1.5,  2.0,  2.5]
        for i, (acc, ep) in enumerate(zip(accs, eps)):
            d = ctrl.observe_and_update(i+1, acc, 2.0 - i*0.1, ep)
            decisions.append((d.action, round(d.new_noise, 6)))
        return decisions

    run1 = run_sequence()
    run2 = run_sequence()
    assert run1 == run2, f"Non-deterministic! run1={run1}, run2={run2}"
    print(f"  [PASS] determinism: two identical runs produce identical decisions")
    for i, (action, noise) in enumerate(run1):
        print(f"         Round {i+1}: {action:20s}  noise={noise}")


def test_summary_counts():
    """Summary action counts must match actual decisions."""
    ctrl = _make_controller(initial_noise=1.0, warmup_rounds=1)
    ctrl.observe_and_update(1, 0.20, 2.3, 0.5)   # WARMUP
    ctrl.observe_and_update(2, 0.40, 2.0, 1.0)   # INCREASE_PRIVACY (rate=1.0)
    ctrl.observe_and_update(3, 0.401, 1.9, 1.5)  # DECREASE_NOISE (rate~0.0025)
    s = ctrl.summary()
    assert s["actions"]["WARMUP"] == 1
    assert s["actions"]["INCREASE_PRIVACY"] == 1
    assert s["actions"]["DECREASE_NOISE"] == 1
    print(f"  [PASS] summary counts: {s['actions']}")


if __name__ == "__main__":
    print("\nRunning Adaptive Controller unit tests...\n")
    test_warmup_no_change()
    test_fast_convergence_increases_noise()
    test_slow_convergence_decreases_noise()
    test_stable_convergence_maintains_noise()
    test_budget_guard_forces_noise_increase()
    test_noise_hard_bounds_respected()
    test_determinism()
    test_summary_counts()
    print("\nAll adaptive controller tests passed.\n")
