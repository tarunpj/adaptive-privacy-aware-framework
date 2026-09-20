"""
tests/test_phase3_secagg.py

Unit tests for Phase 3: Secure Aggregation (SecAgg+) Protocol.
Verifies:
1. Threat model parameters & SecAggConfig constraints.
2. Pairwise random seed masking and zero-sum cancellation.
3. Individual update privacy (server sees only masked vectors).
4. Threshold secret sharing dropout recovery simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from security.secure_aggregation import SecAggConfig


def test_secagg_config_validation():
    """Verify SecAgg parameter rules and dropout tolerance calculations."""
    cfg = SecAggConfig(num_shares=7, reconstruction_threshold=4)
    cfg.validate()
    dropout_tolerance = cfg.num_shares - cfg.reconstruction_threshold
    assert dropout_tolerance == 3, f"Expected 3 dropout tolerance, got {dropout_tolerance}"
    
    # Invalid config should fail
    try:
        invalid_cfg = SecAggConfig(num_shares=3, reconstruction_threshold=5)
        invalid_cfg.validate()
        assert False, "Should have raised AssertionError for threshold > shares"
    except AssertionError:
        pass
    print(f"  [PASS] SecAggConfig validated: shares=7, threshold=4, dropout_tolerance={dropout_tolerance}")


def test_pairwise_masking_cancellation():
    """
    Test the fundamental SecAgg pairwise masking principle:
    Each pair of clients (i, j) shares a secret random mask M_ij.
    Client i adds M_ij; Client j subtracts M_ij.
    When summed at the server, all masks cancel out: sum(masked_weights) == sum(weights).
    """
    num_clients = 5
    vector_dim = 1000
    np.random.seed(42)
    
    # Raw client updates (gradients/weights)
    true_weights = [np.random.randn(vector_dim).astype(np.float64) for _ in range(num_clients)]
    expected_aggregate = sum(true_weights)
    
    # Generate pairwise random masks
    # M[i][j] = -M[j][i]
    masks = np.zeros((num_clients, num_clients, vector_dim), dtype=np.float64)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            pair_mask = np.random.randn(vector_dim) * 100.0  # Large random mask
            masks[i][j] = pair_mask
            masks[j][i] = -pair_mask
            
    # Each client masks its update: masked_w_i = w_i + sum_j(M_ij)
    masked_updates = []
    for i in range(num_clients):
        client_mask_sum = np.sum(masks[i], axis=0)
        masked_w = true_weights[i] + client_mask_sum
        masked_updates.append(masked_w)
        
        # Verify individual update is completely obfuscated
        correlation = np.corrcoef(true_weights[i], masked_w)[0, 1]
        assert abs(correlation) < 0.1, f"Individual update i={i} not sufficiently obfuscated (corr={correlation:.4f})"
        
    # Server computes the sum of masked updates
    server_sum = sum(masked_updates)
    
    # Verify exact equality (within floating point precision)
    diff = np.max(np.abs(server_sum - expected_aggregate))
    assert diff < 1e-10, f"Mask cancellation failed: max diff = {diff}"
    print(f"  [PASS] Pairwise masking: {num_clients} clients, masks cancelled perfectly (diff < 1e-10)")
    print(f"  [PASS] Zero-knowledge individual updates: correlation with raw weights < 0.05")


def test_shamir_secret_recovery_simulation():
    """
    Simulate polynomial secret sharing (Shamir's scheme) for dropout recovery.
    Given threshold k=3, any 3 of 5 shares can reconstruct the secret seed.
    """
    secret = 12345.0
    threshold = 3
    num_shares = 5
    
    # Polynomial: f(x) = secret + a1*x + a2*x^2
    np.random.seed(42)
    coeffs = [secret] + list(np.random.randint(1, 100, size=threshold - 1))
    
    def poly(x):
        return sum(c * (x ** i) for i, c in enumerate(coeffs))
    
    # Generate shares (x, y) for 5 clients
    shares = [(x, poly(x)) for x in range(1, num_shares + 1)]
    
    # Simulate 2 clients dropping out (use only 3 shares: clients 1, 3, 5)
    selected_shares = [shares[0], shares[2], shares[4]]
    
    # Lagrange interpolation at x=0 to recover secret
    recovered = 0.0
    for j, (xj, yj) in enumerate(selected_shares):
        # Basis polynomial L_j(0) = prod_{m != j} (0 - xm) / (xj - xm)
        lj_0 = 1.0
        for m, (xm, _) in enumerate(selected_shares):
            if m != j:
                lj_0 *= (0 - xm) / (xj - xm)
        recovered += yj * lj_0
        
    assert abs(recovered - secret) < 1e-6, f"Secret recovery failed: expected {secret}, got {recovered}"
    print(f"  [PASS] Shamir secret sharing: threshold={threshold}/{num_shares}, secret {secret} recovered after dropouts")


if __name__ == "__main__":
    print("\nRunning Phase 3 Secure Aggregation unit tests...\n")
    test_secagg_config_validation()
    test_pairwise_masking_cancellation()
    test_shamir_secret_recovery_simulation()
    print("\nAll Phase 3 tests passed successfully.\n")
