"""
run_tests.py — Comprehensive Test Suite Runner for Adaptive FL Framework.
Runs all phase unit tests in order and reports overall status.
"""

import subprocess
import sys
from pathlib import Path

TESTS = [
    ("Phase 1: Baseline Architecture & Loops", "tests/test_phase1_baseline.py"),
    ("Phase 2: Differential Privacy (Opacus)", "tests/test_phase2_dp.py"),
    ("Phase 3: Secure Aggregation (SecAgg+)", "tests/test_phase3_secagg.py"),
    ("Phase 4: Adaptive Privacy Controller", "tests/test_adaptive_controller.py"),
    ("Phase 7: Evaluation & MIA Privacy Auditor", "tests/test_phase7_evaluation.py"),
    ("Phase 8: Dataset Manager & Model Factory", "tests/test_phase8_dataset_model.py"),
    ("Phase 9: Experiment Runner & Orchestrator", "tests/test_phase9_experiment_runner.py"),
]

def main() -> int:
    script_dir = Path(__file__).parent.resolve()
    print("=" * 65)
    print("  Adaptive FL Framework - Automated Test Suite Runner")
    print("=" * 65 + "\n")

    failed = []
    for idx, (name, rel_path) in enumerate(TESTS, 1):
        test_file = script_dir / rel_path
        if not test_file.exists():
            print(f"[{idx}/{len(TESTS)}] [FAIL] Missing test file: {rel_path}")
            failed.append(name)
            continue

        print(f"[{idx}/{len(TESTS)}] Running {name}...")
        result = subprocess.run([sys.executable, str(test_file)], cwd=str(script_dir))
        if result.returncode != 0:
            print(f"  [ERROR] {name} failed with exit code {result.returncode}\n")
            failed.append(name)
        else:
            print(f"  [SUCCESS] {name} passed.\n")

    print("=" * 65)
    if failed:
        print(f"  FAILED ({len(failed)}/{len(TESTS)} failed):")
        for f in failed:
            print(f"    - {f}")
        print("=" * 65)
        return 1
    else:
        print(f"  ALL {len(TESTS)} TEST SUITES PASSED SUCCESSFULLY!")
        print("=" * 65)
        return 0

if __name__ == "__main__":
    sys.exit(main())
