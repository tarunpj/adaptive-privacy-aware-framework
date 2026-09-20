"""
run_tests.py — Comprehensive Test Suite Runner for Adaptive FL Framework.
Runs all phase unit tests in order and reports overall status.
"""

import subprocess
import sys
from pathlib import Path

def main() -> int:
    script_dir = Path(__file__).parent.resolve()
    target_dir = script_dir / "adaptive-fl-framework"
    if not target_dir.exists():
        target_dir = script_dir

    runner = target_dir / "run_tests.py"
    return subprocess.run([sys.executable, str(runner)], cwd=str(target_dir)).returncode

if __name__ == "__main__":
    sys.exit(main())
