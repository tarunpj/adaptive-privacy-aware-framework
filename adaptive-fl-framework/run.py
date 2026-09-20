"""
run.py  —  Launcher for the Adaptive FL Framework

Usage:
    python run.py                                         # Phase 1: baseline FedAvg
    python run.py --dp                                    # Phase 2: FedAvg + DP
    python run.py --dp --noise 1.2 --clip 1.0             # Phase 2: custom DP params
    python run.py --secagg                                # Phase 3: FedAvg + DP + SecAgg+
    python run.py --adaptive                              # Phase 4: Adaptive DP
    python run.py --adaptive --alpha 0.1 --warmup 1      # Phase 4: custom controller
    python run.py --rounds 5 --clients 10 --dp            # custom rounds/clients
    python run.py --experiment-id my_run                  # custom results filename
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Dynamic and robust detection of venv Scripts/bin directory & flwr executable
_candidates = [
    Path(sys.executable).parent,
    SCRIPT_DIR.parent / ".venv" / "Scripts",
    SCRIPT_DIR.parent / ".venv" / "bin",
    SCRIPT_DIR / ".venv" / "Scripts",
    SCRIPT_DIR / ".venv" / "bin",
]

FLWR_EXE = None
VENV_SCRIPTS = None

for cand in _candidates:
    for exe_name in ["flwr.exe", "flwr"]:
        p = cand / exe_name
        if p.is_file():
            FLWR_EXE = p
            VENV_SCRIPTS = cand
            break
    if FLWR_EXE:
        break

if not FLWR_EXE:
    which_flwr = shutil.which("flwr")
    if which_flwr:
        FLWR_EXE = Path(which_flwr)
        VENV_SCRIPTS = FLWR_EXE.parent
    else:
        VENV_SCRIPTS = SCRIPT_DIR.parent / ".venv" / "Scripts"
        FLWR_EXE = VENV_SCRIPTS / "flwr.exe"

env = os.environ.copy()
if VENV_SCRIPTS:
    env["PATH"] = str(VENV_SCRIPTS) + os.pathsep + env.get("PATH", "")
env["PYTHONIOENCODING"] = "utf-8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adaptive FL Framework experiment")

    # FL config
    parser.add_argument("--rounds",        type=int,   default=None)
    parser.add_argument("--clients",       type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--local-epochs",  type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--experiment-id", type=str,   default=None)

    # DP config
    parser.add_argument("--dp",    action="store_true", default=False)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--clip",  type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)

    # SecAgg config
    parser.add_argument("--secagg",    action="store_true", default=False)
    parser.add_argument("--shares",    type=int,   default=None)
    parser.add_argument("--threshold", type=int,   default=None)

    # Adaptive controller
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument("--alpha",    type=float, default=None)
    parser.add_argument("--beta",     type=float, default=None)
    parser.add_argument("--warmup",   type=int,   default=None)

    args = parser.parse_args()

    # Mode implications
    if args.secagg or args.adaptive:
        args.dp = True

    # ── Select toml ───────────────────────────────────────────────────────────
    if args.secagg:
        alt_toml = SCRIPT_DIR / "pyproject_secagg.toml"
    elif args.adaptive:
        alt_toml = SCRIPT_DIR / "pyproject_adaptive.toml"
    else:
        alt_toml = None

    # ── Build run-config overrides ────────────────────────────────────────────
    overrides = []

    if args.rounds:
        overrides.append(f"num-server-rounds={args.rounds}")
    if args.lr:
        overrides.append(f"learning-rate={args.lr}")
    if args.local_epochs:
        overrides.append(f"local-epochs={args.local_epochs}")
    if args.batch_size:
        overrides.append(f"batch-size={args.batch_size}")
    if args.experiment_id:
        overrides.append(f'experiment-id="{args.experiment_id}"')

    if args.secagg:
        if args.noise     is not None: overrides.append(f"dp-noise-multiplier={args.noise}")
        if args.clip      is not None: overrides.append(f"dp-max-grad-norm={args.clip}")
        if args.delta     is not None: overrides.append(f"dp-target-delta={args.delta}")
        if args.shares    is not None: overrides.append(f"secagg-num-shares={args.shares}")
        if args.threshold is not None: overrides.append(f"secagg-threshold={args.threshold}")
        overrides.append(f"secagg-num-sampled-clients={args.clients}")
    elif args.adaptive:
        if args.noise  is not None: overrides.append(f"dp-noise-multiplier={args.noise}")
        if args.clip   is not None: overrides.append(f"dp-max-grad-norm={args.clip}")
        if args.delta  is not None: overrides.append(f"dp-target-delta={args.delta}")
        if args.alpha  is not None: overrides.append(f"ctrl-alpha={args.alpha}")
        if args.beta   is not None: overrides.append(f"ctrl-beta={args.beta}")
        if args.warmup is not None: overrides.append(f"ctrl-warmup-rounds={args.warmup}")
    else:
        overrides.append(f"dp-enabled={str(args.dp).lower()}")
        if args.noise is not None: overrides.append(f"dp-noise-multiplier={args.noise}")
        if args.clip  is not None: overrides.append(f"dp-max-grad-norm={args.clip}")
        if args.delta is not None: overrides.append(f"dp-target-delta={args.delta}")

    fed_overrides = [f"num-supernodes={args.clients}"]

    # ── Swap toml if needed, run, restore ─────────────────────────────────────
    default_toml = SCRIPT_DIR / "pyproject.toml"
    backup_toml  = SCRIPT_DIR / "pyproject.toml.bak"
    swapped = False

    try:
        if alt_toml is not None:
            shutil.copy2(default_toml, backup_toml)
            shutil.copy2(alt_toml, default_toml)
            swapped = True

        cmd = [str(FLWR_EXE), "run", "."]
        if overrides:
            cmd += ["--run-config", " ".join(overrides)]
        if fed_overrides:
            cmd += ["--federation-config", " ".join(fed_overrides)]
        cmd += ["--stream"]

        if args.adaptive:
            mode = "Adaptive DP"
        elif args.secagg:
            mode = "DP + SecAgg+"
        elif args.dp:
            mode = "Fixed DP"
        else:
            mode = "Baseline FedAvg"

        print(f"\nMode: {mode}")
        print(f"Launching: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))

    finally:
        if swapped and backup_toml.exists():
            shutil.copy2(backup_toml, default_toml)
            backup_toml.unlink()

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
