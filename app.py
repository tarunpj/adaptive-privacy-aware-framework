"""
app.py — Root Entry Point for Streamlit Community Cloud and Cloud Deployments.

Adaptive Privacy-Aware Federated Learning Framework Dashboard.
Seamlessly delegates execution to adaptive-fl-framework/dashboard/app.py.
"""

import sys
from pathlib import Path

# Resolve absolute repository root and framework paths
REPO_ROOT = Path(__file__).resolve().parent
FRAMEWORK_DIR = REPO_ROOT / "adaptive-fl-framework"

# Ensure adaptive-fl-framework is on sys.path
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

# Target dashboard script
DASHBOARD_SCRIPT = FRAMEWORK_DIR / "dashboard" / "app.py"

if not DASHBOARD_SCRIPT.exists():
    raise FileNotFoundError(f"Dashboard script not found at {DASHBOARD_SCRIPT}")

# Execute the dashboard script in the current global context
with open(DASHBOARD_SCRIPT, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(DASHBOARD_SCRIPT), "exec")
    exec(code, globals())
