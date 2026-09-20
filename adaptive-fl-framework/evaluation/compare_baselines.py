"""
evaluation/compare_baselines.py

Automated 4-Way Multi-Baseline Benchmark Aggregator and Report Generator.

Compares:
1. Baseline FedAvg (No DP, No SecAgg)
2. FedAvg + Fixed DP (Opacus)
3. FedAvg + Fixed DP + SecAgg+
4. Proposed Adaptive DP Controller

Generates:
- Structured comparative benchmark CSV/JSON
- Publication-ready markdown comparison table
- Comparative multi-panel visualization plot
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_4way_comparison_report() -> Dict:
    """Generate side-by-side benchmark summary across all 4 methods."""
    methods = [
        {"name": "Standard FedAvg", "id": "baseline_fedavg", "dp": False, "secagg": False, "adaptive": False},
        {"name": "FedAvg + Fixed DP", "id": "fedavg_dp", "dp": True, "secagg": False, "adaptive": False},
        {"name": "FedAvg + Fixed DP + SecAgg+", "id": "fedavg_dp_sa", "dp": True, "secagg": True, "adaptive": False},
        {"name": "Proposed Adaptive DP", "id": "adaptive_dp", "dp": True, "secagg": False, "adaptive": True},
    ]

    rows = []
    dfs = {}

    for m in methods:
        json_path = RESULTS_DIR / f"{m['id']}_summary.json"
        csv_path = RESULTS_DIR / f"{m['id']}.csv"

        acc, loss, eps, time_s = "N/A", "N/A", "0.0", "N/A"
        
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                acc = round(float(data.get("final_test_acc", 0.0)) * 100, 2)
                loss = round(float(data.get("final_test_loss", 0.0)), 4)
                eps = round(float(data.get("final_epsilon", data.get("cumulative_epsilon", 0.0))), 4) if m["dp"] else "None (0.0)"
                time_s = round(float(data.get("total_time_s", 0.0)), 1)

        if csv_path.exists():
            dfs[m["name"]] = pd.read_csv(csv_path)

        rows.append({
            "Method": m["name"],
            "Final Accuracy (%)": acc,
            "Final Loss": loss,
            "Privacy Budget (ε)": eps,
            "Differential Privacy": "✅ Opacus" if m["dp"] else "❌ None",
            "Secure Aggregation": "✅ SecAgg+" if m["secagg"] else "❌ None",
            "Adaptive Noise": "🧠 Active" if m["adaptive"] else "❌ Fixed",
            "Sim Time (s)": time_s,
        })

    comp_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "multi_baseline_comparison.csv"
    comp_df.to_csv(out_csv, index=False)

    # Multi-panel comparative plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Adaptive Privacy FL: 4-Way Multi-Baseline Benchmark", fontsize=14, fontweight="bold")

    colors = {"Standard FedAvg": "#1f77b4", "FedAvg + Fixed DP": "#ff7f0e", "FedAvg + Fixed DP + SecAgg+": "#d62728", "Proposed Adaptive DP": "#2ca02c"}

    for name, df in dfs.items():
        c = colors.get(name, "#7f7f7f")
        if "test_acc_centralised" in df.columns:
            axes[0].plot(df["round"], df["test_acc_centralised"] * 100, marker="o", label=name, color=c, linewidth=2)
        elif "val_acc" in df.columns:
            axes[0].plot(df["round"], df["val_acc"] * 100, marker="s", linestyle="--", label=name, color=c, linewidth=2)

        if "test_loss_centralised" in df.columns:
            axes[1].plot(df["round"], df["test_loss_centralised"], marker="o", label=name, color=c, linewidth=2)
        elif "train_loss" in df.columns:
            axes[1].plot(df["round"], df["train_loss"], marker="s", linestyle="--", label=name, color=c, linewidth=2)

    axes[0].set_title("Test Accuracy (%) Across Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Loss Trajectory Across Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "multi_baseline_comparison_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\n4-Way Comparison Report saved -> {out_csv}")
    print(f"Comparison plot saved -> {plot_path}\n")
    return {"table": rows, "csv_path": str(out_csv), "plot_path": str(plot_path)}


if __name__ == "__main__":
    generate_4way_comparison_report()
