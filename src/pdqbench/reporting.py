"""Aggregation and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import confidence_half_width


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["case", "rank", "training_noise", "sensor_noise", "sensor_count", "sensor_policy", "method"]
    metric_cols = [
        "sensor_rel_state_error",
        "sensor_rel_energy_norm_error",
        "sensor_rel_operator_residual",
        "full_test_rel_error",
        "train_clean_rel_error",
        "offline_fit_time_sec",
        "online_time_per_case_sec",
        "online_speedup_vs_direct_solve",
        "storage_ratio",
        "kappa_D",
        "sweeps",
    ]
    rows = []
    for keys, g in df.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n"] = len(g)
        for col in metric_cols:
            vals = g[col].replace([float("inf"), -float("inf")], pd.NA).dropna().astype(float)
            rec[f"{col}_mean"] = vals.mean() if len(vals) else float("nan")
            rec[f"{col}_ci95"] = confidence_half_width(vals) if len(vals) else float("nan")
            rec[f"{col}_median"] = vals.median() if len(vals) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols)


def make_main_table(agg: pd.DataFrame) -> pd.DataFrame:
    main = agg[(agg["sensor_policy"] == "variance")].copy()
    cols = [
        "case", "rank", "training_noise", "sensor_noise", "sensor_count", "method",
        "sensor_rel_state_error_mean", "sensor_rel_energy_norm_error_mean",
        "sensor_rel_operator_residual_mean", "full_test_rel_error_mean",
        "online_speedup_vs_direct_solve_mean", "storage_ratio_mean", "kappa_D_mean",
    ]
    return main[cols].sort_values(["case", "rank", "sensor_count", "sensor_noise", "method"])


def save_plots(agg: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    main = agg[agg["sensor_policy"] == "variance"]

    for metric, fname, ylabel in [
        ("sensor_rel_state_error_mean", "sensor_state_error.png", "Sparse-sensor state error"),
        ("sensor_rel_energy_norm_error_mean", "sensor_energy_error.png", "Energy-norm error"),
        ("sensor_rel_operator_residual_mean", "operator_residual.png", "Operator residual"),
        ("online_speedup_vs_direct_solve_mean", "online_speedup.png", "Online speed-up vs direct solve"),
    ]:
        for case, sub_case in main.groupby("case"):
            plt.figure(figsize=(8, 4.8))
            for method, sub in sub_case.groupby("method"):
                sub = sub.sort_values("sensor_count")
                plt.plot(sub["sensor_count"], sub[metric], marker="o", label=method)
            plt.xlabel("Number of sensors")
            plt.ylabel(ylabel)
            plt.title(f"{case}: {ylabel}")
            plt.legend(fontsize=7)
            plt.grid(True, linewidth=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{case}_{fname}", dpi=220)
            plt.close()

    kappa = main.dropna(subset=["kappa_D_mean"])
    if not kappa.empty:
        plt.figure(figsize=(8, 4.8))
        for method, sub in kappa.groupby("method"):
            if "PDQ" not in method:
                continue
            plt.plot(sub["rank"], sub["kappa_D_mean"], marker="o", label=method)
        plt.yscale("log")
        plt.xlabel("Rank")
        plt.ylabel("Core condition number")
        plt.title("PDQ core conditioning and ablations")
        plt.legend(fontsize=8)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "pdq_core_conditioning.png", dpi=220)
        plt.close()
